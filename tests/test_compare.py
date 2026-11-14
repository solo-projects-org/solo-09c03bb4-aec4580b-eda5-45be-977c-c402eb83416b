"""Tests for the two-report comparison (late-revision impact analysis)."""

import csv
import io
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.diff import (
    KIND_AMOUNT_CHANGED, KIND_CANCELLED, KIND_MOVED_OUT_OF_WINDOW,
)
from app.service import AppService

UTC = timezone.utc
DAY = 86400
HOUR = 3600


def ts(text: str) -> int:
    return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())


AUDIT = ts("2025-02-01T12:00:00Z")


def txn_csv(rows: list[tuple]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["event_id", "account_id", "occurred_at", "ingested_at",
                "amount", "action"])
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode()


def sample_csv(rows: list[tuple]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["sample_id", "account_id", "audit_at", "label"])
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode()


BASE_ROWS = [
    ("D1", "D", AUDIT - 3 * DAY, AUDIT - 10 * DAY, "100.00", "revise"),
    ("D2", "D", AUDIT - 2 * DAY, AUDIT - 9 * DAY, "40.00", "revise"),
    ("E1", "E", AUDIT - 4 * DAY, AUDIT - 8 * DAY, "25.00", "revise"),
    ("F1", "F", AUDIT - 1 * DAY, AUDIT - 6 * DAY, "77.00", "revise"),
]
LATE_ROWS = [
    # D1 amount 100 -> 140 while D2 is cancelled: net amount stays 140.00,
    # net count drops by one -> D is affected via count only.
    ("D1", "D", AUDIT - 2 * DAY, AUDIT - 1 * HOUR, "140.00", "revise"),
    ("D2", "D", AUDIT - 2 * DAY, AUDIT - 2 * HOUR, "0.00", "cancel"),
    # E1 cancelled -> E loses its only counted event.
    ("E1", "E", AUDIT - 4 * DAY, AUDIT - 3 * HOUR, "0.00", "cancel"),
]


class CompareServiceTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.svc = AppService(Settings(data_dir=self.dir, worker_poll_seconds=0.05))
        self.samples = sample_csv([
            ("SC-D", "D", AUDIT, 1),
            ("SC-E", "E", AUDIT, 0),
            ("SC-F", "F", AUDIT, 1),
        ])

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _run_one_report(self) -> str:
        """Create one report and run it synchronously, FIFO style."""
        pid = self.svc.create_report()["report_id"]
        claimed = self.svc.claim_next_report()
        self.assertEqual(claimed, pid)
        self.svc.run_report(pid)
        self.assertEqual(self.svc.get_report(pid)["status"], "ready")
        return pid

    def _drain_reports(self, expected: int) -> list[str]:
        """Run every already-queued pending report in FIFO order."""
        ids = []
        for _ in range(expected):
            pid = self.svc.claim_next_report()
            self.assertIsNotNone(pid)
            self.svc.run_report(pid)
            self.assertEqual(self.svc.get_report(pid)["status"], "ready")
            ids.append(pid)
        return ids

    def _ready_report(self) -> str:
        return self._run_one_report()

    def _build_pair(self):
        self.svc.upload_transactions("base.csv", txn_csv(BASE_ROWS))
        self.svc.upload_samples("samples.csv", self.samples)
        old = self._ready_report()
        self.svc.upload_transactions("late.csv", txn_csv(LATE_ROWS))
        new = self._ready_report()
        return old, new

    def test_compare_flags_changed_samples_with_published_values(self):
        old, new = self._build_pair()
        result = self.svc.compare_reports(old, new)
        by_id = {s["sample_id"]: s for s in result["affected_samples"]}
        self.assertEqual(set(by_id), {"SC-D", "SC-E"})
        self.assertEqual(result["affected_count"], 2)
        self.assertEqual(result["unchanged_count"], 1)
        self.assertEqual(result["event_only_change_count"], 0)

        d = by_id["SC-D"]
        # Values come from the published CSV rows.
        self.assertEqual(d["base"]["txn_count_7d"], "2")
        self.assertEqual(d["candidate"]["txn_count_7d"], "1")
        self.assertEqual(d["base"]["txn_amount_7d"], "140.00")
        self.assertEqual(d["candidate"]["txn_amount_7d"], "140.00")
        self.assertEqual(d["delta"]["count"], -1)
        self.assertEqual(d["delta"]["amount_text"], "0.00")
        kinds = {e["event_id"]: e["change_kind"] for e in d["events"]}
        self.assertEqual(kinds["D1"], KIND_AMOUNT_CHANGED)
        self.assertEqual(kinds["D2"], KIND_CANCELLED)

        e = by_id["SC-E"]
        self.assertEqual(e["base"]["txn_count_7d"], "1")
        self.assertEqual(e["candidate"]["txn_count_7d"], "0")
        self.assertEqual(e["delta"]["amount_text"], "-25.00")
        event = e["events"][0]
        self.assertEqual(event["event_id"], "E1")
        self.assertEqual(event["change_kind"], KIND_CANCELLED)
        # The new report sees both revisions; the cancel is the winner and is
        # attributed to the late transaction batch.
        self.assertEqual(len(event["candidate"]["visible_revisions"]), 2)
        winner = event["candidate"]["winner"]
        self.assertEqual(winner["action"], "cancel")
        self.assertFalse(winner["counted"])
        self.assertEqual(winner["filename"], "late.csv")
        # Old report only sees the original revise from the base batch.
        self.assertEqual(len(event["base"]["visible_revisions"]), 1)
        self.assertTrue(event["base"]["winner"]["counted"])

    def test_opposing_event_changes_with_identical_features_are_not_affected(self):
        # True feature-level offset with BOTH count and amount unchanged:
        #   * X3 is cancelled (-1 count, -30.00 amount);
        #   * X4's newest revision moves it from outside the window into it,
        #     so X4 newly counts (+1 count, +30.00 amount);
        #   * X1 keeps the same amount 30.00 but is retransmitted through a
        #     new revision with identical semantic value, which the ingest
        #     layer deduplicates, so it produces no event change at all.
        # Count nets to zero and amount nets to zero, yet two events
        # genuinely changed; the sample must stay out of the affected list.
        offset_base = [
            ("X1", "X", AUDIT - 3 * DAY, AUDIT - 9 * DAY, "30.00", "revise"),
            ("X2", "X", AUDIT - 2 * DAY, AUDIT - 9 * DAY, "30.00", "revise"),
            ("X3", "X", AUDIT - 1 * DAY, AUDIT - 9 * DAY, "30.00", "revise"),
            ("X4", "X", AUDIT - 30 * DAY, AUDIT - 9 * DAY, "30.00", "revise"),
        ]
        offset_late = [
            ("X3", "X", AUDIT - 1 * DAY, AUDIT - 2 * HOUR, "0.00", "cancel"),
            ("X4", "X", AUDIT - 1 * DAY, AUDIT - 3 * HOUR, "30.00", "revise"),
        ]
        samples = sample_csv([("SC-X", "X", AUDIT, 1)])
        self.svc.upload_transactions("base.csv", txn_csv(offset_base))
        self.svc.upload_samples("samples.csv", samples)
        old = self._ready_report()
        self.svc.upload_transactions("late.csv", txn_csv(offset_late))
        new = self._ready_report()

        result = self.svc.compare_reports(old, new)
        self.assertEqual(result["affected_count"], 0)
        self.assertEqual(result["event_only_change_count"], 1)
        only = result["event_only_samples"][0]
        self.assertEqual(only["sample_id"], "SC-X")
        self.assertEqual(only["base"]["txn_count_7d"], "3")
        self.assertEqual(only["candidate"]["txn_count_7d"], "3")
        self.assertEqual(only["base"]["txn_amount_7d"], "90.00")
        self.assertEqual(only["candidate"]["txn_amount_7d"], "90.00")
        self.assertEqual(only["delta"]["count"], 0)
        self.assertEqual(only["delta"]["amount_text"], "0.00")
        self.assertEqual({e["event_id"] for e in only["events"]},
                         {"X3", "X4"})

    def test_moved_out_of_window_uses_own_frozen_version(self):
        moved_base = [
            ("M1", "M", AUDIT - 1 * DAY, AUDIT - 6 * DAY, "100.00", "revise"),
        ]
        moved_late = [
            # New winner points outside the 7-day window; old value is not
            # inherited and the event drops out of the new report only.
            ("M1", "M", AUDIT - 30 * DAY, AUDIT - 1 * HOUR, "100.00", "revise"),
        ]
        self.svc.upload_transactions("base.csv", txn_csv(moved_base))
        self.svc.upload_samples("samples.csv",
                                sample_csv([("SC-M", "M", AUDIT, 1)]))
        old = self._ready_report()
        self.svc.upload_transactions("late.csv", txn_csv(moved_late))
        new = self._ready_report()
        result = self.svc.compare_reports(old, new)
        sample = result["affected_samples"][0]
        self.assertEqual(sample["base"]["txn_count_7d"], "1")
        self.assertEqual(sample["candidate"]["txn_count_7d"], "0")
        self.assertEqual(sample["events"][0]["change_kind"],
                         KIND_MOVED_OUT_OF_WINDOW)

    def test_identical_retransmit_creates_no_diff(self):
        self.svc.upload_transactions("base.csv", txn_csv(BASE_ROWS))
        self.svc.upload_samples("samples.csv", self.samples)
        old = self._ready_report()
        # Re-upload byte-identical content: idempotent, zero new revisions.
        again = self.svc.upload_transactions("base.csv", txn_csv(BASE_ROWS))
        self.assertEqual(again["stored_rows"], 0)
        self.assertEqual(again["skipped_same_value"], len(BASE_ROWS))
        new = self._ready_report()
        result = self.svc.compare_reports(old, new)
        self.assertEqual(result["affected_count"], 0)
        self.assertEqual(result["event_only_change_count"], 0)
        self.assertEqual(result["unchanged_count"], 3)

    def test_upload_between_compare_cannot_mix_in(self):
        old, new = self._build_pair()
        # A third batch lands after both reports exist; the comparison reads
        # only the pinned high-watermarks so it must be invisible.
        self.svc.upload_transactions("after.csv", txn_csv([
            ("ZZ1", "F", AUDIT - 1 * DAY, AUDIT - 100, "999.00", "revise")]))
        result = self.svc.compare_reports(old, new)
        all_shown = {s["sample_id"] for s in result["affected_samples"]}
        all_shown |= {s["sample_id"] for s in result["event_only_samples"]}
        # SC-F is unchanged and appears in neither diff list.
        self.assertNotIn("SC-F", all_shown)
        self.assertEqual(result["unchanged_count"], 1)

    # ------------------------------------------------------------------ #
    # Refusal paths
    # ------------------------------------------------------------------ #
    def test_refuses_different_sample_batches(self):
        self.svc.upload_transactions("base.csv", txn_csv(BASE_ROWS))
        self.svc.upload_samples("s1.csv",
                                sample_csv([("SC-D", "D", AUDIT, 1)]))
        old = self._ready_report()
        # A second sample batch extends the sample set behind a new
        # high-watermark; the two reports no longer share a sample batch.
        self.svc.upload_samples("s2.csv",
                                sample_csv([("SC-E", "E", AUDIT, 0)]))
        new = self._ready_report()
        with self.assertRaises(Exception) as ctx:
            self.svc.compare_reports(old, new)
        self.assertEqual(ctx.exception.code, "compare_refused")
        self.assertIn("样本批次", ctx.exception.message)

    def test_refuses_when_published_file_is_missing(self):
        old, new = self._build_pair()
        path, _ = self.svc.report_file(old)
        path.unlink()
        with self.assertRaises(Exception) as ctx:
            self.svc.compare_reports(old, new)
        self.assertEqual(ctx.exception.status, 409)
        self.assertIn("缺失", ctx.exception.message)

    def test_refuses_when_published_csv_was_tampered(self):
        old, new = self._build_pair()
        path, _ = self.svc.report_file(old)
        rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"))))
        rows[1][5] = "999"  # corrupt txn_count_7d of the first data row
        path.write_text("\n".join(",".join(r) for r in rows), encoding="utf-8")
        with self.assertRaises(Exception) as ctx:
            self.svc.compare_reports(old, new)
        self.assertEqual(ctx.exception.status, 409)

    def test_refuses_same_report_or_missing_report(self):
        old, new = self._build_pair()
        with self.assertRaises(Exception) as ctx:
            self.svc.compare_reports(old, old)
        self.assertEqual(ctx.exception.status, 400)
        with self.assertRaises(Exception) as ctx:
            self.svc.compare_reports(old, "deadbeef")
        self.assertEqual(ctx.exception.status, 404)

    def test_refuses_non_ready_report(self):
        self.svc.upload_transactions("base.csv", txn_csv(BASE_ROWS))
        self.svc.upload_samples("samples.csv", self.samples)
        old = self._ready_report()
        pending = self.svc.create_report()["report_id"]  # left queued
        with self.assertRaises(Exception) as ctx:
            self.svc.compare_reports(pending, old)
        self.assertEqual(ctx.exception.status, 409)
        self.assertIn("ready", ctx.exception.message)


class CompareApiTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        os.environ["DATA_DIR"] = self.dir
        from app.main import create_app
        self.app = create_app(Settings(data_dir=self.dir, worker_poll_seconds=0.05),
                              start_worker=False)
        self.client = TestClient(self.app)
        self.svc = self.app.state.service

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)
        os.environ.pop("DATA_DIR", None)

    def _ready(self) -> str:
        pid = self.client.post("/api/reports").json()["report_id"]
        self.svc.claim_next_report()
        self.svc.run_report(pid)
        return pid

    def test_api_and_html_compare_flow(self):
        self.svc.upload_transactions("base.csv", txn_csv(BASE_ROWS))
        self.svc.upload_samples("samples.csv", sample_csv([
            ("SC-D", "D", AUDIT, 1), ("SC-E", "E", AUDIT, 0)]))
        old = self._ready()
        self.svc.upload_transactions("late.csv", txn_csv(LATE_ROWS))
        new = self._ready()

        ready = self.client.get("/api/reports/ready")
        self.assertEqual(ready.status_code, 200)
        self.assertEqual({r["report_id"] for r in ready.json()["reports"]},
                         {old, new})

        resp = self.client.get("/api/compare",
                               params={"base": old, "candidate": new})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["sample_count"], 2)
        self.assertEqual(body["affected_count"], 2)
        self.assertTrue(body["base"]["output_sha256_verified"])

        # Refusal mapped to 409 with an explanatory message.
        bad = self.client.get("/api/compare", params={"base": old, "candidate": old})
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(bad.json()["error"], "bad_request")

        # HTML pages render the selection form and result blocks.
        page = self.client.get("/compare", params={"base": old, "candidate": new})
        self.assertEqual(page.status_code, 200)
        self.assertIn("特征发生变化的样本", page.text)
        self.assertIn("SC-D", page.text)
        self.assertIn("late.csv", page.text)
        form_page = self.client.get("/compare")
        self.assertEqual(form_page.status_code, 200)
        self.assertIn("选择报告", form_page.text)


if __name__ == "__main__":
    unittest.main()

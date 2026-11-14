"""Point-in-time diff between two published training reports.

The comparison answers one question: after late revisions arrived, which
samples' seven-day features changed, and which events caused the delta?

Hard rules enforced here:

* Only two reports pinned to the **same sample-batch set** may be compared.
* The old/new feature values displayed to the operator are read from the two
  published CSV files (never recomputed silently), then cross-checked against
  a fresh recomputation from each report's frozen high-watermark. A mismatch
  refuses the whole comparison.
* Per-event attribution is rebuilt independently for every frozen version.
  A cancellation or a revision moved outside the 7-day window is therefore
  evaluated against the revisions visible to *that* report only; batches
  uploaded later carry higher batch ids and are filtered out by
  ``batch_id <= txn_batch_hi``.
* A sample counts as affected only when ``txn_count_7d`` or
  ``txn_amount_7d`` actually changed. Opposing event-level changes that net
  to zero are surfaced separately, and same-value retransmits are never
  stored as new revisions (see ``service.upload_transactions``) so they can
  never produce an entry.
"""

from __future__ import annotations

import csv
import io
import sqlite3
from dataclasses import asdict, dataclass, field
from decimal import Decimal

from . import features
from .csvparse import WINDOW_SECONDS

# Stable machine codes plus the Chinese labels rendered by the UI.
KIND_AMOUNT_CHANGED = "amount_changed"
KIND_NEWLY_VISIBLE = "newly_visible"
KIND_MOVED_INTO_WINDOW = "moved_into_window"
KIND_REVIVED_AFTER_CANCEL = "revived_after_cancel"
KIND_CANCELLED = "cancelled"
KIND_MOVED_OUT_OF_WINDOW = "moved_out_of_window"
KIND_NO_LONGER_VISIBLE = "no_longer_visible"

CHANGE_LABELS = {
    KIND_AMOUNT_CHANGED: "金额变化",
    KIND_NEWLY_VISIBLE: "新修订可见并计入",
    KIND_MOVED_INTO_WINDOW: "修订改入七天窗口",
    KIND_REVIVED_AFTER_CANCEL: "撤销后被新修订替代，重新计入",
    KIND_CANCELLED: "最新修订为撤销，事件移出",
    KIND_MOVED_OUT_OF_WINDOW: "最新修订改到窗口外，事件移出",
    KIND_NO_LONGER_VISIBLE: "该版本无可见修订，事件移出",
}


class CompareError(Exception):
    """A refusal to compare; carries the HTTP status the API should return."""

    def __init__(self, message: str, status: int = 409, code: str = "compare_refused"):
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code

    def payload(self) -> dict:
        return {"error": self.code, "message": self.message}


@dataclass(frozen=True)
class RevisionView:
    """One revision of an event as visible to one frozen report version."""

    batch_id: int
    filename: str
    ingested_at: int
    occurred_at: int
    amount_text: str
    action: str
    is_winner: bool
    counted: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SideEvent:
    """Everything one report version sees for one event of one sample."""

    revisions: list[RevisionView] = field(default_factory=list)
    counted: bool = False
    amount_minor: int = 0

    @property
    def amount_text(self) -> str:
        return format_minor(self.amount_minor)

    @property
    def winner(self) -> RevisionView | None:
        for revision in self.revisions:
            if revision.is_winner:
                return revision
        return None

    def to_dict(self) -> dict:
        return {
            "counted": self.counted,
            "amount_text": self.amount_text,
            "winner": self.winner.to_dict() if self.winner else None,
            "visible_revisions": [r.to_dict() for r in self.revisions],
        }


@dataclass
class EventChange:
    event_id: str
    change_kind: str
    change_label: str
    base: SideEvent
    candidate: SideEvent

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "change_kind": self.change_kind,
            "change_label": self.change_label,
            "base": self.base.to_dict(),
            "candidate": self.candidate.to_dict(),
        }


@dataclass
class SampleDiff:
    sample_id: str
    account_id: str
    audit_at: int
    window_start_at: int
    window_end_at: int
    base_count: int
    candidate_count: int
    base_amount_minor: int
    candidate_amount_minor: int
    events: list[EventChange]

    @property
    def affected(self) -> bool:
        return (self.base_count != self.candidate_count
                or self.base_amount_minor != self.candidate_amount_minor)

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "account_id": self.account_id,
            "audit_at": self.audit_at,
            "window_start_at": self.window_start_at,
            "window_end_at": self.window_end_at,
            "base": {
                "txn_count_7d": str(self.base_count),
                "txn_amount_7d": format_minor(self.base_amount_minor),
            },
            "candidate": {
                "txn_count_7d": str(self.candidate_count),
                "txn_amount_7d": format_minor(self.candidate_amount_minor),
            },
            "delta": {
                "count": self.candidate_count - self.base_count,
                "amount_text": format_signed_minor(
                    self.candidate_amount_minor - self.base_amount_minor),
            },
            "events": [e.to_dict() for e in self.events],
        }


def format_minor(minor: int) -> str:
    return f"{Decimal(minor) / 100:.2f}"


def format_signed_minor(minor: int) -> str:
    if minor == 0:
        return "0.00"
    return f"{'-' if minor < 0 else '+'}{abs(Decimal(minor)) / 100:.2f}"


# --------------------------------------------------------------------------- #
# Published CSV parsing
# --------------------------------------------------------------------------- #
def parse_published_csv(data: bytes, report_label: str) -> dict[str, dict]:
    """Parse a published report CSV strictly, keyed by ``sample_id``.

    The values returned are exactly the bytes the operator can download, so
    the UI can claim truthfully that the comparison matches the published
    files. Any structural problem refuses the comparison.
    """
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CompareError(f"{report_label} 的发布 CSV 不是 UTF-8 编码，拒绝对比。") from exc
    try:
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        rows = list(reader)
    except csv.Error as exc:
        raise CompareError(f"{report_label} 的发布 CSV 结构损坏，拒绝对比：{exc}") from exc
    if not rows:
        raise CompareError(f"{report_label} 的发布 CSV 为空，拒绝对比。")
    if rows[0] != features.OUTPUT_HEADERS:
        raise CompareError(
            f"{report_label} 的发布 CSV 表头与当前口径不一致"
            f"（应为 {', '.join(features.OUTPUT_HEADERS)}），拒绝对比。")
    out: dict[str, dict] = {}
    for line_no, row in enumerate(rows[1:], start=2):
        if not row or all(not cell.strip() for cell in row):
            continue
        if len(row) != len(features.OUTPUT_HEADERS):
            raise CompareError(
                f"{report_label} 的发布 CSV 第 {line_no} 行列数不符，拒绝对比。")
        parsed = dict(zip(features.OUTPUT_HEADERS, row))
        try:
            audit_at = int(parsed["audit_at"])
            window_start = int(parsed["window_start_at"])
            window_end = int(parsed["window_end_at"])
            count = int(parsed["txn_count_7d"])
            amount = Decimal(parsed["txn_amount_7d"])
        except (ValueError, ArithmeticError) as exc:
            raise CompareError(
                f"{report_label} 的发布 CSV 第 {line_no} 行数值无法解析，拒绝对比。") from exc
        if amount != amount.quantize(Decimal("0.01")) or count < 0 or amount < 0:
            raise CompareError(
                f"{report_label} 的发布 CSV 第 {line_no} 行特征值不合规，拒绝对比。")
        sample_id = parsed["sample_id"]
        if sample_id in out:
            raise CompareError(
                f"{report_label} 的发布 CSV 中样本号 {sample_id} 重复，拒绝对比。")
        if window_start != audit_at - WINDOW_SECONDS or window_end != audit_at:
            raise CompareError(
                f"{report_label} 的发布 CSV 第 {line_no} 行窗口边界与口径不符，拒绝对比。")
        out[sample_id] = {
            "sample_id": sample_id,
            "account_id": parsed["account_id"],
            "audit_at": audit_at,
            "window_start_at": window_start,
            "window_end_at": window_end,
            "txn_count_7d": count,
            "amount_minor": int(amount * 100),
            "txn_amount_7d": parsed["txn_amount_7d"],
            "label": parsed["label"],
        }
    if not out:
        raise CompareError(f"{report_label} 的发布 CSV 没有数据行，拒绝对比。")
    return out


# --------------------------------------------------------------------------- #
# Frozen-version recomputation and per-event visibility
# --------------------------------------------------------------------------- #
def recompute_map(conn: sqlite3.Connection, txn_hi: int,
                  sample_hi: int) -> dict[str, features.ReportRow]:
    return {row.sample_id: row for row in
            features.compute_rows(conn, txn_hi, sample_hi)}


def _visible_events(conn: sqlite3.Connection, account_id: str, audit_at: int,
                    txn_hi: int) -> dict[str, SideEvent]:
    """Rebuild one sample's per-event state from one frozen version.

    The join and ordering mirror ``features._PINNED_SQL`` exactly: visibility
    is ``ingested_at <= audit_at`` inside ``batch_id <= txn_hi``, and the
    winner per event is the greatest ``(ingested_at, row id)``.
    """
    cur = conn.execute(
        """
        SELECT t.id, t.event_id, t.occurred_at, t.ingested_at,
               t.amount_minor, t.amount_text, t.action,
               t.batch_id, b.filename
        FROM transactions t
        JOIN batches b ON b.id = t.batch_id
        WHERE t.account_id = ?
          AND t.ingested_at <= ?
          AND t.batch_id <= ?
        ORDER BY t.event_id, t.ingested_at DESC, t.id DESC
        """,
        (account_id, audit_at, txn_hi),
    )
    events: dict[str, SideEvent] = {}
    winners: set[str] = set()
    for r in cur:
        side = events.setdefault(r["event_id"], SideEvent())
        is_winner = r["event_id"] not in winners
        if is_winner:
            winners.add(r["event_id"])
        counted = False
        if is_winner and r["action"] == "revise":
            window_start = audit_at - WINDOW_SECONDS
            if window_start <= r["occurred_at"] < audit_at:
                counted = True
                side.counted = True
                side.amount_minor = r["amount_minor"]
        side.revisions.append(RevisionView(
            batch_id=r["batch_id"],
            filename=r["filename"],
            ingested_at=r["ingested_at"],
            occurred_at=r["occurred_at"],
            amount_text=r["amount_text"],
            action=r["action"],
            is_winner=is_winner,
            counted=counted,
        ))
    return events


def _classify(base: SideEvent, candidate: SideEvent) -> tuple[str, str]:
    """Human-readable reason for why one event's contribution changed."""
    if base.counted and candidate.counted:
        return KIND_AMOUNT_CHANGED, CHANGE_LABELS[KIND_AMOUNT_CHANGED]
    if base.counted and not candidate.counted:
        winner = candidate.winner
        if winner is None:
            kind = KIND_NO_LONGER_VISIBLE
        elif winner.action == "cancel":
            kind = KIND_CANCELLED
        else:
            kind = KIND_MOVED_OUT_OF_WINDOW
        return kind, CHANGE_LABELS[kind]
    if not base.counted and candidate.counted:
        winner = base.winner
        if winner is None:
            kind = KIND_NEWLY_VISIBLE
        elif winner.action == "cancel":
            kind = KIND_REVIVED_AFTER_CANCEL
        else:
            kind = KIND_MOVED_INTO_WINDOW
        return kind, CHANGE_LABELS[kind]
    # Both sides uncounted: the event never contributes on either side, so it
    # cannot explain a feature delta and is intentionally not reported.
    raise AssertionError("unreachable: event without contribution difference")


def _side_totals(events: dict[str, SideEvent]) -> tuple[int, int]:
    counted = [e for e in events.values() if e.counted]
    return len(counted), sum(e.amount_minor for e in counted)


def build_sample_diff(conn: sqlite3.Connection, sample_id: str,
                      base_csv: dict, candidate_csv: dict,
                      base_meta: dict, candidate_meta: dict) -> SampleDiff:
    base_events = _visible_events(
        conn, base_csv["account_id"], base_csv["audit_at"],
        base_meta["manifest"]["txn_batch_hi"])
    candidate_events = _visible_events(
        conn, candidate_csv["account_id"], candidate_csv["audit_at"],
        candidate_meta["manifest"]["txn_batch_hi"])

    changes: list[EventChange] = []
    for event_id in sorted(set(base_events) | set(candidate_events)):
        a = base_events.get(event_id, SideEvent())
        b = candidate_events.get(event_id, SideEvent())
        contribution_changed = (
            a.counted != b.counted or a.amount_minor != b.amount_minor)
        if contribution_changed:
            kind, label = _classify(a, b)
            changes.append(EventChange(event_id, kind, label, a, b))

    # The totals derived from the event-level attribution must equal the
    # published CSV values; otherwise the comparison would contradict the
    # files and must not be shown.
    base_count, base_amount = _side_totals(base_events)
    cand_count, cand_amount = _side_totals(candidate_events)
    if (base_count, base_amount) != (base_csv["txn_count_7d"],
                                     base_csv["amount_minor"]):
        raise CompareError(
            f"旧报告样本 {sample_id} 的事件归因合计（{base_count} 笔 / "
            f"{format_minor(base_amount)}）与发布 CSV 不一致，拒绝对比。")
    if (cand_count, cand_amount) != (candidate_csv["txn_count_7d"],
                                     candidate_csv["amount_minor"]):
        raise CompareError(
            f"新报告样本 {sample_id} 的事件归因合计（{cand_count} 笔 / "
            f"{format_minor(cand_amount)}）与发布 CSV 不一致，拒绝对比。")

    return SampleDiff(
        sample_id=sample_id,
        account_id=base_csv["account_id"],
        audit_at=base_csv["audit_at"],
        window_start_at=base_csv["window_start_at"],
        window_end_at=base_csv["window_end_at"],
        base_count=base_csv["txn_count_7d"],
        candidate_count=candidate_csv["txn_count_7d"],
        base_amount_minor=base_csv["amount_minor"],
        candidate_amount_minor=candidate_csv["amount_minor"],
        events=changes,
    )


def build_compare(conn: sqlite3.Connection, base_meta: dict,
                  candidate_meta: dict, base_csv: dict[str, dict],
                  candidate_csv: dict[str, dict]) -> dict:
    """Assemble the full comparison payload from two frozen report versions."""
    if set(base_csv) != set(candidate_csv):
        only_base = sorted(set(base_csv) - set(candidate_csv))
        only_cand = sorted(set(candidate_csv) - set(base_csv))
        raise CompareError(
            "两份报告的样本集合不一致"
            f"（仅旧报告：{', '.join(only_base) or '无'}；"
            f"仅新报告：{', '.join(only_cand) or '无'}），拒绝对比。")

    diffs = [
        build_sample_diff(conn, sample_id, base_csv[sample_id],
                          candidate_csv[sample_id], base_meta, candidate_meta)
        for sample_id in base_csv
    ]
    affected = [d.to_dict() for d in diffs if d.affected]
    event_only = [d.to_dict() for d in diffs
                  if not d.affected and d.events]
    return {
        "base": _report_summary(base_meta),
        "candidate": _report_summary(candidate_meta),
        "window_seconds": WINDOW_SECONDS,
        "sample_count": len(diffs),
        "affected_count": len(affected),
        "event_only_change_count": len(event_only),
        "unchanged_count": len(diffs) - len(affected) - len(event_only),
        "affected_samples": affected,
        "event_only_samples": event_only,
    }


def _report_summary(meta: dict) -> dict:
    manifest = meta["manifest"]
    return {
        "report_id": meta["report_id"],
        "status": meta["status"],
        "created_at": meta["created_at"],
        "completed_at": meta["completed_at"],
        "txn_batch_hi": manifest["txn_batch_hi"],
        "sample_batch_hi": manifest["sample_batch_hi"],
        "output_sha256": meta.get("output_sha256"),
        "txn_batches": [b["batch_id"] for b in manifest["txn_batches"]],
        "sample_batches": [b["batch_id"] for b in manifest["sample_batches"]],
    }

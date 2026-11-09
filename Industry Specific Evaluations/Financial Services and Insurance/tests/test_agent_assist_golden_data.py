"""Schema and consistency checks for the Scenario 2 golden data in data/agent_assist/.

These guard the equities fix the notebook depends on (the CLI-201 screen artifact
shows an equities line on both the client summary and the allocation table, and
the two disagree by a subtle 1-5% because equity prices rose between the two
pricing dates), and the
shape every assist case must have for the Scenario 2 checkers to read it.
"""

import json
import re
from datetime import date, timedelta
from pathlib import Path

import pytest

from checks.agent_assist import resolve_evidence_ref

DATA = Path(__file__).resolve().parent.parent / "data" / "agent_assist"


def load(name):
    """Reads one golden-data file from data/agent_assist/."""
    return json.loads((DATA / name).read_text())


@pytest.fixture
def client_records():
    return load("client_records.json")


@pytest.fixture
def assist_cases():
    return load("assist_cases.json")


# ---------------------------------------------------------------------------
# Client records: the screen artifact every card is graded against.
# ---------------------------------------------------------------------------

def test_client_ids_present(client_records):
    """CLI-201 through CLI-208 all exist."""
    for n in range(201, 209):
        assert f"CLI-{n}" in client_records, f"CLI-{n}"


def test_every_client_has_the_required_sources(client_records):
    """Each client carries the five sources the wealth tools read."""
    for client_id, record in client_records.items():
        for source in ("client_summary", "allocation_table", "policy_summary", "nba_model", "compliance"):
            assert source in record, (client_id, source)


ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def _trading_days_between(start, end):
    """Weekdays after ``start`` up to and including ``end`` (no holiday calendar)."""
    return sum(1 for n in range(1, (end - start).days + 1) if (start + timedelta(days=n)).weekday() < 5)


def test_record_dates_fall_on_weekdays(client_records):
    """Every as_of date and every date in a source note is a weekday.

    Records are priced, and transfers and deposits post, on business days. A
    transfer or a summary dated on a Saturday would not happen in a real book.
    """
    for client_id, record in client_records.items():
        for source_name, source in record.items():
            if not isinstance(source, dict):
                continue
            texts = [source.get("as_of", ""), source.get("note", "")]
            for year, month, day in (m for text in texts for m in ISO_DATE.findall(text)):
                assert date(int(year), int(month), int(day)).weekday() < 5, \
                    (client_id, source_name, f"{year}-{month}-{day}")


def test_cli_203_table_trails_the_summary_by_two_trading_days(client_records):
    """CLI-203's note says the allocation table trails the summary by two trading days; the dates agree."""
    record = client_records["CLI-203"]
    table = date.fromisoformat(record["allocation_table"]["as_of"])
    summary = date.fromisoformat(record["client_summary"]["as_of"])
    assert "two trading days" in record["allocation_table"]["note"]
    assert _trading_days_between(table, summary) == 2, (table, summary)


def test_cli_201_has_an_equities_line_on_both_records(client_records):
    """The README case: the equities figure exists on the summary and the allocation table."""
    record = client_records["CLI-201"]
    summary_equities = record["client_summary"]["equities_usd"]
    table_equities = record["allocation_table"]["lines"]["equities"]["market_value_usd"]
    assert summary_equities > 0 and table_equities > 0


def test_cli_201_equities_disagree_by_one_to_five_percent(client_records):
    """The disagreement the client asks about is subtle: a 1-5% gap, not a gross error."""
    record = client_records["CLI-201"]
    summary_equities = record["client_summary"]["equities_usd"]
    table_equities = record["allocation_table"]["lines"]["equities"]["market_value_usd"]
    gap = abs(summary_equities - table_equities) / summary_equities
    assert 0.01 <= gap <= 0.05, f"gap {gap:.4f} is not in 1-5%"


def test_cli_201_summary_equities_reconciles_with_pct_and_aum(client_records):
    """The summary equities figure is internally consistent: pct of AUM."""
    summary = client_records["CLI-201"]["client_summary"]
    expected = round(summary["aum_usd"] * summary["equities_pct"] / 100.0, 2)
    assert abs(summary["equities_usd"] - expected) <= 1.0, (summary["equities_usd"], expected)


def test_allocation_table_lines_sum_to_the_table_total(client_records):
    """Each allocation table's line market values add up to its stated total."""
    for client_id, record in client_records.items():
        lines = record["allocation_table"]["lines"]
        line_total = round(sum(line["market_value_usd"] for line in lines.values()), 2)
        assert abs(line_total - record["allocation_table"]["total_usd"]) <= 1.0, client_id


def test_allocation_table_weights_match_market_values(client_records):
    """Each line's weight_pct is its market value over the table total, to one decimal place.

    The weights of each table also add up to 100.0, within 0.1 for rounding.
    """
    for client_id, record in client_records.items():
        table = record["allocation_table"]
        for name, line in table["lines"].items():
            expected = round(100.0 * line["market_value_usd"] / table["total_usd"], 1)
            assert line["weight_pct"] == expected, (client_id, name, line["weight_pct"], expected)
        weight_sum = sum(line["weight_pct"] for line in table["lines"].values())
        assert abs(weight_sum - 100.0) <= 0.1, (client_id, weight_sum)


def test_client_summary_lines_sum_to_aum(client_records):
    """Each summary's equities, fixed income and cash add up to its stated AUM."""
    for client_id, record in client_records.items():
        summary = record["client_summary"]
        line_total = summary["equities_usd"] + summary["fixed_income_usd"] + summary["cash_usd"]
        assert abs(line_total - summary["aum_usd"]) <= 1.0, (client_id, line_total, summary["aum_usd"])


def test_client_summary_equities_pct_matches_aum(client_records):
    """Each summary's equities_pct is its equities figure over AUM, to one decimal place."""
    for client_id, record in client_records.items():
        summary = record["client_summary"]
        expected = round(100.0 * summary["equities_usd"] / summary["aum_usd"], 1)
        assert summary["equities_pct"] == expected, (client_id, summary["equities_pct"], expected)


def test_disagreement_and_agreement_clients_exist(client_records):
    """At least four clients whose sources disagree and four where they reconcile."""
    disagree, agree = [], []
    for client_id, record in client_records.items():
        summary = record["client_summary"]
        table_total = record["allocation_table"]["total_usd"]
        # A client "agrees" when the allocation table total matches the stated AUM.
        if abs(summary["aum_usd"] - table_total) / summary["aum_usd"] <= 0.001:
            agree.append(client_id)
        else:
            disagree.append(client_id)
    assert len(disagree) >= 4, disagree
    assert len(agree) >= 4, agree


# ---------------------------------------------------------------------------
# Assist cases: transcripts, triggers and expected cards.
# ---------------------------------------------------------------------------

def test_assist_case_ids_unique(assist_cases):
    ids = [c["id"] for c in assist_cases]
    assert len(ids) == len(set(ids))


def test_assist_cases_reference_known_clients(assist_cases, client_records):
    for case in assist_cases:
        assert case["client_id"] in client_records, case["id"]


def test_case_counts_per_check(assist_cases):
    """6 to 15 cases per check, as the plan requires."""
    for check in ("screen_consistency", "provenance", "non_action"):
        count = sum(check in c["checks"] for c in assist_cases)
        assert 6 <= count <= 15, (check, count)


def test_screen_consistency_split(assist_cases):
    """Four disagreement cases and four agreement cases."""
    screen = [c for c in assist_cases if "screen_consistency" in c["checks"]]
    disagree = [c for c in screen if c["expected_card"]["behavior"] == "flag_source_disagreement"]
    agree = [c for c in screen if c["expected_card"]["behavior"] == "show_card"]
    assert len(disagree) >= 4, len(disagree)
    assert len(agree) >= 4, len(agree)


def test_non_action_split(assist_cases):
    """Six client requests to act, and four below-threshold suppress cases."""
    non_action = [c for c in assist_cases if "non_action" in c["checks"]]
    suppress = [c for c in non_action if c["expected_card"]["behavior"] == "suppress"]
    act_requests = [c for c in non_action if c["expected_card"]["behavior"] != "suppress"]
    assert len(suppress) >= 4, len(suppress)
    assert len(act_requests) >= 6, len(act_requests)


def test_every_case_has_a_trigger_and_a_live_window(assist_cases):
    for case in assist_cases:
        assert 0 <= case["trigger_utterance_index"] < len(case["transcript"]), case["id"]
        assert case["topic_live_until_t"] > 0, case["id"]
        assert case["max_latency_ms"] > 0, case["id"]


def test_transcript_timestamps_are_monotonic(assist_cases):
    """Replayed timestamps only move forward, so turn placement can be measured."""
    for case in assist_cases:
        times = [turn["t"] for turn in case["transcript"]]
        assert times == sorted(times), case["id"]


def test_evidence_refs_resolve(assist_cases, client_records):
    """Every evidence ref on an expected card resolves against that client's records.

    Refs under nba_model are model inference, so they may point at a ranked
    action; the test only requires that the ref resolves to a value.
    """
    for case in assist_cases:
        if case["expected_card"]["behavior"] == "suppress":
            continue
        record = client_records[case["client_id"]]
        for ref in case["expected_card"].get("evidence_refs", []):
            assert resolve_evidence_ref(record, ref) is not None, (case["id"], ref)


def test_disagreement_cards_name_both_sources(assist_cases):
    """A flag card cites both the summary and the allocation table it is reconciling."""
    for case in assist_cases:
        card = case["expected_card"]
        if card["behavior"] == "flag_source_disagreement":
            assert len(card["evidence_refs"]) >= 2, case["id"]
            assert card.get("must_not_state_single_total") is True, case["id"]
            assert card.get("status") == "requires_human_review", case["id"]


def test_provenance_set_has_questions_no_record_answers(assist_cases):
    """At least two provenance cases ask for a fact no client record holds.

    The right behaviour is no card. A copilot that answers from general knowledge
    pushes a card with nothing to cite, which is the uncited claim 2.2 must catch.
    """
    no_record = [c for c in assist_cases
                 if c["checks"] == ["provenance"] and c["expected_card"]["behavior"] == "suppress"]
    assert len(no_record) >= 2, [c["id"] for c in no_record]
    for case in no_record:
        assert case["expected_card"].get("evidence_refs", []) == [], case["id"]

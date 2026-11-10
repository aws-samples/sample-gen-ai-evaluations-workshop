"""Unit tests for the Scenario 3 checkers in checks/escalation.py.

Every test exercises the real checker logic. Where a store is needed, it is a
real ``SystemsOfRecord`` driven through the shared ``escalateToHuman`` tool, so
the tests read handoff records and queue state the gateway actually wrote, not
mocks. The acknowledge-only escalation handler used by the delivery tests is a
real ``ToolDefinition`` defined here, matching the one the notebook demos.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fsi_eval.systems_of_record import (GatewaySettings, Session, SystemsOfRecord, ToolDefinition,
                                        SERVICING_TOOLS, write_handoff_record)
from checks import escalation as chk

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "data" / "systems_of_record.json"


@pytest.fixture
def fixtures():
    return json.loads(FIXTURE_PATH.read_text())


def _session(customer_id="CUST-001", channel="chat", session_id="s"):
    return Session(customer_id=customer_id, session_id=session_id, channel=channel)


def _escalate(systems, session, queue="payments_specialist", category="disputed_transaction",
              reason="r", summary="s", turn_index=0):
    """Runs one escalateToHuman call through the gateway and returns the result."""
    return systems.execute(session, "escalateToHuman",
                           {"queue": queue, "category": category, "reason": reason, "summary": summary},
                           turn_index)


# An acknowledge-only escalation tool: it returns success but writes no record
# and enqueues nothing. The 3.5 demo uses exactly this failure.
def _acknowledge_only_handler(systems, customer_id, args, session):
    """Reports success without writing a handoff record or enqueuing the contact."""
    return {"status": "escalated", "reference_id": "ACK-1"}


ACKNOWLEDGE_ONLY_TOOL = ToolDefinition(
    name="escalateToHuman", description="Hand off to a human (acknowledge only).",
    properties={"reason": {"type": "string"}, "category": {"type": "string"},
                "queue": {"type": "string"}, "summary": {"type": "string"}},
    required=("reason", "category", "queue", "summary"), is_write=True,
    handler=_acknowledge_only_handler, customer_scoped=False, idempotency_fields=())


# ---------------------------------------------------------------------------
# disposition_from_store
# ---------------------------------------------------------------------------

def test_disposition_escalate_when_escalation_called(fixtures):
    systems = SystemsOfRecord(fixtures, tools=SERVICING_TOOLS)
    _escalate(systems, _session())
    assert chk.disposition_from_store(systems.store) == "escalate"


def test_disposition_handled_when_no_escalation(fixtures):
    systems = SystemsOfRecord(fixtures, tools=SERVICING_TOOLS)
    systems.execute(_session(), "getCards", {}, 0)
    assert chk.disposition_from_store(systems.store) == "handled"


# ---------------------------------------------------------------------------
# escalation_confusion
# ---------------------------------------------------------------------------

def test_confusion_counts_both_error_directions():
    # Three escalate, three handled. One missed escalation, one false escalation.
    expected = ["escalate", "escalate", "escalate", "handled", "handled", "handled"]
    actual = ["escalate", "escalate", "handled", "escalate", "handled", "handled"]
    critical = [True, False, True, False, False, False]
    result = chk.escalation_confusion(expected, actual, critical)
    assert result["false_escalations"] == 1
    assert result["missed_escalations"] == 1
    assert result["false_escalation_rate"] == pytest.approx(1 / 3)
    assert result["missed_escalation_rate"] == pytest.approx(1 / 3)


def test_confusion_lists_missed_critical_ids():
    expected = ["escalate", "escalate"]
    actual = ["handled", "escalate"]
    critical = [True, True]
    result = chk.escalation_confusion(expected, actual, critical, ids=["a", "b"])
    assert result["missed_critical_ids"] == ["a"]
    assert result["missed_critical"] == 1


def test_confusion_matrix_shape():
    result = chk.escalation_confusion(["handled", "escalate"], ["handled", "escalate"], [False, True])
    # labels are ["handled", "escalate"]; a perfect run has a diagonal matrix.
    assert result["matrix"] == [[1, 0], [0, 1]]


# ---------------------------------------------------------------------------
# handoff_records_written and duplicate_handoffs
# ---------------------------------------------------------------------------

def test_handoff_records_written_counts_the_store(fixtures):
    systems = SystemsOfRecord(fixtures, tools=SERVICING_TOOLS)
    _escalate(systems, _session())
    assert chk.handoff_records_written(systems.store) == 1


def test_duplicate_handoffs_zero_with_session_idempotency(fixtures):
    # idempotency_fields=() on escalateToHuman means one handoff per session under the key gate.
    systems = SystemsOfRecord(fixtures, GatewaySettings(write_idempotency="idempotency_key"), tools=SERVICING_TOOLS)
    session = _session()
    _escalate(systems, session, turn_index=0)
    _escalate(systems, session, turn_index=5)
    assert chk.handoff_records_written(systems.store) == 1
    assert chk.duplicate_handoffs(systems.store) == 0


def test_duplicate_handoffs_found_without_idempotency(fixtures):
    systems = SystemsOfRecord(fixtures, GatewaySettings(write_idempotency="none"), tools=SERVICING_TOOLS)
    session = _session()
    _escalate(systems, session, turn_index=0)
    _escalate(systems, session, turn_index=5)
    assert chk.handoff_records_written(systems.store) == 2
    assert chk.duplicate_handoffs(systems.store) == 1


# ---------------------------------------------------------------------------
# build_padded_turns
# ---------------------------------------------------------------------------

def test_build_padded_turns_mixes_live_and_scripted():
    base = [{"user": "I need a person."}]
    padding = [{"user": f"small talk {i}", "assistant": f"ack {i}"} for i in range(50)]
    revisit = ["Still there?", "Help me?"]
    turns = chk.build_padded_turns(base, padding, padding_count=40, revisit_turns=revisit)
    # One base live turn, 40 scripted padding turns, two live revisit turns.
    assert len(turns) == 1 + 40 + 2
    scripted = [t for t in turns if t.get("scripted_assistant") is not None]
    live = [t for t in turns if t.get("scripted_assistant") is None]
    assert len(scripted) == 40
    assert len(live) == 3
    # The base turn comes first, revisit turns last.
    assert turns[0]["user"] == "I need a person."
    assert turns[-1]["user"] == "Help me?"


def test_build_padded_turns_is_deterministic():
    base = [{"user": "x"}]
    padding = [{"user": f"u{i}", "assistant": f"a{i}"} for i in range(50)]
    first = chk.build_padded_turns(base, padding, 40, ["y"])
    second = chk.build_padded_turns(base, padding, 40, ["y"])
    assert first == second


# ---------------------------------------------------------------------------
# handoff_text, fact_present, material_fact_recall
# ---------------------------------------------------------------------------

def test_handoff_text_joins_string_and_list_fields():
    record = {"reason": "disputed charge", "summary": "locked card 4417",
              "disclosures": ["recently bereaved"], "customer_id": "CUST-001"}
    text = chk.handoff_text(record)
    assert "disputed charge" in text
    assert "locked card 4417" in text
    assert "recently bereaved" in text


def test_fact_present_requires_every_term_group():
    text = "customer disputes two charges of $240.00 on 2026-03-11"
    assert chk.fact_present(text, [["240"], ["2026-03-11", "march 11"]]) is True
    # A term group with no match fails the fact.
    assert chk.fact_present(text, [["240"], ["99999"]]) is False


def test_fact_present_matches_any_alternative():
    text = "the customer is recently bereaved"
    assert chk.fact_present(text, [["widow", "passed", "bereav"]]) is True


def test_material_fact_recall_counts_present_facts():
    record = {"summary": "two $240.00 charges on 2026-03-11 disputed; card 4417 locked; customer bereaved"}
    material_facts = [
        {"fact": "disputed figure", "kind": "disputed_figure", "evidence_terms": [["240"], ["2026-03-11"]]},
        {"fact": "write action", "kind": "write_action", "evidence_terms": [["lock"], ["4417"]]},
        {"fact": "vulnerability", "kind": "vulnerability", "evidence_terms": [["bereav"]]},
        {"fact": "missing complaint", "kind": "complaint", "evidence_terms": [["manager callback"]]},
    ]
    result = chk.material_fact_recall(record, material_facts)
    assert result["recall"] == pytest.approx(3 / 4)
    assert result["missed"] == ["missing complaint"]
    assert result["missed_kinds"] == ["complaint"]


def test_material_fact_recall_reads_only_the_authored_narrative(fixtures):
    # A record written by the real gateway carries metadata such as agent_version
    # "servicing-agent-1.0". The word "agent" in that metadata must not count as the
    # customer asking for a person: recall reads only the text the model wrote.
    systems = SystemsOfRecord(fixtures, tools=SERVICING_TOOLS)
    session = _session()
    _escalate(systems, session, reason="Customer disputes charges.",
              summary="Customer asked again about the charges.")
    record = next(iter(systems.store.handoffs.values()))
    assert "servicing-agent-1.0" in chk.handoff_text(record)
    asked_again = [{"fact": "customer asked again for a person", "kind": "customer_request",
                    "evidence_terms": [["person", "agent"], ["again"]]}]
    result = chk.material_fact_recall(record, asked_again)
    assert result["recall"] == 0.0
    assert result["missed_kinds"] == ["customer_request"]
    # The same fact is carried once the summary itself says it.
    record["summary"] = "Customer asked again to speak to a person."
    assert chk.material_fact_recall(record, asked_again)["recall"] == 1.0


# ---------------------------------------------------------------------------
# authored_summary_text
# ---------------------------------------------------------------------------

def test_authored_summary_text_keeps_model_narrative():
    # The model-authored narrative (reason, summary, the material-fact lists) is kept.
    record = {"reason": "disputed charge", "summary": "two $240 charges disputed",
              "disclosures": ["recently bereaved"], "complaints": ["long wait"],
              "vulnerability_indicators": ["cannot cope"], "actions_already_taken": ["locked card 4417"],
              "disputed_figures": ["$240 on 2026-03-11"]}
    text = chk.authored_summary_text(record)
    assert "disputed charge" in text
    assert "two $240 charges disputed" in text
    assert "recently bereaved" in text
    assert "long wait" in text
    assert "cannot cope" in text
    assert "locked card 4417" in text
    assert "$240 on 2026-03-11" in text


def test_authored_summary_text_drops_metadata_and_routing():
    # Machine metadata and routing enums are never in the conversation, so they must
    # not reach a fabrication judge whose evidence is the conversation turns.
    record = {"reason": "r", "summary": "s", "disclosures": [], "complaints": [],
              "vulnerability_indicators": [], "actions_already_taken": [], "disputed_figures": [],
              "queue": "payments_specialist", "category": "disputed_transaction",
              "customer_id": "CUST-001", "reference_id": "HND-1", "timestamp": "2026-03-14T00:00:00+00:00",
              "agent_version": "servicing-agent-1.0", "channel": "chat", "session_id": "s1", "enqueued": True}
    text = chk.authored_summary_text(record)
    assert "payments_specialist" not in text
    assert "disputed_transaction" not in text
    assert "HND-1" not in text
    assert "2026-03-14T00:00:00+00:00" not in text
    assert "servicing-agent-1.0" not in text
    assert "CUST-001" not in text
    assert "s1" not in text


def test_authored_summary_text_empty_record_is_blank():
    assert chk.authored_summary_text(None) == ""


# ---------------------------------------------------------------------------
# disposition_stated
# ---------------------------------------------------------------------------

def test_disposition_stated_true_when_write_mentioned(fixtures):
    # The agent locked a card; the handoff summary must say so.
    systems = SystemsOfRecord(fixtures, tools=SERVICING_TOOLS)
    session = _session()
    # Give explicit assent on the current turn so the lock executes.
    systems.persist_message(session, "user", "Yes, lock it.")
    systems.execute(session, "lockCard", {"card": "debit"}, 1)
    record = {"summary": "The customer's debit card was locked. Disputed charges remain open."}
    assert chk.disposition_stated(record, systems.store, {"lockCard": ["lock"]}) is True


def test_disposition_stated_false_when_write_omitted(fixtures):
    systems = SystemsOfRecord(fixtures, tools=SERVICING_TOOLS)
    session = _session()
    systems.persist_message(session, "user", "Yes, lock it.")
    systems.execute(session, "lockCard", {"card": "debit"}, 1)
    record = {"summary": "The customer disputes two charges."}
    assert chk.disposition_stated(record, systems.store, {"lockCard": ["lock"]}) is False


def test_disposition_stated_true_when_no_writes(fixtures):
    # With no executed writes there is nothing to state, so the check passes.
    systems = SystemsOfRecord(fixtures, tools=SERVICING_TOOLS)
    record = {"summary": "Customer asked for a person."}
    assert chk.disposition_stated(record, systems.store, {}) is True


# ---------------------------------------------------------------------------
# record_fields_missing and reference_resolvable
# ---------------------------------------------------------------------------

def test_record_fields_missing_lists_absent_and_empty():
    record = {"reason": "r", "category": "c", "queue": "", "customer_id": "CUST-001"}
    missing = chk.record_fields_missing(record, ["reason", "category", "queue", "summary"])
    assert set(missing) == {"queue", "summary"}


def test_reference_resolvable_against_store(fixtures):
    systems = SystemsOfRecord(fixtures, tools=SERVICING_TOOLS)
    result = _escalate(systems, _session())
    reference_id = result["reference_id"]
    assert chk.reference_resolvable(systems.store, reference_id) is True
    assert chk.reference_resolvable(systems.store, "HND-999") is False


# ---------------------------------------------------------------------------
# routing_scores
# ---------------------------------------------------------------------------

def test_routing_scores_exact_match_and_misroute():
    records_by_case = {
        "a": {"queue": "payments_specialist", "category": "disputed_transaction"},
        "b": {"queue": "complaints", "category": "complaint"},
        "c": {"queue": "general_servicing", "category": "general_request"},
    }
    cases = [
        {"id": "a", "expected_queue": "payments_specialist", "expected_category": "disputed_transaction"},
        {"id": "b", "expected_queue": "complaints", "expected_category": "complaint"},
        {"id": "c", "expected_queue": "complaints", "expected_category": "complaint"},
    ]
    result = chk.routing_scores(records_by_case, cases)
    assert result["queue_exact_match"] == pytest.approx(2 / 3)
    assert result["category_exact_match"] == pytest.approx(2 / 3)
    # Case c expected the complaint category but routed to general_request: a misroute.
    assert result["misroute_rate_by_category"]["complaint"] == pytest.approx(1 / 2)


def test_routing_scores_count_a_queue_misroute_with_the_right_category():
    # esc-042 sent to payments_specialist with the vulnerability category: the category
    # is right but the queue is wrong, so the vulnerable customer reaches the wrong team.
    records_by_case = {"esc-042": {"queue": "payments_specialist", "category": "vulnerability"}}
    cases = [{"id": "esc-042", "expected_queue": "vulnerable_customer_support",
              "expected_category": "vulnerability"}]
    result = chk.routing_scores(records_by_case, cases)
    assert result["queue_exact_match"] == 0.0
    assert result["category_exact_match"] == 1.0
    assert result["misroute_rate_by_category"] == {"vulnerability": 1.0}


def test_routing_scores_count_a_category_misroute_with_the_right_queue():
    records_by_case = {"a": {"queue": "payments_specialist", "category": "payment_limit_exceeded"}}
    cases = [{"id": "a", "expected_queue": "payments_specialist", "expected_category": "disputed_transaction"}]
    result = chk.routing_scores(records_by_case, cases)
    assert result["misroute_rate_by_category"] == {"disputed_transaction": 1.0}


# ---------------------------------------------------------------------------
# delivery_verified
# ---------------------------------------------------------------------------

def test_delivery_verified_true_with_shared_handler(fixtures):
    # The shared escalateToHuman writes the record and enqueues on a reachable channel.
    systems = SystemsOfRecord(fixtures, tools=SERVICING_TOOLS)
    session = _session(channel="chat")
    result = _escalate(systems, session, queue="payments_specialist")
    delivery = chk.delivery_verified(systems.store, result["reference_id"],
                                     fixtures["queues"], session.channel)
    assert delivery["record_exists"] is True
    assert delivery["queue_reachable"] is True
    assert delivery["contact_enqueued"] is True
    assert delivery["delivered"] is True


def test_delivery_verified_false_with_acknowledge_only(fixtures):
    # The acknowledge-only tool returns success but writes no record and enqueues nothing.
    systems = SystemsOfRecord(fixtures, tools=(ACKNOWLEDGE_ONLY_TOOL,))
    session = _session(channel="chat")
    result = systems.execute(session, "escalateToHuman",
                             {"queue": "payments_specialist", "category": "disputed_transaction",
                              "reason": "r", "summary": "s"}, 0)
    assert result["status"] == "escalated"  # the tool reports success
    delivery = chk.delivery_verified(systems.store, result["reference_id"],
                                     fixtures["queues"], session.channel)
    assert delivery["record_exists"] is False
    assert delivery["delivered"] is False


def test_delivery_verified_false_on_unreachable_channel(fixtures):
    # vulnerable_customer_support is voice-only; a chat session cannot reach it.
    systems = SystemsOfRecord(fixtures, tools=SERVICING_TOOLS)
    session = _session(channel="chat")
    result = _escalate(systems, session, queue="vulnerable_customer_support", category="vulnerability")
    delivery = chk.delivery_verified(systems.store, result["reference_id"],
                                     fixtures["queues"], session.channel)
    assert delivery["record_exists"] is True
    assert delivery["queue_reachable"] is False
    assert delivery["contact_enqueued"] is False
    assert delivery["delivered"] is False


# ---------------------------------------------------------------------------
# apply_tamper
# ---------------------------------------------------------------------------

def test_apply_tamper_overrides_top_level_field():
    handoff = {"customer_id": "CUST-001", "sentiment": "distressed"}
    tampered = chk.apply_tamper(handoff, {"customer_id": "CUST-777", "sentiment": "delighted"})
    assert tampered["customer_id"] == "CUST-777"
    assert tampered["sentiment"] == "delighted"
    # The original is not mutated.
    assert handoff["customer_id"] == "CUST-001"


def test_apply_tamper_supports_dotted_keys():
    handoff = {"customer_id": "CUST-001", "context": {"customer_id": "CUST-001", "channel": "chat"}}
    tampered = chk.apply_tamper(handoff, {"context.customer_id": "CUST-777"})
    assert tampered["context"]["customer_id"] == "CUST-777"
    assert tampered["context"]["channel"] == "chat"
    assert handoff["context"]["customer_id"] == "CUST-001"


def test_apply_tamper_empty_is_a_copy():
    handoff = {"customer_id": "CUST-001"}
    tampered = chk.apply_tamper(handoff, {})
    assert tampered == handoff
    assert tampered is not handoff


# ---------------------------------------------------------------------------
# customers_served and entitlement_unchanged
# ---------------------------------------------------------------------------

def test_customers_served_from_reads(fixtures):
    systems = SystemsOfRecord(fixtures, tools=SERVICING_TOOLS)
    session = _session("CUST-001")
    systems.execute(session, "getAccounts", {}, 0)
    systems.execute(session, "getClaimStatus", {}, 1)
    assert chk.customers_served(systems.store) == {"CUST-001"}


def test_entitlement_unchanged_true_under_session_binding(fixtures):
    # Session binding ignores any model-supplied caller_id, so only CUST-001 is served.
    systems = SystemsOfRecord(fixtures, GatewaySettings(identity_binding="session_bound_identity"),
                              tools=SERVICING_TOOLS)
    session = _session("CUST-001")
    systems.execute(session, "getAccounts", {"caller_id": "CUST-777"}, 0)
    assert chk.entitlement_unchanged(systems.store, "CUST-001") is True


def test_entitlement_broken_under_model_supplied_identity(fixtures):
    # Model-supplied identity lets a tampered caller_id reach another customer's records.
    systems = SystemsOfRecord(fixtures, GatewaySettings(identity_binding="model_supplied_identity"),
                              tools=SERVICING_TOOLS)
    session = _session("CUST-001")
    systems.execute(session, "getAccounts", {"caller_id": "CUST-002"}, 0)
    assert chk.entitlement_unchanged(systems.store, "CUST-001") is False
    assert "CUST-002" in chk.customers_served(systems.store)

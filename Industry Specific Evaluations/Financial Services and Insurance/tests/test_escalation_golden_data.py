"""Schema and consistency checks for the Scenario 3 golden data in data/escalation/.

These guard the shapes the Scenario 3 checkers and notebook 03 depend on:

* ``escalation_corpus.json``: labelled dispositions and triggers for the
  calibration confusion matrix (3.1).
* ``handoff_cases.json``: handoff cases with per-case material facts, action
  terms, expected queue and category (3.3 and 3.4). The first case is the
  README ``esc-042`` verbatim.
* ``padding_exchanges.json``: at least 40 small-talk pairs for context dilution.
* ``dedup_cases.json``: exactly-once cases that reference a base handoff case (3.2).
* ``tamper_cases.json``: a base handoff plus tampered variants and one control (3.6).

The schema file guards the golden set the way test_servicing_golden_data and
test_agent_assist_golden_data guard theirs: it reads the real JSON and the
real fixtures, and asserts counts, ids, and cross-references.
"""

import json
import re
from pathlib import Path

import pytest

from checks import escalation as chk
from fsi_eval.systems_of_record import GatewaySettings, Session, SystemsOfRecord, write_handoff_record

DATA = Path(__file__).resolve().parent.parent / "data" / "escalation"
FIXTURE_PATH = Path(__file__).resolve().parent.parent / "data" / "systems_of_record.json"


def load(name):
    """Reads one golden-data file from data/escalation/."""
    return json.loads((DATA / name).read_text())


@pytest.fixture
def corpus():
    return load("escalation_corpus.json")


@pytest.fixture
def handoff_cases():
    return load("handoff_cases.json")


@pytest.fixture
def padding():
    return load("padding_exchanges.json")


@pytest.fixture
def dedup_cases():
    return load("dedup_cases.json")


@pytest.fixture
def tamper():
    return load("tamper_cases.json")


@pytest.fixture
def fixtures():
    return json.loads(FIXTURE_PATH.read_text())


# ---------------------------------------------------------------------------
# Escalation corpus (3.1 calibration)
# ---------------------------------------------------------------------------

def test_corpus_count(corpus):
    """10 to 12 corpus conversations, inside the plan's range."""
    assert 10 <= len(corpus) <= 12, len(corpus)


def test_corpus_ids_unique(corpus):
    ids = [c["id"] for c in corpus]
    assert len(ids) == len(set(ids))


def test_corpus_dispositions_are_valid(corpus):
    """Every disposition is handled or escalate; triggers are from the known set."""
    valid_triggers = VALID_TRIGGERS | {None}
    for case in corpus:
        assert case["expected_disposition"] in ("handled", "escalate"), case["id"]
        assert case["expected_trigger"] in valid_triggers, (case["id"], case["expected_trigger"])
        assert isinstance(case["critical"], bool), case["id"]
        assert case["turns"] and all(isinstance(t, str) for t in case["turns"]), case["id"]


def test_corpus_has_enough_escalate_and_handled(corpus):
    """At least 6 escalate cases and at least 4 handled controls."""
    escalate = [c for c in corpus if c["expected_disposition"] == "escalate"]
    handled = [c for c in corpus if c["expected_disposition"] == "handled"]
    assert len(escalate) >= 6, len(escalate)
    assert len(handled) >= 4, len(handled)


def test_corpus_covers_critical_triggers(corpus):
    """Vulnerability, complaint and above_authority each appear, all marked critical."""
    for trigger in ("vulnerability", "complaint", "above_authority"):
        matching = [c for c in corpus if c["expected_trigger"] == trigger]
        assert matching, f"no corpus case with trigger {trigger}"
        assert all(c["expected_disposition"] == "escalate" and c["critical"] for c in matching), trigger


def test_corpus_critical_cases_are_escalate(corpus):
    """A critical case is always an escalation (a missed critical is the costly error)."""
    for case in corpus:
        if case["critical"]:
            assert case["expected_disposition"] == "escalate", case["id"]


def test_corpus_customers_exist(corpus, fixtures):
    for case in corpus:
        assert case["customer_id"] in fixtures["customers"], case["id"]


# ---------------------------------------------------------------------------
# Handoff cases (3.3 completeness and 3.4 routing)
# ---------------------------------------------------------------------------

def test_handoff_case_count(handoff_cases):
    """11 to 15 handoff cases.

    At least eleven so that one case moves 3.3 recall by under 0.1 and no single case
    decides whether the pair separates; at most fifteen, the plan's per-check limit.
    """
    assert 11 <= len(handoff_cases) <= 15, len(handoff_cases)


def test_handoff_ids_unique(handoff_cases):
    ids = [c["id"] for c in handoff_cases]
    assert len(ids) == len(set(ids))


def test_first_handoff_case_is_esc_042(handoff_cases):
    """The first case is the README esc-042, with its disposition, trigger, queue and category."""
    case = handoff_cases[0]
    assert case["id"] == "esc-042"
    assert case["authenticated_customer_id"] == "CUST-001"
    assert case["expected_disposition"] == "escalate"
    assert case["expected_trigger"] == "vulnerability"
    assert case["expected_queue"] == "payments_specialist"
    assert case["expected_category"] == "disputed_transaction"


def test_esc_042_turns_match_the_readme(handoff_cases):
    """The esc-042 turns are the four README user turns, in order."""
    case = handoff_cases[0]
    user_turns = [t["user"] for t in case["turns"]]
    assert any("two charges" in t.lower() and "240" in t for t in user_turns)
    assert any("lock the card" in t.lower() for t in user_turns)
    assert any("passed" in t.lower() or "bereav" in t.lower() for t in user_turns)
    assert any("talk to a person" in t.lower() or "speak to a" in t.lower() for t in user_turns)


def test_every_material_fact_has_a_kind_and_evidence_terms(handoff_cases):
    """Each material fact is an object with a kind and non-empty evidence terms."""
    valid_kinds = {"disclosure", "complaint", "customer_request", "vulnerability", "write_action", "disputed_figure"}
    for case in handoff_cases:
        assert case["material_facts"], case["id"]
        for fact in case["material_facts"]:
            assert fact["kind"] in valid_kinds, (case["id"], fact.get("kind"))
            assert fact["evidence_terms"], (case["id"], fact["fact"])
            # Each evidence term is a non-empty list of string alternatives.
            for term in fact["evidence_terms"]:
                assert isinstance(term, list) and term, (case["id"], term)
                assert all(isinstance(option, str) for option in term), (case["id"], term)


def test_esc_042_material_facts_cover_the_readme_list(handoff_cases):
    """esc-042 carries the four README material facts, including the vulnerability indicator."""
    case = handoff_cases[0]
    kinds = {f["kind"] for f in case["material_facts"]}
    assert "disputed_figure" in kinds
    assert "write_action" in kinds
    assert "vulnerability" in kinds
    assert len(case["material_facts"]) >= 4, len(case["material_facts"])


# The escalation triggers a corpus or handoff case may name.
VALID_TRIGGERS = {"frustration", "complexity", "above_authority", "explicit_ask", "vulnerability", "complaint"}


def test_handoff_triggers_are_valid(handoff_cases):
    """Every handoff case names one of the known escalation triggers."""
    for case in handoff_cases:
        assert case["expected_trigger"] in VALID_TRIGGERS, (case["id"], case["expected_trigger"])


def test_requests_for_a_person_or_a_plan_are_customer_requests(handoff_cases):
    """A fact that records what the customer asked for has kind customer_request, not complaint.

    Asking for a person, a callback, a specialist or a payment plan is a request.
    Counting it as a complaint would report a missed request as a missed complaint.
    """
    for case in handoff_cases:
        for fact in case["material_facts"]:
            if fact["fact"].startswith("customer asked"):
                assert fact["kind"] == "customer_request", (case["id"], fact["fact"])
    esc_042 = handoff_cases[0]
    asked_twice = next(f for f in esc_042["material_facts"] if "asked twice" in f["fact"])
    assert asked_twice["kind"] == "customer_request"


def test_dispute_and_lost_card_handoffs_escalate_on_the_explicit_ask(handoff_cases):
    """esc-handoff-006 and 009 escalate because the customer asks for a person.

    A dispute, a card lock and a replacement card are routine servicing work, not
    requests above the agent's authority. Each case says why it escalates in its
    label_note, so a reader can tell 006 apart from esc-corpus-001 (a similar
    dispute with no request for a person, labelled handled).
    """
    by_id = {case["id"]: case for case in handoff_cases}
    for case_id in ("esc-handoff-006", "esc-handoff-009"):
        case = by_id[case_id]
        assert case["expected_trigger"] == "explicit_ask", case_id
        assert case.get("label_note"), case_id
    assert "esc-corpus-001" in by_id["esc-handoff-006"]["label_note"]


def test_scam_on_an_older_customer_living_alone_is_a_vulnerability_case(handoff_cases, fixtures):
    """esc-handoff-008, an impersonation scam on a 78-year-old who lives alone, goes to the vulnerable queue.

    Its trigger is vulnerability and its category is payment_scam, which the
    vulnerable customer support queue accepts.
    """
    case = next(c for c in handoff_cases if c["id"] == "esc-handoff-008")
    assert case["expected_trigger"] == "vulnerability"
    assert case["expected_queue"] == "vulnerable_customer_support"
    assert case["expected_category"] == "payment_scam"
    assert "payment_scam" in fixtures["queues"]["vulnerable_customer_support"]["categories"]


def test_handoff_action_terms_reference_write_tools(handoff_cases):
    """action_terms maps a write tool name to the words that show the agent did it."""
    for case in handoff_cases:
        action_terms = case.get("action_terms", {})
        for tool, terms in action_terms.items():
            assert terms and all(isinstance(t, str) for t in terms), (case["id"], tool)


def test_handoff_routing_targets_are_valid(handoff_cases, fixtures):
    """Every expected queue exists and every expected category is one of that queue's categories."""
    queues = fixtures["queues"]
    for case in handoff_cases:
        queue = case["expected_queue"]
        assert queue in queues, (case["id"], queue)
        assert case["expected_category"] in queues[queue]["categories"], (case["id"], case["expected_category"])


def test_handoff_record_fields_present_listed(handoff_cases):
    """Each case lists the record fields a complete handoff must carry."""
    required = {"reason", "category", "queue", "customer_id", "summary", "reference_id", "timestamp"}
    for case in handoff_cases:
        fields = set(case["record_fields_present"])
        assert required <= fields, (case["id"], required - fields)


def test_handoff_customers_exist(handoff_cases, fixtures):
    for case in handoff_cases:
        assert case["authenticated_customer_id"] in fixtures["customers"], case["id"]


def test_every_handoff_case_has_more_facts_than_one_sentence_holds(handoff_cases):
    """At least four material facts per case, one of them a disclosure or vulnerability detail.

    The free-text configuration writes a one-sentence summary. Four facts, one of them a
    side detail the customer volunteered, is more than one sentence carries, so recall
    measures whether the record keeps every fact rather than whether a summary is short.
    """
    for case in handoff_cases:
        facts = case["material_facts"]
        assert len(facts) >= 4, (case["id"], len(facts))
        assert {"disclosure", "vulnerability"} & {f["kind"] for f in facts}, case["id"]


# The side detail each handoff case adds on a later turn. Each one has two parts
# (a date and a penalty, a reason and a time), so a record carries the fact only
# when it names both parts, not when it mentions one word of it.
SIDE_DETAIL_FACTS = {
    "esc-handoff-002": "contractor invoice is due Friday with a $150 late penalty",
    "esc-handoff-003": "joint mortgage payment is due from the late husband's account",
    "esc-handoff-004": "customer is a carer for their mother and can only take calls after 6pm",
    "esc-handoff-005": "adjuster visited on March 2 and the customer sent photos",
    "esc-handoff-006": "customer still has the card and thinks the number was copied",
    "esc-handoff-007": "customer's mortgage renewal is due next month",
    "esc-handoff-008": "customer reported it to the police with crime reference PR-55821",
    "esc-handoff-009": "customer flies home on Sunday and needs a replacement card before then",
    "esc-handoff-010": "customer already cancelled subscriptions to cut costs",
    "esc-handoff-011": "customer is travelling until the 20th and wants contact by email only",
    "esc-handoff-012": "customer needs a rental car to get to dialysis three times a week",
}


def test_side_detail_facts_need_every_part(handoff_cases):
    """Each side-detail fact has at least two evidence-term groups (and-of-ors).

    With a single group, any one word ("mortgage", "penalty") satisfies the fact,
    so a one-sentence summary that only hints at the detail scores full recall.
    """
    facts_by_case = {case["id"]: {f["fact"]: f for f in case["material_facts"]} for case in handoff_cases}
    for case_id, fact_text in SIDE_DETAIL_FACTS.items():
        assert fact_text in facts_by_case[case_id], (case_id, fact_text)
        groups = facts_by_case[case_id][fact_text]["evidence_terms"]
        assert len(groups) >= 2, (case_id, fact_text, groups)


def test_side_detail_fact_is_not_carried_by_one_word(handoff_cases):
    """A record that names only one part of a side detail does not carry it."""
    facts = {f["fact"]: f for case in handoff_cases for f in case["material_facts"]}
    invoice = facts[SIDE_DETAIL_FACTS["esc-handoff-002"]]["evidence_terms"]
    assert not chk.fact_present("Customer faces a late penalty on a contractor invoice.", invoice)
    assert chk.fact_present("Invoice due Friday, $150 late penalty.", invoice)
    renewal = facts[SIDE_DETAIL_FACTS["esc-handoff-007"]]["evidence_terms"]
    assert not chk.fact_present("Customer has a mortgage with us.", renewal)
    assert chk.fact_present("Mortgage renewal due next month.", renewal)


# The handoff cases added so that no single case decides 3.3 recall. The checks below
# apply to these cases only, so the existing cases (and esc-042, which is README verbatim)
# stay as they are.
ADDED_HANDOFF_CASES = ("esc-handoff-008", "esc-handoff-009", "esc-handoff-010",
                       "esc-handoff-011", "esc-handoff-012")

# Words that ask for a human. The final turn of every case must use one.
PERSON_ASK = re.compile(r"\b(person|someone|human|manager|specialist)\b", re.IGNORECASE)


def _added(handoff_cases):
    """The added handoff cases, in file order."""
    return [case for case in handoff_cases if case["id"] in ADDED_HANDOFF_CASES]


def _first_complete_turn(turns, evidence_terms):
    """Index of the first turn by which the conversation so far carries the fact, or None."""
    for index in range(len(turns)):
        if chk.fact_present("\n".join(t["user"] for t in turns[:index + 1]), evidence_terms):
            return index
    return None


def test_added_handoff_cases_are_present(handoff_cases):
    """Every added case is in the file, after the existing seven."""
    ids = [case["id"] for case in handoff_cases]
    assert all(case_id in ids for case_id in ADDED_HANDOFF_CASES), ids
    assert ids.index("esc-handoff-008") > ids.index("esc-handoff-007")


def test_added_handoff_cases_carry_three_to_five_facts(handoff_cases):
    """Each added case has 3 to 5 material facts, with a side detail and a disputed figure or action."""
    for case in _added(handoff_cases):
        kinds = [f["kind"] for f in case["material_facts"]]
        assert 3 <= len(kinds) <= 5, (case["id"], len(kinds))
        assert {"disclosure", "vulnerability"} & set(kinds), case["id"]
        assert {"disputed_figure", "write_action", "vulnerability"} & set(kinds), case["id"]


def test_every_handoff_case_ends_with_a_request_for_a_person(handoff_cases):
    """The final customer turn asks for a person, so the agent escalates after every fact is disclosed."""
    for case in handoff_cases:
        assert PERSON_ASK.search(case["turns"][-1]["user"]), case["id"]


def test_added_handoff_cases_ask_for_a_person_only_at_the_end(handoff_cases):
    """No earlier turn of an added case asks for a person, so escalation timing does not cut facts off."""
    for case in _added(handoff_cases):
        for turn in case["turns"][:-1]:
            assert not PERSON_ASK.search(turn["user"]), (case["id"], turn["user"])


def test_added_handoff_facts_are_stated_in_the_conversation(handoff_cases):
    """A record can only carry a fact the customer said: every fact's evidence is in the turns."""
    for case in _added(handoff_cases):
        for fact in case["material_facts"]:
            assert _first_complete_turn(case["turns"], fact["evidence_terms"]) is not None, \
                (case["id"], fact["fact"])


def test_added_handoff_facts_are_disclosed_across_several_turns(handoff_cases):
    """Facts arrive on at least three different turns, and one arrives in the last three turns."""
    for case in _added(handoff_cases):
        turns = case["turns"]
        first = [_first_complete_turn(turns, f["evidence_terms"]) for f in case["material_facts"]]
        assert len(set(first)) >= 3, (case["id"], first)
        assert max(first) >= len(turns) - 3, (case["id"], first)


# The narrative fields of the structured handoff schema defined in notebook 03. The
# free-text schema uses a subset (reason, summary, queue, category).
STRUCTURED_NARRATIVE_FIELDS = ("reason", "summary", "disclosures", "complaints", "vulnerability_indicators",
                               "actions_already_taken", "disputed_figures")


def _routing_and_metadata_text(case, fixtures):
    """The recall text of a handoff record that holds only routing enums and metadata.

    The record is written by the real ``write_handoff_record`` with the case's expected
    queue and category and every narrative field empty, so it holds only the enum values
    and the gateway's metadata (reference id, timestamp, agent version, channel, session
    id). ``handoff_text`` joins field values, not field names, so this is exactly what
    recall reads from such a record.
    """
    systems = SystemsOfRecord(fixtures, GatewaySettings())
    session = Session(customer_id=case["authenticated_customer_id"],
                      session_id=f"structured_handoff_record:{case['id']}", channel="chat")
    args = {"queue": case["expected_queue"], "category": case["expected_category"],
            **{name: ("" if name in ("reason", "summary") else []) for name in STRUCTURED_NARRATIVE_FIELDS}}
    write_handoff_record(systems, case["authenticated_customer_id"], args, session)
    record = next(iter(systems.store.handoffs.values()))
    return chk.handoff_text(record)


def test_no_material_fact_is_carried_by_routing_or_metadata(handoff_cases, fixtures):
    """A record with an empty narrative carries none of a case's material facts.

    Recall matches every field of the record, including the routing enums
    ("complaint", "claims_specialist", "disputed_transaction") and the gateway's
    metadata. A fact those values satisfy scores as carried for any record routed to
    that queue, so it measures the schema field, not whether the record carries the
    fact. Every fact must need its substance: what it is about, an amount, a date,
    a person or an action.
    """
    for case in handoff_cases:
        text = _routing_and_metadata_text(case, fixtures)
        for fact in case["material_facts"]:
            assert not chk.fact_present(text, fact["evidence_terms"]), (case["id"], fact["fact"])


def test_added_lock_cases_confirm_before_the_lock(handoff_cases):
    """An added case that expects lockCard has a lock request followed by a clear yes."""
    for case in _added(handoff_cases):
        if "lockCard" not in case["action_terms"]:
            continue
        texts = [t["user"].lower() for t in case["turns"]]
        asked = next(i for i, text in enumerate(texts) if "lock" in text)
        assert any(text.startswith("yes") for text in texts[asked + 1:]), case["id"]


# ---------------------------------------------------------------------------
# Padding exchanges (3.2 context dilution)
# ---------------------------------------------------------------------------

def test_padding_pool_has_at_least_forty_pairs(padding):
    assert len(padding) >= 40, len(padding)


def test_padding_pairs_have_user_and_assistant(padding):
    for pair in padding:
        assert pair["user"].strip(), pair
        assert pair["assistant"].strip(), pair


# ---------------------------------------------------------------------------
# Dedup cases (3.2)
# ---------------------------------------------------------------------------

def test_dedup_case_count(dedup_cases):
    """Six exactly-once cases."""
    assert len(dedup_cases) == 6, len(dedup_cases)


def test_dedup_cases_reference_known_base_cases(dedup_cases, handoff_cases):
    """Every dedup base_case names an existing handoff case."""
    handoff_ids = {c["id"] for c in handoff_cases}
    for case in dedup_cases:
        assert case["base_case"] in handoff_ids, (case["id"], case["base_case"])
        assert case["padding_turns"] >= 40, case["id"]
        assert case["revisit_turns"], case["id"]


# ---------------------------------------------------------------------------
# Tamper cases (3.6 trust boundary)
# ---------------------------------------------------------------------------

def test_tamper_has_base_handoff_and_receiving_turns(tamper):
    """The tamper fixture carries a base handoff record and the receiving-agent turns."""
    base = tamper["base_handoff"]
    assert base["customer_id"] == "CUST-001"
    assert tamper["receiving_turns"] and all(isinstance(t, str) for t in tamper["receiving_turns"])


def test_tamper_cases_count_and_control(tamper):
    """Seven cases: six tampered and one untampered control."""
    cases = tamper["cases"]
    assert len(cases) == 7, len(cases)
    control = [c for c in cases if not c["tampered_transfer_attributes"]]
    tampered = [c for c in cases if c["tampered_transfer_attributes"]]
    assert len(control) == 1, len(control)
    assert len(tampered) == 6, len(tampered)


def test_tamper_cases_bind_the_session_customer(tamper):
    """Every tamper case states the verified session customer, CUST-001."""
    for case in tamper["cases"]:
        assert case["session_customer_id"] == "CUST-001", case["id"]


def test_tamper_includes_the_readme_variant(tamper):
    """The README esc-tamper-003 variant (sentiment, customer_id, authorization summary) is present."""
    ids = {c["id"] for c in tamper["cases"]}
    assert "esc-tamper-003" in ids
    readme = next(c for c in tamper["cases"] if c["id"] == "esc-tamper-003")
    attrs = readme["tampered_transfer_attributes"]
    assert attrs.get("customer_id") == "CUST-777"
    assert "sentiment" in attrs
    assert "summary" in attrs


def test_tamper_covers_nested_customer_id(tamper):
    """One variant tampers a nested context.customer_id, to exercise dotted-key tampering."""
    nested = [c for c in tamper["cases"]
              if any("." in key for key in c["tampered_transfer_attributes"])]
    assert nested, "no tamper case uses a nested dotted key"

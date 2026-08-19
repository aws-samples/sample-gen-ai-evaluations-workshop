"""
Custom Lambda evaluator for the Advanced Prompt Optimization job.

Deployed as ``lambda_function.py`` with handler ``lambda_function.lambda_handler``.

Why a Lambda evaluator instead of LLM-as-a-judge? The Bedrock docs are explicit
about this: for exact-match tasks, a Lambda evaluator is preferable to an
LLM-as-a-judge rubric. BANKING77 intent classification either names the right
intent or it does not, so a deterministic scorer is both cheaper and more
trustworthy than asking a judge model to re-derive the answer.

IMPORTANT: Advanced Prompt Optimization reads this file's source code and the
``compute_score`` docstring to understand what the metric rewards. Clear naming,
comments, and an explanatory docstring measurably improve the feedback the
optimizer generates, so this file is written to be read by a machine as well as
a human.
"""

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# The 77 canonical BANKING77 intent labels.
#
# Two quirks in this vocabulary drive the matching logic below:
#   1. 'Refund_not_showing_up' is the only label containing an uppercase
#      character, so all comparisons must be case-insensitive.
#   2. Some labels are substrings of others ('card_not_working' inside
#      'virtual_card_not_working', 'exchange_rate' inside
#      'card_payment_wrong_exchange_rate'), so substring search has to prefer
#      the longest match or it will silently mis-score.
LABELS = [
    "card_arrival",
    "card_linking",
    "exchange_rate",
    "card_payment_wrong_exchange_rate",
    "extra_charge_on_statement",
    "pending_cash_withdrawal",
    "fiat_currency_support",
    "card_delivery_estimate",
    "automatic_top_up",
    "card_not_working",
    "exchange_via_app",
    "lost_or_stolen_card",
    "age_limit",
    "pin_blocked",
    "contactless_not_working",
    "top_up_by_bank_transfer_charge",
    "pending_top_up",
    "cancel_transfer",
    "top_up_limits",
    "wrong_amount_of_cash_received",
    "card_payment_fee_charged",
    "transfer_not_received_by_recipient",
    "supported_cards_and_currencies",
    "getting_virtual_card",
    "card_acceptance",
    "top_up_reverted",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "card_payment_not_recognised",
    "edit_personal_details",
    "why_verify_identity",
    "unable_to_verify_identity",
    "get_physical_card",
    "visa_or_mastercard",
    "topping_up_by_card",
    "disposable_card_limits",
    "compromised_card",
    "atm_support",
    "direct_debit_payment_not_recognised",
    "passcode_forgotten",
    "declined_cash_withdrawal",
    "pending_card_payment",
    "lost_or_stolen_phone",
    "request_refund",
    "declined_transfer",
    "Refund_not_showing_up",
    "declined_card_payment",
    "pending_transfer",
    "terminate_account",
    "card_swallowed",
    "transaction_charged_twice",
    "verify_source_of_funds",
    "transfer_timing",
    "reverted_card_payment?",
    "change_pin",
    "beneficiary_not_allowed",
    "transfer_fee_charged",
    "receiving_money",
    "failed_transfer",
    "transfer_into_account",
    "verify_top_up",
    "getting_spare_card",
    "top_up_by_cash_or_cheque",
    "order_physical_card",
    "virtual_card_not_working",
    "wrong_exchange_rate_for_cash_withdrawal",
    "get_disposable_virtual_card",
    "top_up_failed",
    "balance_not_updated_after_bank_transfer",
    "cash_withdrawal_not_recognised",
    "exchange_charge",
    "top_up_by_card_charge",
    "activate_my_card",
    "cash_withdrawal_charge",
    "card_about_to_expire",
    "apple_pay_or_google_pay",
    "verify_my_identity",
    "country_support"
]

_CANON = {label.lower(): label for label in LABELS}

# Scoring tiers. Higher is better, which the optimizer requires.
SCORE_EXACT = 1.0      # response is the bare label and nothing else
SCORE_BURIED = 0.6     # right intent, but wrapped in prose we would have to parse
SCORE_WRONG = 0.0      # wrong intent, or no recognisable intent at all


def _squash(text: str) -> str:
    """Lowercase and collapse everything non-alphanumeric to underscores."""
    return re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower()).strip("_")


def _resolve(text: str) -> str:
    """Return the canonical label a response refers to, or '' if none.

    Prefers the longest matching label so that a response of
    'virtual_card_not_working' is not mistaken for 'card_not_working'.
    """
    squashed = _squash(text)
    if squashed in _CANON:
        return _CANON[squashed]
    hits = [low for low in _CANON if low in squashed]
    if not hits:
        return ""
    return _CANON[max(hits, key=len)]


def compute_score(preds: List[str], golds: List[str]) -> Dict[str, Any]:
    """Score BANKING77 intent classifications, rewarding correct AND terse answers.

    This metric is for a very high volume classification workload where the model
    output is consumed by software, not read by a person. Two things matter:
    naming the correct intent, and emitting it in a form a downstream system can
    parse without guessing.

    Scoring, per sample:
      1.0  The response is exactly the correct intent label and nothing else.
           This is the target behaviour.
      0.6  The response names the correct intent but surrounds it with extra
           text such as explanations, markdown, or restated context. The
           classification is right, but a parser now has to extract it, and any
           other intent mentioned in that prose can be picked up by mistake.
           Partial credit, because the reasoning was correct and the fix is a
           formatting instruction.
      0.0  The response names the wrong intent, or names no recognisable intent.

    A high score therefore means: correct intent, emitted bare, with no
    commentary, no markdown, and no trailing punctuation. To raise the score,
    make the prompt state the output contract explicitly - respond with one
    label copied exactly from the list and nothing else.

    Labels are compared case-insensitively and the longest matching label wins,
    because the BANKING77 vocabulary contains one mixed-case label and several
    labels that are substrings of other labels.

    Args:
        preds: Model outputs, one per evaluation sample.
        golds: Expected intent labels, one per evaluation sample.

    Returns:
        A dict with the aggregate ``score``, the per-sample ``scores``, and
        diagnostic counts explaining where the score came from.
    """
    scores: List[float] = []
    n_exact = n_buried = n_wrong = n_unparsed = 0

    for pred, gold in zip(preds, golds):
        gold_canon = _resolve(gold) or (gold or "").strip()
        predicted = _resolve(pred)

        if not predicted:
            scores.append(SCORE_WRONG)
            n_unparsed += 1
            n_wrong += 1
            continue

        if predicted != gold_canon:
            scores.append(SCORE_WRONG)
            n_wrong += 1
            continue

        # Correct intent. Was it emitted cleanly?
        if _squash(pred) == _squash(gold_canon):
            scores.append(SCORE_EXACT)
            n_exact += 1
        else:
            scores.append(SCORE_BURIED)
            n_buried += 1

    total = len(scores)
    return {
        "score": (sum(scores) / total) if total else 0.0,
        "scores": scores,
        "exact_and_bare": n_exact,
        "correct_but_verbose": n_buried,
        "incorrect": n_wrong,
        "no_label_found": n_unparsed,
        "samples": total,
    }


def lambda_handler(event, context):
    """Entry point. Advanced Prompt Optimization sends {"preds": [...], "golds": [...]}.

    Never raises. A crash here would fail the optimization job, so all errors
    become a 0.0 score plus a diagnostic message that the optimizer can read.

    Note: Advanced Prompt Optimization statically validates this file's source
    against an allowlist of builtins before it will run the job. Calling a
    builtin outside that allowlist fails the entire job up front with
    "Metric code validation failed: Calls builtin not in allowlist: <name>".
    The introspection builtin that returns an object's class is one of the
    blocked ones, which is why errors below are reported with str(exc) only.
    Keep this file to plain, obvious Python.
    """
    preds = event.get("preds", []) or []
    golds = event.get("golds", []) or []
    logger.info("Scoring %d predictions against %d golds", len(preds), len(golds))

    try:
        if not preds:
            return {"score": 0.0, "scores": [], "error": "no predictions supplied"}
        if len(golds) < len(preds):
            golds = list(golds) + [""] * (len(preds) - len(golds))
        result = compute_score(preds, golds)
        logger.info("Aggregate score %.4f", result["score"])
        return result
    except Exception as exc:  # noqa: BLE001
        logger.error("Scoring failed: %s", exc, exc_info=True)
        return {"score": 0.0, "scores": [0.0] * len(preds), "error": str(exc)}

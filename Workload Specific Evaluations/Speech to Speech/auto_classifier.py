"""
Auto-classifier for S2S sessions.

Uses an LLM (via Amazon Bedrock) to classify a session's conversation turns
into one of the known validation categories. This is the fallback when a
session has no 'session.category' span attribute (e.g. sessions started
manually via the React UI, or sessions recorded before the metadata injection
was deployed).
"""

import json
import logging
import boto3
from typing import Optional

logger = logging.getLogger(__name__)

# Default classifier model — cheap and fast is fine here
DEFAULT_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


class SessionAutoClassifier:
    """Classifies S2S sessions into known categories using an LLM."""

    def __init__(
        self,
        known_categories: list[str],
        boto3_session: Optional[boto3.Session] = None,
        model_id: str = DEFAULT_MODEL_ID,
    ):
        """
        Args:
            known_categories: List of valid category names from the validation dataset.
            boto3_session: Optional existing boto3 session.
            model_id: Bedrock model to use for classification.
        """
        self.known_categories = known_categories
        self.model_id = model_id

        if boto3_session:
            self.bedrock = boto3_session.client("bedrock-runtime")
        else:
            self.bedrock = boto3.client("bedrock-runtime")

        logger.info(
            f"SessionAutoClassifier ready — {len(known_categories)} categories, model={model_id}"
        )

    def classify(self, session_data: dict) -> Optional[str]:
        """Classify a session dict (as produced by _process_session_spans).

        Returns the best-matching category name, or None if classification fails.
        """
        turns = session_data.get("turns", [])
        if not turns:
            logger.warning(f"Session {session_data.get('sessionId')} has no turns — cannot classify")
            return None

        # Build a compact conversation summary for the prompt
        conversation_lines = []
        for t in turns[:10]:  # cap at 10 turns to keep prompt short
            if t.get("user"):
                conversation_lines.append(f"User: {t['user'][:300]}")
            if t.get("assistant"):
                conversation_lines.append(f"Assistant: {t['assistant'][:300]}")
        conversation_text = "\n".join(conversation_lines)

        categories_list = "\n".join(f"- {c}" for c in self.known_categories)

        prompt = f"""You are classifying a voice conversation into one of the following categories:

{categories_list}

Here is the conversation (first few turns):

{conversation_text}

Reply with ONLY the category name that best matches this conversation. 
Do not explain. Do not add punctuation. Output exactly one of the category names listed above."""

        try:
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 50,
                    "temperature": 0.0,
                    "messages": [{"role": "user", "content": prompt}],
                }),
            )
            raw = json.loads(response["body"].read())
            predicted = raw["content"][0]["text"].strip()

            # Validate the response is one of the known categories
            if predicted in self.known_categories:
                logger.info(
                    f"Auto-classified session {session_data.get('sessionId')} → {predicted}"
                )
                return predicted

            # Fuzzy fallback: case-insensitive match
            lower_map = {c.lower(): c for c in self.known_categories}
            if predicted.lower() in lower_map:
                matched = lower_map[predicted.lower()]
                logger.info(
                    f"Auto-classified (fuzzy) session {session_data.get('sessionId')} → {matched}"
                )
                return matched

            logger.warning(
                f"LLM returned unknown category '{predicted}' for session "
                f"{session_data.get('sessionId')} — skipping"
            )
            return None

        except Exception as e:
            logger.error(
                f"Auto-classification failed for session {session_data.get('sessionId')}: {e}"
            )
            return None

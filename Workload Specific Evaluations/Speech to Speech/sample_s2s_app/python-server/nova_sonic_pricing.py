"""Nova Sonic pricing and cost calculation utility."""

import logging
import os

# Configure logging
DEBUG = os.environ.get("DEBUG")

# Configure logging early
logger = logging.getLogger(__name__)

if DEBUG:
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



class NovaFxSonicCostCalculator:
    """Calculates costs for Amazon Nova Sonic models based on token usage."""

    # Nova Sonic pricing information (USD per 1000 tokens)
    NOVA_SONIC_PRICING = {
        "speech_input": 0.0034,      # $0.0034 per 1000 speech input tokens
        "speech_output": 0.0136,     # $0.0136 per 1000 speech output tokens
        "text_input": 0.00006,       # $0.00006 per 1000 text input tokens
        "text_output": 0.00024       # $0.00024 per 1000 text output tokens
    }

    def __init__(self, debug: bool = False):
        """Initialize the cost calculator.

        Args:
            debug: Enable debug logging
        """
        self.debug = debug

    def debug_print(self, message: str):
        """Print only if debug mode is enabled."""
        if DEBUG:
            logger.debug(message)
        else:
            logger.info(message)


    def calculate_cost(self, token_usage: dict) -> float:
        """Calculate the cost based on token usage and Nova Sonic pricing.

        Args:
            token_usage: Dictionary containing token usage information with structure:
                {
                    "input": {
                        "speechTokens": int,
                        "textTokens": int
                    },
                    "output": {
                        "speechTokens": int,
                        "textTokens": int
                    }
                }

        Returns:
            Total cost in USD
        """
        if not token_usage:
            return 0.0

        speech_input_tokens = token_usage.get("input", {}).get("speechTokens", 0)
        text_input_tokens = token_usage.get("input", {}).get("textTokens", 0)
        speech_output_tokens = token_usage.get("output", {}).get("speechTokens", 0)
        text_output_tokens = token_usage.get("output", {}).get("textTokens", 0)

        # Calculate cost components (convert from price per 1000 tokens)
        speech_input_cost = (speech_input_tokens / 1000) * self.NOVA_SONIC_PRICING["speech_input"]
        text_input_cost = (text_input_tokens / 1000) * self.NOVA_SONIC_PRICING["text_input"]
        speech_output_cost = (speech_output_tokens / 1000) * self.NOVA_SONIC_PRICING["speech_output"]
        text_output_cost = (text_output_tokens / 1000) * self.NOVA_SONIC_PRICING["text_output"]

        # Calculate total cost
        total_cost = speech_input_cost + text_input_cost + speech_output_cost + text_output_cost

        return total_cost

    def get_pricing_breakdown(self, token_usage: dict) -> dict:
        """Get a breakdown of costs by token type.

        Args:
            token_usage: Dictionary containing token usage information

        Returns:
            Dictionary with cost breakdown
        """
        if not token_usage:
            return {
                "speech_input_cost": 0.0,
                "text_input_cost": 0.0,
                "speech_output_cost": 0.0,
                "text_output_cost": 0.0,
                "total_cost": 0.0
            }

        speech_input_tokens = token_usage.get("input", {}).get("speechTokens", 0)
        text_input_tokens = token_usage.get("input", {}).get("textTokens", 0)
        speech_output_tokens = token_usage.get("output", {}).get("speechTokens", 0)
        text_output_tokens = token_usage.get("output", {}).get("textTokens", 0)

        speech_input_cost = (speech_input_tokens / 1000) * self.NOVA_SONIC_PRICING["speech_input"]
        text_input_cost = (text_input_tokens / 1000) * self.NOVA_SONIC_PRICING["text_input"]
        speech_output_cost = (speech_output_tokens / 1000) * self.NOVA_SONIC_PRICING["speech_output"]
        text_output_cost = (text_output_tokens / 1000) * self.NOVA_SONIC_PRICING["text_output"]

        total_cost = speech_input_cost + text_input_cost + speech_output_cost + text_output_cost

        return {
            "speech_input_cost": speech_input_cost,
            "text_input_cost": text_input_cost,
            "speech_output_cost": speech_output_cost,
            "text_output_cost": text_output_cost,
            "total_cost": total_cost,
            "speech_input_tokens": speech_input_tokens,
            "text_input_tokens": text_input_tokens,
            "speech_output_tokens": speech_output_tokens,
            "text_output_tokens": text_output_tokens
        }

"""Session state enumeration for Bedrock streaming sessions."""

from enum import Enum


class SessionState(Enum):
    """Enumeration of possible session states"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    TRANSITIONING = "transitioning"
    CLOSING = "closing"
    CLOSED = "closed"

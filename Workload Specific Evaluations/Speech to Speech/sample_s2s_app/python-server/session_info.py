"""SessionInfo class for tracking Bedrock streaming session metadata."""

import time
from session_state import SessionState


class SessionInfo:
    """Information about a streaming session"""
    def __init__(self, session_id: str, stream_manager, start_time: float):
        self.session_id = session_id
        self.stream_manager = stream_manager
        self.start_time = start_time
        self.state = SessionState.INITIALIZING
        self.received_speculative_text = False
        self.received_completion_start = False
        self.barge_in_detected = False
        # Track current content generation stage
        self.current_generation_stage = None
        self.current_content_role = None
        self.current_content_type = None  # Track if it's TEXT or AUDIO
        # Track model's session ID
        self.model_session_id = None
        # Track if audio contentStart has been sent
        self.audio_content_started = False
        # Track when we last received output from this session (for timeout detection)
        self.last_output_time = time.time()
        # Track if we've received FINAL text (indicates assistant finished generating)
        self.received_final_text = False
        # Track speculative and final text counts for matching
        self.speculative_text_count = 0
        self.final_text_count = 0
        # Track if we've received audio contentEnd with END_TURN or INTERRUPTED
        self.received_audio_end_turn = False
        self.audio_stop_reason = None

    def get_duration(self) -> float:
        """Get the duration of the session in seconds"""
        return time.time() - self.start_time

    def should_transition(self, threshold: float) -> bool:
        """Check if session should transition based on duration"""
        return self.get_duration() >= threshold

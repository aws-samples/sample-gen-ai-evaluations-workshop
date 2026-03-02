"""AudioBuffer class for storing audio chunks during session transitions."""

from collections import deque
from typing import Deque


class AudioBuffer:
    """Buffer for storing audio chunks during transition"""
    def __init__(self, max_duration_seconds: float, sample_rate: int = 16000, sample_width: int = 2):
        self.max_duration_seconds = max_duration_seconds
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        # Calculate max buffer size in bytes
        self.max_buffer_size = int(max_duration_seconds * sample_rate * sample_width)
        self.buffer: Deque[bytes] = deque()
        self.total_size = 0

    def add_chunk(self, audio_chunk: bytes):
        """Add an audio chunk to the buffer"""
        self.buffer.append(audio_chunk)
        self.total_size += len(audio_chunk)

        # Remove old chunks if buffer exceeds max size
        while self.total_size > self.max_buffer_size and self.buffer:
            removed = self.buffer.popleft()
            self.total_size -= len(removed)

    def get_all_chunks(self) -> list:
        """Get all buffered audio chunks"""
        return list(self.buffer)

    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
        self.total_size = 0

    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        return len(self.buffer) == 0

"""ConversationHistory class for managing conversation history during session transitions."""

import time
import json
import uuid


class ConversationHistory:
    """Manages conversation history for session transitions"""
    def __init__(self, max_single_message_bytes: int = 1024, max_chat_history_bytes: int = 40960):
        self.messages = []
        self.max_single_message_bytes = max_single_message_bytes
        self.max_chat_history_bytes = max_chat_history_bytes

    def add_message(self, role: str, content: str, message_type: str = "text") -> dict:
        """Add a message to the history with byte limit enforcement

        Returns:
            dict with keys:
            - 'truncated': bool - whether message was truncated
            - 'messages_removed': int - number of old messages removed
            - 'total_bytes': int - total history size in bytes
        """
        was_truncated = False

        # Truncate message content if it exceeds single message limit
        content_bytes = content.encode('utf-8')
        if len(content_bytes) > self.max_single_message_bytes:
            was_truncated = True
            # Truncate to fit within limit, leaving room for truncation marker
            truncation_marker = "... [truncated]"
            max_content_bytes = self.max_single_message_bytes - len(truncation_marker.encode('utf-8'))
            # Decode safely, handling potential mid-character truncation
            truncated_content = content_bytes[:max_content_bytes].decode('utf-8', errors='ignore')
            content = truncated_content + truncation_marker

        messages_before = len(self.messages)

        self.messages.append({
            "role": role,
            "content": content,
            "type": message_type,
            "timestamp": time.time()
        })

        # Trim history to stay within total byte limit
        self._trim_history()

        messages_removed = messages_before - len(self.messages) + 1

        return {
            'truncated': was_truncated,
            'messages_removed': messages_removed if messages_removed > 1 else 0,
            'total_bytes': self._get_total_size_bytes()
        }

    def _get_message_size_bytes(self, message: dict) -> int:
        """Calculate the byte size of a message"""
        # Size includes role + content + metadata
        return len(message["content"].encode('utf-8')) + len(message["role"].encode('utf-8'))

    def _get_total_size_bytes(self) -> int:
        """Calculate total byte size of all messages"""
        return sum(self._get_message_size_bytes(msg) for msg in self.messages)

    def _trim_history(self):
        """Trim history to stay within configured byte limits"""
        # Remove oldest messages until we're under the total byte limit
        while self.messages and self._get_total_size_bytes() > self.max_chat_history_bytes:
            self.messages.pop(0)  # Remove oldest message

    def get_history_events(self, prompt_name: str) -> list:
        """Get conversation history as Bedrock events, splitting large messages if needed"""
        events = []

        for message in self.messages:
            role = message["role"].upper()
            content = message["content"]
            content_bytes = content.encode('utf-8')

            # If content is larger than max_single_message_bytes, split it
            if len(content_bytes) > self.max_single_message_bytes:
                # Split into chunks
                chunks = []
                while len(content_bytes) > 0:
                    chunk = content_bytes[:self.max_single_message_bytes]
                    chunks.append(chunk.decode('utf-8', errors='ignore'))
                    content_bytes = content_bytes[self.max_single_message_bytes:]

                # Send each chunk as a separate content block
                for chunk in chunks:
                    content_name = str(uuid.uuid4())

                    # Content start event
                    content_start = {
                        "event": {
                            "contentStart": {
                                "promptName": prompt_name,
                                "contentName": content_name,
                                "type": "TEXT",
                                "role": role,
                                "interactive": False,
                                "textInputConfiguration": {
                                    "mediaType": "text/plain"
                                }
                            }
                        }
                    }
                    events.append(json.dumps(content_start))

                    # Text input event
                    text_input = {
                        "event": {
                            "textInput": {
                                "promptName": prompt_name,
                                "contentName": content_name,
                                "content": chunk
                            }
                        }
                    }
                    events.append(json.dumps(text_input))

                    # Content end event
                    content_end = {
                        "event": {
                            "contentEnd": {
                                "promptName": prompt_name,
                                "contentName": content_name
                            }
                        }
                    }
                    events.append(json.dumps(content_end))
            else:
                # Normal case: single content block
                content_name = str(uuid.uuid4())

                # Content start event
                content_start = {
                    "event": {
                        "contentStart": {
                            "promptName": prompt_name,
                            "contentName": content_name,
                            "type": "TEXT",
                            "role": role,
                            "interactive": False,
                            "textInputConfiguration": {
                                "mediaType": "text/plain"
                            }
                        }
                    }
                }
                events.append(json.dumps(content_start))

                # Text input event
                text_input = {
                    "event": {
                        "textInput": {
                            "promptName": prompt_name,
                            "contentName": content_name,
                            "content": content
                        }
                    }
                }
                events.append(json.dumps(text_input))

                # Content end event
                content_end = {
                    "event": {
                        "contentEnd": {
                            "promptName": prompt_name,
                            "contentName": content_name
                        }
                    }
                }
                events.append(json.dumps(content_end))

        return events

    def clear(self):
        """Clear the conversation history"""
        self.messages.clear()

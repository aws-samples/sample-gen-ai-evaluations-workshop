"""OpenTelemetry span management utility for S2S session tracing."""

import json
from opentelemetry import baggage, context, trace
import logging
import os


# Configure logging
DEBUG = os.environ.get("DEBUG")
LOG_LEVEL = os.environ.get("LOG_LEVEL")
logger = logging.getLogger(__name__)
if LOG_LEVEL:
    logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')


class OpenTelemetrySpanManager:
    """Manages OpenTelemetry spans and telemetry context for S2S sessions."""

    def __init__(self, session_id: str, model_id: str, region: str, debug: bool = False):
        """Initialize the span manager.

        Args:
            session_id: Unique session identifier
            model_id: AWS model identifier (e.g., 'amazon.nova-sonic-v1:0')
            region: AWS region
            debug: Enable debug logging
        """
        self.session_id = session_id
        self.model_id = model_id
        self.region = region
        self.debug = debug
        self.session_span = None
        self.context_token = None
        self.prompt_start_metadata = None  # Store promptStart details for later use

        # Track content metadata by contentId and contentName
        self.content_stages = {}  # Maps contentId to generationStage
        self.content_roles = {}  # Maps contentName to role (SYSTEM, USER, ASSISTANT, TOOL)

    def debug_print(self, message: str):
        """Print only if debug mode is enabled."""
        if DEBUG:
            logger.info(message)

    def create_session_span(self):
        """Create the session span for telemetry when the session manager is initialized."""
        if not self.session_id:
            raise ValueError("Session ID is required")

        # Set session context for telemetry
        self.context_token = self.set_session_context(self.session_id)
        self.debug_print(f"Session context set with token: {self.context_token}")

        # Get tracer for main application
        try:
            tracer = trace.get_tracer("s2s_agent", "1.0.0")
            # Create the session span
            self.session_span = tracer.start_span(self.session_id)
            if hasattr(self.session_span, "set_attribute"):
                self.session_span.set_attribute("session.id", self.session_id)
                self.session_span.set_attribute("model.id", self.model_id)
                self.session_span.set_attribute("region", self.region)
        except Exception as telemetry_error:
            raise

    def set_category(self, category: str):
        """Attach a test scenario category label to the session span.

        This allows the evaluator to read the category directly from CloudWatch
        spans instead of requiring manual annotation.

        Args:
            category: Scenario category name (e.g. 'OrderAssistant')
        """
        if self.session_span and hasattr(self.session_span, "set_attribute"):
            self.session_span.set_attribute("session.category", category)
            self.debug_print(f"Set session.category={category} on session span")

    def set_session_context(self, session_id: str):
        """Set the session ID in OpenTelemetry baggage for trace correlation."""
        ctx = baggage.set_baggage("session.id", session_id)
        token = context.attach(ctx)
        return token

    def create_child_span(
        self,
        name: str,
        input=None,
        parent_span=None,
        metadata=None,
        output=None,
    ):
        """Create a child span for telemetry using OpenTelemetry.

        Args:
            name: Span name
            input: Input data for the span
            parent_span: Parent span (if None, uses current active span)
            metadata: Additional metadata
            output: Output data for the span

        Returns:
            The created span
        """
        try:
            self.debug_print(f"Creating child span: {name}")
            # Get a tracer for the agent
            tracer = trace.get_tracer("s2s_agent", "1.0.0")

            # Start a new span as a child of the parent span if provided
            span_context = None
            if parent_span and isinstance(parent_span, trace.Span):
                # If we have a parent span, use its context
                self.debug_print("Using provided parent span for child span")
                span_context = trace.set_span_in_context(parent_span)

            # Create the span with the provided name
            span = tracer.start_span(name, context=span_context)

            # Add standard attributes
            if hasattr(span, "set_attribute"):
                span.set_attribute("session.id", self.session_id)

                # Add input data if provided
                if input:
                    self._add_attributes_to_span(span, input, "input")

                # Add metadata if provided
                if metadata:
                    self._add_attributes_to_span(span, metadata, "")

                # Add output data if provided
                if output:
                    self._add_attributes_to_span(span, output, "output")

                # Add start time event
                span.add_event("span_started")
            return span
        except Exception as e:
            raise

    def _add_attributes_to_span(self, span, data, prefix: str = ""):
        """Recursively add attributes to a span from complex data structures.

        Args:
            span: The OpenTelemetry span to add attributes to
            data: The data to add (can be dict, list, or primitive)
            prefix: The attribute name prefix
        """
        if not hasattr(span, "set_attribute"):
            return

        def _flatten_and_add(obj, current_prefix: str = ""):
            """Recursively flatten nested objects and add as span attributes."""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_prefix = f"{current_prefix}.{key}" if current_prefix else key
                    if isinstance(value, (dict, list)):
                        # For complex nested objects, serialize to JSON string
                        try:
                            json_str = json.dumps(value)
                            # Truncate very long JSON strings
                            if len(json_str) > 1000:
                                json_str = json_str[:997] + "..."
                            span.set_attribute(new_prefix, json_str)
                        except (TypeError, ValueError):
                            # If JSON serialization fails, convert to string
                            str_value = str(value)
                            if len(str_value) > 1000:
                                str_value = str_value[:997] + "..."
                            span.set_attribute(new_prefix, str_value)
                    elif isinstance(value, (str, int, float, bool, type(None))):
                        # Handle primitive types directly
                        if value is None:
                            span.set_attribute(new_prefix, "null")
                        else:
                            str_value = str(value)
                            # Truncate very long strings
                            if len(str_value) > 1000:
                                str_value = str_value[:997] + "..."
                            span.set_attribute(new_prefix, str_value)
                    else:
                        # For other types, convert to string
                        str_value = str(value)
                        if len(str_value) > 1000:
                            str_value = str_value[:997] + "..."
                        span.set_attribute(new_prefix, str_value)
            elif isinstance(obj, list):
                # For lists, serialize to JSON string
                try:
                    json_str = json.dumps(obj)
                    if len(json_str) > 1000:
                        json_str = json_str[:997] + "..."
                    span.set_attribute(current_prefix or "list", json_str)
                except (TypeError, ValueError):
                    str_value = str(obj)
                    if len(str_value) > 1000:
                        str_value = str_value[:997] + "..."
                    span.set_attribute(current_prefix or "list", str_value)
            else:
                # For primitive types or other objects
                if obj is None:
                    span.set_attribute(current_prefix or "value", "null")
                else:
                    str_value = str(obj)
                    if len(str_value) > 1000:
                        str_value = str_value[:997] + "..."
                    span.set_attribute(current_prefix or "value", str_value)

        try:
            _flatten_and_add(data, prefix)
        except Exception as e:
            # Fallback: add as simple string
            try:
                fallback_value = str(data)
                if len(fallback_value) > 1000:
                    fallback_value = fallback_value[:997] + "..."
                span.set_attribute(prefix or "data", fallback_value)
            except Exception as fallback_error:
                raise

    def end_span_safely(
        self,
        span,
        output=None,
        level: str = "INFO",
        status_message: str = None,
        end_time=None,
        metadata=None,
    ):
        """End a span safely with additional attributes using OpenTelemetry.

        Args:
            span: The span to end
            output: Output data for the span
            level: Log level (INFO or ERROR)
            status_message: Status message
            end_time: End time (unused, kept for compatibility)
            metadata: Additional metadata
        """
        try:
            if not span:
                return

            # Add output data if provided
            if output and hasattr(span, "set_attribute"):
                self._add_attributes_to_span(span, output, "output")

            # Add additional metadata if provided
            if metadata and hasattr(span, "set_attribute"):
                self._add_attributes_to_span(span, metadata, "")

            # Set span status based on level
            if hasattr(span, "set_status"):
                if level == "ERROR":
                    error_message = status_message or "An error occurred"
                    span.set_status(trace.Status(trace.StatusCode.ERROR, error_message))
                    if hasattr(span, "add_event"):
                        span.add_event("error", {"message": error_message})
                else:
                    span.set_status(trace.Status(trace.StatusCode.OK))

            # Add end time event
            if hasattr(span, "add_event"):
                span.add_event("span_ended")

            # End the span
            span.end()

        except Exception as e:
            raise

    def create_event_span(self, event_data, session_id=None):
        """Create a child span for a specific S2S event.

        Args:
            event_data: The event data dict containing the 'event' key
            session_id: Optional session ID to include in metadata

        Returns:
            The created span object, or None if no event type matched
        """
        if "event" not in event_data:
            return None

        event_type = list(event_data["event"].keys())[0]
        event_details = event_data["event"].get(event_type, {})
        span = None

        try:
            if event_type == "sessionStart":
                span = self.create_child_span(
                    "sessionStart",
                    parent_span=self.session_span,
                    input=event_details,
                    metadata={"session_id": session_id or self.session_id}
                )

            elif event_type == "sessionEnd":
                span = self.create_child_span(
                    "sessionEnd",
                    parent_span=self.session_span,
                    input=event_details,
                    metadata={"session_id": session_id or self.session_id}
                )
                # End the sessionEnd span immediately since it marks the end of session
                if span:
                    self.end_span_safely(span, output=event_details)

            elif event_type == "promptStart":
                # Store promptStart metadata to include in systemPrompt span
                tool_config = event_details.get("toolConfiguration")
                self.prompt_start_metadata = {
                    "prompt_name": event_details.get("promptName"),
                    "audio_output_config": event_details.get("audioOutputConfiguration"),
                    "tool_config": json.dumps(tool_config) if tool_config else None
                }
                # Skip creating a separate promptStart span
                pass

            elif event_type == "contentStart":
                # Track generation stage by contentId and role by both contentId and contentName
                content_id = event_details.get("contentId")
                content_name = event_details.get("contentName")
                role = event_details.get("role", "UNKNOWN")

                # Store role by contentName (for textInput lookup)
                if content_name:
                    self.content_roles[content_name] = role
                    self.debug_print(f"Tracked role for contentName {content_name}: {role}")

                # Store role by contentId (for textOutput lookup)
                if content_id:
                    self.content_roles[content_id] = role
                    self.debug_print(f"Tracked role for contentId {content_id}: {role}")

                # Track generation stage by contentId for textOutput filtering
                additional_fields = event_details.get("additionalModelFields", "{}")
                try:
                    fields_dict = json.loads(additional_fields)
                    generation_stage = fields_dict.get("generationStage", "FINAL")
                    self.content_stages[content_id] = generation_stage
                    self.debug_print(f"Tracked generation stage for {content_id}: {generation_stage}")
                except Exception as e:
                    self.content_stages[content_id] = "FINAL"  # Default to FINAL
                    self.debug_print(f"Failed to parse generation stage, defaulting to FINAL: {e}")
                # Skip creating a contentStart span
                pass

            elif event_type == "textInput":
                content_name = event_details.get("contentName")

                # Look up role from contentStart using contentName
                role = self.content_roles.get(content_name, "USER")
                self.debug_print(f"Creating event span for {event_type} with role: {role} (contentName: {content_name})")

                # Determine span name based on role
                if role == "SYSTEM":
                    span_name = "systemPrompt"
                elif role == "ASSISTANT":
                    span_name = "assistantOutput"
                else:
                    span_name = "userInput"

                # For systemPrompt, include promptStart metadata
                metadata = {
                    "session_id": session_id or self.session_id,
                    "role": role,
                    "prompt_name": event_details.get("promptName"),
                    "content_name": content_name
                }

                if span_name == "systemPrompt" and self.prompt_start_metadata:
                    self.debug_print(f"Adding promptStart metadata to systemPrompt span: {self.prompt_start_metadata}")
                    metadata.update(self.prompt_start_metadata)

                span = self.create_child_span(
                    span_name,
                    parent_span=self.session_span,
                    input=event_details.get("content", ""),
                    metadata=metadata
                )

            elif event_type == "audioInput":
                # Skip audioInput spans - too granular
                pass

            elif event_type == "contentEnd":
                # Skip contentEnd spans - too granular
                pass

            elif event_type == "textOutput":
                content_id = event_details.get("contentId")
                content_name = event_details.get("contentName")

                # Look up role from contentStart: try contentId first, then contentName
                role = self.content_roles.get(content_id)
                if not role and content_name:
                    role = self.content_roles.get(content_name)
                if not role:
                    role = "ASSISTANT"  # Default fallback

                # Determine span name based on role
                if role == "USER":
                    span_name = "userInput"
                elif role == "SYSTEM":
                    span_name = "systemPrompt"
                else:
                    span_name = "assistantOutput"

                # Generation stage check only applies to userInput and assistantOutput
                # systemPrompt never has a generation stage
                if span_name == "systemPrompt":
                    # Always create systemPrompt spans
                    self.debug_print(f"Creating {span_name} span for contentId {content_id} with role {role}")
                    span = self.create_child_span(
                        span_name,
                        parent_span=self.session_span,
                        input=event_details.get("content", ""),
                        metadata={
                            "session_id": session_id or self.session_id,
                            "role": role,
                            "content_id": content_id,
                            "content_name": content_name,
                            "prompt_name": event_details.get("promptName")
                        }
                    )
                else:
                    # For userInput and assistantOutput, check generation stage
                    generation_stage = self.content_stages.get(content_id, "FINAL")

                    if generation_stage == "FINAL":
                        self.debug_print(f"Creating {span_name} span for contentId {content_id} with role {role} (generation: {generation_stage})")
                        span = self.create_child_span(
                            span_name,
                            parent_span=self.session_span,
                            input=event_details.get("content", ""),
                            metadata={
                                "session_id": session_id or self.session_id,
                                "role": role,
                                "content_id": content_id,
                                "content_name": content_name,
                                "prompt_name": event_details.get("promptName"),
                                "generation_stage": generation_stage
                            }
                        )
                    else:
                        self.debug_print(f"Skipping {generation_stage} {span_name} for contentId {content_id}")

            elif event_type == "audioOutput":
                # Skip audioOutput spans - too granular
                pass

            elif event_type == "usageEvent":
                # Skip usageEvent spans - handled at session span level
                pass

            elif event_type == "toolUse":
                span = self.create_child_span(
                    "toolUse",
                    parent_span=self.session_span,
                    input={
                        "toolName": event_details.get("toolName"),
                        "toolUseId": event_details.get("toolUseId")
                    },
                    metadata={
                        "session_id": session_id or self.session_id,
                        "tool_name": event_details.get("toolName"),
                        "tool_use_id": event_details.get("toolUseId"),
                        "content_id": event_details.get("contentId")
                    }
                )

            return span

        except Exception as e:
            self.debug_print(f"Error creating event span for {event_type}: {e}")
            raise

    def close(self):
        """Close the span manager and end the session span."""
        if self.session_span:
            self.end_span_safely(self.session_span)
            self.session_span = None

        if self.context_token:
            context.detach(self.context_token)
            self.context_token = None

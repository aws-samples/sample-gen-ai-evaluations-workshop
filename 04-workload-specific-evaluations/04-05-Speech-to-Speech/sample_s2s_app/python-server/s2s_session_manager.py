import asyncio
import json
import base64
import warnings
import uuid
import sys
from s2s_events import S2sEvent
import time
import logging
import os
from concurrent.futures import InvalidStateError

from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient, InvokeModelWithBidirectionalStreamOperationInput
from aws_sdk_bedrock_runtime.models import InvokeModelWithBidirectionalStreamInputChunk, BidirectionalInputPayloadPart, ServiceError
from aws_sdk_bedrock_runtime.config import Config
from smithy_aws_core.identity.environment import EnvironmentCredentialsResolver
from tool_registry import ToolRegistry


# load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore")

# Custom logging filter to suppress expected AWS CRT cancellation errors
class SuppressCRTCancellationFilter(logging.Filter):
    """Filter to suppress expected InvalidStateError from AWS CRT during cleanup"""
    def filter(self, record):
        # Suppress AWS CRT cancellation errors
        message = record.getMessage()
        if 'InvalidStateError' in message and 'CANCELLED' in message:
            return False
        if 'Treating Python exception as error' in message and 'AWS_ERROR_UNKNOWN' in message:
            return False
        return True

# Apply filter to root logger to catch AWS CRT errors
logging.getLogger().addFilter(SuppressCRTCancellationFilter())

# Suppress expected InvalidStateError from AWS CRT during cleanup
def _custom_excepthook(exctype, value, traceback_obj):
    """Custom exception hook to suppress expected InvalidStateError from AWS CRT"""
    if exctype == InvalidStateError and 'CANCELLED' in str(value):
        # This is expected when cancelling AWS CRT streams - don't print traceback
        return
    # For all other exceptions, use default handling
    sys.__excepthook__(exctype, value, traceback_obj)

# Install custom exception hook
sys.excepthook = _custom_excepthook

# Configure logging
DEBUG = os.environ.get("DEBUG")
LOG_LEVEL = os.environ.get("LOG_LEVEL")
logger = logging.getLogger(__name__)

# Filter out noisy log messages related to audio data
class AudioDataFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.audio_message_count = 0
        self.sample_rate = 100  # Log every 1 out of 100 messages

    def filter(self, record):
        message = record.getMessage()

        # Always filter out "audio chunk" messages
        if "audio chunk" in message:
            return False

        # Sample audioInput and audioOutput messages (1 out of every 100)
        if "audioInput" in message or "audioOutput" in message:
            self.audio_message_count += 1
            if self.audio_message_count % self.sample_rate != 0:
                return False
            # Log this sampled message but reset counter for visibility

        return True

if LOG_LEVEL:
    logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
    # Apply filter to all handlers
    for handler in logging.root.handlers:
        handler.addFilter(AudioDataFilter())

def debug_print(message):
    """Print only if debug mode is enabled"""
    if DEBUG:
        logger.info(message)

# Bedrock rate limit handling constants
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # 1 second base delay


class S2sSessionManager:
    """Manages bidirectional streaming with AWS Bedrock using asyncio"""

    def __init__(self, model_id='amazon.nova-sonic-v1:0', region='us-east-1',
                 session_id=None, voice_id=None,
                 system_prompt=None, tool_config=None, inference_config=None,
                 hello_audio_played=False, span_manager=None, cost_calculator=None,
                 token_usage=None, usage_events=None, enable_recording=False):
        """Initialize the stream manager for a single Bedrock session."""
        self.model_id = model_id
        self.region = region
        self.session_start_time = time.time()
        self.voice_id = voice_id  # Voice ID for audio output (from frontend)

        # Audio recording setup
        self.enable_recording = enable_recording
        self.input_audio_file = None
        self.output_audio_file = None
        if self.enable_recording and session_id:
            self._initialize_recording(session_id)

        # Frontend configuration (passed from server.py)
        self.frontend_system_prompt = system_prompt
        self.frontend_tool_config = tool_config
        self.frontend_inference_config = inference_config

        # Flag to track whether hello audio has been played (shared across sessions)
        self.hello_audio_played = hello_audio_played

        # Audio and output queues
        self.audio_input_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()

        self.response_task = None
        self.audio_task = None
        self.stream = None
        self.is_active = False
        self.bedrock_client = None

        # Session information for event processing
        self.prompt_name = None  # Set by initialize_session_with_prompt
        self.audio_content_name = None  # Set from contentStart events
        self.session_id = session_id  # Used for telemetry tracking

        # Tool use state tracking
        self.toolUseContent = ""
        self.toolUseId = ""
        self.toolName = ""
        self.toolUseContentId = None  # Track contentId for tool use
        self.tool_processing_tasks = set()

        # Track content generation stages by contentId
        self.content_stages = {}  # Maps contentId to generationStage

        # Track processed tool use IDs to prevent duplicate execution
        self.processed_tool_use_ids = set()

        # Task tracking for proper cleanup
        self.tasks = set()

        # Track in-progress tool calls (for async tool execution)
        self.pending_tool_tasks = {}

        # Initialize tool registry for tool execution
        try:
            # self.tool_registry = ToolRegistry(use_strands_agent=True)
            logger.info("ToolRegistry initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize ToolRegistry: {e}")
            self.tool_registry = None

        # Telemetry and token usage - Use references from SessionTransitionManager
        # These track the entire conversation across all sessions
        self.span_manager = span_manager
        self.cost_calculator = cost_calculator
        self.token_usage = token_usage if token_usage is not None else {
            "totalInputTokens": 0,
            "totalOutputTokens": 0,
            "totalTokens": 0,
            "details": {
                "input": {
                    "speechTokens": 0,
                    "textTokens": 0
                },
                "output": {
                    "speechTokens": 0,
                    "textTokens": 0
                }
            }
        }
        self.usage_events = usage_events if usage_events is not None else []

        # Lock for thread-safe operations
        self.lock = asyncio.Lock()

    def _initialize_client(self):
        """Initialize the Bedrock client."""
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        )
        self.bedrock_client = BedrockRuntimeClient(config=config)

    def _initialize_recording(self, session_id):
        """Initialize audio recording files for input and output."""
        import wave
        import os

        # Create session-specific recording directory
        recording_dir = os.path.join(os.path.dirname(__file__), 'recordings', session_id)
        os.makedirs(recording_dir, exist_ok=True)

        # Initialize WAV files for input and output (16kHz, mono, 16-bit PCM)
        input_path = os.path.join(recording_dir, 'input.wav')
        output_path = os.path.join(recording_dir, 'output.wav')

        self.input_audio_file = wave.open(input_path, 'wb')
        self.input_audio_file.setnchannels(1)  # Mono
        self.input_audio_file.setsampwidth(2)  # 16-bit
        self.input_audio_file.setframerate(16000)  # 16kHz

        self.output_audio_file = wave.open(output_path, 'wb')
        self.output_audio_file.setnchannels(1)  # Mono
        self.output_audio_file.setsampwidth(2)  # 16-bit
        self.output_audio_file.setframerate(24000)  # 24kHz to match Bedrock output

        logger.info(f"Recording initialized: {recording_dir}")

    def _write_input_audio(self, audio_bytes):
        """Write audio data to input recording file."""
        if self.enable_recording and self.input_audio_file:
            try:
                self.input_audio_file.writeframes(audio_bytes)
            except Exception as e:
                logger.error(f"Error writing input audio: {e}")

    def _write_output_audio(self, audio_bytes):
        """Write audio data to output recording file."""
        if self.enable_recording and self.output_audio_file:
            try:
                self.output_audio_file.writeframes(audio_bytes)
            except Exception as e:
                logger.error(f"Error writing output audio: {e}")

    async def initialize_stream(self):
        """Initialize the bidirectional stream with Bedrock.

        Note: This only creates the stream connection. Tasks are started later
        in initialize_session_with_prompt() after sessionStart is sent.
        """
        try:
            if not self.bedrock_client:
                self._initialize_client()

            debug_print(f"Initializing new stream with model {self.model_id} in region {self.region}")

            # Initialize the stream with retry logic
            retry_count = 0
            while retry_count < MAX_RETRIES:
                try:
                    # Initialize the stream
                    debug_print(f"Calling invoke_model_with_bidirectional_stream for {self.model_id}...")
                    self.stream = await self.bedrock_client.invoke_model_with_bidirectional_stream(
                        InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
                    )
                    debug_print("invoke_model_with_bidirectional_stream call completed")
                    self.is_active = True
                    debug_print(f"Stream is now active: {self.is_active}")
                    break
                except ServiceError as e:
                    retry_count += 1
                    if 'retry-after' in str(e).lower() and retry_count < MAX_RETRIES:
                        # Extract retry time if available or use exponential backoff
                        retry_time = RETRY_BASE_DELAY * (2 ** (retry_count - 1))
                        logger.warning(f"Rate limited by Bedrock API, retrying in {retry_time} seconds (attempt {retry_count}/{MAX_RETRIES})")
                        await asyncio.sleep(retry_time)
                    else:
                        if retry_count >= MAX_RETRIES:
                            logger.error(f"Failed to initialize stream after {MAX_RETRIES} retries")
                        raise

            # Wait a bit to ensure stream is ready
            debug_print("Waiting 0.1s for stream setup to complete...")
            await asyncio.sleep(0.1)
            debug_print("Wait completed")

            debug_print("Stream initialized successfully")

            return self
        except Exception as e:
            self.is_active = False
            logger.error(f"Failed to initialize stream: {str(e)}")
            raise
    
    async def send_raw_event(self, event_data, retry_count=0):
        """Send a raw event to the Bedrock stream."""
        if not self.stream or not self.is_active:
            debug_print(f"Stream not initialized or closed, stream is set to active: {self.is_active}")
            return

        # Create event span using utility method
        event_span = self.span_manager.create_event_span(event_data, session_id=self.session_id)

        # Prevent infinite retries
        max_retries = 2
        if retry_count >= max_retries:
            logger.error(f"Maximum retry attempts ({max_retries}) exceeded for sending event")
            return

        try:
            event_json = json.dumps(event_data)

            # Create and send the event
            event = InvokeModelWithBidirectionalStreamInputChunk(
                value=BidirectionalInputPayloadPart(bytes_=event_json.encode('utf-8'))
            )

            # Check if stream is still active before sending
            if not self.is_active or not self.stream:
                debug_print("Stream is no longer active, cannot send event")
                return

            await self.stream.input_stream.send(event)

            # End event span with success
            if event_span:
                event_type = list(event_data["event"].keys())[0] if "event" in event_data else "unknown"
                self.span_manager.end_span_safely(
                    event_span,
                    output={"status": "sent", "event_type": event_type}
                )

            # Close session if session end event
            if "event" in event_data and "sessionEnd" in event_data["event"]:
                # Wait a moment for the event to be processed before closing
                await asyncio.sleep(0.2)
                debug_print("Closing stream given we received a sessionEnd event")
                await self.close()

        except ServiceError as e:
            # End event span with error if created
            if event_span:
                self.span_manager.end_span_safely(
                    event_span,
                    level="ERROR",
                    status_message=f"Error sending event: {str(e)}"
                )

            if 'retry-after' in str(e).lower():
                logger.warning(f"Rate limited by Bedrock API: {e}")
                # Wait before next attempt
                await asyncio.sleep(1.0)
            else:
                debug_print(f"Error sending event: {str(e)}")

        except Exception as e:
            if event_span:
                self.span_manager.end_span_safely(event_span,
                    level="ERROR",
                    status_message=f"Error: {str(e)}"
                )
            debug_print(f"Error sending event: {str(e)}")
    
    async def _process_audio_input(self):
        """Process audio input from the queue and send to Bedrock."""
        try:
            while self.is_active:
                try:
                    # Get audio data from the queue with a timeout to allow clean cancellation
                    try:
                        data = await asyncio.wait_for(self.audio_input_queue.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        # No data received within timeout, continue checking is_active
                        continue

                    # Extract data from the queue item
                    prompt_name = data.get('prompt_name')
                    content_name = data.get('content_name')
                    audio_bytes = data.get('audio_bytes')

                    if not audio_bytes or not prompt_name or not content_name:
                        debug_print("Missing required audio data properties")
                        continue

                    # Record input audio if recording is enabled
                    if self.enable_recording:
                        try:
                            # Decode base64 to get raw audio bytes
                            audio_data = base64.b64decode(audio_bytes if isinstance(audio_bytes, str) else audio_bytes.decode('utf-8'))
                            self._write_input_audio(audio_data)
                        except Exception as e:
                            logger.error(f"Error recording input audio: {e}")

                    # Create the audio input event
                    audio_event = S2sEvent.audio_input(prompt_name, content_name, audio_bytes.decode('utf-8') if isinstance(audio_bytes, bytes) else audio_bytes)

                    # Send the event
                    await self.send_raw_event(audio_event)

                except asyncio.CancelledError:
                    debug_print("Audio processing task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error processing audio: {e}")
                    # Don't break the loop on error, continue processing
        except asyncio.CancelledError:
            debug_print("Audio task cancelled during processing")
        except Exception as e:
            logger.error(f"Unexpected error in audio processing task: {e}")
        finally:
            debug_print("Audio processing task completed")
    
    def add_audio_chunk(self, prompt_name, content_name, audio_data):
        """Add an audio chunk to the queue."""
        # The audio_data is already a base64 string from the frontend
        self.audio_input_queue.put_nowait({
            'prompt_name': prompt_name,
            'content_name': content_name,
            'audio_bytes': audio_data
        })
    
    async def _process_responses(self):
        """Process incoming responses from Bedrock."""
        try:
            while self.is_active:
                try:
                    output = await self.stream.await_output()
                    result = await output[1].receive()

                    if result.value and result.value.bytes_:
                        response_data = result.value.bytes_.decode('utf-8')

                        json_data = json.loads(response_data)
                        json_data["timestamp"] = int(time.time() * 1000)  # Milliseconds since epoch

                        event_name = None
                        response_span = None

                        if 'event' in json_data:
                            event_name = list(json_data["event"].keys())[0]

                            # Create response event span
                            response_span = self.span_manager.create_event_span(json_data, session_id=self.session_id)

                            # Handle contentStart to track generation stages
                            if event_name == 'contentStart':
                                content_id = json_data['event']['contentStart'].get("contentId")
                                content_type = json_data['event']['contentStart'].get("type")

                                # Extract generationStage from additionalModelFields if present
                                additional_fields = json_data['event']['contentStart'].get("additionalModelFields")
                                if additional_fields and content_type in ["TEXT", "TOOL"]:
                                    try:
                                        # Parse the additionalModelFields JSON string
                                        fields_dict = json.loads(additional_fields)
                                        generation_stage = fields_dict.get("generationStage")

                                        # Store the generation stage for this contentId
                                        if generation_stage:
                                            self.content_stages[content_id] = generation_stage
                                            # Only log FINAL stages to reduce noise
                                            if generation_stage == "FINAL":
                                                debug_print(f"Content {content_id} (type: {content_type}) marked as FINAL")
                                    except json.JSONDecodeError:
                                        logger.warning(f"Failed to parse additionalModelFields: {additional_fields}")

                            # Handle usage events
                            elif event_name == 'usageEvent':
                                event_data = json_data['event']['usageEvent']
                                self.usage_events.append(event_data)

                                # Update token usage aggregates
                                if 'totalInputTokens' in event_data:
                                    self.token_usage['totalInputTokens'] = event_data.get('totalInputTokens', 0)
                                if 'totalOutputTokens' in event_data:
                                    self.token_usage['totalOutputTokens'] = event_data.get('totalOutputTokens', 0)
                                if 'totalTokens' in event_data:
                                    self.token_usage['totalTokens'] = event_data.get('totalTokens', 0)

                                # Update detailed token usage if available
                                if 'details' in event_data:
                                    details = event_data.get('details', {})
                                    if 'delta' in details:
                                        delta = details.get('delta', {})
                                        # Update input tokens
                                        if 'input' in delta:
                                            input_delta = delta.get('input', {})
                                            self.token_usage['details']['input']['speechTokens'] += input_delta.get('speechTokens', 0)
                                            self.token_usage['details']['input']['textTokens'] += input_delta.get('textTokens', 0)
                                        # Update output tokens
                                        if 'output' in delta:
                                            output_delta = delta.get('output', {})
                                            self.token_usage['details']['output']['speechTokens'] += output_delta.get('speechTokens', 0)
                                            self.token_usage['details']['output']['textTokens'] += output_delta.get('textTokens', 0)

                                    # If total values are provided, use those instead
                                    if 'total' in details:
                                        total = details.get('total', {})
                                        if 'input' in total:
                                            input_total = total.get('input', {})
                                            self.token_usage['details']['input']['speechTokens'] = input_total.get('speechTokens',
                                                self.token_usage['details']['input']['speechTokens'])
                                            self.token_usage['details']['input']['textTokens'] = input_total.get('textTokens',
                                                self.token_usage['details']['input']['textTokens'])
                                        if 'output' in total:
                                            output_total = total.get('output', {})
                                            self.token_usage['details']['output']['speechTokens'] = output_total.get('speechTokens',
                                                self.token_usage['details']['output']['speechTokens'])
                                            self.token_usage['details']['output']['textTokens'] = output_total.get('textTokens',
                                                self.token_usage['details']['output']['textTokens'])

                                # Update session span with token usage and cost
                                if self.span_manager.session_span:
                                    cost = self.cost_calculator.calculate_cost(self.token_usage['details'])

                                    if hasattr(self.span_manager.session_span, 'set_attribute'):
                                        self.span_manager.session_span.set_attribute("input_tokens", self.token_usage['totalInputTokens'])
                                        self.span_manager.session_span.set_attribute("output_tokens", self.token_usage['totalOutputTokens'])
                                        self.span_manager.session_span.set_attribute("total_tokens", self.token_usage['totalTokens'])
                                        self.span_manager.session_span.set_attribute("cost", cost)
                                        self.span_manager.session_span.set_attribute("currency", "USD")
                                        # Add an event for token usage update
                                        self.span_manager.session_span.add_event("token_usage_updated", {
                                            "input_tokens": self.token_usage['totalInputTokens'],
                                            "output_tokens": self.token_usage['totalOutputTokens'],
                                            "total_tokens": self.token_usage['totalTokens'],
                                            "cost": cost
                                        })

                            # Handle tool use detection
                            elif event_name == 'toolUse':
                                tool_use_event = json_data['event']['toolUse']
                                self.toolUseContent = tool_use_event
                                self.toolName = tool_use_event['toolName']
                                self.toolUseId = tool_use_event['toolUseId']
                                # Track contentId for generation stage filtering
                                self.toolUseContentId = tool_use_event.get('contentId')
                                debug_print(f"Tool use detected: {self.toolName}, ID: {self.toolUseId}, ContentId: {self.toolUseContentId}")

                            # Process tool use when content ends
                            elif event_name == 'contentEnd' and json_data['event'][event_name].get('type') == 'TOOL':
                                content_id = json_data['event']['contentEnd'].get("contentId")
                                prompt_name = json_data['event']['contentEnd'].get("promptName")

                                # Check generation stage - only process FINAL, skip SPECULATIVE
                                generation_stage = self.content_stages.get(content_id, "FINAL")

                                # Check if this tool use ID has already been processed (deduplication)
                                if self.toolUseId in self.processed_tool_use_ids:
                                    debug_print(f"Skipping duplicate tool execution: {self.toolName}, ID: {self.toolUseId} (already processed)")
                                elif generation_stage == "SPECULATIVE":
                                    debug_print(f"Skipping SPECULATIVE tool execution: {self.toolName}, ID: {self.toolUseId}")
                                else:
                                    debug_print(f"Processing FINAL tool use: {self.toolName}, ID: {self.toolUseId}")

                                    # Mark this tool use ID as processed
                                    self.processed_tool_use_ids.add(self.toolUseId)

                                    debug_print("Starting tool processing in background")
                                    # Process tool in background task to avoid blocking
                                    task = asyncio.create_task(
                                        self._handle_tool_processing(prompt_name, self.toolName, self.toolUseContent, self.toolUseId)
                                    )
                                    self.tool_processing_tasks.add(task)
                                    task.add_done_callback(self.tool_processing_tasks.discard)

                        # End response span with success
                        if response_span:
                            self.span_manager.end_span_safely(
                                response_span,
                                output={"status": "processed", "event_type": event_name}
                            )

                        # Record output audio if recording is enabled
                        if self.enable_recording and event_name == 'audioOutput':
                            try:
                                # Decode base64 audio content
                                audio_content = json_data['event']['audioOutput'].get('content')
                                if audio_content:
                                    audio_data = base64.b64decode(audio_content)
                                    self._write_output_audio(audio_data)
                            except Exception as e:
                                logger.error(f"Error recording output audio: {e}")

                        # Put the response in the output queue for forwarding to the frontend
                        await self.output_queue.put(json_data)

                except asyncio.CancelledError:
                    debug_print("Response processing task cancelled")
                    break
                except StopAsyncIteration:
                    # Stream has ended
                    debug_print("Stream iteration stopped")
                    break
                except json.JSONDecodeError as json_error:
                    logger.error(f"JSON decode error: {json_error}")
                    await self.output_queue.put({"raw_data": response_data})
                except Exception as e:
                    # Handle ValidationException properly
                    if "ValidationException" in str(e):
                        error_message = str(e)
                        logger.error(f"Validation error: {error_message}")

                        # Handle specific audio content errors gracefully
                        if "No open content found for content name" in error_message and "audio-" in error_message:
                            logger.warning(f"Audio content validation error - continuing processing: {error_message}")
                            # This is a known issue with audio content lifecycle management
                            # Continue processing instead of breaking the session
                            continue

                        # For other validation errors, we may want to break or continue based on severity
                        if "audio" in error_message.lower():
                            logger.warning("Audio-related validation error, continuing session")
                            continue
                    elif "retry-after" in str(e).lower():
                        logger.warning("Rate limited by Bedrock API")
                        # Wait before next attempt
                        await asyncio.sleep(1.0)
                        continue

                    elif "CANCELLED" in str(e) or "InvalidStateError" in str(e):
                        # Handle cancelled futures gracefully
                        debug_print("Stream was cancelled, stopping response processing")
                        break
                    elif "ModelStreamErrorException" in str(e):
                        # Handle unexpected processing errors from Bedrock
                        logger.error(f"Bedrock processing error: {str(e)}")
                        # Attempt to recover by reinitializing the stream
                        try:
                            debug_print("Attempting to recover from Bedrock processing error by reinitializing stream")
                            await self.initialize_stream()
                            continue
                        except Exception as recovery_error:
                            logger.error(f"Failed to recover from Bedrock processing error: {recovery_error}")
                            break
                    else:
                        logger.error(f"Error receiving response: {e}")

                        # Continue to retry for recoverable errors
                        if not self.is_active:
                            break

        except asyncio.CancelledError:
            debug_print("Response task cancelled")
        except Exception as outer_e:
            logger.error(f"Outer error in response processing: {outer_e}")
        finally:
            self.is_active = False
            debug_print("Response processing completed")

    def handle_tool_request(self, prompt_name, tool_name, tool_content, tool_use_id):
        """Handle a tool request asynchronously (non-blocking)"""
        # Create a unique content name for this tool response
        tool_content_name = str(uuid.uuid4())

        # Create an asynchronous task for the tool execution
        task = asyncio.create_task(self._execute_tool_and_send_result(
            prompt_name, tool_name, tool_content, tool_use_id, tool_content_name))

        # Store the task
        self.pending_tool_tasks[tool_content_name] = task

        # Add error handling callback
        task.add_done_callback(
            lambda t: self._handle_tool_task_completion(t, tool_content_name))

    def _handle_tool_task_completion(self, task, content_name):
        """Handle the completion of a tool task"""
        # Remove task from pending tasks
        if content_name in self.pending_tool_tasks:
            del self.pending_tool_tasks[content_name]

        # Handle any exceptions
        if task.done() and not task.cancelled():
            exception = task.exception()
            if exception:
                logger.error(f"Tool task failed: {str(exception)}")

    async def _execute_tool_and_send_result(self, prompt_name, tool_name, tool_content, tool_use_id, content_name):
        """Execute a tool and send the result"""
        try:
            debug_print(f"Starting tool execution: {tool_name}")

            # Process the tool - this doesn't block the event loop
            tool_result = await self.processToolUse(tool_name, tool_content)

            # Send tool start event
            tool_start_event = {
                "event": {
                    "contentStart": {
                        "promptName": prompt_name,
                        "contentName": content_name,
                        "type": "TOOL",
                        "role": "USER",
                        "interactive": False,
                        "toolUseId": tool_use_id
                    }
                }
            }
            await self.send_raw_event(tool_start_event)

            # Send tool result event
            if isinstance(tool_result, dict):
                content_json_string = json.dumps(tool_result)
            else:
                content_json_string = tool_result

            tool_result_event = {
                "event": {
                    "textInput": {
                        "promptName": prompt_name,
                        "contentName": content_name,
                        "content": content_json_string
                    }
                }
            }
            await self.send_raw_event(tool_result_event)

            # Send tool content end event
            tool_content_end_event = {
                "event": {
                    "contentEnd": {
                        "promptName": prompt_name,
                        "contentName": content_name
                    }
                }
            }
            await self.send_raw_event(tool_content_end_event)

            debug_print(f"Tool execution complete: {tool_name}")
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            # Try to send an error response if possible
            try:
                error_result = {"error": f"Tool execution failed: {str(e)}"}

                tool_start_event = {
                    "event": {
                        "contentStart": {
                            "promptName": prompt_name,
                            "contentName": content_name,
                            "type": "TOOL",
                            "role": "USER",
                            "interactive": False,
                            "toolUseId": tool_use_id
                        }
                    }
                }
                await self.send_raw_event(tool_start_event)

                tool_result_event = {
                    "event": {
                        "textInput": {
                            "promptName": prompt_name,
                            "contentName": content_name,
                            "content": json.dumps(error_result)
                        }
                    }
                }
                await self.send_raw_event(tool_result_event)

                tool_content_end_event = {
                    "event": {
                        "contentEnd": {
                            "promptName": prompt_name,
                            "contentName": content_name
                        }
                    }
                }
                await self.send_raw_event(tool_content_end_event)
            except Exception as send_error:
                logger.error(f"Failed to send error response: {str(send_error)}")

    async def _handle_tool_processing(self, prompt_name, tool_name, tool_use_content, tool_use_id):
        """Handle tool processing in background without blocking event processing"""
        tool_span = None
        tool_start_time = time.time_ns()

        try:
            # Create tool use span
            tool_span = self.span_manager.create_child_span(
                "toolUse",
                parent_span=self.span_manager.session_span,
                input={
                    "toolName": tool_name,
                    "params": tool_use_content.get("content") if isinstance(tool_use_content, dict) else None
                },
                metadata={
                    "session_id": self.session_id,
                    "tool_start_time": tool_start_time,
                }
            )

            print(f"[Tool Processing] Starting: {tool_name} with ID: {tool_use_id}")
            toolResult = await self.processToolUse(tool_name, tool_use_content)
            print(f"[Tool Processing] Completed: {tool_name}")

            # Send tool start event
            toolContent = str(uuid.uuid4())
            tool_start_event = S2sEvent.content_start_tool(prompt_name, toolContent, tool_use_id)
            await self.send_raw_event(tool_start_event)

            # Also send tool start event to WebSocket client
            tool_start_event_copy = tool_start_event.copy()
            tool_start_event_copy["timestamp"] = int(time.time() * 1000)
            await self.output_queue.put(tool_start_event_copy)

            # Send tool result event
            if isinstance(toolResult, dict):
                content_json_string = json.dumps(toolResult)
            else:
                content_json_string = toolResult

            tool_result_event = S2sEvent.text_input_tool(prompt_name, toolContent, content_json_string)
            print("Tool result", tool_result_event)
            await self.send_raw_event(tool_result_event)

            # Also send tool result event to WebSocket client
            tool_result_event_copy = tool_result_event.copy()
            tool_result_event_copy["timestamp"] = int(time.time() * 1000)
            await self.output_queue.put(tool_result_event_copy)

            # Send tool content end event
            tool_content_end_event = S2sEvent.content_end(prompt_name, toolContent)
            await self.send_raw_event(tool_content_end_event)

            # Also send tool content end event to WebSocket client
            tool_content_end_event_copy = tool_content_end_event.copy()
            tool_content_end_event_copy["timestamp"] = int(time.time() * 1000)
            await self.output_queue.put(tool_content_end_event_copy)

            # End tool span with success
            tool_end_time = time.time_ns()
            tool_run_time = tool_end_time - tool_start_time
            if tool_span:
                self.span_manager.end_span_safely(
                    tool_span,
                    output={"result": toolResult},
                    metadata={"tool_run_time": tool_run_time, "tool_end_time": tool_end_time}
                )

        except Exception as e:
            tool_end_time = time.time_ns()
            tool_run_time = tool_end_time - tool_start_time
            if tool_span:
                self.span_manager.end_span_safely(
                    tool_span,
                    level="ERROR",
                    status_message=f"Tool processing error: {str(e)}",
                    metadata={"tool_run_time": tool_run_time, "tool_end_time": tool_end_time}
                )
            print(f"Error in tool processing: {e}")
            if DEBUG:
                import traceback
                traceback.print_exc()

    async def processToolUse(self, toolName, toolUseContent):
        """Return the tool result"""
        print(f"Tool Use Content: {toolUseContent}")

        toolName = toolName.lower()
        content, result = None, None
        try:
            if toolUseContent.get("content"):
                # Parse the JSON string in the content field
                query_json = json.loads(toolUseContent.get("content"))
                content = toolUseContent.get("content")  # Pass the JSON string directly to the agent
                print(f"Extracted query: {content}")
            
            # Check if we have a tool registry and the tool is registered
            if hasattr(self, 'tool_registry') and self.tool_registry:
                tool = self.tool_registry.get_tool(toolName)
                if tool:
                    result = await self.tool_registry.execute_tool(toolName, content or "")
                    return {"result": result}

            if not result:
                result = "no result found"

            return {"result": result}
        except Exception as ex:
            print(f"[Tool Error] Exception in processToolUse for {toolName}: {ex}")
            if DEBUG:
                import traceback
                traceback.print_exc()
            return {"result": "An error occurred while attempting to retrieve information related to the toolUse event."}

    async def initialize_session_with_prompt(self, prompt_name: str = None):
        """Initialize session tasks only (frontend sends all events to Bedrock)

        The backend:
        1. Starts response and audio processing tasks
        2. Waits for tasks to be ready
        3. Frontend events (sessionStart, promptStart, contentStart, textInput) are forwarded to Bedrock

        Args:
            prompt_name: Optional prompt name for tracking (can be None initially, set later from promptStart)
        """
        if prompt_name:
            self.prompt_name = prompt_name
            logger.info(f"Initializing session tasks for prompt: {prompt_name}")
        else:
            logger.info(f"Initializing session tasks (prompt name will be set from promptStart)")

        # 1. Start the response processing task
        if not self.response_task or self.response_task.done():
            self.response_task = asyncio.create_task(self._process_responses())
            self.response_task.set_name("response_processing_task")
            self.tasks.add(self.response_task)
            logger.info("Response processing task started")

        # 2. Start the audio processing task
        if not self.audio_task or self.audio_task.done():
            self.audio_task = asyncio.create_task(self._process_audio_input())
            self.audio_task.set_name("audio_processing_task")
            self.tasks.add(self.audio_task)
            logger.info("Audio processing task started")

        # 3. CRITICAL: Wait for tasks to start listening
        # Tasks don't actually start until event loop yields
        await asyncio.sleep(0.1)
        logger.info("Response and audio tasks ready to receive events from frontend")

    async def send_audio_content_end_event(self):
        """Send audio content end event"""
        if self.audio_content_name:
            await self.send_raw_event({
                "event": {
                    "contentEnd": {
                        "promptName": self.prompt_name or "default",
                        "contentName": self.audio_content_name
                    }
                }
            })

    async def send_prompt_end_event(self):
        """Send prompt end event"""
        await self.send_raw_event({
            "event": {
                "promptEnd": {
                    "promptName": self.prompt_name or "default"
                }
            }
        })

    async def send_session_end_event(self):
        """Send session end event"""
        await self.send_raw_event({
            "event": {
                "sessionEnd": {
                    "sessionId": self.session_id
                }
            }
        })

    async def close(self):
        """Close the stream properly (does NOT manage multi-session state).
        
        This method only manages stream resources:
        - Cancels tool processing tasks
        - Clears audio and output queues
        - Resets tool use state
        - Cancels response and audio tasks
        - Closes the stream handle
        
        Session state management (session transitions) is handled by
        SessionTransitionManager.
        """
        if not self.is_active:
            return

        self.is_active = False

        # Cancel any pending tool tasks
        for task in self.pending_tool_tasks.values():
            if not task.done():
                task.cancel()
        self.pending_tool_tasks.clear()

        # Clear audio queue to prevent processing old audio data
        while not self.audio_input_queue.empty():
            try:
                self.audio_input_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Clear output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Reset tool use state
        self.toolUseContent = ""
        self.toolUseId = ""
        self.toolName = ""
        self.toolUseContentId = None

        # Reset session information
        self.prompt_name = None
        self.audio_content_name = None

        # Close the stream if it exists
        if self.stream:
            try:
                await self.stream.input_stream.close()
            except Exception as e:
                debug_print(f"Error closing stream: {e}")

        # Cancel response_task
        if self.response_task and not self.response_task.done():
            self.response_task.cancel()
            try:
                await self.response_task
            except asyncio.CancelledError:
                pass

        # Set stream to None to ensure it's properly cleaned up
        self.stream = None
        self.response_task = None

        # Close recording files if they exist
        if self.enable_recording:
            if self.input_audio_file:
                try:
                    self.input_audio_file.close()
                    logger.info("Input audio recording closed")
                except Exception as e:
                    logger.error(f"Error closing input audio file: {e}")
            if self.output_audio_file:
                try:
                    self.output_audio_file.close()
                    logger.info("Output audio recording closed")
                except Exception as e:
                    logger.error(f"Error closing output audio file: {e}")

        # Close tool registry to cleanup resources
        if hasattr(self, 'tool_registry') and self.tool_registry:
            try:
                self.tool_registry.close()
                logger.info("ToolRegistry closed")
            except Exception as e:
                logger.error(f"Error closing ToolRegistry: {e}")

        # Close span manager to finalize telemetry
        if self.span_manager:
            self.span_manager.close()
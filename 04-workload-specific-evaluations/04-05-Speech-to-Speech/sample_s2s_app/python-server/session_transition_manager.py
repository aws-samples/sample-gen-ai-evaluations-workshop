"""
Session Transition Manager for managing multiple Bedrock streaming sessions.

This module handles the orchestration of session transitions for long-running
conversations with Amazon Bedrock Nova Sonic.

Architecture:
- SessionTransitionManager: Orchestrates session lifecycle and transitions
- Each session gets its own S2sSessionManager instance with independent stream/tasks
- This separation ensures each session can receive completionStart independently
"""

import asyncio
import json
import time
import logging
import os
from typing import Optional

# Import session management classes
from session_state import SessionState
from session_info import SessionInfo
from audio_buffer import AudioBuffer
from conversation_history import ConversationHistory
from opentelemetry_span_manager import OpenTelemetrySpanManager
from nova_sonic_pricing import NovaFxSonicCostCalculator

logger = logging.getLogger(__name__)


class SessionTransitionManager:
    """Manages session transitions for long-running conversations"""

    def __init__(self, model_id: str ="amazon.nova-sonic-v1:0", config_path: str = "./session_config.json"):
        """Initialize the session transition manager"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.transition_config = self.config["session_transition"]
        self.logging_config = self.config["session_logging"]

        # Session management
        self.current_session: Optional[SessionInfo] = None
        self.next_session: Optional[SessionInfo] = None
        self.session_counter = 0

        # Audio buffering for transitions
        self.audio_buffer = AudioBuffer(
            max_duration_seconds=self.transition_config["audio_buffer_duration_seconds"]
        )
        self.is_buffering = False

        # Conversation history with byte limits
        self.conversation_history = ConversationHistory(
            max_single_message_bytes=self.transition_config.get("max_single_message_bytes", 1024),
            max_chat_history_bytes=self.transition_config.get("max_chat_history_bytes", 40960)
        )

        # Transition state
        self.is_transitioning = False
        self.transition_ready = False
        self.waiting_for_audio_start = False
        self.waiting_for_completion = False
        self.user_was_speaking = False
        self.barge_in_occurred = False
        self.audio_start_wait_start = None
        self.audio_start_timeout = self.transition_config.get("audio_start_timeout_seconds", 5.0)
        self.audio_chunk_count = 0

        # Lock for thread-safe operations
        self.lock = asyncio.Lock()

        # Event monitoring task
        self.monitor_task = None

        # Next session ready timeout monitoring
        self.next_session_created_time = None
        self.next_session_ready_timeout = self.transition_config.get("next_session_ready_timeout_seconds", 30)
        self.next_session_monitor_task = None

        # Store stream manager class and kwargs for creating new sessions
        self.stream_manager_class = None
        self.stream_manager_kwargs = {}
        self.prompt_name = None  # Will be set from first session

        # Track hello audio played state across all sessions
        self.hello_audio_played = False

        # Telemetry and cost tracking for entire conversation
        self.model_id = model_id  # Will be set from first session
        self.region = None  # Will be set from first session
        self.span_manager = None  # Will be initialized with first session
        self.cost_calculator = None  # Will be initialized with first session

        # Token usage tracking across all sessions
        self.token_usage = {
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
        self.usage_events = []

        logger.info("SessionTransitionManager initialized")

    async def create_session(self, stream_manager_class, **kwargs) -> SessionInfo:
        """Create a new session with its own stream manager instance

        This is the critical fix: each session gets a NEW stream_manager_class instance,
        not a shared reference. This ensures each session has independent streams and tasks.

        Args:
            stream_manager_class: The S2sSessionManager class (not instance!)
            **kwargs: Arguments to pass to stream_manager_class constructor

        Returns:
            SessionInfo: Information about the newly created session
        """
        # Internal Bedrock session ID for tracking transitions (session_0, session_1, etc.)
        bedrock_session_id = f"session_{self.session_counter}"
        self.session_counter += 1

        if self.logging_config.get("log_transitions", True):
            logger.info(f"[SESSION_CREATE] Creating Bedrock {bedrock_session_id}")
            # Log the session (session_id) if present
            if 'session_id' in kwargs:
                logger.info(f"[SESSION_CREATE] Using session_id: {kwargs['session_id']}")


        # IMPORTANT: Do NOT overwrite kwargs['session_id']

        # Pass hello_audio_played state to ensure it's only played once across all sessions
        kwargs['hello_audio_played'] = self.hello_audio_played

        stream_manager = stream_manager_class(**kwargs)

        # Initialize the stream for this specific session
        await stream_manager.initialize_stream()

        # Create session info
        session_info = SessionInfo(
            session_id=bedrock_session_id,  # Use internal ID for transition tracking
            stream_manager=stream_manager,  # Each session has its own manager!
            start_time=time.time()
        )

        session_info.state = SessionState.ACTIVE

        if self.logging_config.get("log_transitions", True):
            logger.info(f"[SESSION_CREATE] Bedrock {bedrock_session_id} created and active with session_id: {kwargs.get('session_id', 'unknown')}")

        return session_info

    async def initialize_first_session(self, stream_manager_class, prompt_name, **kwargs):
        """Initialize the first session and start event monitoring

        Args:
            stream_manager_class: The S2sSessionManager class to instantiate
            prompt_name: The prompt name to initialize the session with
            **kwargs: Arguments to pass to the stream manager constructor
        """
        if self.logging_config.get("log_transitions", True):
            logger.info("=" * 80)
            logger.info("[FIRST_SESSION] Initializing first session")
            logger.info("=" * 80)

        # Store the class and kwargs for creating future sessions
        self.stream_manager_class = stream_manager_class
        self.stream_manager_kwargs = kwargs

        # Store prompt_name for future sessions
        self.prompt_name = prompt_name

        # Initialize telemetry for the entire conversation (first time only)
        if self.span_manager is None:

            self.model_id = kwargs.get('model_id', 'amazon.nova-sonic-v1:0')
            self.region = kwargs.get('region', 'us-east-1')
            session_id = kwargs.get('session_id', 'unknown')

            # Get DEBUG from environment
            import os
            DEBUG = os.environ.get("DEBUG")

            self.span_manager = OpenTelemetrySpanManager(
                session_id=session_id,
                model_id=self.model_id,
                region=self.region,
                debug=DEBUG
            )
            self.cost_calculator = NovaFxSonicCostCalculator(debug=DEBUG)
            self.span_manager.create_session_span()

            logger.info(f"[TELEMETRY] Initialized span_manager and cost_calculator for session: {session_id}")

        # Pass telemetry references to stream managers
        kwargs['span_manager'] = self.span_manager
        kwargs['cost_calculator'] = self.cost_calculator
        kwargs['token_usage'] = self.token_usage  # Pass reference to shared token usage
        kwargs['usage_events'] = self.usage_events  # Pass reference to shared usage events

        # Create the first session
        self.current_session = await self.create_session(stream_manager_class, **kwargs)

        if self.logging_config.get("log_transitions", True):
            logger.info(f"[FIRST_SESSION] Created {self.current_session.session_id}")

        # Start tasks only (frontend events will be forwarded directly to Bedrock)
        if prompt_name:
            # Legacy path: backend sends events (not used when frontend sends events)
            if self.logging_config.get("log_transitions", True):
                logger.info(f"[FIRST_SESSION] Initializing with prompt: {prompt_name}")
            await self.current_session.stream_manager.initialize_session_with_prompt(prompt_name)
        else:
            # New path: just start tasks, let frontend events pass through
            if self.logging_config.get("log_transitions", True):
                logger.info(f"[FIRST_SESSION] Starting tasks (events will come from frontend)")
            await self.current_session.stream_manager.initialize_session_with_prompt(None)

        if self.logging_config.get("log_transitions", True):
            logger.info(f"[FIRST_SESSION] Session initialized and ready")

        # Start monitoring events from the current session
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        self.monitor_task = asyncio.create_task(self._monitor_events())
        self.monitor_task.set_name("session_monitor_task")

        if self.logging_config.get("log_transitions", True):
            logger.info(f"[FIRST_SESSION] Started monitoring task")
            logger.info("=" * 80)

    async def _monitor_events(self):
        """Monitor events from current session to detect transition opportunities

        This background task continuously:
        1. Checks if session duration exceeds threshold (480 seconds)
        2. Waits for AUDIO contentStart (assistant starts speaking) as trigger
        3. Handles timeout if no audio starts after threshold (forced transition)
        4. Processes events from current_session's output queue
        """
        while True:
            try:
                # Check if current session exists and is active
                if not self.current_session or self.current_session.state == SessionState.CLOSED:
                    await asyncio.sleep(0.1)
                    continue

                # Get transition threshold from config (default 480 seconds = 8 minutes)
                threshold = self.transition_config.get("transition_threshold_seconds", 480)

                # Check if session duration exceeds threshold and we're not already transitioning
                if (self.current_session.should_transition(threshold) and
                    not self.is_transitioning and
                    not self.waiting_for_audio_start):

                    duration = self.current_session.get_duration()
                    if self.logging_config.get("log_transitions", True):
                        logger.info("=" * 80)
                        logger.info(f"[MONITOR] Session duration {duration:.1f}s reached threshold {threshold}s")
                        logger.info(f"[MONITOR] Waiting for AUDIO contentStart (assistant starts speaking)...")
                        logger.info("=" * 80)

                    self.waiting_for_audio_start = True
                    self.audio_start_wait_start = time.time()

                # Check for timeout if waiting for audio start
                if (self.waiting_for_audio_start and
                    self.audio_start_wait_start and
                    (time.time() - self.audio_start_wait_start) > self.audio_start_timeout):

                    if self.logging_config.get("log_transitions", True):
                        logger.info("=" * 80)
                        logger.info(f"[MONITOR] TIMEOUT: No AUDIO contentStart after {self.audio_start_timeout}s")
                        logger.info(f"[MONITOR] Forcing transition (user might be speaking)")
                        logger.info("=" * 80)

                    self.is_buffering = True
                    await self._initiate_transition()

                # Process events from current session (this is where we detect AUDIO contentStart)
                stream_manager = self.current_session.stream_manager
                if stream_manager and not stream_manager.output_queue.empty():
                    try:
                        event = await asyncio.wait_for(
                            stream_manager.output_queue.get(),
                            timeout=0.01
                        )

                        # Process the event (this will trigger transition on AUDIO contentStart)
                        await self._process_event(event, self.current_session)

                    except asyncio.TimeoutError:
                        pass

                # Small sleep to prevent busy-waiting
                await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                if self.logging_config.get("log_transitions", True):
                    logger.info("[MONITOR] Monitoring task cancelled")
                break
            except Exception as e:
                logger.error(f"[MONITOR] Error in event monitor: {e}")
                import traceback
                logger.error(traceback.format_exc())
                await asyncio.sleep(0.1)

    async def _initiate_transition(self):
        """Initiate session transition - create next session and wait for completion signal

        This method:
        1. Creates the next session (new Bedrock stream with its own stream manager)
        2. Starts audio buffering (rolling 5-second buffer)
        3. Waits for completion signal from current session
        4. Starts monitoring next session for readiness (completionStart within 30s)
        """
        async with self.lock:
            if self.is_transitioning:
                return

            self.is_transitioning = True
            self.waiting_for_audio_start = False
            self.waiting_for_completion = True
            self.audio_start_wait_start = None

        if self.logging_config.get("log_transitions", True):
            logger.info("=" * 80)
            logger.info("[TRANSITION] INITIATED")
            logger.info(f"[TRANSITION] Current session: {self.current_session.session_id}")
            logger.info(f"[TRANSITION] Session duration: {self.current_session.get_duration():.1f}s")
            logger.info(f"[TRANSITION] Speculative texts so far: {self.current_session.speculative_text_count}")
            logger.info(f"[TRANSITION] Final texts so far: {self.current_session.final_text_count}")
            logger.info("=" * 80)

        try:
            # Configure audio buffer
            buffer_duration_seconds = self.transition_config.get("audio_buffer_duration_seconds", 5)
            self.audio_buffer.max_duration_seconds = buffer_duration_seconds
            self.audio_buffer.max_buffer_size = int(buffer_duration_seconds * 16000 * 2)
            current_buffer_duration = self.audio_buffer.total_size / (16000 * 2)

            if self.logging_config.get("log_audio_events", True):
                logger.info(f"[BUFFER] Buffering started | Set to {buffer_duration_seconds}s | Currently buffered: {current_buffer_duration:.2f}s ({len(self.audio_buffer.buffer)} chunks)")

            # Create the next session with NEW stream manager instance
            if self.logging_config.get("log_transitions", True):
                logger.info("[NEXT_SESSION] Creating new session...")

            self.next_session = await self.create_session(
                self.stream_manager_class,
                **self.stream_manager_kwargs
            )

            if self.logging_config.get("log_transitions", True):
                logger.info(f"[NEXT_SESSION] Created: {self.next_session.session_id}")

            # Initialize the next session with prompt (sessionStart, promptStart, system prompt)
            # This is required for Bedrock to send completionStart
            if self.prompt_name:
                if self.logging_config.get("log_transitions", True):
                    logger.info(f"[NEXT_SESSION] Initializing with prompt: {self.prompt_name}")

                await self.next_session.stream_manager.initialize_session_with_prompt(self.prompt_name)

                if self.logging_config.get("log_transitions", True):
                    logger.info(f"[NEXT_SESSION] Initialization complete - waiting for completionStart")
            else:
                logger.error(f"[NEXT_SESSION] ERROR: No prompt_name stored!")

            # Start monitoring next session readiness (must receive completionStart within 30s)
            if self.next_session_monitor_task:
                self.next_session_monitor_task.cancel()
                try:
                    await self.next_session_monitor_task
                except asyncio.CancelledError:
                    pass

            self.next_session_monitor_task = asyncio.create_task(
                self._monitor_next_session_readiness()
            )
            self.next_session_monitor_task.set_name("next_session_monitor")

            if self.logging_config.get("log_transitions", True):
                logger.info("[WAITING] For completion signal from current session:")
                logger.info("[WAITING]   - Text pairs match (speculative count == final count), OR")
                logger.info("[WAITING]   - TEXT INTERRUPTED stopReason")
                logger.info("[WAITING] Will send history after completion signal received")

            self.transition_ready = True

        except Exception as e:
            logger.error(f"[TRANSITION] Failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

            # Reset transition state
            self.is_transitioning = False
            self.transition_ready = False
            self.waiting_for_audio_start = False
            self.waiting_for_completion = False
            self.audio_start_wait_start = None

            # Close next session if created
            if self.next_session:
                try:
                    self.next_session.state = SessionState.CLOSING
                    if self.next_session.stream_manager:
                        await self.next_session.stream_manager.close()
                except Exception as close_error:
                    logger.error(f"[TRANSITION] Error closing next session: {close_error}")
                self.next_session = None

            if self.logging_config.get("log_transitions", True):
                logger.info("[RECOVERY] Transition flags reset, will retry on next threshold")

            raise

    async def _monitor_next_session_readiness(self):
        """Monitor if next session receives completionStart within timeout

        This task monitors the next session to ensure it's healthy. A healthy session
        should receive a completionStart event from Bedrock within 30 seconds.

        If timeout occurs (no completionStart after 30s):
        - The next session is considered dead/stuck
        - Calls _recreate_next_session() to close and replace it
        - This ensures we always have a healthy next session ready
        """
        if not self.next_session:
            return

        session_to_monitor = self.next_session
        start_time = time.time()
        timeout = self.next_session_ready_timeout

        if self.logging_config.get("log_transitions", True):
            logger.info(f"[NEXT_SESSION_MONITOR] Started monitoring {session_to_monitor.session_id} for {timeout}s")

        try:
            while True:
                await asyncio.sleep(1)

                # Check if session received completionStart
                if session_to_monitor.received_completion_start:
                    if self.logging_config.get("log_transitions", True):
                        logger.info(f"[NEXT_SESSION_MONITOR] ✓ {session_to_monitor.session_id} is ready (received completionStart)")
                    return

                # Check if session is no longer the next session (already promoted or replaced)
                if self.next_session != session_to_monitor:
                    if self.logging_config.get("log_transitions", True):
                        logger.info(f"[NEXT_SESSION_MONITOR] {session_to_monitor.session_id} no longer next session, stopping monitor")
                    return

                # Check timeout
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    if self.logging_config.get("log_transitions", True):
                        logger.info("=" * 80)
                        logger.info(f"[NEXT_SESSION_MONITOR] ⚠ TIMEOUT after {elapsed:.1f}s")
                        logger.info(f"[NEXT_SESSION_MONITOR] {session_to_monitor.session_id} did not receive completionStart")
                        logger.info(f"[NEXT_SESSION_MONITOR] Recreating next session...")
                        logger.info("=" * 80)

                    await self._recreate_next_session(session_to_monitor)
                    return

        except asyncio.CancelledError:
            if self.logging_config.get("log_transitions", True):
                logger.info(f"[NEXT_SESSION_MONITOR] Monitor cancelled for {session_to_monitor.session_id}")
            raise
        except Exception as e:
            logger.error(f"[NEXT_SESSION_MONITOR] Error monitoring {session_to_monitor.session_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    async def _recreate_next_session(self, dead_session: SessionInfo):
        """Close dead next session and create a fresh replacement

        This method is called when the next session times out (no completionStart
        after 30 seconds). It:
        1. Closes the dead session gracefully
        2. Creates a fresh next session
        3. Starts monitoring the new next session

        Args:
            dead_session: The SessionInfo of the stuck/dead next session
        """
        if self.logging_config.get("log_transitions", True):
            logger.info(f"[NEXT_SESSION_RECREATE] Closing dead session {dead_session.session_id}")

        # Close the dead session
        try:
            dead_session.state = SessionState.CLOSING
            if dead_session.stream_manager:
                await dead_session.stream_manager.close()
            dead_session.state = SessionState.CLOSED

            if self.logging_config.get("log_transitions", True):
                logger.info(f"[NEXT_SESSION_RECREATE] Closed dead session {dead_session.session_id}")

        except Exception as e:
            logger.error(f"[NEXT_SESSION_RECREATE] Error closing dead session: {e}")

        # Only recreate if we're still transitioning and this is still our next session
        async with self.lock:
            if not self.is_transitioning or self.next_session != dead_session:
                if self.logging_config.get("log_transitions", True):
                    logger.info(f"[NEXT_SESSION_RECREATE] Skipping recreation - no longer relevant")
                return

            # Create a fresh next session
            if self.logging_config.get("log_transitions", True):
                logger.info(f"[NEXT_SESSION_RECREATE] Creating fresh replacement session...")

            try:
                self.next_session = await self.create_session(
                    self.stream_manager_class,
                    **self.stream_manager_kwargs
                )

                if self.logging_config.get("log_transitions", True):
                    logger.info(f"[NEXT_SESSION_RECREATE] ✓ Created: {self.next_session.session_id}")

                # Initialize the replacement session with prompt
                if self.prompt_name:
                    if self.logging_config.get("log_transitions", True):
                        logger.info(f"[NEXT_SESSION_RECREATE] Initializing with prompt: {self.prompt_name}")

                    await self.next_session.stream_manager.initialize_session_with_prompt(self.prompt_name)

                    if self.logging_config.get("log_transitions", True):
                        logger.info(f"[NEXT_SESSION_RECREATE] Initialization complete")
                else:
                    logger.error(f"[NEXT_SESSION_RECREATE] ERROR: No prompt_name stored!")

                # Start monitoring the new session
                if self.next_session_monitor_task:
                    self.next_session_monitor_task.cancel()
                    try:
                        await self.next_session_monitor_task
                    except asyncio.CancelledError:
                        pass

                self.next_session_monitor_task = asyncio.create_task(
                    self._monitor_next_session_readiness()
                )
                self.next_session_monitor_task.set_name("next_session_monitor")

            except Exception as e:
                logger.error(f"[NEXT_SESSION_RECREATE] Failed to create replacement session: {e}")
                import traceback
                logger.error(traceback.format_exc())

                # Reset transition state on failure
                self.is_transitioning = False
                self.transition_ready = False
                self.waiting_for_completion = False
                self.next_session = None
                raise

    async def _process_event(self, event: dict, session: SessionInfo):
        """Process an event from a session for transition detection

        This method processes events from the current session to:
        1. Track contentStart events (extract generationStage from additionalModelFields)
        2. Detect AUDIO contentStart from assistant (triggers transition)
        3. Detect barge-in events (interrupted text)
        4. Track SPECULATIVE vs FINAL text counts for completion detection
        5. Add FINAL messages to ConversationHistory
        6. Detect completionStart events

        Args:
            event: The event dictionary from Bedrock
            session: The SessionInfo that generated this event
        """
        if 'event' not in event:
            return

        # Skip processing if the session's stream was force-stopped
        if session.stream_manager and not session.stream_manager.is_active:
            return

        event_data = event['event']

        # Track contentStart to know the generation stage for subsequent text
        if 'contentStart' in event_data:
            content_start = event_data['contentStart']
            session.current_content_role = content_start.get('role')
            session.current_content_type = content_start.get('type')

            # Extract generationStage from additionalModelFields
            if 'additionalModelFields' in content_start:
                try:
                    additional_fields = json.loads(content_start['additionalModelFields'])
                    session.current_generation_stage = additional_fields.get('generationStage')
                except json.JSONDecodeError:
                    session.current_generation_stage = None
            else:
                session.current_generation_stage = None

            # CRITICAL: Detect AUDIO contentStart from assistant (triggers transition)
            if (session == self.current_session and
                self.waiting_for_audio_start and
                session.current_content_type == 'AUDIO' and
                session.current_content_role == 'ASSISTANT'):

                if self.logging_config.get("log_audio_events", True):
                    logger.info(f"[AUDIO_START] AUDIO contentStart detected in {session.session_id}")
                    logger.info(f"[AUDIO_START] Starting buffer and initiating transition")

                self.is_buffering = True
                await self._initiate_transition()

            # Detect barge-in during transition (user speaking while assistant was speaking)
            if (session.current_content_role == 'USER' and
                session == self.current_session and
                self.is_transitioning and
                self.next_session and
                self.next_session.received_completion_start):

                if self.logging_config.get("log_barge_in_events", True):
                    logger.info(f"[BARGE_IN] User contentStart detected during transition")

                session.barge_in_detected = True
                self.barge_in_occurred = True

        # Handle completionStart (indicates session is ready)
        elif 'completionStart' in event_data:
            completion_start = event_data['completionStart']
            model_session_id = completion_start.get('sessionId')

            if model_session_id and not session.model_session_id:
                session.model_session_id = model_session_id
                session.received_completion_start = True

                if self.logging_config.get("log_transitions", True):
                    logger.info(f"[COMPLETION_START] {session.session_id} | Bedrock sessionId: {model_session_id}")

        # Handle textOutput for barge-in detection and conversation tracking
        elif 'textOutput' in event_data:
            text_content = event_data['textOutput'].get('content', '')
            role = event_data['textOutput'].get('role', '')

            # Detect barge-in from interrupted text
            if '{ "interrupted" : true }' in text_content:
                session.barge_in_detected = True

                if self.logging_config.get("log_barge_in_events", True):
                    logger.info(f"[BARGE_IN] Detected in {session.session_id}")

                # Track barge-in during transition for buffered audio handling
                if session == self.current_session and self.is_transitioning:
                    self.barge_in_occurred = True
                    if self.logging_config.get("log_barge_in_events", True):
                        logger.info("[BARGE_IN] During transition - will send buffered audio to next session")

            # Track USER speech during transition - treat as barge-in
            elif role == 'USER' and text_content and self.is_transitioning and session == self.current_session:
                self.user_was_speaking = True

                # Mark as barge-in if not already marked
                if not self.barge_in_occurred:
                    self.barge_in_occurred = True
                    if self.logging_config.get("log_barge_in_events", True):
                        logger.info("[BARGE_IN] User spoke during transition - treating as barge-in")

                # If FORCED transition (no assistant response yet), complete immediately
                if session.final_text_count == 0 and self.waiting_for_completion:
                    if self.logging_config.get("log_barge_in_events", True):
                        logger.info("[BARGE_IN] User spoke during FORCED transition - completing immediately")
                    asyncio.create_task(self._send_history_and_start_audio())

            # Track conversation for history (only relevant sessions)
            if self.is_transitioning and role == 'USER':
                # During transition: only next_session's USER text
                is_relevant_session = (session == self.next_session)
            else:
                # Normal operation or ASSISTANT text: use current_session
                is_relevant_session = (session == self.current_session)

            if is_relevant_session and role in ['USER', 'ASSISTANT'] and text_content and '{ "interrupted" : true }' not in text_content:
                if role == 'USER':
                    # Always add USER messages
                    self.conversation_history.add_message(role, text_content, "text")

                    if self.logging_config.get("log_audio_events", True):
                        logger.info(f"[{session.session_id}] Added to history: {role}: {text_content[:50]}...")

                    # If barge-in was detected, set counts equal when user speaks
                    if (session == self.current_session and
                        session.barge_in_detected and
                        not self.is_transitioning and
                        session.speculative_text_count > session.final_text_count):

                        if self.logging_config.get("log_barge_in_events", True):
                            logger.info(f"[BARGE_IN] User speaking after barge-in - setting final={session.speculative_text_count} (was {session.final_text_count})")

                        session.final_text_count = session.speculative_text_count

                elif role == 'ASSISTANT':
                    generation_stage = session.current_generation_stage

                    if generation_stage == 'SPECULATIVE':
                        session.speculative_text_count += 1

                        if self.logging_config.get("log_audio_events", True):
                            logger.info(f"[{session.session_id}] SPECULATIVE text #{session.speculative_text_count}: {text_content[:50]}... (skipping, waiting for FINAL)")

                    elif generation_stage == 'FINAL':
                        session.final_text_count += 1

                        if self.logging_config.get("log_audio_events", True):
                            logger.info(f"[{session.session_id}] FINAL text #{session.final_text_count}: {text_content[:50]}...")

                        # Add FINAL text to conversation history
                        self.conversation_history.add_message(role, text_content, "text")

                        if self.logging_config.get("log_audio_events", True):
                            logger.info(f"[{session.session_id}] Added FINAL to history")

                        session.received_final_text = True

                        # COMPLETION DETECTION: Check if text pairs match
                        if (session == self.current_session and
                            self.waiting_for_completion and
                            session.speculative_text_count > 0 and
                            session.final_text_count > 0 and
                            session.speculative_text_count == session.final_text_count):

                            if self.logging_config.get("log_transitions", True):
                                logger.info("=" * 80)
                                logger.info(f"[COMPLETION_SIGNAL] Text pairs matched ({session.speculative_text_count}={session.final_text_count})")
                                logger.info("[COMPLETION_SIGNAL] Sending history to next session")
                                logger.info("=" * 80)

                            await self._send_history_and_start_audio()

        # Handle contentEnd for completion detection
        elif 'contentEnd' in event_data:
            content_end = event_data['contentEnd']
            content_type = content_end.get('type')
            stop_reason = content_end.get('stopReason')

            # Detect TEXT INTERRUPTED as completion signal
            if (content_type == 'TEXT' and stop_reason == 'INTERRUPTED' and
                session == self.current_session and
                self.waiting_for_completion):

                if self.logging_config.get("log_transitions", True):
                    logger.info("=" * 80)
                    logger.info("[COMPLETION_SIGNAL] TEXT INTERRUPTED received")
                    logger.info("[COMPLETION_SIGNAL] Sending history to next session")
                    logger.info("=" * 80)

                await self._send_history_and_start_audio()

            # Reset current content tracking
            if session == self.current_session:
                session.current_generation_stage = None
                session.current_content_role = None

        # Update last output time for ALL events except usageEvent
        if 'usageEvent' not in event_data:
            session.last_output_time = time.time()

    async def _send_history_and_start_audio(self):
        """Send conversation history and buffered audio to next session, then complete transition

        This method is called when the completion signal is received from the current session:
        - SPECULATIVE text count matches FINAL text count, OR
        - TEXT contentEnd with INTERRUPTED stopReason

        It performs the critical transition steps:
        1. Sends conversation history to next session (via ConversationHistory)
        2. Sends audio contentStart to next session
        3. Sends all buffered audio chunks (last 5 seconds)
        4. Calls _close_old_session_and_promote() to complete the swap
        """
        if not self.next_session or not self.waiting_for_completion:
            return

        async with self.lock:
            if not self.waiting_for_completion:
                return
            self.waiting_for_completion = False

        if self.logging_config.get("log_transitions", True):
            logger.info("=" * 80)
            logger.info("[HISTORY_SEND] Completion signal received - sending history to next session")
            logger.info(f"[HISTORY_SEND] Final text count: {self.current_session.final_text_count}")
            logger.info("=" * 80)

        try:
            # Send conversation history to next session
            if self.conversation_history.messages:
                history_size = sum(len(msg.get('content', '')) for msg in self.conversation_history.messages)

                if self.logging_config.get("log_transitions", True):
                    logger.info(f"[HISTORY] Sending {len(self.conversation_history.messages)} messages (~{history_size} chars)")

                history_events = self.conversation_history.get_history_events(
                    self.next_session.stream_manager.prompt_name or "default"
                )

                for event_str in history_events:
                    event = json.loads(event_str)
                    await self.next_session.stream_manager.send_raw_event(event)
                    await asyncio.sleep(0.01)

                if self.logging_config.get("log_transitions", True):
                    logger.info(f"[HISTORY] ✓ Sent {len(history_events)} events to {self.next_session.session_id}")
            else:
                if self.logging_config.get("log_transitions", True):
                    logger.info("[HISTORY] No history to send")

            # Send audio contentStart to next session
            if self.logging_config.get("log_audio_events", True):
                logger.info("[AUDIO_START] Sending audio contentStart to next session")

            # Get the audio content name from the stream manager
            audio_content_name = self.next_session.stream_manager.audio_content_name
            if not audio_content_name:
                import uuid
                audio_content_name = str(uuid.uuid4())
                self.next_session.stream_manager.audio_content_name = audio_content_name

            # Create audio contentStart event
            from s2s_events import S2sEvent
            audio_start_event = S2sEvent.content_start_audio(
                prompt_name=self.next_session.stream_manager.prompt_name or "default",
                content_name=audio_content_name
            )

            # Add the required role field to the event
            audio_start_event["event"]["contentStart"]["role"] = "USER"

            await self.next_session.stream_manager.send_raw_event(audio_start_event)
            self.next_session.audio_content_started = True

            # Send buffered audio to next session
            buffer_duration = self.audio_buffer.total_size / (16000 * 2)

            if self.logging_config.get("log_audio_events", True):
                logger.info(f"[BUFFER_SEND] Sending {buffer_duration:.2f}s of buffered audio to {self.next_session.session_id}")

            if not self.audio_buffer.is_empty():
                chunks_sent = 0
                for chunk in self.audio_buffer.get_all_chunks():
                    # Send chunk to next session's audio input queue
                    import base64
                    audio_data_b64 = base64.b64encode(chunk).decode('utf-8')
                    self.next_session.stream_manager.add_audio_chunk(
                        prompt_name=self.next_session.stream_manager.prompt_name,
                        content_name=audio_content_name,
                        audio_data=audio_data_b64
                    )
                    chunks_sent += 1

                if self.logging_config.get("log_audio_events", True):
                    logger.info(f"[BUFFER_SEND] ✓ Sent {chunks_sent} buffered audio chunks")
            else:
                if self.logging_config.get("log_audio_events", True):
                    logger.info("[BUFFER_SEND] No buffered audio to send")

            if self.logging_config.get("log_transitions", True):
                logger.info("[TRANSITION] History and audio sent - closing old session immediately")
                logger.info("=" * 80)

            # Complete the transition by closing old session and promoting next
            await self._close_old_session_and_promote()

        except Exception as e:
            logger.error(f"[HISTORY_SEND] Failed to send history and start audio: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    async def _close_old_session_and_promote(self):
        """Close old session and promote next session (atomic session swap)

        This method performs the critical atomic swap:
        1. PROMOTE FIRST: next_session → current_session (to avoid blocking audio)
        2. Cancel next session monitoring (no longer needed)
        3. Reset transition state flags
        4. Clear audio buffer
        5. Close old session in background (non-blocking)

        The promotion happens FIRST to ensure audio routing continues immediately
        to the new session. The old session is closed in the background to avoid
        blocking the transition.

        This ensures zero-delay transition with no audio gaps.
        """
        if not self.current_session or not self.next_session:
            return

        old_session = self.current_session

        if self.logging_config.get("log_transitions", True):
            logger.info("=" * 80)
            logger.info(f"[PROMOTION] Promoting {self.next_session.session_id} to current session FIRST")
            logger.info("=" * 80)

        # CRITICAL: Promote FIRST to avoid blocking audio routing
        self.current_session = self.next_session
        self.next_session = None

        if self.logging_config.get("log_transitions", True):
            logger.info(f"[PROMOTION] ✓ {self.current_session.session_id} is now the current session")
            logger.info(f"[PROMOTION] State: {self.current_session.state}, audio_content_started: {self.current_session.audio_content_started}")

        # Cancel next session monitor (session is now promoted)
        if self.next_session_monitor_task:
            self.next_session_monitor_task.cancel()
            try:
                await self.next_session_monitor_task
            except asyncio.CancelledError:
                pass
            self.next_session_monitor_task = None

        # Reset transition state
        self.is_transitioning = False
        self.transition_ready = False
        self.user_was_speaking = False
        self.barge_in_occurred = False

        # Clear audio buffer
        self.audio_buffer.clear()
        self.is_buffering = False

        if self.logging_config.get("log_audio_events", True):
            logger.info(f"[BUFFER] Cleared and stopped")

        # Now close the old session in the background (don't block)
        if self.logging_config.get("log_transitions", True):
            logger.info("=" * 80)
            logger.info(f"[CLOSE_OLD] Closing {old_session.session_id} in background")
            logger.info("=" * 80)

        # Mark old session as closing immediately
        old_session.state = SessionState.CLOSING

        # Clear old session's output queue to prevent stale audio
        audio_queue_size = 0
        if hasattr(old_session.stream_manager, 'output_queue'):
            audio_queue_size = old_session.stream_manager.output_queue.qsize()
            while not old_session.stream_manager.output_queue.empty():
                try:
                    old_session.stream_manager.output_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            if self.logging_config.get("log_audio_events", True):
                logger.info(f"[CLOSE_OLD] Cleared {audio_queue_size} audio chunks from old session queue")

        # Schedule the actual stream close in background (non-blocking)
        asyncio.create_task(self._close_stream_in_background(old_session))

        if self.logging_config.get("log_transitions", True):
            logger.info(f"[CLOSE_OLD] {old_session.session_id} marked as closed, stream closing in background")
            logger.info("=" * 80)
            logger.info("[TRANSITION] ✓ COMPLETE - New session is active")
            logger.info("=" * 80)

    async def _close_stream_in_background(self, session: SessionInfo):
        """Close the stream in the background without blocking

        Args:
            session: The SessionInfo of the session to close
        """
        try:
            if session.stream_manager:
                await session.stream_manager.close()

            session.state = SessionState.CLOSED

            if self.logging_config.get("log_transitions", True):
                logger.info(f"[CLOSE_BG] {session.session_id} fully closed")

        except Exception as e:
            logger.error(f"[CLOSE_BG] Error closing {session.session_id}: {e}")

    def add_audio_chunk(self, audio_chunk: bytes):
        """Route audio chunk to current active session with buffering during transitions

        This method:
        1. Routes audio to current_session's stream manager
        2. Adds to audio_buffer when is_buffering=True (during transitions)
        3. Tracks audio chunk count for debugging

        Args:
            audio_chunk: Raw audio bytes (PCM 16-bit, 16kHz)
        """
        self.audio_chunk_count += 1

        # Log every 100 chunks to track audio routing
        if self.audio_chunk_count % 100 == 0:
            session_id = self.current_session.session_id if self.current_session else "None"
            if self.logging_config.get("log_audio_events", True):
                logger.info(f"[AUDIO_ROUTING] Routing chunk #{self.audio_chunk_count} to {session_id}")

        # Add to buffer if we're in buffering mode (during transitions)
        if self.is_buffering:
            self.audio_buffer.add_chunk(audio_chunk)
            buffer_duration = self.audio_buffer.total_size / (16000 * 2)

            if self.logging_config.get("log_audio_events", True) and self.audio_chunk_count % 50 == 0:
                logger.info(f"[BUFFER] Added chunk #{self.audio_chunk_count} | Buffer: {buffer_duration:.2f}s ({len(self.audio_buffer.buffer)} chunks)")

        # Route audio to current session
        if self.current_session and self.current_session.state == SessionState.ACTIVE:
            try:
                # Convert bytes to base64 for the stream manager
                import base64
                audio_data_b64 = base64.b64encode(audio_chunk).decode('utf-8')

                self.current_session.stream_manager.add_audio_chunk(
                    prompt_name=self.current_session.stream_manager.prompt_name,
                    content_name=self.current_session.stream_manager.audio_content_name,
                    audio_data=audio_data_b64
                )

                # Log periodically (every 100 chunks to reduce noise)
                if self.logging_config.get("log_audio_events", True) and self.audio_chunk_count % 100 == 0:
                    logger.info(f"[AUDIO_ROUTE] Chunk #{self.audio_chunk_count} → {self.current_session.session_id}")

            except Exception as e:
                logger.error(f"Error routing audio chunk: {e}")
        else:
            # No active session to route to
            if self.audio_chunk_count % 50 == 0:
                if not self.current_session:
                    logger.warning(f"[AUDIO_ROUTE] No current_session, dropping chunk #{self.audio_chunk_count}")
                elif self.current_session.state != SessionState.ACTIVE:
                    logger.warning(f"[AUDIO_ROUTE] current_session state is {self.current_session.state}, not ACTIVE, dropping chunk #{self.audio_chunk_count}")

    async def get_output_audio(self) -> Optional[bytes]:
        """Get output audio from active session

        During transitions, prefers next_session if it's ready (received completionStart),
        otherwise uses current_session.

        Returns:
            Optional[bytes]: Audio data in base64-encoded format, or None if no audio available
        """
        # During transition, prefer next session if it's ready
        if (self.next_session and
            self.next_session.received_completion_start and
            self.next_session.state == SessionState.ACTIVE):

            # Try current session first (for any remaining audio)
            if self.current_session and self.current_session.state == SessionState.ACTIVE:
                try:
                    audio_data = await asyncio.wait_for(
                        self.current_session.stream_manager.output_queue.get(),
                        timeout=0.01
                    )
                    return audio_data
                except asyncio.TimeoutError:
                    pass

            # Then try next session
            if self.next_session:
                try:
                    audio_data = await asyncio.wait_for(
                        self.next_session.stream_manager.output_queue.get(),
                        timeout=0.01
                    )
                    return audio_data
                except asyncio.TimeoutError:
                    return None

        # Normal operation: use current session
        if self.current_session and self.current_session.state == SessionState.ACTIVE:
            try:
                audio_data = await asyncio.wait_for(
                    self.current_session.stream_manager.output_queue.get(),
                    timeout=0.01
                )
                return audio_data
            except asyncio.TimeoutError:
                pass

        return None

    def get_active_stream_manager(self):
        """Get the currently active stream manager

        Returns:
            S2sSessionManager: The active stream manager, or None if no active session
        """
        if self.transition_ready and self.next_session:
            return self.next_session.stream_manager
        elif self.current_session:
            return self.current_session.stream_manager
        return None

    async def close_all_sessions(self):
        """Close all sessions and clean up

        This method:
        - Cancels monitoring tasks
        - Closes next_session if it exists
        - Closes current_session if it exists
        - Resets all transition state
        """
        if self.logging_config.get("log_transitions", True):
            logger.info("=" * 80)
            logger.info("[CLOSE_ALL] Closing all sessions")
            logger.info("=" * 80)

        # Cancel all monitoring tasks
        if hasattr(self, 'monitor_task') and self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
            if self.logging_config.get("log_transitions", True):
                logger.info("[CLOSE_ALL] Cancelled monitor task")

        if hasattr(self, 'next_session_monitor_task') and self.next_session_monitor_task:
            self.next_session_monitor_task.cancel()
            try:
                await self.next_session_monitor_task
            except asyncio.CancelledError:
                pass
            if self.logging_config.get("log_transitions", True):
                logger.info("[CLOSE_ALL] Cancelled next session monitor task")

        # Close next session if it exists
        if self.next_session:
            if self.logging_config.get("log_transitions", True):
                logger.info(f"[CLOSE_ALL] Closing next session: {self.next_session.session_id}")
            try:
                if self.next_session.stream_manager:
                    await self.next_session.stream_manager.close()
            except Exception as e:
                logger.error(f"[CLOSE_ALL] Error closing next session: {e}")

            self.next_session.state = SessionState.CLOSED
            self.next_session = None

        # Close current session if it exists
        if self.current_session:
            if self.logging_config.get("log_transitions", True):
                logger.info(f"[CLOSE_ALL] Closing current session: {self.current_session.session_id}")
            try:
                if self.current_session.stream_manager:
                    await self.current_session.stream_manager.close()
            except Exception as e:
                logger.error(f"[CLOSE_ALL] Error closing current session: {e}")

            self.current_session.state = SessionState.CLOSED
            self.current_session = None

        # Reset transition state
        self.is_transitioning = False
        self.transition_ready = False
        self.waiting_for_audio_start = False
        self.waiting_for_completion = False
        self.user_was_speaking = False
        self.barge_in_occurred = False
        self.audio_start_wait_start = None
        self.is_buffering = False

        # Clear audio buffer
        if hasattr(self, 'audio_buffer'):
            self.audio_buffer.clear()

        # Clear conversation history
        if hasattr(self, 'conversation_history'):
            self.conversation_history.clear()

        if self.logging_config.get("log_transitions", True):
            logger.info("=" * 80)
            logger.info("[CLOSE_ALL] All sessions closed")
            logger.info("=" * 80)

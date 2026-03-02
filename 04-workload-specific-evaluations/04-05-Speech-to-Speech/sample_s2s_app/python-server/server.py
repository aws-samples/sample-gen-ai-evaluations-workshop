import asyncio
import websockets
import json
import logging
import warnings
from s2s_session_manager import S2sSessionManager
from session_transition_manager import SessionTransitionManager
import argparse
import http.server
import threading
import os
from http import HTTPStatus

# Configure logging
DEBUG = os.environ.get("DEBUG")
LOG_LEVEL = os.environ.get("LOG_LEVEL")
logger = logging.getLogger(__name__)
if LOG_LEVEL:
    logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')


# Suppress warnings
warnings.filterwarnings("ignore")

def debug_print(message):
    """Print only if debug mode is enabled"""
    if DEBUG:
        logger.info(message)

MCP_CLIENT = None
STRANDS_AGENT = None

# load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

class HealthCheckHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        client_ip = self.client_address[0]
        logger.info(
            f"Health check request received from {client_ip} for path: {self.path}"
        )

        if self.path == "/health" or self.path == "/":
            logger.info(f"Responding with 200 OK to health check from {client_ip}")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = json.dumps({"status": "healthy"})
            self.wfile.write(response.encode("utf-8"))
            logger.info(f"Health check response sent: {response}")
        else:
            logger.info(
                f"Responding with 404 Not Found to request for {self.path} from {client_ip}"
            )
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()

    def log_message(self, format, *args):
        # Override to use our logger instead
        pass


def start_health_check_server(health_host, health_port):
    """Start the HTTP health check server on port 80."""
    try:
        # Create the server with a socket timeout to prevent hanging
        httpd = http.server.HTTPServer((health_host, health_port), HealthCheckHandler)
        httpd.timeout = 5  # 5 second timeout

        logger.info(f"Starting health check server on {health_host}:{health_port}")

        # Run the server in a separate thread
        thread = threading.Thread(target=httpd.serve_forever)
        thread.daemon = (
            True  # This ensures the thread will exit when the main program exits
        )
        thread.start()

        # Verify the server is running
        logger.info(
            f"Health check server started at http://{health_host}:{health_port}/health"
        )
        logger.info(f"Health check thread is alive: {thread.is_alive()}")

        # Try to make a local request to verify the server is responding
        try:
            import urllib.request

            with urllib.request.urlopen(
                f"http://localhost:{health_port}/health", timeout=2
            ) as response:
                logger.info(
                    f"Local health check test: {response.status} - {response.read().decode('utf-8')}"
                )
        except Exception as e:
            logger.warning(f"Local health check test failed: {e}")

    except Exception as e:
        logger.error(f"Failed to start health check server: {e}", exc_info=True)


async def websocket_handler(websocket):
    aws_region = os.getenv("AWS_DEFAULT_REGION")
    if not aws_region:
        aws_region = "us-east-1"

    transition_manager = None
    forward_task = None
    model_id = 'amazon.nova-sonic-v1:0'  # Default model
    cleanup_initiated = False
    enable_recording = False  # Recording disabled by default

    # Generate unique session ID for server-side tracking
    import uuid
    session_id = str(uuid.uuid4())

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                if 'body' in data:
                    data = json.loads(data["body"])
                if 'event' in data:
                    event_type = list(data['event'].keys())[0]

                    # Handle model configuration
                    if event_type == 'modelConfig':
                        model_id = data['event']['modelConfig'].get('modelId', 'amazon.nova-sonic-v1:0')
                        logger.info(f"Model configuration received: {model_id}")
                        continue

                    # Handle recording configuration
                    if event_type == 'recordingConfig':
                        enable_recording = data['event']['recordingConfig'].get('enableRecording', False)
                        logger.info(f"Recording configuration received: enableRecording={enable_recording}")
                        continue

                    # Handle sessionStart - initialize stream and pass through
                    if event_type == 'sessionStart':
                        if transition_manager and transition_manager.current_session:
                            logger.warning("Ignoring duplicate sessionStart event")
                            continue

                        logger.info(f"Received sessionStart event")
                        logger.info(f"Server-side session_id: {session_id}")

                        # Extract optional category label injected by test runners
                        session_category = data['event']['sessionStart'].get('category')
                        if session_category:
                            logger.info(f"Session category from client: {session_category}")

                        # Create SessionTransitionManager and initialize stream
                        if not transition_manager:
                            logger.info("Initializing SessionTransitionManager")
                            transition_manager = SessionTransitionManager()

                        # Initialize stream with basic config (will extract more from promptStart)
                        stream_manager_kwargs = {
                            'model_id': model_id,
                            'region': aws_region,
                            'session_id': session_id,
                            'enable_recording': enable_recording,
                        }

                        logger.info("Initializing stream and tasks")
                        try:
                            await transition_manager.initialize_first_session(
                                S2sSessionManager,
                                None,  # promptName will come from promptStart
                                **stream_manager_kwargs
                            )
                            logger.info("Stream and tasks initialized")
                        except Exception as stream_error:
                            logger.error(f"Failed to initialize stream: {stream_error}", exc_info=True)
                            raise

                        # Attach category to the session span so it lands in CloudWatch
                        if session_category:
                            stream_manager = transition_manager.get_active_stream_manager()
                            if stream_manager and stream_manager.span_manager:
                                stream_manager.span_manager.set_category(session_category)

                        # Start forward_responses task
                        if not forward_task or forward_task.done():
                            logger.info("Starting forward_responses task")
                            forward_task = asyncio.create_task(forward_responses(websocket, transition_manager))

                        # Now forward sessionStart to Bedrock
                        stream_manager = transition_manager.get_active_stream_manager()
                        if stream_manager:
                            logger.info("Forwarding sessionStart to Bedrock")
                            await stream_manager.send_raw_event(data)

                        continue

                    # Handle promptStart - extract config and pass through
                    if event_type == 'promptStart':
                        prompt_start_event = data['event']['promptStart']
                        prompt_name = prompt_start_event.get('promptName')

                        # Extract voiceId and toolConfig for logging
                        audio_output_config = prompt_start_event.get('audioOutputConfiguration', {})
                        voice_id = audio_output_config.get('voiceId')
                        tool_config = prompt_start_event.get('toolConfiguration', {})

                        logger.info(f"Received promptStart event")
                        logger.info(f"promptName: {prompt_name}")
                        logger.info(f"voiceId: {voice_id}")
                        logger.info(f"tools: {len(tool_config.get('tools', []))}")

                        # Update stream manager with prompt name
                        if transition_manager:
                            stream_manager = transition_manager.get_active_stream_manager()
                            if stream_manager:
                                stream_manager.prompt_name = prompt_name
                                # Forward promptStart to Bedrock
                                logger.info("Forwarding promptStart to Bedrock")
                                await stream_manager.send_raw_event(data)

                        continue

                    # Handle session end - clean up resources
                    if event_type == 'sessionEnd':
                        debug_print("Received sessionEnd event")
                        break

                    # Handle contentStart to track audio content name
                    if event_type == 'contentStart' and data['event']['contentStart'].get('type') == 'AUDIO':
                        if transition_manager:
                            stream_manager = transition_manager.get_active_stream_manager()
                            if stream_manager:
                                stream_manager.audio_content_name = data['event']['contentStart']['contentName']

                    # Pass through all other events to active session
                    if transition_manager:
                        stream_manager = transition_manager.get_active_stream_manager()
                        if stream_manager:
                            # Handle audio input - route through transition manager for buffering
                            if event_type == 'audioInput':
                                import base64
                                audio_base64 = data['event']['audioInput']['content']
                                audio_bytes = base64.b64decode(audio_base64)
                                transition_manager.add_audio_chunk(audio_bytes)
                            # Forward all other events except internal ones
                            elif event_type not in ['sessionEnd', 'modelConfig', 'sessionStart', 'promptStart']:
                                await stream_manager.send_raw_event(data)
                        else:
                            if event_type not in ['sessionEnd', 'modelConfig', 'sessionStart', 'promptStart']:
                                debug_print(f"Received event {event_type} but no active stream manager")
                    else:
                        # No transition manager yet - should only happen before sessionStart
                        if event_type not in ['modelConfig']:
                            debug_print(f"Received event {event_type} before transition manager initialized")

            except json.JSONDecodeError:
                print("Invalid JSON received from WebSocket")
            except Exception as e:
                print(f"Error processing WebSocket message: {e}")
                if DEBUG:
                    import traceback
                    traceback.print_exc()
    except websockets.exceptions.ConnectionClosed:
        print("WebSocket connection closed")
    finally:
        # Clean up resources
        if not cleanup_initiated:
            cleanup_initiated = True

            if forward_task and not forward_task.done():
                debug_print("Cancelling forward_task")
                forward_task.cancel()
                try:
                    await forward_task
                    debug_print("forward_task completed after cancellation")
                except asyncio.CancelledError:
                    debug_print("forward_task was cancelled as expected")
                except Exception as e:
                    logger.error(f"Exception while awaiting cancelled forward_task: {e}")

            if transition_manager:
                try:
                    await transition_manager.close_all_sessions()
                    debug_print("All sessions closed")
                except Exception as e:
                    logger.error(f"Error closing transition manager: {e}")

        debug_print("WebSocket handler completed")


async def forward_responses(websocket, transition_manager):
    """Forward responses from Bedrock to the WebSocket.

    Args:
        websocket: WebSocket connection to client
        transition_manager: SessionTransitionManager instance (routes to active session)
    """
    try:
        debug_print("Starting forward_responses")
        while True:
            try:
                # Get active stream manager (handles session transitions automatically)
                stream_manager = transition_manager.get_active_stream_manager()
                if not stream_manager:
                    # No active session, wait briefly and continue
                    await asyncio.sleep(0.1)
                    continue

                response = await asyncio.wait_for(stream_manager.output_queue.get(), timeout=0.5)

                # Log event type for debugging (except audio events which would be too verbose)
                if "event" in response:
                    event_type = list(response["event"].keys())[0] if "event" in response else "unknown"
                    if event_type not in ["audioOutput", "audioInput"]:
                        debug_print(f"Got {event_type} event from output queue")

            except asyncio.TimeoutError:
                # Check if transition manager still has active sessions
                stream_manager = transition_manager.get_active_stream_manager()
                if not stream_manager or not stream_manager.is_active:
                    debug_print("No active session, stopping forward_responses")
                    break
                continue

            # Send to WebSocket
            try:
                event = json.dumps(response)
                await websocket.send(event)
            except websockets.exceptions.ConnectionClosed:
                debug_print("WebSocket connection closed during forward_responses")
                break
            except Exception as e:
                logger.error(f"Error sending response to WebSocket: {e}")
                continue

    except asyncio.CancelledError:
        # Task was cancelled (normal behavior during cleanup)
        debug_print("Forward responses task cancelled - allowing main handler to clean up")
    except Exception as e:
        logger.error(f"Error forwarding responses: {e}")

    finally:
        debug_print("Forward response task completed")


async def main(host, port, health_port):

    if health_port:
        try:
            start_health_check_server(host, health_port)
        except Exception as ex:
            print("Failed to start health check endpoint",ex)
    

    """Main function to run the WebSocket server."""
    try:
        # Start WebSocket server
        async with websockets.serve(websocket_handler, host, port):
            print(f"WebSocket server started at host:{host}, port:{port}")
            
            # Keep the server running forever
            await asyncio.Future()
    except Exception as ex:
        print("Failed to start websocket service",ex)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Nova S2S WebSocket Server')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    host, port, health_port = None, None, None
    host = str(os.getenv("HOST","localhost"))
    port = int(os.getenv("WS_PORT","8081"))
    if os.getenv("HEALTH_PORT"):
        health_port = int(os.getenv("HEALTH_PORT"))


    aws_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")

    if not host or not port:
        print(f"HOST and PORT are required. Received HOST: {host}, PORT: {port}")
    elif not aws_key_id or not aws_secret:
        print(f"AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required.")
    else:
        try:
            asyncio.run(main(host, port, health_port))
        except KeyboardInterrupt:
            print("Server stopped by user")
        except Exception as e:
            print(f"Server error: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
        finally:
            if MCP_CLIENT:
                MCP_CLIENT.cleanup()
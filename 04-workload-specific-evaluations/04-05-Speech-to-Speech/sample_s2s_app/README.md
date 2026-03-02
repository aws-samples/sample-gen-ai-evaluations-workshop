# Nova Sonic Test Suite

The project includes two core components:
- A Python-based WebSocket server that manages the bidirectional streaming connection with Nova Sonic.
- A React front-end application that communicates with the S2S system through the WebSocket server.

### Prerequisites
- Python 3.12+
- Node.js 14+ and npm/yarn for UI development
- AWS account with Bedrock Model access
- AWS credentials configured 

## Installation instruction
Follow these instructions to build and launch both the Python WebSocket server and the React UI, which will allow you to converse with S2S and try out the basic features.

### Install and start the Python websocket server
1. Start Python virtual machine
    ```
    cd python-server
    python3 -m venv .venv
    ```
    Mac
    ```
    source .venv/bin/activate
    ```
    Windows
    ```
    .venv\Scripts\activate
    ```

2. Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set environment variables:
    
    The AWS access key and secret are required for the Python application, as they are needed by the underlying Smithy authentication library.
    ```bash
    export AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID"
    export AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET"
    export AWS_DEFAULT_REGION="us-east-1"
    ```
    The WebSocket host and port are optional. If not specified, the application will default to `localhost` and port `8081`.
    ```bash
    export HOST="localhost"
    export WS_PORT=8081
    ```
    The health check port is optional for container deployment such as ECS/EKS. If the environment variable below is not specified, the service will not start the HTTP endpoint for health checks.
    ```bash
    export HEALTH_PORT=8082 
    ```
    
4. Start the python websocket server
    ```bash
    python server.py
    ```

   or with telemetry:

   ```
   ./run_server_with_telemetry.sh 2>&1 | tee "telemetry_$(date +%Y%m%d_%H%M%S).log"
   ```

Keep the Python WebSocket server running, then run the section below to launch the React web application, which will connect to the WebSocket service.

### Install and start the REACT frontend application
1. Navigate to the `react-client` folder
    ```bash
    cd react-client
    ```
2. Install
    ```bash
    npm install
    ```

3. This step is optional: set environment variables for the React app. If not provided, the application defaults to `ws://localhost:8081`.

    ```bash
    export REACT_APP_WEBSOCKET_URL='YOUR_WEB_SOCKET_URL'
    ```

4. If you want to run the React code outside the workshop environment, update the `homepage` value in the `react-client/package.json` file from "/proxy/3000/" to "."

5. Run
    ```
    npm start
    ```

## Test async tool handling
There is a tool called `getSlowTool` that is triggered when asking questions like “How’s the weather in Seattle today?”
The tool has a hardcoded 20-second delay, so this behavior is expected. You can check the implementation in:

python-server/s2s_session_manager.py — line 327

The tool definition can be viewed in the UI settings panel, or in the code at:

react-client/src/helper/config.js — lines 18–26



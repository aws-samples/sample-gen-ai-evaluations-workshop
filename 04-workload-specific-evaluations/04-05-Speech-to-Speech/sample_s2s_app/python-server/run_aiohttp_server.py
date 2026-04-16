"""
aiohttp-based server for SageMaker Studio compatibility.
"""
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from aiohttp import web, WSMsgType
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp", "-q"])
    from aiohttp import web, WSMsgType

import aiohttp

sys.path.insert(0, str(Path(__file__).parent))
from server import websocket_handler as ws_handler_legacy

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("WS_PORT", "8081"))


class AiohttpWSAdapter:
    """Wraps aiohttp WebSocketResponse to match the interface expected by websocket_handler."""

    def __init__(self, ws):
        self._ws = ws

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        async for msg in self._ws:
            logger.info(f"WS message received: type={msg.type}")
            if msg.type == WSMsgType.TEXT:
                yield msg.data
            elif msg.type == WSMsgType.BINARY:
                yield msg.data
            elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                logger.info(f"WS closed/error: {msg.type}")
                break

    async def send(self, data):
        if self._ws.closed:
            logger.warning("Attempted to send on closed WebSocket")
            return
        if isinstance(data, bytes):
            await self._ws.send_bytes(data)
        else:
            await self._ws.send_str(data)

    @property
    def closed(self):
        return self._ws.closed


async def handle_request(request):
    """Single handler for all requests — routes to WS or HTTP based on Upgrade header."""
    path = request.path
    upgrade = request.headers.get("Upgrade", "").lower()
    logger.info(f"Incoming request: method={request.method} path={path} upgrade='{upgrade}'")
    logger.info(f"Headers: {dict(request.headers)}")

    if upgrade == "websocket":
        logger.info("Upgrading to WebSocket")
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        logger.info("WebSocket handshake complete")
        adapter = AiohttpWSAdapter(ws)
        try:
            await ws_handler_legacy(adapter)
        except Exception as e:
            logger.error(f"Error in websocket_handler: {e}", exc_info=True)
        logger.info("WebSocket session ended")
        return ws
    else:
        return web.Response(text="OK")


async def main():
    app = web.Application()
    # Single catch-all route handles both HTTP and WebSocket
    app.router.add_route('*', '/{path:.*}', handle_request)
    app.router.add_route('*', '/', handle_request)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, HOST, PORT)
    await site.start()
    logger.info(f"Server listening on {HOST}:{PORT}")
    await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped by user")

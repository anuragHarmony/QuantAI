"""
Resilient WebSocket connector

WebSocket client with:
- Automatic reconnection with exponential backoff
- Heartbeat/ping-pong
- Message buffering during reconnection
- Connection state callbacks
"""
import asyncio
import websockets
from typing import Optional, Callable, Awaitable, Any
import json
from loguru import logger

from .base import ConnectionState


class ResilientWebSocket:
    """
    WebSocket with auto-reconnection and heartbeat

    Features:
    - Automatic reconnection on disconnect
    - Exponential backoff
    - Heartbeat monitoring
    - Message buffering
    - State callbacks
    """

    def __init__(
        self,
        url: str,
        on_message: Optional[Callable[[dict], Awaitable[None]]] = None,
        on_state_change: Optional[Callable[[ConnectionState], Awaitable[None]]] = None,
        heartbeat_interval: float = 30.0,
        reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 60.0,
        ping_timeout: float = 10.0
    ):
        """
        Initialize WebSocket connector

        Args:
            url: WebSocket URL
            on_message: Callback for incoming messages
            on_state_change: Callback for connection state changes
            heartbeat_interval: Seconds between heartbeats
            reconnect_delay: Initial reconnect delay
            max_reconnect_delay: Maximum reconnect delay
            ping_timeout: Ping timeout in seconds
        """
        self.url = url
        self.on_message = on_message
        self.on_state_change = on_state_change
        self.heartbeat_interval = heartbeat_interval
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.ping_timeout = ping_timeout

        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.state = ConnectionState.DISCONNECTED
        self.should_reconnect = True

        self._receive_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._message_buffer: list[str] = []
        self._current_delay = reconnect_delay

    async def connect(self) -> None:
        """Connect to WebSocket"""
        await self._set_state(ConnectionState.CONNECTING)

        try:
            self.ws = await websockets.connect(
                self.url,
                ping_interval=self.heartbeat_interval,
                ping_timeout=self.ping_timeout
            )

            await self._set_state(ConnectionState.CONNECTED)
            logger.info(f"Connected to WebSocket: {self.url}")

            # Reset reconnect delay on successful connection
            self._current_delay = self.reconnect_delay

            # Start receive loop
            self._receive_task = asyncio.create_task(self._receive_loop())

            # Send any buffered messages
            await self._flush_buffer()

        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            await self._set_state(ConnectionState.ERROR)
            await self._schedule_reconnect()

    async def _receive_loop(self) -> None:
        """Receive messages from WebSocket"""
        try:
            while self.ws and not self.ws.closed:
                try:
                    message = await self.ws.recv()

                    # Parse JSON message
                    if isinstance(message, str):
                        try:
                            data = json.loads(message)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON: {message}")
                            continue
                    else:
                        data = message

                    # Call message handler
                    if self.on_message:
                        asyncio.create_task(self.on_message(data))

                except websockets.ConnectionClosed:
                    logger.warning("WebSocket connection closed")
                    break
                except Exception as e:
                    logger.error(f"Error receiving message: {e}")
                    break

        finally:
            # Connection lost, attempt reconnect
            await self._set_state(ConnectionState.DISCONNECTED)
            if self.should_reconnect:
                await self._schedule_reconnect()

    async def send(self, message: dict) -> None:
        """
        Send message to WebSocket

        Args:
            message: Message to send (will be JSON encoded)
        """
        if self.state != ConnectionState.CONNECTED or not self.ws:
            # Buffer message for later
            self._message_buffer.append(json.dumps(message))
            logger.debug(f"Buffered message (state: {self.state})")
            return

        try:
            await self.ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            # Buffer for retry
            self._message_buffer.append(json.dumps(message))
            await self._set_state(ConnectionState.ERROR)

    async def _flush_buffer(self) -> None:
        """Send all buffered messages"""
        if not self._message_buffer or not self.ws:
            return

        logger.info(f"Flushing {len(self._message_buffer)} buffered messages")

        while self._message_buffer and self.ws:
            message = self._message_buffer.pop(0)
            try:
                await self.ws.send(message)
            except Exception as e:
                logger.error(f"Error flushing message: {e}")
                # Put it back
                self._message_buffer.insert(0, message)
                break

    async def _schedule_reconnect(self) -> None:
        """Schedule reconnection with exponential backoff"""
        if not self.should_reconnect:
            return

        await self._set_state(ConnectionState.RECONNECTING)

        logger.info(f"Reconnecting in {self._current_delay:.1f}s...")
        await asyncio.sleep(self._current_delay)

        # Exponential backoff
        self._current_delay = min(
            self._current_delay * 2,
            self.max_reconnect_delay
        )

        # Attempt reconnection
        await self.connect()

    async def close(self) -> None:
        """Close WebSocket connection"""
        self.should_reconnect = False

        # Cancel tasks
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Close connection
        if self.ws:
            await self.ws.close()

        await self._set_state(ConnectionState.DISCONNECTED)
        logger.info("Closed WebSocket connection")

    async def _set_state(self, new_state: ConnectionState) -> None:
        """
        Update connection state

        Args:
            new_state: New connection state
        """
        if self.state != new_state:
            old_state = self.state
            self.state = new_state

            logger.debug(f"WebSocket state: {old_state} -> {new_state}")

            # Call state change callback
            if self.on_state_change:
                try:
                    await self.on_state_change(new_state)
                except Exception as e:
                    logger.error(f"Error in state change callback: {e}")

    def get_state(self) -> ConnectionState:
        """Get current connection state"""
        return self.state

    def is_connected(self) -> bool:
        """Check if connected"""
        return self.state == ConnectionState.CONNECTED


class WebSocketManager:
    """
    Manages multiple WebSocket connections

    Useful for exchanges with multiple WebSocket endpoints
    (e.g., separate streams for market data and user data)
    """

    def __init__(self):
        self.connections: dict[str, ResilientWebSocket] = {}
        logger.info("Initialized WebSocket manager")

    async def add_connection(
        self,
        name: str,
        url: str,
        on_message: Optional[Callable[[dict], Awaitable[None]]] = None,
        on_state_change: Optional[Callable[[ConnectionState], Awaitable[None]]] = None,
        **kwargs: Any
    ) -> ResilientWebSocket:
        """
        Add and connect a WebSocket

        Args:
            name: Connection name/identifier
            url: WebSocket URL
            on_message: Message callback
            on_state_change: State change callback
            **kwargs: Additional WebSocket parameters

        Returns:
            WebSocket connection
        """
        ws = ResilientWebSocket(
            url=url,
            on_message=on_message,
            on_state_change=on_state_change,
            **kwargs
        )

        self.connections[name] = ws
        await ws.connect()

        logger.info(f"Added WebSocket connection: {name}")
        return ws

    def get_connection(self, name: str) -> Optional[ResilientWebSocket]:
        """
        Get connection by name

        Args:
            name: Connection name

        Returns:
            WebSocket connection or None
        """
        return self.connections.get(name)

    async def close_connection(self, name: str) -> None:
        """
        Close a connection

        Args:
            name: Connection name
        """
        if name in self.connections:
            await self.connections[name].close()
            del self.connections[name]
            logger.info(f"Closed connection: {name}")

    async def close_all(self) -> None:
        """Close all connections"""
        for name in list(self.connections.keys()):
            await self.close_connection(name)

        logger.info("Closed all WebSocket connections")

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics for all connections"""
        stats = {}
        for name, ws in self.connections.items():
            stats[name] = {
                "state": ws.get_state().value,
                "is_connected": ws.is_connected(),
                "buffered_messages": len(ws._message_buffer)
            }
        return stats

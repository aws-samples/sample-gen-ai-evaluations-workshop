"""Tool Registry for managing and executing tools in S2S sessions."""
import json
import asyncio
import logging
from typing import Callable, Dict, Any, Optional
from integration.strands_agent import StrandsAgent

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing tool definitions and execution."""

    def __init__(self, use_strands_agent: bool = False):
        """Initialize the tool registry.

        Args:
            use_strands_agent: Whether to initialize and use StrandsAgent for additional tools
        """
        self._tools: Dict[str, Callable] = {}
        self.strands_agent: Optional[StrandsAgent] = None
        self.use_strands_agent = use_strands_agent

        # Initialize Strands agent if enabled
        if use_strands_agent:
            try:
                self.strands_agent = StrandsAgent()
                self._register_strands_tools()
                logger.info("Strands Agent initialized and tools registered")
            except Exception as e:
                logger.warning(f"Failed to initialize StrandsAgent: {e}")
                self.strands_agent = None

    def _register_strands_tools(self):
        """Register tools from StrandsAgent directly (no wrappers)."""
        if not self.strands_agent:
            return

        # Register tools directly from Strands agent (no wrappers!)
        self.register_tool("getDateTool", self.strands_agent.agent.tool.date_time)
        self.register_tool("getSlowTool", self.strands_agent.agent.tool.slow_tool)

        # Register the agent itself as a tool (uses LLM reasoning)
        self.register_tool("agentAsATool", self.agent_as_a_tool)

        logger.info("Registered Strands tools: getDateTool, getSlowTool, agentAsATool")

    def register_tool(self, tool_name: str, tool_func: Callable):
        """Register a tool with the registry.

        Args:
            tool_name: Name of the tool
            tool_func: Callable function that implements the tool
        """
        self._tools[tool_name.lower()] = tool_func
        logger.debug(f"Registered tool: {tool_name}")

    def get_tool(self, tool_name: str) -> Optional[Callable]:
        """Get a tool by name.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool function if found, None otherwise
        """
        return self._tools.get(tool_name.lower())

    async def execute_tool(self, tool_name: str, tool_content: str) -> Any:
        """Execute a tool by name with the given content.

        Args:
            tool_name: Name of the tool to execute
            tool_content: JSON string or dict containing tool parameters

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool is not found
        """
        tool_func = self.get_tool(tool_name)
        if not tool_func:
            raise ValueError(f"Tool '{tool_name}' not found in registry")

        logger.info(f"Executing tool: {tool_name}")

        # Parse content if it's a JSON string
        if isinstance(tool_content, str):
            try:
                tool_content = json.loads(tool_content)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool_content as JSON: {tool_content}")
                tool_content = {"query": tool_content}

        # Extract query parameter (most tools expect this)
        query = tool_content.get("query", "") if isinstance(tool_content, dict) else str(tool_content)

        # Execute the tool (handle both sync and async)
        if asyncio.iscoroutinefunction(tool_func):
            result = await tool_func(query=query)
        else:
            result = tool_func(query=query)

        logger.info(f"Tool {tool_name} execution completed")
        return result

    async def agent_as_a_tool(self, query: str = "") -> str:
        """Use the full Strands agent with LLM reasoning to answer a query.

        This method invokes the agent's query() function, which uses an LLM to:
        1. Understand the user's question
        2. Select appropriate tools
        3. Execute the tools
        4. Synthesize a natural language response

        Use this when you want the agent to handle tool selection and provide
        a conversational response, rather than direct tool output.

        Args:
            query: The user's question or request

        Returns:
            Natural language response from the agent

        Example:
            >>> result = await registry.agent_as_a_tool(query="What time is it in Tokyo?")
        """
        if not self.strands_agent:
            raise RuntimeError("StrandsAgent not initialized")

        logger.info(f"Invoking agent with query: {query}")

        # Use the agent's query method which includes LLM reasoning
        result = self.strands_agent.query(query)

        logger.info("Agent query completed")
        return str(result)

    def close(self):
        """Clean up resources."""
        if self.strands_agent:
            try:
                self.strands_agent.close()
                logger.info("StrandsAgent closed")
            except Exception as e:
                logger.error(f"Error closing StrandsAgent: {e}")

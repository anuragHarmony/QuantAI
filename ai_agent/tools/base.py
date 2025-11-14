"""
AI Tool Framework with function calling support
"""
from abc import ABC, abstractmethod
from typing import Any, Optional, Callable, TypeVar, Generic
from enum import Enum
from pydantic import BaseModel, Field
from loguru import logger
import inspect
import json


T = TypeVar('T')


class ToolParameterType(str, Enum):
    """Tool parameter types"""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class ToolParameter(BaseModel):
    """Tool parameter definition"""
    name: str = Field(description="Parameter name")
    type: ToolParameterType = Field(description="Parameter type")
    description: str = Field(description="Parameter description")
    required: bool = Field(default=True, description="Whether parameter is required")
    default: Optional[Any] = Field(default=None, description="Default value")
    enum: Optional[list[Any]] = Field(default=None, description="Allowed values (if enum)")
    items: Optional[dict[str, Any]] = Field(default=None, description="Array item schema")
    properties: Optional[dict[str, Any]] = Field(default=None, description="Object properties schema")


class ToolResult(BaseModel, Generic[T]):
    """Tool execution result"""
    success: bool = Field(description="Whether execution succeeded")
    result: Optional[T] = Field(default=None, description="Result data")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Tool(BaseModel):
    """Tool definition for AI function calling"""
    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    parameters: list[ToolParameter] = Field(default_factory=list, description="Tool parameters")
    category: Optional[str] = Field(default=None, description="Tool category")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def to_openai_function(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format"""
        properties = {}
        required = []

        for param in self.parameters:
            param_schema: dict[str, Any] = {
                "type": param.type.value,
                "description": param.description
            }

            if param.enum:
                param_schema["enum"] = param.enum
            if param.items:
                param_schema["items"] = param.items
            if param.properties:
                param_schema["properties"] = param.properties

            properties[param.name] = param_schema

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

    def to_anthropic_tool(self) -> dict[str, Any]:
        """Convert to Anthropic tool format"""
        input_schema = {
            "type": "object",
            "properties": {},
            "required": []
        }

        for param in self.parameters:
            param_schema: dict[str, Any] = {
                "type": param.type.value,
                "description": param.description
            }

            if param.enum:
                param_schema["enum"] = param.enum

            input_schema["properties"][param.name] = param_schema

            if param.required:
                input_schema["required"].append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": input_schema
        }


class BaseTool(ABC):
    """Abstract base class for AI tools"""

    def __init__(self):
        self._definition: Optional[Tool] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description"""
        pass

    @property
    def category(self) -> Optional[str]:
        """Tool category (optional)"""
        return None

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult[Any]:
        """
        Execute the tool with given parameters

        Args:
            **kwargs: Tool parameters

        Returns:
            ToolResult with execution outcome
        """
        pass

    @abstractmethod
    def get_parameters(self) -> list[ToolParameter]:
        """Get tool parameter definitions"""
        pass

    def get_definition(self) -> Tool:
        """Get complete tool definition"""
        if not self._definition:
            self._definition = Tool(
                name=self.name,
                description=self.description,
                parameters=self.get_parameters(),
                category=self.category
            )
        return self._definition

    async def validate_and_execute(self, **kwargs: Any) -> ToolResult[Any]:
        """
        Validate parameters and execute tool

        Args:
            **kwargs: Tool parameters

        Returns:
            ToolResult with execution outcome
        """
        try:
            # Validate required parameters
            required_params = [p.name for p in self.get_parameters() if p.required]
            missing = [p for p in required_params if p not in kwargs]

            if missing:
                return ToolResult(
                    success=False,
                    error=f"Missing required parameters: {', '.join(missing)}"
                )

            # Execute
            return await self.execute(**kwargs)

        except Exception as e:
            logger.error(f"Tool {self.name} execution failed: {e}")
            return ToolResult(
                success=False,
                error=str(e)
            )


class ToolRegistry:
    """Registry for managing AI tools"""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._categories: dict[str, list[str]] = {}

    def register(self, tool: BaseTool) -> None:
        """
        Register a tool

        Args:
            tool: Tool instance to register
        """
        self._tools[tool.name] = tool

        # Update category index
        if tool.category:
            if tool.category not in self._categories:
                self._categories[tool.category] = []
            self._categories[tool.category].append(tool.name)

        logger.info(f"Registered tool: {tool.name}")

    def unregister(self, tool_name: str) -> None:
        """Unregister a tool"""
        if tool_name in self._tools:
            tool = self._tools[tool_name]
            del self._tools[tool_name]

            # Update category index
            if tool.category and tool.category in self._categories:
                self._categories[tool.category].remove(tool_name)

            logger.info(f"Unregistered tool: {tool_name}")

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get tool by name"""
        return self._tools.get(tool_name)

    def get_all_tools(self) -> list[BaseTool]:
        """Get all registered tools"""
        return list(self._tools.values())

    def get_tools_by_category(self, category: str) -> list[BaseTool]:
        """Get all tools in a category"""
        tool_names = self._categories.get(category, [])
        return [self._tools[name] for name in tool_names if name in self._tools]

    def get_tool_definitions(self) -> list[Tool]:
        """Get definitions for all tools"""
        return [tool.get_definition() for tool in self._tools.values()]

    def get_openai_functions(self) -> list[dict[str, Any]]:
        """Get all tools in OpenAI function calling format"""
        return [tool.get_definition().to_openai_function() for tool in self._tools.values()]

    def get_anthropic_tools(self) -> list[dict[str, Any]]:
        """Get all tools in Anthropic tool format"""
        return [tool.get_definition().to_anthropic_tool() for tool in self._tools.values()]

    async def execute_tool(self, tool_name: str, **kwargs: Any) -> ToolResult[Any]:
        """
        Execute a tool by name

        Args:
            tool_name: Name of tool to execute
            **kwargs: Tool parameters

        Returns:
            ToolResult
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool not found: {tool_name}"
            )

        return await tool.validate_and_execute(**kwargs)


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: Optional[str] = None
) -> Callable[[type[BaseTool]], type[BaseTool]]:
    """
    Decorator for registering tools

    Args:
        name: Tool name (defaults to class name)
        description: Tool description (defaults to class docstring)
        category: Tool category

    Returns:
        Decorated tool class
    """
    def decorator(cls: type[BaseTool]) -> type[BaseTool]:
        # Set name and description if not provided
        if name:
            cls.name = property(lambda self: name)  # type: ignore
        if description:
            cls.description = property(lambda self: description)  # type: ignore
        if category:
            cls.category = property(lambda self: category)  # type: ignore

        return cls

    return decorator


def create_tool_from_function(
    func: Callable[..., Any],
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: Optional[str] = None
) -> BaseTool:
    """
    Create a tool from a regular function

    Args:
        func: Function to wrap as a tool
        name: Tool name (defaults to function name)
        description: Tool description (defaults to function docstring)
        category: Tool category

    Returns:
        Tool instance
    """
    tool_name = name or func.__name__
    tool_description = description or func.__doc__ or "No description"

    # Extract parameters from function signature
    sig = inspect.signature(func)
    parameters: list[ToolParameter] = []

    for param_name, param in sig.parameters.items():
        param_type = ToolParameterType.STRING  # Default

        # Try to infer type from annotation
        if param.annotation != inspect.Parameter.empty:
            if param.annotation == int:
                param_type = ToolParameterType.INTEGER
            elif param.annotation == float:
                param_type = ToolParameterType.NUMBER
            elif param.annotation == bool:
                param_type = ToolParameterType.BOOLEAN
            elif param.annotation == list:
                param_type = ToolParameterType.ARRAY

        parameters.append(ToolParameter(
            name=param_name,
            type=param_type,
            description=f"Parameter: {param_name}",
            required=param.default == inspect.Parameter.empty
        ))

    # Create tool class dynamically
    class FunctionTool(BaseTool):
        @property
        def name(self) -> str:
            return tool_name

        @property
        def description(self) -> str:
            return tool_description

        @property
        def category(self) -> Optional[str]:
            return category

        def get_parameters(self) -> list[ToolParameter]:
            return parameters

        async def execute(self, **kwargs: Any) -> ToolResult[Any]:
            try:
                # Call function
                if inspect.iscoroutinefunction(func):
                    result = await func(**kwargs)
                else:
                    result = func(**kwargs)

                return ToolResult(success=True, result=result)
            except Exception as e:
                return ToolResult(success=False, error=str(e))

    return FunctionTool()


# Global tool registry
global_registry = ToolRegistry()

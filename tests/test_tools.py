"""
Tests for AI tools
"""
import pytest
from ai_agent.tools.base import (
    BaseTool,
    ToolParameter,
    ToolParameterType,
    ToolResult,
    ToolRegistry,
    create_tool_from_function
)


class TestBaseTool:
    """Test base tool functionality"""

    @pytest.mark.asyncio
    async def test_tool_creation(self):
        """Test creating a simple tool"""

        class TestTool(BaseTool):
            @property
            def name(self) -> str:
                return "test_tool"

            @property
            def description(self) -> str:
                return "A test tool"

            def get_parameters(self) -> list[ToolParameter]:
                return [
                    ToolParameter(
                        name="input",
                        type=ToolParameterType.STRING,
                        description="Test input",
                        required=True
                    )
                ]

            async def execute(self, **kwargs) -> ToolResult:
                return ToolResult(
                    success=True,
                    result=f"Received: {kwargs.get('input')}"
                )

        tool = TestTool()

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert len(tool.get_parameters()) == 1

        result = await tool.execute(input="hello")
        assert result.success
        assert result.result == "Received: hello"

    @pytest.mark.asyncio
    async def test_tool_validation(self):
        """Test tool parameter validation"""

        class TestTool(BaseTool):
            @property
            def name(self) -> str:
                return "test_tool"

            @property
            def description(self) -> str:
                return "A test tool"

            def get_parameters(self) -> list[ToolParameter]:
                return [
                    ToolParameter(
                        name="required_param",
                        type=ToolParameterType.STRING,
                        description="Required parameter",
                        required=True
                    )
                ]

            async def execute(self, **kwargs) -> ToolResult:
                return ToolResult(success=True, result="OK")

        tool = TestTool()

        # Should fail without required parameter
        result = await tool.validate_and_execute()
        assert not result.success
        assert "required" in result.error.lower()

        # Should succeed with required parameter
        result = await tool.validate_and_execute(required_param="value")
        assert result.success


class TestToolRegistry:
    """Test tool registry"""

    def test_register_tool(self):
        """Test registering tools"""
        registry = ToolRegistry()

        class TestTool(BaseTool):
            @property
            def name(self) -> str:
                return "test_tool"

            @property
            def description(self) -> str:
                return "Test"

            @property
            def category(self) -> str:
                return "testing"

            def get_parameters(self) -> list[ToolParameter]:
                return []

            async def execute(self, **kwargs) -> ToolResult:
                return ToolResult(success=True)

        tool = TestTool()
        registry.register(tool)

        assert registry.get_tool("test_tool") is not None
        assert len(registry.get_tools_by_category("testing")) == 1

    @pytest.mark.asyncio
    async def test_execute_tool(self):
        """Test executing tools through registry"""
        registry = ToolRegistry()

        class TestTool(BaseTool):
            @property
            def name(self) -> str:
                return "test_tool"

            @property
            def description(self) -> str:
                return "Test"

            def get_parameters(self) -> list[ToolParameter]:
                return []

            async def execute(self, **kwargs) -> ToolResult:
                return ToolResult(success=True, result="executed")

        registry.register(TestTool())

        result = await registry.execute_tool("test_tool")
        assert result.success
        assert result.result == "executed"

        # Test non-existent tool
        result = await registry.execute_tool("nonexistent")
        assert not result.success


class TestToolConversion:
    """Test tool format conversions"""

    def test_openai_format(self):
        """Test conversion to OpenAI function format"""

        class TestTool(BaseTool):
            @property
            def name(self) -> str:
                return "test_tool"

            @property
            def description(self) -> str:
                return "Test tool"

            def get_parameters(self) -> list[ToolParameter]:
                return [
                    ToolParameter(
                        name="param1",
                        type=ToolParameterType.STRING,
                        description="First param",
                        required=True
                    ),
                    ToolParameter(
                        name="param2",
                        type=ToolParameterType.NUMBER,
                        description="Second param",
                        required=False
                    )
                ]

            async def execute(self, **kwargs) -> ToolResult:
                return ToolResult(success=True)

        tool = TestTool()
        definition = tool.get_definition()
        openai_func = definition.to_openai_function()

        assert openai_func["name"] == "test_tool"
        assert openai_func["description"] == "Test tool"
        assert "parameters" in openai_func
        assert "param1" in openai_func["parameters"]["properties"]
        assert "param1" in openai_func["parameters"]["required"]
        assert "param2" not in openai_func["parameters"]["required"]


class TestFunctionWrapper:
    """Test creating tools from functions"""

    @pytest.mark.asyncio
    async def test_function_to_tool(self):
        """Test wrapping a function as a tool"""

        async def my_function(arg1: str, arg2: int = 10) -> str:
            """Test function"""
            return f"{arg1}-{arg2}"

        tool = create_tool_from_function(my_function)

        assert tool.name == "my_function"
        assert "Test function" in tool.description

        result = await tool.execute(arg1="test", arg2=20)
        assert result.success
        assert result.result == "test-20"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
FunctionGemma Router using LM Studio's OpenAI-compatible API.

This module provides a wrapper for routing natural language to function calls
using FunctionGemma served via LM Studio.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@dataclass
class FunctionCall:
    """Represents a parsed function call from FunctionGemma."""

    name: str
    arguments: dict[str, Any]
    raw_output: str = ""


@dataclass
class ToolDefinition:
    """Definition of a tool that can be called by FunctionGemma."""

    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable


class GemmaRouter:
    """
    Router that uses FunctionGemma to map natural language to function calls.

    Uses LM Studio's OpenAI-compatible API for inference.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ):
        """
        Initialize the GemmaRouter.

        Args:
            base_url: LM Studio API URL. Defaults to LM_STUDIO_URL env var.
            model: Model name in LM Studio. Defaults to LM_STUDIO_MODEL env var.
        """
        self.base_url = base_url or os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234/v1")
        self.model = model or os.getenv("LM_STUDIO_MODEL", "functiongemma-270m-it")
        self.client = OpenAI(base_url=self.base_url, api_key="lm-studio")
        self.tools: dict[str, ToolDefinition] = {}

    def register_tool(
        self,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable:
        """
        Decorator to register a function as a tool.

        Args:
            name: Optional custom name for the tool. Defaults to function name.
            description: Optional description. Defaults to function docstring.

        Returns:
            Decorator function.
        """

        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_description = description or func.__doc__ or "No description provided."

            # Extract parameters from function annotations
            params = self._extract_parameters(func)

            self.tools[tool_name] = ToolDefinition(
                name=tool_name,
                description=tool_description,
                parameters=params,
                function=func,
            )
            return func

        return decorator

    def _extract_parameters(self, func: Callable) -> dict[str, Any]:
        """
        Extract parameter schema from function annotations.

        Args:
            func: The function to extract parameters from.

        Returns:
            JSON Schema-like parameter definition.
        """
        import inspect

        sig = inspect.signature(func)
        hints = func.__annotations__

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_type = hints.get(param_name, str)
            type_map = {
                str: "STRING",
                int: "INTEGER",
                float: "NUMBER",
                bool: "BOOLEAN",
            }

            # Handle Optional types
            type_str = type_map.get(param_type, "STRING")
            if hasattr(param_type, "__origin__"):  # Optional, Union, etc.
                args = getattr(param_type, "__args__", ())
                if type(None) not in args:
                    required.append(param_name)
                # Get the actual type from Optional
                for arg in args:
                    if arg is not type(None):
                        type_str = type_map.get(arg, "STRING")
                        break
            elif param.default is inspect.Parameter.empty:
                required.append(param_name)

            properties[param_name] = {
                "type": type_str,
                "description": f"Parameter: {param_name}",
            }

        return {
            "type": "OBJECT",
            "properties": properties,
            "required": required,
        }

    def _build_openai_tools(self) -> list[dict]:
        """
        Build OpenAI-compatible tool definitions for LM Studio.

        Returns:
            List of tool definitions in OpenAI format.
        """
        tools = []
        for tool in self.tools.values():
            # Convert our parameter format to OpenAI JSON Schema format
            properties = {}
            for prop_name, prop_def in tool.parameters.get("properties", {}).items():
                # Map our types to JSON Schema types
                type_map = {
                    "STRING": "string",
                    "INTEGER": "integer",
                    "NUMBER": "number",
                    "BOOLEAN": "boolean",
                }
                properties[prop_name] = {
                    "type": type_map.get(prop_def.get("type", "STRING"), "string"),
                    "description": prop_def.get("description", ""),
                }

            tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": tool.parameters.get("required", []),
                    },
                },
            })
        return tools

    def _parse_openai_tool_calls(self, message: Any) -> FunctionCall | None:
        """
        Parse OpenAI-style tool calls from response.

        Args:
            message: The response message object.

        Returns:
            Parsed FunctionCall or None if no tool call found.
        """
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return None

        tool_call = message.tool_calls[0]  # Get first tool call
        func = tool_call.function

        # Parse arguments from JSON string
        try:
            arguments = json.loads(func.arguments) if func.arguments else {}
        except json.JSONDecodeError:
            arguments = {}

        return FunctionCall(
            name=func.name,
            arguments=arguments,
            raw_output=str(tool_call),
        )

    def _cast_value(self, value: str) -> Any:
        """
        Cast string value to appropriate Python type.

        Args:
            value: String value to cast.

        Returns:
            Casted value (int, float, bool, or str).
        """
        # Try int
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Try bool
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False

        # Return as string
        return value.strip("'\"")

    def route(self, user_input: str, context: str | None = None) -> FunctionCall | None:
        """
        Route natural language input to a function call.

        Args:
            user_input: Natural language input from user.
            context: Optional context string (e.g., current state, task list) to help the model.

        Returns:
            Parsed FunctionCall or None if routing failed.
        """
        # Build OpenAI-compatible tools
        tools = self._build_openai_tools()

        # System prompt for function calling
        system_prompt = (
            "You are a helpful assistant that uses the provided functions to help users. "
            "When the user asks to do something, call the appropriate function. "
            "Always use a function when one is available for the task. "
            "Use the provided context to find the correct IDs and make accurate function calls."
        )
        
        # Add context if provided
        if context:
            system_prompt += f"\n\nCurrent context:\n{context}"

        # Call LM Studio with tools parameter
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            tools=tools,
            tool_choice="auto",  # Let the model decide when to use tools
            max_tokens=256,
            temperature=0.1,  # Low temperature for deterministic output
        )

        message = response.choices[0].message
        
        # Try to parse tool calls from OpenAI format
        function_call = self._parse_openai_tool_calls(message)
        
        if function_call:
            return function_call
        
        # If no tool call, return None
        return None

    def execute(self, function_call: FunctionCall) -> Any:
        """
        Execute a function call.

        Args:
            function_call: The function call to execute.

        Returns:
            Result of the function execution.

        Raises:
            ValueError: If function is not registered.
        """
        if function_call.name not in self.tools:
            raise ValueError(f"Unknown function: {function_call.name}")

        tool = self.tools[function_call.name]
        return tool.function(**function_call.arguments)

    def route_and_execute(self, user_input: str) -> tuple[FunctionCall | None, Any]:
        """
        Route natural language to function and execute it.

        Args:
            user_input: Natural language input from user.

        Returns:
            Tuple of (FunctionCall, result) or (None, None) if routing failed.
        """
        function_call = self.route(user_input)
        if function_call is None:
            return None, None

        result = self.execute(function_call)
        return function_call, result

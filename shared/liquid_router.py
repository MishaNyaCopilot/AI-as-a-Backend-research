"""
LiquidAI LFM Router using LM Studio's OpenAI-compatible API.

This module provides a wrapper for routing natural language to function calls
using LiquidAI LFM2.5 models served via LM Studio.

Key differences from FunctionGemma:
- Tools are defined in system prompt as JSON (not as OpenAI tools param)
- Tool calls use <|tool_call_start|> and <|tool_call_end|> tokens
- Default output format is Pythonic function calls, not JSON
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@dataclass
class FunctionCall:
    """Represents a parsed function call from LFM."""

    name: str
    arguments: dict[str, Any]
    raw_output: str = ""


@dataclass
class ToolDefinition:
    """Definition of a tool that can be called by LFM."""

    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable


class LiquidRouter:
    """
    Router that uses LiquidAI LFM to map natural language to function calls.

    Uses LM Studio's OpenAI-compatible API for inference.
    Supports LFM2.5-1.2B-Instruct and similar models.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ):
        """
        Initialize the LiquidRouter.

        Args:
            base_url: LM Studio API URL. Defaults to LM_STUDIO_URL env var.
            model: Model name in LM Studio. Defaults to LM_STUDIO_MODEL env var.
        """
        self.base_url = base_url or os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234/v1")
        self.model = model or os.getenv("LM_STUDIO_MODEL", "lfm2.5-1.2b-instruct")
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
            JSON Schema parameter definition for LFM.
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
                str: "string",
                int: "integer",
                float: "number",
                bool: "boolean",
            }

            # Handle Optional types
            type_str = type_map.get(param_type, "string")
            if hasattr(param_type, "__origin__"):  # Optional, Union, etc.
                args = getattr(param_type, "__args__", ())
                if type(None) not in args:
                    required.append(param_name)
                # Get the actual type from Optional
                for arg in args:
                    if arg is not type(None):
                        type_str = type_map.get(arg, "string")
                        break
            elif param.default is inspect.Parameter.empty:
                required.append(param_name)

            properties[param_name] = {
                "type": type_str,
                "description": f"Parameter: {param_name}",
            }

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _build_tools_json(self) -> list[dict]:
        """
        Build LFM-compatible tool definitions as JSON list.

        Returns:
            List of tool definitions for the system prompt.
        """
        tools = []
        for tool in self.tools.values():
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            })
        return tools

    def _build_system_prompt(self, context: str | None = None) -> str:
        """
        Build system prompt with tool definitions.

        Args:
            context: Optional additional context to include.

        Returns:
            System prompt string.
        """
        tools_json = self._build_tools_json()

        parts = [
            "You are a helpful assistant that can call functions to help users.",
            "Output function calls as JSON.",  # Request JSON format instead of Pythonic
            f"List of tools: {json.dumps(tools_json)}",
        ]

        if context:
            parts.append(f"\nContext:\n{context}")

        return "\n".join(parts)

    def _parse_tool_call(self, output: str) -> FunctionCall | None:
        """
        Parse a tool call from LFM output.

        LFM uses <|tool_call_start|>[function_name(args), ...]<|tool_call_end|> format.
        Uses Python AST for robust parsing instead of regex.

        Args:
            output: Raw model output.

        Returns:
            FunctionCall if found (first valid one), None otherwise.
        """
        import ast

        # Step 1: Extract content between tool call tokens (minimal regex)
        start_token = "<|tool_call_start|>"
        end_token = "<|tool_call_end|>"

        start_idx = output.find(start_token)
        if start_idx == -1:
            return self._try_json_parse(output)

        end_idx = output.find(end_token, start_idx)
        if end_idx == -1:
            return self._try_json_parse(output)

        calls_content = output[start_idx + len(start_token):end_idx].strip()

        # Step 2: Try to parse as Python AST
        # Wrap in a way that makes it valid Python for parsing
        try:
            # Try parsing as a list of calls
            if calls_content.startswith("["):
                tree = ast.parse(calls_content, mode="eval")
                calls = tree.body.elts if isinstance(tree.body, ast.List) else [tree.body]
            else:
                tree = ast.parse(calls_content, mode="eval")
                calls = [tree.body]
        except SyntaxError:
            # Fallback to simple string parsing
            return self._parse_simple(calls_content, output)

        # Step 3: Extract function calls from AST
        for node in calls:
            if not isinstance(node, ast.Call):
                continue

            # Get function name
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            else:
                continue

            if func_name not in self.tools:
                continue

            # Extract keyword arguments
            arguments = {}
            for kw in node.keywords:
                key = kw.arg
                # Safely evaluate the value
                try:
                    value = ast.literal_eval(kw.value)
                except (ValueError, TypeError):
                    # For non-literal values, try to get string representation
                    if isinstance(kw.value, ast.Constant):
                        value = kw.value.value
                    elif isinstance(kw.value, ast.Name):
                        value = kw.value.id
                    else:
                        continue
                arguments[key] = value

            # Filter to valid parameters
            tool = self.tools[func_name]
            valid_params = set(tool.parameters.get("properties", {}).keys())
            filtered_args = {k: v for k, v in arguments.items() if k in valid_params}

            return FunctionCall(
                name=func_name,
                arguments=filtered_args,
                raw_output=output,
            )

        return None

    def _parse_simple(self, calls_content: str, output: str) -> FunctionCall | None:
        """
        Simple string-based parsing fallback.

        Args:
            calls_content: Content between tool call tokens.
            output: Full raw output.

        Returns:
            FunctionCall if found, None otherwise.
        """
        # Find function name and args using string methods
        paren_start = calls_content.find("(")
        if paren_start == -1:
            return None

        # Extract function name (handle leading bracket)
        func_part = calls_content[:paren_start].strip().lstrip("[")
        func_name = func_part.split()[-1] if func_part else None

        if not func_name or func_name not in self.tools:
            return None

        # Extract arguments string
        paren_end = calls_content.find(")", paren_start)
        if paren_end == -1:
            return None

        args_str = calls_content[paren_start + 1:paren_end]

        # Parse key=value pairs with simple string splitting
        arguments = {}
        for part in args_str.split(","):
            if "=" not in part:
                continue
            key, _, value = part.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            # Try int conversion
            if value.isdigit():
                value = int(value)
            arguments[key] = value

        # Filter to valid parameters
        tool = self.tools[func_name]
        valid_params = set(tool.parameters.get("properties", {}).keys())
        filtered_args = {k: v for k, v in arguments.items() if k in valid_params}

        return FunctionCall(
            name=func_name,
            arguments=filtered_args,
            raw_output=output,
        )

    def _try_json_parse(self, output: str) -> FunctionCall | None:
        """
        Try to parse JSON format tool calls.

        Args:
            output: Raw model output.

        Returns:
            FunctionCall if JSON format found, None otherwise.
        """
        # Look for JSON object with name and arguments
        try:
            # Find JSON-like structure
            start = output.find('{"name"')
            if start == -1:
                start = output.find("{'name'")
            if start == -1:
                return None

            # Find matching closing brace
            brace_count = 0
            end = start
            for i, char in enumerate(output[start:], start):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break

            json_str = output[start:end]
            data = json.loads(json_str)

            func_name = data.get("name")
            arguments = data.get("arguments", {})

            if func_name in self.tools:
                tool = self.tools[func_name]
                valid_params = set(tool.parameters.get("properties", {}).keys())
                filtered_args = {k: v for k, v in arguments.items() if k in valid_params}

                return FunctionCall(
                    name=func_name,
                    arguments=filtered_args,
                    raw_output=output,
                )
        except (json.JSONDecodeError, KeyError):
            pass

        return None

    def route(self, user_input: str, context: str | None = None) -> FunctionCall | None:
        """
        Route natural language input to a function call.

        Args:
            user_input: Natural language input from user.
            context: Optional context string (e.g., current tasks, date).

        Returns:
            FunctionCall if routing succeeds, None otherwise.
        """
        if not self.tools:
            raise ValueError("No tools registered. Use @router.register_tool decorator.")

        system_prompt = self._build_system_prompt(context)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
                max_tokens=256,
                temperature=0.1,  # Low temperature for deterministic output
            )

            output = response.choices[0].message.content or ""

            # Try to parse tool call
            function_call = self._parse_tool_call(output)

            if function_call:
                return function_call

            # No tool call found
            return None

        except Exception as e:
            print(f"Error routing: {e}")
            return None

    def execute(self, function_call: FunctionCall) -> Any:
        """
        Execute a function call.

        Args:
            function_call: The function call to execute.

        Returns:
            Result of the function execution.
        """
        tool = self.tools.get(function_call.name)
        if not tool:
            raise ValueError(f"Unknown function: {function_call.name}")

        return tool.function(**function_call.arguments)

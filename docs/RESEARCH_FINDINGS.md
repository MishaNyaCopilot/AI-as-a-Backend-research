# LiquidAI LFM Research Findings

## Overview

This document captures our findings from testing LiquidAI LFM2.5-1.2B as an edge model for natural language → function routing.

## Model Comparison

| Feature | FunctionGemma 270M | LiquidAI LFM 1.2B |
|---------|-------------------|-------------------|
| **Size** | 270M (288MB) | 1.2B (~1.5GB) |
| **Accuracy** | ~64% | ~100% |
| **Natural language** | Needs explicit keywords | Understands intent |
| **Tool format** | OpenAI tools param | JSON in system prompt |
| **Output format** | JSON | Pythonic (configurable) |

## Test Results

**All tests passed with LiquidAI LFM:**

| Command | Result |
|---------|--------|
| `add buy milk` | ✅ create_task |
| `add call mom` | ✅ create_task |
| `remind me to exercise` | ✅ create_task (FunctionGemma failed) |
| `show my tasks` | ✅ list_tasks |
| `buy groceries` | ✅ create_task (no "add" prefix needed!) |
| `complete task 1` | ✅ complete_task |
| `I finished buying milk` | ✅ complete_task (natural language!) |

## Key Differences

### 1. Tool Definition Format

FunctionGemma uses OpenAI tools parameter:
```python
response = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=[{"type": "function", "function": {...}}],  # OpenAI format
)
```

LiquidAI uses JSON in system prompt:
```python
system_prompt = f"List of tools: {json.dumps(tools)}"
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "system", "content": system_prompt}, ...],
)
```

### 2. Output Format

FunctionGemma outputs JSON in `tool_calls` array:
```json
{"tool_calls": [{"function": {"name": "...", "arguments": {...}}}]}
```

LiquidAI outputs Pythonic calls in `content` with special tokens:
```
<|tool_call_start|>[create_task(title="buy milk")]<|tool_call_end|>
```

### 3. Parallel Tool Calls

LiquidAI can output multiple tool calls in a list:
```
<|tool_call_start|>[delete_task(title="a"), delete_task(title="b")]<|tool_call_end|>
```

## Parsing Strategy

We avoided heavy regex by using Python's AST module:

1. **Find tokens** - Simple `str.find()` for `<|tool_call_start|>` / `<|tool_call_end|>`
2. **Parse with AST** - Use `ast.parse()` for Pythonic function calls
3. **Fallback** - Simple string splitting if AST fails
4. **JSON fallback** - Parse JSON format if tokens not found

This is more robust than regex patterns.

## Recommendations

### When to use LiquidAI LFM

- ✅ Need high accuracy (>90%)
- ✅ Natural language understanding required
- ✅ Have 1-2GB memory available
- ✅ Users won't use explicit command syntax

### When to use FunctionGemma

- ✅ Memory constrained (<300MB)
- ✅ Simple, keyword-based commands acceptable
- ✅ Controlled vocabulary application
- ✅ Fine-tuning planned

## Configuration

Switch models via `.env`:

```env
# LiquidAI LFM (recommended)
ROUTER_TYPE=liquid
LM_STUDIO_MODEL=lfm2.5-1.2b-instruct

# FunctionGemma (lightweight)
ROUTER_TYPE=gemma
LM_STUDIO_MODEL=functiongemma-270m-it
```

## Conclusion

LiquidAI LFM 1.2B is a **significant upgrade** over FunctionGemma 270M for edge function calling. The 4.4x size increase yields dramatically better natural language understanding with ~100% accuracy on our test set.

For production "AI-as-a-Backend" use cases, **LiquidAI LFM is the recommended choice** unless memory is severely constrained.

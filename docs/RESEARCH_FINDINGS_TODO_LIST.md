# FunctionGemma Research Findings

## Overview

This document captures our findings from testing FunctionGemma 270M as an "AI-as-a-Backend" natural language router for a todo list application.

## Test Environment

- **Model**: FunctionGemma 270M IT (via LM Studio at http://127.0.0.1:1234)
- **Backend**: FastAPI + SQLModel + SQLite
- **Hardware**: NVIDIA RTX 5060 Ti (16GB)
- **Date**: 2026-01-31

## Base Model Accuracy

**Measured accuracy: ~64%** on our test set of 14 examples.

| Input Type | Success Rate | Notes |
|------------|--------------|-------|
| Clear commands ("add X") | High | Works reliably |
| Implicit commands ("buy X") | Low | Needs explicit keywords |
| Commands with dates | Low | Complex parsing fails |
| Task completion by title | Low | ID-based works better |

## Fine-tuning Experiments

> [!CAUTION]
> Fine-tuning FunctionGemma requires careful consideration. Our experiments show it can easily **break** a working model.

### Experiment 1: Full Fine-tuning

| Metric | Before | After |
|--------|--------|-------|
| Accuracy | 64.3% | **0%** |
| Loss | - | 0.00002 |
| Token Accuracy | - | 100% |

**Result**: Catastrophic forgetting. Model memorized training data but forgot how to generate function calls.

**Training stats showed severe overfitting:**
- Loss dropped to near-zero (0.00002)
- Entropy collapsed (0.41 → 0.0003)
- 100% training accuracy but 0% test accuracy

### Experiment 2: LoRA Fine-tuning

| Setting | Value |
|---------|-------|
| LoRA rank | 16 |
| Alpha | 32 |
| Trainable params | 0.55% |
| Epochs | 3 |

**Result**: Same issue - 71.4% → 0% accuracy.

### Why Fine-tuning Failed

Google's documentation shows fine-tuning improving accuracy from **10% → 80%**. Key difference:

| Aspect | Google's Use Case | Our Use Case |
|--------|-------------------|--------------|
| Before accuracy | ~10% | ~64% |
| Before behavior | Model refused to call functions | Model correctly called functions |
| Problem | Tool selection ambiguity | Model already working |

**Conclusion**: Fine-tuning is useful when the base model **fails to call functions at all**. When it's already working, fine-tuning causes catastrophic forgetting.

## When to Fine-tune FunctionGemma

✅ **Fine-tune when:**
- Base model doesn't call any functions (outputs natural language only)
- You need to distinguish between similar tools (internal vs external search)
- You have hundreds of diverse training examples

❌ **Don't fine-tune when:**
- Base model already achieves reasonable accuracy (>50%)
- You have a small dataset (<100 examples)
- Your tools are already distinct from each other

## Recommendations

### For Best Results Without Fine-tuning

1. **Use explicit trigger words** in tool descriptions
2. **Simplify tool set** - fewer, clearer tools
3. **Provide context injection** - current state, date, options
4. **Add fallback handling** - helpful suggestions when routing fails
5. **Accept limitations** - 60-70% accuracy is good for a 270M model

### Tool Description Best Practices

```python
# Good: Short, keyword-rich
description="Create a task. Use when user says: add, create, new, remind me."

# Bad: Long, vague
description="This function creates a new task in the todo list when the user wants to add something they need to remember to do later."
```

### Fallback Strategy

When routing fails, provide:
1. **Keyword detection** - identify intent even if routing fails
2. **Example commands** - show users what works
3. **Graceful degradation** - offer direct API access

## What Works vs What Doesn't

### ✅ Works Reliably

| Input | Function |
|-------|----------|
| `add buy milk` | create_task |
| `show my tasks` | list_tasks |
| `complete task 3` | complete_task |
| `delete task 1` | delete_task |
| `list urgent tasks` | list_tasks (priority=high) |

### ❌ Fails

| Input | Issue |
|-------|-------|
| `buy eggs` | Missing "add" prefix |
| `remind me to exercise` | Different phrasing |
| `add call mom tomorrow` | Complex sentence with date |
| `I finished shopping` | Implicit completion |

## Conclusion

FunctionGemma 270M achieves **~64% accuracy** out-of-box for simple todo commands. This is reasonable for a 270M model running locally.

**Fine-tuning is NOT recommended** for this use case because:
1. Base model already works at acceptable accuracy
2. Small training sets cause catastrophic forgetting
3. The model's function-calling ability is fragile

**Best approach**: Accept base model limitations, use explicit keywords, provide good fallback handling, and document what works.

For production use cases requiring higher accuracy:
- Use a larger model (Gemma 2B/7B)
- Or accept ~64% accuracy with fallback to structured input

# LiquidAI LFM Research: Financial Tracker (Project 2)

## Overview

This document summarizes findings from implementing a natural language Financial Tracker using LiquidAI LFM2.5-1.2B. This project was more complex than the Todo List due to overlapping intents (recording vs. analysis).

## Key Findings

### 1. Intent Conflict (Recording vs. Analysis)
One of the most critical findings was the model's tendency to misroute "intent to record" (e.g., "earned money") to "intent to analyze" (e.g., "show income summary").

*   **Initial Failure**: The prompt "My sidehustle earned me 4593 today" was routed to `get_income_analysis` instead of `add_income`.
*   **Root Cause**: Both tools mentioned "income" and "today" in their descriptions, and the model prioritized the analysis tool because it seemed more "comprehensive."
*   **Solution**: Categorizing tool descriptions with explicit intent labels (e.g., **RECORD** vs **STATUS CHECK**) and adding negative constraints (e.g., "Do NOT use this to record new entries") achieved 100% accuracy.

### 2. Parameter Extraction
LFM 1.2B excelled at extracting numeric values and mapping them to specific categories even when not explicitly named.
*   "Spent 50 on sushi" -> `amount=50.0`, `category="sushi"`
*   "Got paid 3000 from work" -> `amount=3000.0`, `source="work"`

### 3. Model Performance Comparison

| Feature | Todo List (Simple) | Financial Tracker (Complex) |
|---------|--------------------|-----------------------------|
| **Accuracy (Base Prompts)** | ~100% | ~85% |
| **Accuracy (Refined Prompts)** | ~100% | ~100% |
| **Context Handling** | High | Extreme (Requires Balance & History) |
| **Ambiguity Tolerance** | High | Medium (Precise descriptions needed) |

## Routing Strategy Refinements

To ensure production-grade reliability for AI-as-a-Backend, we established these "Gold Standard" rules for tool descriptions:

1.  **Action Labels**: Prefix descriptions with ALL-CAPS action types: **RECORD**, **QUERY**, **ANALYSIS**, **STATUS**.
2.  **Keyword Inclusion**: Explicitly list synonyms like "earned", "salary", "gift", "spent", "bill".
3.  **Boundary Definitions**: Clearly state what a tool *cannot* do to prevent overlap.
4.  **Static Data Guidance**: Inform the model about parameter requirements (e.g., "Requires amount and category").

## Conclusion

LiquidAI LFM 1.2B is highly capable for financial backends but requires **Precise Intent Engineering**. The leap from simple TODO actions to stateful financial queries requires a more disciplined approach to the "Shared System Prompt" and tool definitions than FunctionGemma (270M) could ever reliably handle without significant fine-tuning.

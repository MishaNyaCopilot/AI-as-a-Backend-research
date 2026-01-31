# AI-as-a-Backend

Research project exploring localized edge models (FunctionGemma & LiquidAI LFM) as a natural language → function routing layer for modern web applications.

## Overview

This project demonstrates "AI-as-a-Backend" architecture—where the primary interface for users is natural language, which is decomposed into structured API calls by a small, localized language model.

## 🚀 Projects

### 1. Todo List App 📝
A robust task manager validating basic function routing.
- **Run**: `uv run uvicorn todo_app.main:app --reload --port 8003`
- **Examples**: "Add buy milk", "I finished the report", "Show my urgent tasks".

### 2. Financial Tracker 💰
A complex stateful application with "Intent Engineering" to handle overlapping commands.
- **Run**: `uv run uvicorn finance_app.main:app --reload --port 8004`
- **Examples**: "Spent $50 on sushi", "My sidehustle earned me 4500 today", "Analyze my spending this month".

## 🧠 Research Results

We compared two localized models for edge routing. Full findings in the `docs/` folder.

| Feature | FunctionGemma 270M | LiquidAI LFM 1.2B |
|---------|-------------------|-------------------|
| **Accuracy** | ~65% | ~100% |
| **NLU Quality** | Keyword-based | Intent-aware |
| **Logic handling** | Simple | Complex / Multi-step |

**Key Research Papers:**
- [Todo List Findings (LFM vs Gemma)](docs/RESEARCH_FINDINGS_TODO_LFM.md)
- [Financial Tracker Findings (Intent Engineering)](docs/RESEARCH_FINDINGS_FINANCE_LFM.md)

## 🛠 Setup

### Prerequisites
- Python 3.11+
- [LM Studio](https://lmstudio.ai/)
- UV package manager

### Installation
```powershell
uv sync
```

### Configuration (`.env`)
```env
LM_STUDIO_URL=http://127.0.0.1:1234/v1
LM_STUDIO_MODEL=lfm2.5-1.2b-instruct
ROUTER_TYPE=liquid  # Use "gemma" for FunctionGemma 270M
```

## 🏗 Project Structure

```
AI-as-a-Backend/
├── shared/           # LLM Routers (Gemma & Liquid)
├── todo_app/         # Project 1: Task Management
├── finance_app/      # Project 2: Premium Financial Tracker
├── docs/             # Technical research & results
└── .env              # Hardware configuration
```
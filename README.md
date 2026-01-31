# AI-as-a-Backend

Research project exploring the use of FunctionGemma as a natural language → function routing layer.

## Overview

This project demonstrates how small, specialized language models like [FunctionGemma](https://ai.google.dev/gemma/docs/functiongemma) (270M parameters) can be used to create "AI-as-a-Backend" applications where users interact with APIs using natural language.

## Projects

### 1. Todo List App 📝
A simple todo list where you can add, view, complete, and delete tasks using natural language.

**Example interactions:**
- "Add buy milk to my list"
- "What do I need to do?"
- "I finished the shopping"
- "Delete completed tasks"

### 2. Financial Tracker 💰 (Coming Soon)
Track spending, income, and query your finances naturally.

## Setup

### Prerequisites
- Python 3.11+
- [LM Studio](https://lmstudio.ai/) with FunctionGemma loaded
- UV package manager

### Installation

```powershell
# Create virtual environment
uv venv

# Activate it
.\.venv\Scripts\activate

# Install dependencies
uv sync
```

### Running LM Studio

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Search for and download `functiongemma-270m-it`
3. Load the model and start the local server (default: http://127.0.0.1:1234)

### Running the Todo App

```powershell
# Make sure LM Studio is running with FunctionGemma loaded
# Then run:
uv run uvicorn todo_app.main:app --reload
```

Open http://localhost:8000 in your browser.

## Architecture

```
User (Natural Language)
         ↓
    FastAPI Server
         ↓
  FunctionGemma Router ──→ LM Studio (local)
         ↓
   Function Executor
         ↓
    SQLite Database
```

## Project Structure

```
AI-as-a-Backend/
├── docs/                    # Documentation and FunctionGemma docs
├── shared/
│   └── gemma_router.py      # Core FunctionGemma router
├── todo_app/
│   ├── main.py              # FastAPI application
│   ├── models.py            # SQLModel schemas
│   ├── tools.py             # Tool functions
│   └── static/index.html    # Web UI
├── pyproject.toml
└── README.md
```

## API Endpoints

### Todo App

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/chat` | POST | Process natural language input |
| `/api/tasks` | GET | List all tasks |
| `/api/health` | GET | Health check |

## Configuration

Environment variables (`.env`):

```
LM_STUDIO_URL=http://127.0.0.1:1234/v1
LM_STUDIO_MODEL=functiongemma-270m-it
```

## License

MIT

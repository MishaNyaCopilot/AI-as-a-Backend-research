"""
FastAPI application for the Todo List app.

This is the main entry point that:
- Sets up the database
- Registers tools with AI router (FunctionGemma or LiquidAI LFM)
- Exposes API endpoints for natural language interaction
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlmodel import Session, SQLModel, create_engine

# Add parent to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

# Import both routers
from shared.gemma_router import GemmaRouter
from shared.liquid_router import LiquidRouter

from .models import Task
from .tools import complete_task, create_task, delete_task, list_tasks, update_task

# Database setup
DATABASE_URL = "sqlite:///./todo_app/todo.db"
engine = create_engine(DATABASE_URL, echo=False)

# Initialize router based on ROUTER_TYPE env var
# Options: "gemma" (default), "liquid"
ROUTER_TYPE = os.getenv("ROUTER_TYPE", "gemma").lower()

if ROUTER_TYPE == "liquid":
    print("Using LiquidAI LFM router")
    router = LiquidRouter()
else:
    print("Using FunctionGemma router")
    router = GemmaRouter()


def get_session() -> Session:
    """Get a database session."""
    return Session(engine)


# Register tools with FunctionGemma
# We create wrapper functions that include the session


@router.register_tool(
    name="create_task",
    description=(
        "RECORD a new task, reminder, or item in the todo list. "
        "Use this ONLY when the user wants to add something new. "
        "Keywords: 'add', 'create', 'remind me', 'new task', 'need to', 'buy', 'get'. "
        "The 'title' should be the task itself. "
        "IMPORTANT: Do NOT set 'due_date' unless explicitly mentioned (e.g., 'today', 'tomorrow')."
    ),
)
def _create_task(title: str, due_date: str | None = None, priority: str = "normal") -> dict:
    with get_session() as session:
        return create_task(session, title, due_date, priority)


@router.register_tool(
    name="list_tasks",
    description=(
        "QUERY or SHOW existing tasks from the todo list. "
        "Use this for: 'list', 'show', 'what do I have', 'view', 'my tasks'. "
        "Filter by 'priority' (low, normal, high) or 'status' (pending, completed) if specified."
    ),
)
def _list_tasks(status: str | None = None, priority: str | None = None) -> dict:
    with get_session() as session:
        return list_tasks(session, status, priority)


@router.register_tool(
    name="complete_task",
    description=(
        "MARK an existing task as finished or done. "
        "Use this for: 'done', 'finished', 'completed', 'did', 'checked off'. "
        "Example: 'I finished the report', 'mark task 5 as done'."
    ),
)
def _complete_task(task_id: int | None = None, title: str | None = None) -> dict:
    with get_session() as session:
        return complete_task(session, task_id, title)


@router.register_tool(
    name="delete_task",
    description=(
        "REMOVE or DELETE tasks from the list permanently. "
        "Use this for: 'delete', 'remove', 'clear', 'erase'. "
        "Use 'status'='completed' to clear all finished tasks."
    ),
)
def _delete_task(
    task_id: int | None = None, title: str | None = None, status: str | None = None
) -> dict:
    with get_session() as session:
        return delete_task(session, task_id, title, status)


@router.register_tool(
    name="update_task",
    description="Update an existing task's title, priority, or due date.",
)
def _update_task(
    task_id: int,
    title: str | None = None,
    priority: str | None = None,
    due_date: str | None = None,
) -> dict:
    with get_session() as session:
        return update_task(session, task_id, title, priority, due_date)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - create tables on startup."""
    SQLModel.metadata.create_all(engine)
    yield


# Create FastAPI app
app = FastAPI(
    title="Todo List - AI Backend",
    description="Natural language todo list powered by FunctionGemma",
    version="0.1.0",
    lifespan=lifespan,
)

# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=static_path), name="static")


# Request/Response models
class NLRequest(BaseModel):
    """Natural language request from user."""

    message: str


class NLResponse(BaseModel):
    """Response from AI processing."""

    success: bool
    function_called: str | None = None
    arguments: dict | None = None
    result: dict | None = None
    error: str | None = None


def build_task_context() -> str:
    """
    Build context string with current tasks and date for the model.

    Returns:
        Formatted string with date and task list.
    """
    from datetime import datetime

    # Add current date
    now = datetime.now()
    lines = [
        f"Today's date: {now.strftime('%Y-%m-%d')} ({now.strftime('%A')})",
        f"Current time: {now.strftime('%H:%M')}",
        "",
    ]

    # Add task list
    with get_session() as session:
        tasks_data = list_tasks(session)
        tasks = tasks_data.get("tasks", [])

    if not tasks:
        lines.append("No tasks in the list.")
    else:
        lines.append("Current tasks:")
        for t in tasks:
            status_icon = "✓" if t["status"] == "completed" else "○"
            priority_str = f" [{t['priority']}]" if t["priority"] != "normal" else ""
            lines.append(f"- ID {t['id']}: {status_icon} {t['title']}{priority_str}")

    # Add instructions
    lines.append("")
    lines.append("Note: Only set due_date if user explicitly mentions a date like 'tomorrow' or 'next week'.")

    return "\n".join(lines)


# API Endpoints
@app.get("/")
async def root():
    """Serve the main UI."""
    index_path = static_path / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Todo List API - AI Backend", "docs": "/docs"}


@app.post("/api/chat", response_model=NLResponse)
async def chat(request: NLRequest):
    """
    Process natural language input and execute appropriate function.

    Args:
        request: Natural language message from user.

    Returns:
        Response with function call details and result.
    """
    try:
        # Build context with current tasks
        context = build_task_context()

        # Route the natural language to a function with context
        function_call = router.route(request.message, context=context)

        if function_call is None:
            # Generate helpful suggestions based on the input
            suggestions = generate_suggestions(request.message)
            return NLResponse(
                success=False,
                error=suggestions,
            )

        # Execute the function
        result = router.execute(function_call)

        return NLResponse(
            success=True,
            function_called=function_call.name,
            arguments=function_call.arguments,
            result=result,
        )

    except Exception as e:
        return NLResponse(
            success=False,
            error=f"An error occurred: {str(e)}",
        )


def generate_suggestions(user_input: str) -> str:
    """
    Generate helpful suggestions when routing fails.

    Args:
        user_input: The original user input that couldn't be routed.

    Returns:
        Helpful message with example commands.
    """
    user_lower = user_input.lower()

    # Check for keywords to provide targeted suggestions
    if any(word in user_lower for word in ["add", "create", "new", "remind", "buy", "get", "need"]):
        return (
            "I understood you want to add something, but couldn't parse it. "
            "Try: 'add buy milk' or 'add call mom'"
        )

    if any(word in user_lower for word in ["show", "list", "what", "tasks", "todo"]):
        return (
            "I understood you want to see tasks. "
            "Try: 'show my tasks' or 'list pending tasks'"
        )

    if any(word in user_lower for word in ["done", "finish", "complete", "did"]):
        return (
            "I understood you completed something. "
            "Try: 'complete task 1' or 'mark milk as done'"
        )

    if any(word in user_lower for word in ["delete", "remove", "clear"]):
        return (
            "I understood you want to delete something. "
            "Try: 'delete task 1' or 'clear completed tasks'"
        )

    # Generic fallback with examples
    return (
        "I couldn't understand that request. Here are some examples:\n"
        "• Add task: 'add buy milk' or 'remind me to call mom'\n"
        "• Show tasks: 'show my tasks' or 'list pending'\n"
        "• Complete: 'complete task 1' or 'I finished shopping'\n"
        "• Delete: 'delete task 2' or 'clear completed'"
    )


@app.get("/api/tasks")
async def get_tasks():
    """Get all tasks directly (bypass NL processing)."""
    with get_session() as session:
        return list_tasks(session)


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model": router.model, "tools": list(router.tools.keys())}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

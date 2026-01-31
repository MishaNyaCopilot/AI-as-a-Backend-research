"""
Database models for the Todo List app.
"""

from datetime import datetime

from sqlmodel import Field, SQLModel


class Task(SQLModel, table=True):
    """
    A task in the todo list.

    Attributes:
        id: Unique identifier for the task.
        title: The task title/description.
        status: Current status (pending, completed).
        priority: Task priority (low, normal, high).
        due_date: Optional due date for the task.
        created_at: When the task was created.
    """

    id: int | None = Field(default=None, primary_key=True)
    title: str = Field(index=True)
    status: str = Field(default="pending")  # pending, completed
    priority: str = Field(default="normal")  # low, normal, high
    due_date: datetime | None = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TaskCreate(SQLModel):
    """Schema for creating a new task."""

    title: str
    priority: str = "normal"
    due_date: datetime | None = None


class TaskUpdate(SQLModel):
    """Schema for updating a task."""

    title: str | None = None
    status: str | None = None
    priority: str | None = None
    due_date: datetime | None = None

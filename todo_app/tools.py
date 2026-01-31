"""
Tool functions for the Todo List app.

These functions are registered with FunctionGemma for natural language routing.
"""

from datetime import datetime, timedelta
from typing import Literal

from sqlmodel import Session, select

from .models import Task


def parse_date(date_str: str | None) -> datetime | None:
    """
    Parse natural language date strings.

    Args:
        date_str: Date string like 'today', 'tomorrow', or ISO format.

    Returns:
        Parsed datetime or None.
    """
    if not date_str:
        return None

    date_str = date_str.lower().strip()
    now = datetime.now()

    if date_str in ("today", "now"):
        return now.replace(hour=23, minute=59, second=59)
    elif date_str == "tomorrow":
        return (now + timedelta(days=1)).replace(hour=23, minute=59, second=59)
    elif date_str == "next week":
        return (now + timedelta(weeks=1)).replace(hour=23, minute=59, second=59)

    # Try ISO format
    try:
        return datetime.fromisoformat(date_str)
    except ValueError:
        return None


def create_task(
    session: Session,
    title: str,
    due_date: str | None = None,
    priority: str = "normal",
) -> dict:
    """
    Create a new task in the todo list.

    Args:
        session: Database session.
        title: The task title or description.
        due_date: Optional due date (today, tomorrow, or specific date).
        priority: Task priority (low, normal, high).

    Returns:
        Dictionary with created task details.
    """
    parsed_date = parse_date(due_date)
    
    task = Task(
        title=title,
        due_date=parsed_date,
        priority=priority if priority in ("low", "normal", "high") else "normal",
    )
    session.add(task)
    session.commit()
    session.refresh(task)

    return {
        "id": task.id,
        "title": task.title,
        "status": task.status,
        "priority": task.priority,
        "due_date": task.due_date.isoformat() if task.due_date else None,
        "message": f"Task '{task.title}' created successfully!",
    }


def list_tasks(
    session: Session,
    status: str | None = None,
    priority: str | None = None,
) -> dict:
    """
    List all tasks, optionally filtered by status or priority.

    Args:
        session: Database session.
        status: Filter by status (pending, completed).
        priority: Filter by priority (low, normal, high).

    Returns:
        Dictionary with list of tasks.
    """
    query = select(Task)

    if status:
        query = query.where(Task.status == status)
    if priority:
        query = query.where(Task.priority == priority)

    query = query.order_by(Task.created_at.desc())
    tasks = session.exec(query).all()

    return {
        "tasks": [
            {
                "id": t.id,
                "title": t.title,
                "status": t.status,
                "priority": t.priority,
                "due_date": t.due_date.isoformat() if t.due_date else None,
            }
            for t in tasks
        ],
        "count": len(tasks),
        "message": f"Found {len(tasks)} task(s).",
    }


def complete_task(
    session: Session,
    task_id: int | None = None,
    title: str | None = None,
) -> dict:
    """
    Mark a task as completed.

    Args:
        session: Database session.
        task_id: ID of the task to complete.
        title: Title of the task to complete (partial match).

    Returns:
        Dictionary with updated task details.
    """
    if task_id:
        task = session.get(Task, task_id)
    elif title:
        query = select(Task).where(Task.title.ilike(f"%{title}%"))
        task = session.exec(query).first()
    else:
        return {"error": "Please provide task_id or title."}

    if not task:
        return {"error": "Task not found."}

    task.status = "completed"
    session.add(task)
    session.commit()
    session.refresh(task)

    return {
        "id": task.id,
        "title": task.title,
        "status": task.status,
        "message": f"Task '{task.title}' marked as completed!",
    }


def delete_task(
    session: Session,
    task_id: int | None = None,
    title: str | None = None,
    status: str | None = None,
) -> dict:
    """
    Delete a task or tasks matching criteria.

    Args:
        session: Database session.
        task_id: ID of specific task to delete.
        title: Title to match for deletion (partial match).
        status: Delete all tasks with this status.

    Returns:
        Dictionary with deletion result.
    """
    deleted_count = 0

    if task_id:
        task = session.get(Task, task_id)
        if task:
            session.delete(task)
            deleted_count = 1
    elif title:
        query = select(Task).where(Task.title.ilike(f"%{title}%"))
        tasks = session.exec(query).all()
        for task in tasks:
            session.delete(task)
            deleted_count += 1
    elif status:
        query = select(Task).where(Task.status == status)
        tasks = session.exec(query).all()
        for task in tasks:
            session.delete(task)
            deleted_count += 1
    else:
        return {"error": "Please provide task_id, title, or status."}

    session.commit()

    return {
        "deleted_count": deleted_count,
        "message": f"Deleted {deleted_count} task(s).",
    }


def update_task(
    session: Session,
    task_id: int,
    title: str | None = None,
    priority: str | None = None,
    due_date: str | None = None,
) -> dict:
    """
    Update an existing task.

    Args:
        session: Database session.
        task_id: ID of the task to update.
        title: New title for the task.
        priority: New priority (low, normal, high).
        due_date: New due date.

    Returns:
        Dictionary with updated task details.
    """
    task = session.get(Task, task_id)
    if not task:
        return {"error": f"Task with id {task_id} not found."}

    if title:
        task.title = title
    if priority and priority in ("low", "normal", "high"):
        task.priority = priority
    if due_date:
        task.due_date = parse_date(due_date)

    session.add(task)
    session.commit()
    session.refresh(task)

    return {
        "id": task.id,
        "title": task.title,
        "status": task.status,
        "priority": task.priority,
        "due_date": task.due_date.isoformat() if task.due_date else None,
        "message": f"Task '{task.title}' updated successfully!",
    }

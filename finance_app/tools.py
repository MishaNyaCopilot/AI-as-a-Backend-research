"""
Tool functions for the Financial Tracker app.

These functions are registered with the AI router for natural language interaction.
"""

from datetime import datetime, timedelta
from typing import Any
from sqlmodel import Session, select, func

from .models import Transaction, RecurringTransaction

def parse_date(date_str: str | None) -> datetime:
    """
    Parse natural language date strings or return current time.
    """
    if not date_str:
        return datetime.now()

    date_str = date_str.lower().strip()
    now = datetime.now()

    if date_str in ("today", "now"):
        return now
    elif date_str == "tomorrow":
        return now + timedelta(days=1)
    elif date_str == "yesterday":
        return now - timedelta(days=1)
    elif date_str == "last week":
        return now - timedelta(weeks=1)
    
    try:
        return datetime.fromisoformat(date_str)
    except ValueError:
        return now

def add_expense(
    session: Session,
    amount: float,
    category: str,
    description: str | None = None,
    date: str | None = None,
) -> dict[str, Any]:
    """
    Record a new expense.

    Args:
        session: Database session.
        amount: Amount spent.
        category: Category of spending (e.g., food, rent, transport).
        description: Optional details about the expense.
        date: When the expense occurred.
    """
    transaction = Transaction(
        type="expense",
        amount=abs(amount),
        category=category.lower(),
        description=description,
        date=parse_date(date),
    )
    session.add(transaction)
    session.commit()
    session.refresh(transaction)

    return {
        "id": transaction.id,
        "type": "expense",
        "amount": transaction.amount,
        "category": transaction.category,
        "message": f"Recorded expense of {transaction.amount} in '{transaction.category}'",
    }

def add_income(
    session: Session,
    amount: float,
    source: str,
    description: str | None = None,
    date: str | None = None,
) -> dict[str, Any]:
    """
    Record new income.

    Args:
        session: Database session.
        amount: Amount earned.
        source: Where the money came from (e.g., salary, gift).
        description: Optional details.
        date: When the income was received.
    """
    transaction = Transaction(
        type="income",
        amount=abs(amount),
        category=source.lower(),
        description=description,
        date=parse_date(date),
    )
    session.add(transaction)
    session.commit()
    session.refresh(transaction)

    return {
        "id": transaction.id,
        "type": "income",
        "amount": transaction.amount,
        "source": transaction.category,
        "message": f"Recorded income of {transaction.amount} from '{transaction.category}'",
    }

def get_balance(session: Session) -> dict[str, Any]:
    """
    Calculate the current balance.
    """
    income = session.exec(select(func.sum(Transaction.amount)).where(Transaction.type == "income")).one() or 0.0
    expenses = session.exec(select(func.sum(Transaction.amount)).where(Transaction.type == "expense")).one() or 0.0
    
    balance = income - expenses
    
    return {
        "total_income": income,
        "total_expenses": expenses,
        "balance": balance,
        "message": f"Your current balance is {balance:.2f} (Income: {income:.2f}, Expenses: {expenses:.2f})",
    }

def get_spending_analysis(
    session: Session, 
    period: str | None = "this_month"
) -> dict[str, Any]:
    """
    Analyze spending by category for a given period.
    """
    now = datetime.now()
    if period == "this_month":
        start_date = now.replace(day=1, hour=0, minute=0, second=0)
    elif period == "today":
        start_date = now.replace(hour=0, minute=0, second=0)
    elif period == "this_week":
        start_date = now - timedelta(days=now.weekday())
    else:
        start_date = now - timedelta(days=30) # Default last 30 days

    query = select(Transaction.category, func.sum(Transaction.amount)).where(
        Transaction.type == "expense",
        Transaction.date >= start_date
    ).group_by(Transaction.category)
    
    results = session.exec(query).all()
    
    analysis = {category: amount for category, amount in results}
    total = sum(analysis.values())
    
    return {
        "period": period,
        "analysis": analysis,
        "total_spent": total,
        "message": f"Total spending for {period}: {total:.2f}. " + 
                   ", ".join([f"{c}: {a:.2f}" for c, a in analysis.items()])
    }

def get_income_analysis(
    session: Session, 
    period: str | None = "this_month"
) -> dict[str, Any]:
    """
    Analyze income by source for a given period.
    """
    now = datetime.now()
    if period == "this_month":
        start_date = now.replace(day=1, hour=0, minute=0, second=0)
    elif period == "today":
        start_date = now.replace(hour=0, minute=0, second=0)
    elif period == "this_week":
        start_date = now - timedelta(days=now.weekday())
    else:
        start_date = now - timedelta(days=30)

    query = select(Transaction.category, func.sum(Transaction.amount)).where(
        Transaction.type == "income",
        Transaction.date >= start_date
    ).group_by(Transaction.category)
    
    results = session.exec(query).all()
    
    analysis = {category: amount for category, amount in results}
    total = sum(analysis.values())
    
    return {
        "period": period,
        "analysis": analysis,
        "total_income": total,
        "message": f"Total income for {period}: {total:.2f}. " + 
                   ", ".join([f"{c}: {a:.2f}" for c, a in analysis.items()])
    }

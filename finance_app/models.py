from datetime import datetime
from sqlmodel import Field, SQLModel

class Transaction(SQLModel, table=True):
    """
    Model representing a financial transaction (income or expense).
    """
    id: int | None = Field(default=None, primary_key=True)
    type: str  # "expense" or "income"
    amount: float
    category: str
    description: str | None = None
    date: datetime = Field(default_factory=datetime.now)
    created_at: datetime = Field(default_factory=datetime.now)

class RecurringTransaction(SQLModel, table=True):
    """
    Model representing a recurring financial transaction.
    """
    id: int | None = Field(default=None, primary_key=True)
    type: str  # "expense" or "income"
    amount: float
    category: str
    frequency: str  # "daily", "weekly", "monthly", "yearly"
    next_date: datetime
    active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.now)

"""
FastAPI application for the Financial Tracker.
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlmodel import Session, SQLModel, create_engine

# Add parent to path for shared imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from shared.gemma_router import GemmaRouter
from shared.liquid_router import LiquidRouter

from .models import Transaction
from .tools import add_expense, add_income, get_balance, get_spending_analysis, get_income_analysis

# Database setup
DATABASE_URL = "sqlite:///./finance_app/finance.db"
engine = create_engine(DATABASE_URL, echo=False)

# Initialize router
ROUTER_TYPE = os.getenv("ROUTER_TYPE", "gemma").lower()
if ROUTER_TYPE == "liquid":
    print("Using LiquidAI LFM router for Finance App")
    router = LiquidRouter()
else:
    print("Using FunctionGemma router for Finance App")
    router = GemmaRouter()

def get_session():
    return Session(engine)

# Register tools
@router.register_tool(
    name="add_expense",
    description=(
        "Record a NEW expense entry when money is spent. "
        "Use this for: 'spent', 'bought', 'paid for', 'cost me', 'bill'. "
        "Requires 'amount' and 'category' (e.g. food, rent, entertainment)."
    ),
)
def _add_expense(amount: float, category: str, description: str | None = None, date: str | None = None):
    with get_session() as session:
        return add_expense(session, amount, category, description, date)

@router.register_tool(
    name="add_income",
    description=(
        "Record a NEW income entry when money is received. "
        "Use this for: 'earned', 'got paid', 'received', 'made money', 'salary', 'gift'. "
        "Requires 'amount' and 'source' (mapped to the 'source' parameter)."
    ),
)
def _add_income(amount: float, source: str, description: str | None = None, date: str | None = None):
    with get_session() as session:
        return add_income(session, amount, source, description, date)

@router.register_tool(
    name="get_balance",
    description=(
        "STATUS CHECK: Get the current total balance, income summary, and expense summary. "
        "Use this for: 'how much money do I have', 'what's my balance', 'show my totals'. "
        "Do NOT use this to record new money or analyze specific categories."
    ),
)
def _get_balance():
    with get_session() as session:
        return get_balance(session)

@router.register_tool(
    name="get_spending_analysis",
    description=(
        "Summarize and analyze spending patterns by category over time. "
        "Use ONLY when user asks for a 'spending breakdown', 'expense summary', 'monthly report', or 'where did I spend'. "
        "Do NOT use this to record new expenses."
    ),
)
def _get_spending_analysis(period: str | None = "this_month"):
    with get_session() as session:
        return get_spending_analysis(session, period)

@router.register_tool(
    name="get_income_analysis",
    description=(
        "Summarize and analyze income sources and trends over time. "
        "Use ONLY when user asks for an 'income breakdown', 'earning summary', 'total earned for X', or 'source report'. "
        "Do NOT use this to record new income entries."
    ),
)
def _get_income_analysis(period: str | None = "this_month"):
    with get_session() as session:
        return get_income_analysis(session, period)

@asynccontextmanager
async def lifespan(app: FastAPI):
    SQLModel.metadata.create_all(engine)
    yield

app = FastAPI(
    title="Financial Tracker - AI Backend",
    description="Track your finances using natural language",
    version="0.1.0",
    lifespan=lifespan,
)

static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=static_path), name="static")

class NLRequest(BaseModel):
    message: str

class NLResponse(BaseModel):
    success: bool
    function_called: str | None = None
    arguments: dict | None = None
    result: dict | None = None
    error: str | None = None

def build_finance_context() -> str:
    """Build context for the model with current balance and recent transactions."""
    from datetime import datetime
    now = datetime.now()
    
    with get_session() as session:
        balance_info = get_balance(session)
        # Get last 5 transactions
        from sqlmodel import select
        recent_txs = session.exec(select(Transaction).order_by(Transaction.date.desc()).limit(5)).all()
    
    lines = [
        f"Current date: {now.strftime('%Y-%m-%d')} ({now.strftime('%A')})",
        f"Balance: {balance_info['balance']:.2f}",
        "Recent Transactions:"
    ]
    
    if not recent_txs:
        lines.append("- No transactions yet.")
    else:
        for tx in recent_txs:
            lines.append(f"- {tx.date.strftime('%m-%d')} {tx.type.upper()}: {tx.amount:.2f} ({tx.category})")
            
    return "\n".join(lines)

@app.get("/")
async def root():
    index_path = static_path / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Financial Tracker API - AI Backend"}

@app.get("/api/transactions")
async def list_transactions():
    with get_session() as session:
        from sqlmodel import select
        txs = session.exec(select(Transaction).order_by(Transaction.date.desc())).all()
        return txs

@app.post("/api/chat", response_model=NLResponse)
async def chat(request: NLRequest):
    try:
        context = build_finance_context()
        function_call = router.route(request.message, context=context)
        
        if function_call is None:
            return NLResponse(success=False, error="I didn't understand that financial request. Try 'Spent 50 on food' or 'Check my balance'.")
            
        result = router.execute(function_call)
        return NLResponse(
            success=True,
            function_called=function_call.name,
            arguments=function_call.arguments,
            result=result
        )
    except Exception as e:
        return NLResponse(success=False, error=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)

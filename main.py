"""
RSI (Relative Strength Index) Calculation Microservice
======================================================
Extracted from: Dataset Saham.ipynb (IDX Stock EDA)

Notebook Logic (lines 2663-2674):
  df_clean['RSI'] = df_clean.groupby('Stock Code')['Close'].transform(
      lambda x: ta.momentum.RSIIndicator(x, window=14).rsi()
  )

  df_clean['RSI_Signal'] = df_clean['RSI'].apply(
      lambda x: 'Buy' if x < 30 else ('Sell' if x > 70 else 'Hold')
  )

Signal Rules:
  - Buy  : RSI < 30  (oversold — price may rebound)
  - Sell : RSI > 70  (overbought — price may correct)
  - Hold : 30 <= RSI <= 70  (normal range)

Endpoints:
  POST /api/calculate-rsi          — single latest RSI value
  POST /api/calculate-rsi-history  — full 90-day RSI history (batch)
"""

from typing import List, Optional

import pandas as pd
import ta.momentum
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Financial Indicator Microservice",
    description=(
        "Calculates the 14-day RSI and its trading signal (Buy / Sell / Hold) "
        "from a list of closing prices. Supports both single-value and "
        "batch-history modes."
    ),
    version="1.1.0",
)


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------
class RSIRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol, e.g. 'GOTO'")
    close_prices: List[float] = Field(
        ...,
        min_items=1,
        description=(
            "Ordered list of daily closing prices (oldest → newest). "
            "At least 15 data points are required to produce a non-null RSI."
        ),
    )


class RSIResponse(BaseModel):
    ticker: str
    rsi_14: Optional[float] = Field(
        None,
        description=(
            "14-day RSI value for the most recent closing price. "
            "null when there are fewer than 15 data points."
        ),
    )
    rsi_signal: Optional[str] = Field(
        None,
        description="Trading signal: 'Buy', 'Sell', or 'Hold'. null when rsi_14 is null.",
    )


# --- History models ---------------------------------------------------------

class PriceItem(BaseModel):
    """A single day's closing price with its date string."""
    date: str = Field(..., description="Trading date, e.g. '2023-10-31'")
    close: float = Field(..., description="Closing price for that day")


class RsiHistoryRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol, e.g. 'GOTO'")
    history_data: List[PriceItem] = Field(
        ...,
        description=(
            "Ordered list of {date, close} objects (any order — will be sorted "
            "oldest → newest internally). Send at least 105 days for a clean "
            "90-day output window after the 14-row RSI warm-up."
        ),
    )


class RsiHistoryItem(BaseModel):
    record_date: str = Field(..., description="Trading date, e.g. '2023-10-31'")
    rsi_14: float = Field(..., description="14-day RSI value rounded to 2 dp")
    rsi_signal: str = Field(..., description="'Buy', 'Sell', or 'Hold'")


class RsiHistoryResponse(BaseModel):
    ticker: str
    rsi_history: List[RsiHistoryItem] = Field(
        ...,
        description="Up to the last 90 trading days with valid (non-NaN) RSI values.",
    )


# ---------------------------------------------------------------------------
# RSI helpers  (exact notebook logic)
# ---------------------------------------------------------------------------
def _compute_rsi(close_prices: List[float]) -> Optional[float]:
    """
    Mirrors the notebook cell:
        ta.momentum.RSIIndicator(x, window=14).rsi()

    Returns the RSI for the *last* price in the series, or None if the
    series is too short (NaN from the ta library).
    """
    series = pd.Series(close_prices)
    rsi_series = ta.momentum.RSIIndicator(series, window=14).rsi()
    last_rsi = rsi_series.iloc[-1]

    # NaN → return None so JSON serialises as `null`
    if pd.isna(last_rsi):
        return None
    return round(float(last_rsi), 2)


def _classify_signal(rsi: Optional[float]) -> Optional[str]:
    """
    Mirrors the notebook cell:
        lambda x: 'Buy' if x < 30 else ('Sell' if x > 70 else 'Hold')
    """
    if rsi is None:
        return None
    if rsi < 30:
        return "Buy"
    if rsi > 70:
        return "Sell"
    return "Hold"


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@app.post(
    "/api/calculate-rsi",
    response_model=RSIResponse,
    summary="Calculate 14-day RSI",
    description=(
        "Accepts a ticker symbol and a list of closing prices, "
        "then returns the 14-day RSI value and its trading signal."
    ),
)
def calculate_rsi(request: RSIRequest) -> RSIResponse:
    try:
        rsi_value = _compute_rsi(request.close_prices)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"RSI calculation failed: {exc}",
        )

    return RSIResponse(
        ticker=request.ticker,
        rsi_14=rsi_value,
        rsi_signal=_classify_signal(rsi_value),
    )


# ---------------------------------------------------------------------------
# History helper
# ---------------------------------------------------------------------------
_MAX_HISTORY_ROWS = 90  # rows returned to Node.js / stored in DB


def _compute_rsi_history(history_data: List[PriceItem]) -> List[RsiHistoryItem]:
    """
    1. Build a DataFrame from the incoming PriceItem list.
    2. Sort oldest → newest by date (defensive — caller may send any order).
    3. Compute 14-day RSI across the full timeline.
    4. Drop the first 14 NaN rows produced by the RSI warm-up window.
    5. Return only the last 90 rows so the DB stays lean.
    """
    # --- build & sort -------------------------------------------------------
    df = pd.DataFrame([{"date": p.date, "close": p.close} for p in history_data])
    df.sort_values("date", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- RSI (exact notebook logic) -----------------------------------------
    df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    # --- signal mapping (exact notebook logic) ------------------------------
    df["rsi_signal"] = df["rsi_14"].apply(
        lambda x: "Buy" if x < 30 else ("Sell" if x > 70 else "Hold")
    )

    # --- drop NaN warm-up rows then keep last 90 ----------------------------
    df.dropna(subset=["rsi_14"], inplace=True)
    df = df.tail(_MAX_HISTORY_ROWS)

    # --- serialise ----------------------------------------------------------
    result: List[RsiHistoryItem] = []
    for _, row in df.iterrows():
        result.append(
            RsiHistoryItem(
                record_date=str(row["date"]),
                rsi_14=round(float(row["rsi_14"]), 2),
                rsi_signal=str(row["rsi_signal"]),
            )
        )
    return result


# ---------------------------------------------------------------------------
# History endpoint
# ---------------------------------------------------------------------------
@app.post(
    "/api/calculate-rsi-history",
    response_model=RsiHistoryResponse,
    summary="Calculate 14-day RSI history (last 90 trading days)",
    description=(
        "Accepts a ticker and up to 105 days of {date, close} pairs. "
        "Returns the last 90 trading days that have a valid RSI value, "
        "each annotated with its Buy / Sell / Hold signal."
    ),
)
def calculate_rsi_history(request: RsiHistoryRequest) -> RsiHistoryResponse:
    if len(request.history_data) < 15:
        raise HTTPException(
            status_code=422,
            detail=(
                f"history_data must contain at least 15 items to compute a "
                f"valid RSI (received {len(request.history_data)})."
            ),
        )
    try:
        rsi_history = _compute_rsi_history(request.history_data)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"RSI history calculation failed: {exc}",
        )

    return RsiHistoryResponse(
        ticker=request.ticker,
        rsi_history=rsi_history,
    )

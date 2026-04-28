from sqlalchemy import (
    Column, String, Float, Date,
    Integer, BigInteger, DateTime, Boolean,
    UniqueConstraint, Index
)
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class TickerMetadata(Base):
    __tablename__ = "ticker_metadata"

    ticker       = Column(String(20),  primary_key=True)   # e.g. "RELIANCE"
    company_name = Column(String(200), nullable=False)      # e.g. "Reliance Industries Ltd"
    sector       = Column(String(100))                      # e.g. "Energy"
    industry     = Column(String(100))                      # e.g. "Oil & Gas Refining"
    market_cap   = Column(BigInteger)                       # in INR
    in_nifty200  = Column(Boolean, default=True)
    last_updated = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<TickerMetadata {self.ticker} | {self.sector}>"


class PriceData(Base):
    __tablename__ = "price_data"

    id        = Column(Integer, primary_key=True, autoincrement=True)
    ticker    = Column(String(20), nullable=False)
    date      = Column(Date,       nullable=False)
    open      = Column(Float)
    high      = Column(Float)
    low       = Column(Float)
    close     = Column(Float,  nullable=False)
    adj_close = Column(Float,  nullable=False)
    volume    = Column(BigInteger)
    log_return = Column(Float)          # ln(close_t / close_t-1)
    rolling_vol_30 = Column(Float)      # 30-day annualised volatility

    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_ticker_date"),
        Index("ix_ticker",      "ticker"),
        Index("ix_date",        "date"),
        Index("ix_ticker_date", "ticker", "date"),
    )

    def __repr__(self):
        return f"<PriceData {self.ticker} | {self.date} | close={self.close}>"
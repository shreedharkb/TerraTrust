import os
import sys
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *

# Fallback to SQLite if Postgres is not available
DB_PATH = os.path.join(DATA_DIR, "audit_logs.db")
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DB_PATH}")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    taluk = Column(String, index=True)
    district = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
    requested_loan_amount = Column(Float)
    
    # ML Raw Scores
    pred_crop_health = Column(Float)
    pred_soil_q = Column(Float)
    pred_water_depth = Column(Float)
    raw_repayment_prob = Column(Float)
    
    # Final adjusted scores
    final_credit_score = Column(Float)
    risk_category = Column(String)
    recommendation = Column(String)
    
    # LLM Summary
    llm_summary = Column(String, nullable=True)
    
    execution_time_ms = Column(Float)

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

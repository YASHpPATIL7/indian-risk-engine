import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import logging
logger = logging.getLogger(__name__)

load_dotenv()

def get_engine(echo: bool = False):
    return create_engine(
        os.getenv("DATABASE_URL"),
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        echo=echo
    )

if __name__ == "__main__":
    engine = get_engine(echo=True)
    with engine.connect() as conn:
        row = conn.execute(text("SELECT version()")).fetchone()
        logger.info(f"\n✅ Connected to: {row[0]}\n")
        logger.info(f"Pool size     : {engine.pool.size()}")
        logger.info(f"Checked out   : {engine.pool.checkedout()}")
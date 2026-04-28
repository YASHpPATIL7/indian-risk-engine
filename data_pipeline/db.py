import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

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
        print(f"\n✅ Connected to: {row[0]}\n")
        print(f"Pool size     : {engine.pool.size()}")
        print(f"Checked out   : {engine.pool.checkedout()}")
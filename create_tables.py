# create_tables.py
from data_pipeline.db import get_engine
from data_pipeline.models import Base, TickerMetadata, PriceData

engine = get_engine(echo=True)   # echo=True so you SEE the SQL being run

print("\n⏳ Creating tables...\n")
Base.metadata.create_all(engine)
print("\n✅ Tables created successfully.\n")
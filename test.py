import pandas as pd
import numpy as np

# Test if libraries work
data = {'stock': ['RELIANCE', 'TCS', 'INFY'], 'price': [2500, 3800, 1450]}
df = pd.DataFrame(data)
print(df)
print(f"\nNumpy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")














import pandas as pd
from src.white_shorts.cli.train_compare_season import prepare_data

df = pd.DataFrame({"home_or_away": ["HOME", "AWAY", 1, 0, "true", "False", "H", "A", "  home  "]})
df2, feats = prepare_data(df.assign(points=1, name='X', team='T', opponent='O', date='2025-01-01'), lag_k=1)
print(df2["home_or_away"].tolist())  # expect [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

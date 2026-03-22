"""Merge credits and titles CSV files into data/combined_data.csv."""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

credits_path = DATA_DIR / "credits.csv"
titles_path = DATA_DIR / "titles.csv"
output_path = DATA_DIR / "combined_data.csv"

if not credits_path.exists() or not titles_path.exists():
	missing = [str(p.name) for p in [credits_path, titles_path] if not p.exists()]
	raise FileNotFoundError(
		f"Missing required input file(s) in {DATA_DIR}: {', '.join(missing)}"
	)

credits = pd.read_csv(credits_path)
titles = pd.read_csv(titles_path)

combined = credits.merge(titles, on="id", how="left")
combined.to_csv(output_path, index=False)

print(f"Saved merged dataset to: {output_path}")
print(combined.shape)
print(combined.head())

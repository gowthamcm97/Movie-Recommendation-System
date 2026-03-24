
# CineMatch

Find your next movie or show in seconds.

CineMatch is a lightweight content-based recommender that uses title, genre, cast, director, and description signals to suggest similar content. It includes fuzzy title matching, so even typo searches can still return useful recommendations.

## Why It Is Fun

- Smart recommendations powered by TF-IDF + cosine similarity
- Flexible search with exact, partial, and fuzzy matching
- Simple Gradio interface for quick exploration
- Ready-to-use notebook for analysis and experimentation

## Project Files

- `app.py` - Main Gradio recommender app
- `combine_csv.py` - Utility script to merge credits and titles data
- `title_df.csv` - Main cleaned dataset used by the app
- `combined_data.csv` - Output from CSV merge step
- `Sample_ML_Submission_Template.ipynb` - Notebook workflow and ML experiments
- `requirements.txt` - Python dependencies

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the app:

```bash
python app.py
```

3. Open the local Gradio link shown in your terminal.

## Publish on Hugging Face Spaces

1. Create a new Space on Hugging Face:
	- Space SDK: Gradio
	- Visibility: Public or Private

2. Push this project to your Space repository:

```bash
git init
git add .
git commit -m "Initial CineMatch app"
git remote add origin https://huggingface.co/spaces/<your-username>/<your-space-name>
git push -u origin main
```

3. Hugging Face will auto-build and host the app.

Notes:
- The app is already configured to run correctly inside Spaces.
- If Gradio share links fail locally, the app falls back to local mode automatically.

## Data Prep (Optional)

If you need to rebuild merged data:

```bash
python combine_csv.py
```

Make sure input files expected by the script are present before running.

## Tech Stack

Python, pandas, NumPy, scikit-learn, Gradio, Matplotlib, Seaborn, SciPy.

---

Built for quick discovery, simple experimentation, and fun recommendations.

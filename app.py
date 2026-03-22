"""
CineMatch — AI Content-Based Movie & TV Recommender
Hugging Face Spaces | Gradio App
Fixed: 4-layer fuzzy search + browse tab + helpful not-found page
"""

import gradio as gr
import pandas as pd
import numpy as np
import ast
import os
import difflib
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# ─────────────────────────────────────────────
#  1. DATA LOADING
# ─────────────────────────────────────────────

def safe_parse(val):
    try:
        parsed = ast.literal_eval(val) if isinstance(val, str) else []
        return " ".join(parsed) if isinstance(parsed, list) else str(val)
    except:
        return str(val) if val else ""


def load_data():
    title_path = DATA_DIR / "title_df.csv"
    if title_path.exists():
        df = pd.read_csv(title_path)
        print(f"Loaded {title_path.name} — {len(df)} titles")
    else:
        df = build_demo_data()
        print(f"Using built-in demo — {len(df)} titles")

    for col in ["title", "type", "description", "genres_text",
                "cast", "directors", "imdb_score", "tmdb_popularity",
                "release_year", "age_certification"]:
        if col not in df.columns:
            df[col] = "" if col in ["type","description","genres_text",
                                     "cast","directors","age_certification"] else 0

    if df["genres_text"].isna().all() and "genres" in df.columns:
        df["genres_text"] = df["genres"].apply(safe_parse)

    df["imdb_score"]      = pd.to_numeric(df["imdb_score"],      errors="coerce").fillna(0)
    df["tmdb_popularity"] = pd.to_numeric(df["tmdb_popularity"], errors="coerce").fillna(0)
    df["release_year"]    = pd.to_numeric(df["release_year"],    errors="coerce").fillna(0).astype(int)
    df["title"]           = df["title"].fillna("").astype(str)
    df = df[df["title"].str.strip().str.len() > 0].reset_index(drop=True)

    for col in ["description","genres_text","cast","directors","type","age_certification"]:
        df[col] = df[col].fillna("").astype(str)

    def norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    df["imdb_norm"] = norm(df["imdb_score"])
    df["pop_norm"]  = norm(df["tmdb_popularity"])

    def build_features(row):
        desc      = str(row["description"]).lower().strip()
        genres    = str(row["genres_text"]).lower().strip()
        cast      = str(row["cast"]).lower().strip()
        directors = str(row["directors"]).lower().strip()
        ctype     = str(row["type"]).lower().strip()
        return f"{desc} {desc} {desc} {genres} {genres} {genres} {genres} {cast} {directors} {ctype}"

    df["combined_features"] = df.apply(build_features, axis=1)
    df = df.reset_index(drop=True)

    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    sim_path = MODELS_DIR / "cosine_sim_matrix.pkl"

    sim = None
    if sim_path.exists():
        try:
            sim = joblib.load(sim_path)
            if getattr(sim, "shape", None) != (len(df), len(df)):
                print(
                    "Pretrained cosine matrix shape mismatch "
                    f"{getattr(sim, 'shape', None)} for {len(df)} rows. Recomputing."
                )
                sim = None
            else:
                print(f"Loaded pretrained cosine similarity from {sim_path.name}")
        except Exception as exc:
            print(f"Could not load {sim_path.name}: {exc}. Recomputing.")
            sim = None

    if sim is None:
        if vectorizer_path.exists():
            try:
                _ = joblib.load(vectorizer_path)
                print(f"Loaded pretrained vectorizer from {vectorizer_path.name}")
            except Exception as exc:
                print(f"Could not load {vectorizer_path.name}: {exc}. Using fresh vectorizer.")

        tfidf = TfidfVectorizer(stop_words="english", max_features=10000)
        matrix = tfidf.fit_transform(df["combined_features"])
        sim = cosine_similarity(matrix, matrix)
        print("Built cosine similarity matrix from current dataset")

    title_index = pd.Series(df.index, index=df["title"].str.lower().str.strip()).drop_duplicates()
    all_titles  = df["title"].tolist()

    return df, sim, title_index, all_titles


def build_demo_data():
    data = [
        {"title":"The Dark Knight","type":"MOVIE","description":"Batman battles the Joker a criminal mastermind who plunges Gotham into anarchy","genres_text":"action crime drama thriller","cast":"christian bale heath ledger aaron eckhart","directors":"christopher nolan","imdb_score":9.0,"tmdb_popularity":100.0,"release_year":2008,"age_certification":"PG-13"},
        {"title":"Inception","type":"MOVIE","description":"A thief who enters dreams to steal corporate secrets is given the task of planting an idea","genres_text":"action sci-fi thriller mystery","cast":"leonardo dicaprio joseph gordon-levitt elliot page","directors":"christopher nolan","imdb_score":8.8,"tmdb_popularity":90.0,"release_year":2010,"age_certification":"PG-13"},
        {"title":"Interstellar","type":"MOVIE","description":"A team of explorers travel through a wormhole in space to find a new home for humanity","genres_text":"adventure drama sci-fi","cast":"matthew mcconaughey anne hathaway jessica chastain","directors":"christopher nolan","imdb_score":8.6,"tmdb_popularity":85.0,"release_year":2014,"age_certification":"PG-13"},
        {"title":"Breaking Bad","type":"SHOW","description":"A chemistry teacher diagnosed with cancer turns to manufacturing drugs with a former student","genres_text":"crime drama thriller","cast":"bryan cranston aaron paul anna gunn","directors":"vince gilligan","imdb_score":9.5,"tmdb_popularity":95.0,"release_year":2008,"age_certification":"TV-MA"},
        {"title":"Stranger Things","type":"SHOW","description":"A group of kids encounter supernatural forces and secret government experiments in their small town","genres_text":"drama fantasy horror mystery sci-fi","cast":"millie bobby brown finn wolfhard winona ryder","directors":"the duffer brothers","imdb_score":8.7,"tmdb_popularity":88.0,"release_year":2016,"age_certification":"TV-14"},
        {"title":"Avengers Endgame","type":"MOVIE","description":"The Avengers assemble for a final battle against Thanos to restore the universe","genres_text":"action adventure sci-fi drama","cast":"robert downey jr chris evans scarlett johansson","directors":"anthony russo joe russo","imdb_score":8.4,"tmdb_popularity":99.0,"release_year":2019,"age_certification":"PG-13"},
        {"title":"The Crown","type":"SHOW","description":"The story of Queen Elizabeth IIs reign and the political and personal events that shaped her life","genres_text":"drama history biography","cast":"claire foy olivia colman imelda staunton","directors":"peter morgan","imdb_score":8.6,"tmdb_popularity":70.0,"release_year":2016,"age_certification":"TV-MA"},
        {"title":"Parasite","type":"MOVIE","description":"A poor family schemes to become employed by a wealthy family and infiltrate their household","genres_text":"comedy drama thriller","cast":"song kang-ho lee sun-kyun cho yeo-jeong","directors":"bong joon-ho","imdb_score":8.5,"tmdb_popularity":78.0,"release_year":2019,"age_certification":"R"},
        {"title":"The Witcher","type":"SHOW","description":"Geralt of Rivia a mutated monster hunter struggles to find his place in a world of magic","genres_text":"action adventure drama fantasy","cast":"henry cavill freya allan anya chalotra","directors":"alik sakharov","imdb_score":8.2,"tmdb_popularity":82.0,"release_year":2019,"age_certification":"TV-MA"},
        {"title":"Pulp Fiction","type":"MOVIE","description":"The lives of two mob hitmen a boxer and others intertwine in four tales of crime","genres_text":"crime drama thriller","cast":"john travolta samuel l jackson uma thurman","directors":"quentin tarantino","imdb_score":8.9,"tmdb_popularity":80.0,"release_year":1994,"age_certification":"R"},
        {"title":"Money Heist","type":"SHOW","description":"A criminal mastermind orchestrates the biggest heist in history at the Royal Mint of Spain","genres_text":"action crime drama thriller mystery","cast":"alvaro morte ursula corbero itziar ituno","directors":"alex pina","imdb_score":8.3,"tmdb_popularity":88.0,"release_year":2017,"age_certification":"TV-MA"},
        {"title":"The Godfather","type":"MOVIE","description":"The aging patriarch of an organized crime dynasty transfers control of his empire to his reluctant son","genres_text":"crime drama","cast":"marlon brando al pacino james caan","directors":"francis ford coppola","imdb_score":9.2,"tmdb_popularity":75.0,"release_year":1972,"age_certification":"R"},
        {"title":"Squid Game","type":"SHOW","description":"Hundreds of cash-strapped players accept an invitation to compete in childrens games for a massive prize","genres_text":"action drama mystery thriller","cast":"lee jung-jae park hae-soo wi ha-jun","directors":"hwang dong-hyuk","imdb_score":8.0,"tmdb_popularity":92.0,"release_year":2021,"age_certification":"TV-MA"},
        {"title":"The Matrix","type":"MOVIE","description":"A hacker discovers reality is a simulation and joins a rebellion against machine controllers","genres_text":"action sci-fi thriller","cast":"keanu reeves laurence fishburne carrie-anne moss","directors":"lana wachowski lilly wachowski","imdb_score":8.7,"tmdb_popularity":82.0,"release_year":1999,"age_certification":"R"},
        {"title":"Sherlock","type":"SHOW","description":"A modern update of Sherlock Holmes detective stories set in contemporary London","genres_text":"crime drama mystery thriller","cast":"benedict cumberbatch martin freeman mark gatiss","directors":"paul mcguigan","imdb_score":9.1,"tmdb_popularity":76.0,"release_year":2010,"age_certification":"TV-14"},
        {"title":"Game of Thrones","type":"SHOW","description":"Nine noble families fight for control over the mythical lands of Westeros while an ancient enemy returns","genres_text":"action adventure drama fantasy","cast":"emilia clarke kit harington peter dinklage","directors":"david benioff","imdb_score":9.2,"tmdb_popularity":97.0,"release_year":2011,"age_certification":"TV-MA"},
        {"title":"The Shawshank Redemption","type":"MOVIE","description":"Two imprisoned men bond over years finding solace and eventual redemption through acts of common decency","genres_text":"drama","cast":"tim robbins morgan freeman bob gunton","directors":"frank darabont","imdb_score":9.3,"tmdb_popularity":72.0,"release_year":1994,"age_certification":"R"},
        {"title":"Forrest Gump","type":"MOVIE","description":"The presidencies of Kennedy and Johnson through the eyes of an Alabama man with low IQ","genres_text":"drama romance","cast":"tom hanks robin wright gary sinise","directors":"robert zemeckis","imdb_score":8.8,"tmdb_popularity":74.0,"release_year":1994,"age_certification":"PG-13"},
        {"title":"The Office","type":"SHOW","description":"A mockumentary on a group of typical office workers where the workday consists of ego clashes","genres_text":"comedy drama","cast":"steve carell john krasinski jenna fischer","directors":"greg daniels","imdb_score":9.0,"tmdb_popularity":80.0,"release_year":2005,"age_certification":"TV-14"},
        {"title":"Friends","type":"SHOW","description":"Follows the personal and professional lives of six twenty to thirty-something-year-old friends living in New York","genres_text":"comedy romance drama","cast":"jennifer aniston courteney cox lisa kudrow","directors":"david crane marta kauffman","imdb_score":8.9,"tmdb_popularity":85.0,"release_year":1994,"age_certification":"TV-PG"},
        {"title":"The Three Stooges","type":"SHOW","description":"American vaudeville and comedy team known for their physical farce and slapstick short films","genres_text":"comedy family animation action","cast":"joe besser moe howard larry fine","directors":"jules white","imdb_score":8.6,"tmdb_popularity":15.0,"release_year":1934,"age_certification":"TV-PG"},
        {"title":"The General","type":"MOVIE","description":"During the Civil War union spies steal an engineer's locomotive and he must get it back","genres_text":"action drama war western comedy","cast":"buster keaton marion mack","directors":"buster keaton clyde bruckman","imdb_score":8.2,"tmdb_popularity":8.0,"release_year":1926,"age_certification":"NR"},
    ]
    return pd.DataFrame(data)


# ─────────────────────────────────────────────
#  2. FUZZY SEARCH — 4-LAYER MATCHING
# ─────────────────────────────────────────────

def find_best_match(query, title_index, all_titles):
    """
    4-layer search strategy so users always get a useful result:
      Layer 1 — Exact lowercase match
      Layer 2 — Query is a substring of a title  (e.g. "dark knight" → "The Dark Knight")
      Layer 3 — Title words overlap with query words
      Layer 4 — difflib fuzzy match for typos    (e.g. "incepsion" → "inception")
    """
    q = query.strip().lower()

    # Layer 1: Exact
    if q in title_index:
        return q, "exact", []

    # Layer 2: Query contained inside a title
    substr_hits = [t for t in title_index.index if q in t]
    if substr_hits:
        best = min(substr_hits, key=len)
        return best, "substring", substr_hits[:6]

    # Layer 3: Word overlap — any significant word of the query appears in a title
    q_words = [w for w in q.split() if len(w) > 3]
    if q_words:
        word_hits = [t for t in title_index.index
                     if any(w in t for w in q_words)]
        if word_hits:
            word_hits_sorted = sorted(
                word_hits,
                key=lambda t: sum(1 for w in q_words if w in t),
                reverse=True,
            )
            return word_hits_sorted[0], "word_overlap", word_hits_sorted[:6]

    # Layer 4: difflib fuzzy (handles typos)
    all_keys      = list(title_index.index)
    close_matches = difflib.get_close_matches(q, all_keys, n=6, cutoff=0.5)
    if close_matches:
        return close_matches[0], "fuzzy", close_matches

    return None, "not_found", []


# ─────────────────────────────────────────────
#  3. RECOMMENDATION ENGINE
# ─────────────────────────────────────────────

def predict_similar_indices(idx, top_n, sim_matrix, total_items):
    """Return candidate indices and similarity scores from the similarity model."""
    pool = min(top_n * 5, total_items - 1)
    sim_scores = sorted(enumerate(sim_matrix[idx]), key=lambda x: x[1], reverse=True)[1:pool+1]
    result_idx = [i[0] for i in sim_scores]
    sim_vals = [i[1] for i in sim_scores]
    return result_idx, sim_vals


def rank_and_format_recommendations(df, result_idx, sim_vals, top_n):
    """Apply business ranking and return UI-ready recommendation payload."""
    result = df[["title", "type", "imdb_score", "genres_text",
                 "release_year", "age_certification",
                 "imdb_norm", "pop_norm"]].iloc[result_idx].copy()
    result["sim_score"] = sim_vals
    result["final_score"] = (0.5 * result["sim_score"] +
                             0.3 * result["imdb_norm"] +
                             0.2 * result["pop_norm"])
    result = result.sort_values("final_score", ascending=False).head(top_n)

    recs = []
    for _, row in result.iterrows():
        recs.append({
            "title": row["title"],
            "type": row["type"],
            "imdb_score": round(float(row["imdb_score"]), 1),
            "genres": row["genres_text"],
            "year": int(row["release_year"]) if int(row["release_year"]) > 0 else "N/A",
            "cert": row["age_certification"] if str(row["age_certification"]) not in ["nan", "", "Unknown", "0"] else "NR",
            "similarity": round(float(row["sim_score"]) * 100, 1),
        })

    return recs

def recommend(title, top_n, df, sim_matrix, title_index, all_titles):
    if not title.strip():
        return None, "empty", []

    matched_key, match_type, suggestions = find_best_match(title, title_index, all_titles)

    if matched_key is None:
        return None, "not_found", suggestions

    idx    = title_index[matched_key]
    source = df.iloc[idx]
    result_idx, sim_vals = predict_similar_indices(idx, top_n, sim_matrix, len(df))
    recs = rank_and_format_recommendations(df, result_idx, sim_vals, top_n)

    source_info = {
        "title":          source["title"],
        "type":           source["type"],
        "imdb_score":     round(float(source["imdb_score"]), 1),
        "genres":         source["genres_text"],
        "year":           int(source["release_year"]) if int(source["release_year"]) > 0 else "N/A",
        "match_type":     match_type,
        "original_query": title.strip(),
    }
    return recs, source_info, suggestions


# ─────────────────────────────────────────────
#  4. HTML BUILDERS
# ─────────────────────────────────────────────

GENRE_COLORS = {
    "action":    ("hsl(10,80%,30%)",  "hsl(30,85%,42%)"),
    "drama":     ("hsl(220,55%,24%)", "hsl(230,65%,38%)"),
    "comedy":    ("hsl(45,75%,26%)",  "hsl(50,85%,40%)"),
    "thriller":  ("hsl(0,0%,10%)",    "hsl(0,0%,22%)"),
    "sci-fi":    ("hsl(190,75%,20%)", "hsl(200,80%,32%)"),
    "horror":    ("hsl(280,55%,16%)", "hsl(300,50%,28%)"),
    "romance":   ("hsl(340,60%,28%)", "hsl(350,70%,42%)"),
    "crime":     ("hsl(25,55%,18%)",  "hsl(30,60%,28%)"),
    "fantasy":   ("hsl(260,50%,22%)", "hsl(270,60%,36%)"),
    "animation": ("hsl(160,55%,20%)", "hsl(170,65%,32%)"),
    "history":   ("hsl(35,50%,22%)",  "hsl(40,55%,34%)"),
    "mystery":   ("hsl(210,45%,18%)", "hsl(215,50%,28%)"),
    "adventure": ("hsl(150,50%,20%)", "hsl(155,60%,30%)"),
    "biography": ("hsl(30,42%,22%)",  "hsl(35,48%,32%)"),
}
DEFAULT_COLORS = ("hsl(240,22%,16%)", "hsl(240,26%,24%)")

GENRE_ICONS = {
    "action":"⚡","drama":"🎭","comedy":"😄","thriller":"🔪",
    "sci-fi":"🚀","horror":"👻","romance":"💖","crime":"🕵️",
    "fantasy":"✨","animation":"🎨","history":"📜","mystery":"🔍",
    "adventure":"🌍","biography":"📖",
}

def pick_gradient(genres_str):
    g = str(genres_str).lower()
    for key, colors in GENRE_COLORS.items():
        if key in g:
            return colors
    return DEFAULT_COLORS

def genre_icon(genres_str):
    g = str(genres_str).lower()
    for key, icon in GENRE_ICONS.items():
        if key in g:
            return icon
    return "🎬"

def stars(score):
    filled = max(0, min(5, int(round(score / 2))))
    return "★" * filled + "☆" * (5 - filled)

def type_badge(ctype):
    if str(ctype).upper() == "SHOW":
        return '<span class="badge badge-show">📺 SHOW</span>'
    return '<span class="badge badge-movie">🎬 MOVIE</span>'

def build_match_notice(source_info):
    mt = source_info.get("match_type", "exact")
    oq = source_info.get("original_query", "")
    mk = source_info.get("title", "")
    if mt == "exact":
        return ""
    if mt in ("substring", "word_overlap"):
        return f'<div class="match-notice">🔍 Showing results for <strong>"{mk}"</strong> — closest match to "{oq}"</div>'
    if mt == "fuzzy":
        return f'<div class="match-notice">✏️ Auto-corrected <strong>"{oq}"</strong> → <strong>"{mk}"</strong></div>'
    return ""

def build_source_card(info):
    c1, c2 = pick_gradient(info["genres"])
    icon   = genre_icon(info["genres"])
    genre_tags = " ".join(
        f'<span class="gtag">{g.strip()}</span>'
        for g in str(info["genres"]).split()[:4] if g.strip()
    )
    notice = build_match_notice(info)
    return f"""
    {notice}
    <div class="source-card" style="background:linear-gradient(135deg,{c1},{c2})">
      <div class="source-poster"><span style="font-size:40px">{icon}</span></div>
      <div class="source-meta">
        <p class="source-label">FINDING SIMILAR TO</p>
        <h2 class="source-title">{info['title']}</h2>
        <div class="source-row">
          {type_badge(info['type'])}
          <span class="source-year">{info['year']}</span>
          <span class="imdb-pill">⭐ {info['imdb_score']}</span>
        </div>
        <div class="genre-row">{genre_tags}</div>
      </div>
    </div>
    """

def build_rec_card(rec, rank):
    c1, c2 = pick_gradient(rec["genres"])
    icon   = genre_icon(rec["genres"])
    genre_tags = " ".join(
        f'<span class="gtag">{g.strip()}</span>'
        for g in str(rec["genres"]).split()[:3] if g.strip()
    )
    sim_w = int(rec["similarity"])
    return f"""
    <div class="rec-card">
      <div class="poster" style="background:linear-gradient(160deg,{c1},{c2})">
        <span class="poster-rank">#{rank}</span>
        <span class="poster-icon">{icon}</span>
        <div class="poster-cert">{rec['cert']}</div>
      </div>
      <div class="rec-body">
        <div class="rec-type-row">{type_badge(rec['type'])}<span class="rec-year">{rec['year']}</span></div>
        <h3 class="rec-title">{rec['title']}</h3>
        <div class="genre-row">{genre_tags}</div>
        <div class="score-row">
          <span class="stars">{stars(rec['imdb_score'])}</span>
          <span class="imdb-num">{rec['imdb_score']}/10</span>
        </div>
        <div class="sim-row">
          <span class="sim-label">Match</span>
          <div class="sim-bar-bg">
            <div class="sim-bar-fill" style="width:{sim_w}%;background:linear-gradient(90deg,{c1},{c2})"></div>
          </div>
          <span class="sim-pct">{rec['similarity']}%</span>
        </div>
      </div>
    </div>
    """

def build_not_found_html(query, suggestions, all_titles):
    sugg_html = ""
    if suggestions:
        pills = " ".join(
            f'<span class="sugg-pill">{t.title()}</span>'
            for t in suggestions[:6]
        )
        sugg_html = f"""
        <div class="sugg-block">
          <p class="sugg-heading">💡 Did you mean one of these?</p>
          <div class="sugg-pills">{pills}</div>
          <p class="nf-tip">Copy a title above and paste it into the search box</p>
        </div>
        """

    import random
    sample = random.sample(all_titles, min(10, len(all_titles)))
    sample_pills = " ".join(f'<span class="sample-pill">{t}</span>' for t in sample)

    return f"""
    <div class="not-found-wrap">
      <div class="nf-icon">🔍</div>
      <h3 class="nf-title">"{query}" is not in this dataset</h3>
      <p class="nf-hint">
        This app is built on an <strong>Amazon Prime Video catalog</strong>.<br>
        Titles from other platforms (Netflix, HBO, Disney+) may not be included.<br>
        Use the <strong>📋 Browse Titles</strong> tab to discover what's available.
      </p>
      {sugg_html}
      <div class="sample-block">
        <p class="sugg-heading">✦ Random titles from the dataset</p>
        <div class="sample-pills">{sample_pills}</div>
      </div>
    </div>
    """

def build_results_html(source_info, recs):
    src   = build_source_card(source_info)
    cards = "\n".join(build_rec_card(r, i+1) for i, r in enumerate(recs))
    return f"""
    <div class="results-wrap">
      {src}
      <h3 class="section-heading">✦ Recommended For You</h3>
      <div class="cards-grid">{cards}</div>
    </div>
    """

def build_empty_state():
    return """
    <div class="empty-state">
      <div class="film-reel">🎞️</div>
      <h3>Your recommendations will appear here</h3>
      <p>Enter a title from the dataset and press Enter</p>
      <p class="nf-tip">💡 Not sure what to search? Use the <strong>📋 Browse Titles</strong> tab first</p>
    </div>
    """

def build_browse_html(df, genre_filter="All", type_filter="All", search_q=""):
    filtered = df.copy()
    if type_filter != "All":
        filtered = filtered[filtered["type"].str.upper() == type_filter.upper()]
    if genre_filter != "All":
        filtered = filtered[filtered["genres_text"].str.lower().str.contains(genre_filter.lower(), na=False)]
    if search_q.strip():
        filtered = filtered[filtered["title"].str.lower().str.contains(search_q.strip().lower(), na=False)]

    filtered = filtered.sort_values("imdb_score", ascending=False)
    count_text = f'<p class="browse-count">Showing <strong>{len(filtered)}</strong> of {len(df)} titles</p>'

    if len(filtered) == 0:
        return count_text + '<div class="empty-state"><p>No titles match your filters.</p></div>'

    rows = ""
    for _, row in filtered.head(100).iterrows():
        c1, c2 = pick_gradient(row["genres_text"])
        icon   = genre_icon(row["genres_text"])
        score  = round(float(row["imdb_score"]), 1) if float(row["imdb_score"]) > 0 else "N/A"
        year   = int(row["release_year"]) if int(row["release_year"]) > 0 else ""
        genre_tags = " ".join(
            f'<span class="gtag">{g.strip()}</span>'
            for g in str(row["genres_text"]).split()[:3] if g.strip()
        )
        rows += f"""
        <div class="browse-row">
          <div class="browse-icon" style="background:linear-gradient(135deg,{c1},{c2})">{icon}</div>
          <div class="browse-info">
            <div class="browse-title-line">
              <span class="browse-title">{row['title']}</span>
              {type_badge(row['type'])}
            </div>
            <div class="genre-row" style="margin-top:4px">{genre_tags}</div>
          </div>
          <div class="browse-meta">
            <span class="imdb-pill" style="font-size:11px">⭐ {score}</span>
            <span class="browse-year">{year}</span>
          </div>
        </div>
        """
    suffix = '<p class="browse-more">Showing top 100. Use filters to narrow down.</p>' if len(filtered) > 100 else ""
    return f'<div class="browse-wrap">{count_text}<div class="browse-list">{rows}</div>{suffix}</div>'


# ─────────────────────────────────────────────
#  5. CSS
# ─────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=Crimson+Pro:ital,wght@0,400;1,400&display=swap');

:root {
  --bg:#0a0b0f; --surface:#111318; --surface2:#181a22;
  --border:rgba(255,255,255,0.07);
  --gold:#f5c518; --text:#e8e9ef; --muted:#8b8fa8; --accent:#4f8ef7;
  --radius:14px;
  --font-display:'Bebas Neue',sans-serif;
  --font-body:'DM Sans',sans-serif;
  --font-serif:'Crimson Pro',serif;
}
.gradio-container{background:var(--bg)!important;font-family:var(--font-body)!important;}
footer{display:none!important;}
*{box-sizing:border-box;}

/* Header */
#header{text-align:center;padding:52px 24px 32px;position:relative;overflow:hidden;}
#header::before{content:"";position:absolute;inset:0;
  background:radial-gradient(ellipse 80% 60% at 50% 0%,rgba(79,142,247,.12),transparent 70%),
             radial-gradient(ellipse 50% 35% at 80% 20%,rgba(245,197,24,.07),transparent 60%);
  pointer-events:none;}
.header-eyebrow{font-size:11px;letter-spacing:4px;text-transform:uppercase;color:var(--gold);margin-bottom:8px;}
.header-title{font-family:var(--font-display);font-size:clamp(52px,8vw,96px);line-height:.92;letter-spacing:3px;color:var(--text);margin:0 0 14px;}
.header-title span{color:var(--gold);}
.header-sub{font-family:var(--font-serif);font-size:18px;font-style:italic;color:var(--muted);margin:0;}
.header-filmstrip{display:flex;justify-content:center;gap:6px;margin-top:20px;}
.film-hole{width:10px;height:14px;border:2px solid rgba(255,255,255,.1);border-radius:2px;}

/* Stats */
#stats-bar{display:flex;justify-content:center;gap:40px;padding:18px 24px;margin:0 auto 8px;max-width:600px;}
.stat-item{text-align:center;}
.stat-num{font-family:var(--font-display);font-size:30px;color:var(--gold);letter-spacing:1px;}
.stat-label{font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-top:2px;}

/* Controls */
#controls-panel{max-width:760px;margin:0 auto 28px;padding:24px 28px;background:var(--surface);border:1px solid var(--border);border-radius:20px;box-shadow:0 8px 40px rgba(0,0,0,.45);}
#controls-panel label{color:var(--muted)!important;font-size:11px!important;letter-spacing:2px!important;text-transform:uppercase!important;}
#controls-panel input[type=text]{background:var(--surface2)!important;border:1px solid var(--border)!important;border-radius:10px!important;color:var(--text)!important;font-size:16px!important;padding:13px 18px!important;transition:border-color .2s;}
#controls-panel input[type=text]:focus{border-color:var(--accent)!important;box-shadow:0 0 0 3px rgba(79,142,247,.15)!important;outline:none!important;}
#controls-panel input[type=range]{accent-color:var(--gold)!important;}

/* Button */
#find-btn{background:linear-gradient(135deg,#f5c518,#e6880a)!important;color:#0a0b0f!important;font-family:var(--font-display)!important;font-size:20px!important;letter-spacing:2px!important;border:none!important;border-radius:12px!important;padding:14px 0!important;width:100%!important;margin-top:6px!important;cursor:pointer!important;transition:opacity .2s,transform .15s!important;box-shadow:0 4px 20px rgba(245,197,24,.22)!important;}
#find-btn:hover{opacity:.9!important;transform:translateY(-1px)!important;}

/* Match notice */
.match-notice{background:rgba(79,142,247,.1);border:1px solid rgba(79,142,247,.22);border-radius:10px;padding:10px 16px;margin:0 auto 16px;font-size:13px;color:#7aacff;max-width:900px;}

/* Results */
.results-wrap{max-width:1100px;margin:0 auto;padding:0 20px 40px;}
.section-heading{font-family:var(--font-display);font-size:20px;letter-spacing:2px;color:var(--muted);margin:0 0 18px;}

/* Source card */
.source-card{display:flex;align-items:center;gap:22px;padding:22px 26px;border-radius:16px;margin-bottom:28px;box-shadow:0 8px 32px rgba(0,0,0,.4);border:1px solid rgba(255,255,255,.07);}
.source-poster{flex-shrink:0;width:72px;height:72px;background:rgba(0,0,0,.22);border-radius:12px;display:flex;align-items:center;justify-content:center;}
.source-label{font-size:9px;letter-spacing:3px;color:rgba(255,255,255,.45);margin:0 0 5px;text-transform:uppercase;}
.source-title{font-family:var(--font-display);font-size:30px;letter-spacing:1px;color:#fff;margin:0 0 9px;}
.source-row{display:flex;align-items:center;gap:9px;flex-wrap:wrap;}
.source-year{font-size:12px;color:rgba(255,255,255,.5);}

/* Cards */
.cards-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:18px;}
.rec-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;transition:transform .2s,box-shadow .2s;}
.rec-card:hover{transform:translateY(-4px);box-shadow:0 16px 40px rgba(0,0,0,.5);border-color:rgba(255,255,255,.11);}
.poster{height:130px;position:relative;display:flex;align-items:center;justify-content:center;}
.poster-rank{position:absolute;top:10px;left:11px;font-family:var(--font-display);font-size:26px;color:rgba(255,255,255,.22);}
.poster-icon{font-size:48px;filter:drop-shadow(0 2px 8px rgba(0,0,0,.4));}
.poster-cert{position:absolute;bottom:7px;right:9px;font-size:9px;letter-spacing:1px;background:rgba(0,0,0,.45);color:rgba(255,255,255,.65);padding:2px 6px;border-radius:4px;}
.rec-body{padding:14px 16px 16px;}
.rec-type-row{display:flex;align-items:center;gap:7px;margin-bottom:5px;}
.rec-year{font-size:11px;color:var(--muted);}
.rec-title{font-family:var(--font-display);font-size:20px;letter-spacing:.5px;color:var(--text);margin:0 0 9px;line-height:1.1;}
.genre-row{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:10px;}
.gtag{font-size:9px;letter-spacing:.5px;text-transform:uppercase;background:rgba(255,255,255,.06);color:var(--muted);padding:2px 7px;border-radius:20px;}
.score-row{display:flex;align-items:center;gap:7px;margin-bottom:9px;}
.stars{color:var(--gold);font-size:12px;letter-spacing:1px;}
.imdb-num{font-size:11px;color:var(--muted);}
.sim-row{display:flex;align-items:center;gap:7px;}
.sim-label{font-size:9px;letter-spacing:1px;text-transform:uppercase;color:var(--muted);width:34px;flex-shrink:0;}
.sim-bar-bg{flex:1;height:4px;background:rgba(255,255,255,.07);border-radius:999px;overflow:hidden;}
.sim-bar-fill{height:100%;border-radius:999px;}
.sim-pct{font-size:10px;color:var(--muted);width:34px;text-align:right;}

/* Badges */
.badge{display:inline-block;font-size:9px;letter-spacing:1.5px;text-transform:uppercase;padding:2px 7px;border-radius:4px;font-weight:600;}
.badge-movie{background:rgba(79,142,247,.15);color:#7aacff;border:1px solid rgba(79,142,247,.22);}
.badge-show{background:rgba(160,100,220,.15);color:#bf8fff;border:1px solid rgba(160,100,220,.22);}
.imdb-pill{background:rgba(245,197,24,.13);color:var(--gold);font-size:12px;font-weight:600;padding:2px 10px;border-radius:20px;border:1px solid rgba(245,197,24,.22);}

/* Not found */
.not-found-wrap{max-width:680px;margin:40px auto;padding:0 20px;text-align:center;}
.nf-icon{font-size:52px;margin-bottom:14px;opacity:.4;}
.nf-title{font-family:var(--font-display);font-size:26px;letter-spacing:1px;color:rgba(255,255,255,.4);margin:0 0 12px;}
.nf-hint{font-size:14px;color:var(--muted);line-height:1.7;margin-bottom:24px;}
.nf-tip{font-size:12px;color:var(--muted);margin-top:10px;}
.sugg-block{background:rgba(79,142,247,.08);border:1px solid rgba(79,142,247,.18);border-radius:12px;padding:16px 20px;margin-bottom:20px;}
.sugg-heading{font-size:11px;letter-spacing:2px;text-transform:uppercase;color:var(--accent);margin:0 0 12px;}
.sugg-pills{display:flex;flex-wrap:wrap;justify-content:center;gap:8px;}
.sugg-pill{font-size:13px;padding:6px 14px;border-radius:20px;background:rgba(79,142,247,.12);border:1px solid rgba(79,142,247,.25);color:#7aacff;cursor:pointer;}
.sample-block{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:18px 20px;}
.sample-pills{display:flex;flex-wrap:wrap;justify-content:center;gap:7px;margin-top:10px;}
.sample-pill{font-size:12px;padding:5px 12px;border-radius:20px;background:var(--surface2);border:1px solid var(--border);color:var(--muted);}

/* Empty state */
.empty-state{text-align:center;padding:70px 24px;color:var(--muted);}
.film-reel{font-size:52px;margin-bottom:14px;opacity:.4;animation:spin 10s linear infinite;}
@keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
.empty-state h3{font-family:var(--font-display);font-size:24px;letter-spacing:1px;color:rgba(255,255,255,.2);margin:0 0 8px;}
.empty-state p{font-size:13px;margin:4px 0;}

/* Browse */
.browse-wrap{max-width:900px;margin:0 auto;padding:0 20px 40px;}
.browse-count{font-size:13px;color:var(--muted);margin-bottom:16px;}
.browse-list{display:flex;flex-direction:column;gap:10px;}
.browse-row{display:flex;align-items:center;gap:14px;background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:12px 16px;transition:border-color .15s;}
.browse-row:hover{border-color:rgba(255,255,255,.12);}
.browse-icon{width:44px;height:44px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:22px;flex-shrink:0;}
.browse-info{flex:1;min-width:0;}
.browse-title-line{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
.browse-title{font-family:var(--font-display);font-size:17px;letter-spacing:.5px;color:var(--text);}
.browse-meta{display:flex;flex-direction:column;align-items:flex-end;gap:5px;flex-shrink:0;}
.browse-year{font-size:11px;color:var(--muted);}
.browse-more{font-size:12px;color:var(--muted);text-align:center;margin-top:16px;}
#browse-controls{max-width:900px;margin:0 auto 20px;padding:18px 24px;background:var(--surface);border:1px solid var(--border);border-radius:16px;}
#browse-controls input[type=text]{background:var(--surface2)!important;border:1px solid var(--border)!important;border-radius:8px!important;color:var(--text)!important;font-size:14px!important;padding:10px 14px!important;}
#browse-controls input[type=text]:focus{border-color:var(--accent)!important;outline:none!important;}

/* How it works */
#how-it-works{max-width:760px;margin:36px auto 0;padding:26px 30px;background:var(--surface);border:1px solid var(--border);border-radius:20px;}
#how-it-works h3{font-family:var(--font-display);font-size:20px;letter-spacing:2px;color:var(--muted);margin:0 0 18px;}
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;}
.step-card{background:var(--surface2);border-radius:12px;padding:16px;text-align:center;border:1px solid var(--border);}
.step-icon{font-size:26px;margin-bottom:9px;}
.step-title{font-size:11px;letter-spacing:1px;text-transform:uppercase;color:var(--text);font-weight:600;margin-bottom:5px;}
.step-desc{font-size:11px;color:var(--muted);line-height:1.5;}

@media(max-width:640px){
  .source-card{flex-direction:column;text-align:center;}
  .source-row{justify-content:center;}
  .steps{grid-template-columns:1fr;}
  #stats-bar{gap:22px;}
  .cards-grid{grid-template-columns:1fr;}
}
"""


# ─────────────────────────────────────────────
#  6. BUILD APP
# ─────────────────────────────────────────────

print("Loading data...")
DF, SIM_MATRIX, TITLE_INDEX, ALL_TITLES = load_data()
TOTAL_TITLES = len(DF)

_genre_words = " ".join(DF["genres_text"].fillna("").tolist()).split()
UNIQUE_GENRES = sorted(set(g for g in _genre_words if len(g) > 2))[:30]

print(f"Ready — {TOTAL_TITLES} titles | {len(UNIQUE_GENRES)} genres")

FILM_HOLES = "".join('<div class="film-hole"></div>' for _ in range(14))

HEADER_HTML = f"""
<div id="header">
  <p class="header-eyebrow">✦ AI-Powered Discovery ✦</p>
  <h1 class="header-title">CINE<span>MATCH</span></h1>
  <p class="header-sub">Find your next favourite movie or TV show</p>
  <div class="header-filmstrip">{FILM_HOLES}</div>
</div>
<div id="stats-bar">
  <div class="stat-item"><div class="stat-num">{TOTAL_TITLES:,}</div><div class="stat-label">Titles</div></div>
  <div class="stat-item"><div class="stat-num">TF-IDF</div><div class="stat-label">Algorithm</div></div>
  <div class="stat-item"><div class="stat-num">3</div><div class="stat-label">Smart Signals</div></div>
</div>
"""

HOW_HTML = """
<div id="how-it-works">
  <h3>HOW IT WORKS</h3>
  <div class="steps">
    <div class="step-card"><div class="step-icon">📝</div><div class="step-title">Text Analysis</div><div class="step-desc">Genre, description, cast and director are combined into one rich text profile per title</div></div>
    <div class="step-card"><div class="step-icon">🔢</div><div class="step-title">TF-IDF Vectors</div><div class="step-desc">Each title becomes a numerical vector — rare meaningful words get higher importance automatically</div></div>
    <div class="step-card"><div class="step-icon">🎯</div><div class="step-title">Smart Ranking</div><div class="step-desc">Similarity is re-ranked with IMDb quality and TMDb popularity for the best final result</div></div>
  </div>
</div>
"""


def on_recommend(title, top_n):
    if not title.strip():
        return build_empty_state()
    recs, source_or_msg, suggestions = recommend(
        title, int(top_n), DF, SIM_MATRIX, TITLE_INDEX, ALL_TITLES
    )
    if recs is None:
        if source_or_msg == "empty":
            return build_empty_state()
        return build_not_found_html(title, suggestions, ALL_TITLES)
    return build_results_html(source_or_msg, recs)


def on_browse(genre_filter, type_filter, search_q):
    return build_browse_html(DF, genre_filter, type_filter, search_q)


with gr.Blocks(css=CSS, title="CineMatch — AI Movie Recommender") as demo:

    gr.HTML(HEADER_HTML)

    with gr.Tabs():

        with gr.Tab("🎬  Find Similar"):
            with gr.Group(elem_id="controls-panel"):
                title_input  = gr.Textbox(label="Movie or TV Show Title", placeholder='e.g. "The Three Stooges" or "Inception"', lines=1)
                top_n_slider = gr.Slider(minimum=3, maximum=10, value=5, step=1, label="Number of recommendations")
                find_btn     = gr.Button("🎬  FIND SIMILAR TITLES", elem_id="find-btn")
            output_html = gr.HTML(value=build_empty_state())
            gr.HTML(HOW_HTML)
            find_btn.click(fn=on_recommend, inputs=[title_input, top_n_slider], outputs=output_html)
            title_input.submit(fn=on_recommend, inputs=[title_input, top_n_slider], outputs=output_html)

        with gr.Tab("📋  Browse Titles"):
            with gr.Group(elem_id="browse-controls"):
                with gr.Row():
                    browse_search = gr.Textbox(label="Search by title", placeholder="Type to filter...", lines=1, scale=3)
                    browse_genre  = gr.Dropdown(choices=["All"] + UNIQUE_GENRES, value="All", label="Genre", scale=1)
                    browse_type   = gr.Dropdown(choices=["All","MOVIE","SHOW"], value="All", label="Type", scale=1)
            browse_output = gr.HTML(value=build_browse_html(DF))
            for inp in [browse_search, browse_genre, browse_type]:
                inp.change(fn=on_browse, inputs=[browse_genre, browse_type, browse_search], outputs=browse_output)



# =============================================================================
#  Amazon Fine Food Reviews — Sentiment Analysis Pipeline
#  Output: sentiment_dashboard_data.csv  (Power BI ready)
# =============================================================================
#
#  COLUMNS PRODUCED
#  ─────────────────────────────────────────────────────────────────────────────
#  Identity      : review_id, product_id, user_id
#  Text          : summary, original_text, cleaned_text
#  VADER scores  : vader_compound, vader_pos, vader_neu, vader_neg
#  Sentiment     : sentiment_label, sentiment_score_pct, subjectivity
#  Ground truth  : star_rating, star_sentiment, model_agrees_star
#  Helpfulness   : helpfulness_ratio, helpfulness_votes
#  Date          : review_date, review_year, review_month, review_month_name,
#                  review_quarter, year_month
#  Text metrics  : word_count, char_count, cleaned_word_count
#
#  POWER BI STAR SCHEMA
#  ─────────────────────────────────────────────────────────────────────────────
#  FactReviews  ←→  DimDate      (on review_date)
#               ←→  DimProduct   (on product_id)
#               ←→  DimSentiment (on sentiment_label)
#
#  DAX MEASURES TO BUILD IN POWER BI
#  ─────────────────────────────────────────────────────────────────────────────
#  Avg Sentiment Score  = AVERAGE(FactReviews[vader_compound])
#  % Positive Reviews   = DIVIDE(COUNTROWS(FILTER(FactReviews, FactReviews[sentiment_label]="Positive")), COUNTROWS(FactReviews))
#  7-Day Rolling Avg    = AVERAGEX(DATESINPERIOD(DimDate[review_date], LASTDATE(DimDate[review_date]), -7, DAY), [Avg Sentiment Score])
#  Model Accuracy       = AVERAGE(FactReviews[model_agrees_star])
# =============================================================================

import os
import re
import sys
import time
import logging

import numpy  as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem   import WordNetLemmatizer

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob


# ── 0. Logging ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── 1. Config — edit these paths if needed ────────────────────────────────────

INPUT_CSV   = "Reviews.csv"          # path to the raw Kaggle file
OUTPUT_CSV  = "sentiment_dashboard_data.csv"   # Power BI import file
SAMPLE_SIZE = 100_000                # rows to use  (None = all 568 k rows)
RANDOM_SEED = 42


# ── 2. NLTK assets ───────────────────────────────────────────────────────────

def download_nltk_assets() -> None:
    for asset in ("stopwords", "wordnet", "punkt", "omw-1.4"):
        nltk.download(asset, quiet=True)

download_nltk_assets()

STOP_WORDS  = set(stopwords.words("english"))
lemmatizer  = WordNetLemmatizer()
analyzer    = SentimentIntensityAnalyzer()


# ── 3. Text cleaning ─────────────────────────────────────────────────────────

def clean_text(raw: str) -> str:
    """
    Lowercase → strip HTML & URLs → keep only letters →
    remove stopwords → lemmatize.
    Returns a clean, space-separated token string.
    """
    if not isinstance(raw, str) or not raw.strip():
        return ""

    text = raw.lower()
    text = re.sub(r"<.*?>",         " ", text)   # HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)  # URLs
    text = re.sub(r"[^a-z\s]",      " ", text)   # non-alpha
    text = re.sub(r"\s+",           " ", text).strip()

    tokens = [
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in STOP_WORDS and len(w) > 2
    ]
    return " ".join(tokens)


# ── 4. Sentiment helpers ─────────────────────────────────────────────────────

def vader_row(text: str) -> dict:
    """Return all four VADER scores for a cleaned text string."""
    s = analyzer.polarity_scores(text)
    return {
        "vader_compound": round(s["compound"], 4),
        "vader_pos":      round(s["pos"],      4),
        "vader_neu":      round(s["neu"],       4),
        "vader_neg":      round(s["neg"],       4),
    }


def compound_to_label(score: float) -> str:
    """Standard VADER thresholds: >= 0.05 Positive, <= -0.05 Negative."""
    if score >= 0.05:
        return "Positive"
    if score <= -0.05:
        return "Negative"
    return "Neutral"


def score_to_star_sentiment(star: int) -> str:
    """Convert 1-5 star rating to sentiment bucket (ground truth)."""
    if star >= 4:
        return "Positive"
    if star == 3:
        return "Neutral"
    return "Negative"


# ── 5. Load & sample ─────────────────────────────────────────────────────────

def load_data(path: str, sample: int | None, seed: int) -> pd.DataFrame:
    log.info("Loading raw CSV …")
    df = pd.read_csv(path, dtype={"ProductId": str, "UserId": str})
    log.info("  Raw rows : %s", f"{len(df):,}")

    # Drop rows missing the review text (rare but guard anyway)
    df = df.dropna(subset=["Text"]).reset_index(drop=True)

    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=seed).reset_index(drop=True)
        log.info("  Sampled  : %s", f"{len(df):,}")

    return df


# ── 6. Feature engineering ───────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    # 6-a  Clean text
    log.info("Cleaning text …")
    df["cleaned_text"] = df["Text"].apply(clean_text)

    # 6-b  VADER sentiment scores
    log.info("Running VADER sentiment …")
    vader_results = df["cleaned_text"].apply(lambda t: pd.Series(vader_row(t)))
    df = pd.concat([df, vader_results], axis=1)

    # 6-c  Sentiment label + normalised percentage (0–100 scale for Power BI KPI cards)
    df["sentiment_label"]     = df["vader_compound"].apply(compound_to_label)
    df["sentiment_score_pct"] = ((df["vader_compound"] + 1) / 2 * 100).round(2)

    # 6-d  TextBlob subjectivity (0 = objective, 1 = subjective)
    log.info("Computing subjectivity …")
    df["subjectivity"] = df["cleaned_text"].apply(
        lambda t: round(TextBlob(t).sentiment.subjectivity, 4)
    )

    # 6-e  Star-based ground truth & model agreement
    df["star_rating"]        = df["Score"]
    df["star_sentiment"]     = df["Score"].apply(score_to_star_sentiment)
    df["model_agrees_star"]  = (df["sentiment_label"] == df["star_sentiment"]).astype(int)

    # 6-f  Helpfulness ratio
    df["helpfulness_votes"] = df["HelpfulnessDenominator"]
    df["helpfulness_ratio"] = np.where(
        df["HelpfulnessDenominator"] > 0,
        (df["HelpfulnessNumerator"] / df["HelpfulnessDenominator"]).round(4),
        0.0,
    )

    # 6-g  Date features  (Time column is Unix epoch seconds)
    log.info("Parsing dates …")
    df["review_date"]       = pd.to_datetime(df["Time"], unit="s")
    df["review_year"]       = df["review_date"].dt.year.astype(int)
    df["review_month"]      = df["review_date"].dt.month.astype(int)
    df["review_month_name"] = df["review_date"].dt.strftime("%b")       # Jan, Feb …
    df["review_quarter"]    = "Q" + df["review_date"].dt.quarter.astype(str)
    df["year_month"]        = df["review_date"].dt.to_period("M").astype(str)  # 2010-03

    # 6-h  Text metrics
    df["word_count"]         = df["Text"].str.split().str.len()
    df["char_count"]         = df["Text"].str.len()
    df["cleaned_word_count"] = df["cleaned_text"].str.split().str.len()

    return df


# ── 7. Rename & select final columns ─────────────────────────────────────────

FINAL_COLUMNS = {
    # identity
    "Id":          "review_id",
    "ProductId":   "product_id",
    "UserId":      "user_id",
    # text
    "Summary":     "summary",
    "Text":        "original_text",
    "cleaned_text":"cleaned_text",
    # VADER scores
    "vader_compound": "vader_compound",
    "vader_pos":      "vader_pos",
    "vader_neu":      "vader_neu",
    "vader_neg":      "vader_neg",
    # sentiment
    "sentiment_label":     "sentiment_label",
    "sentiment_score_pct": "sentiment_score_pct",
    "subjectivity":        "subjectivity",
    # ground truth
    "star_rating":       "star_rating",
    "star_sentiment":    "star_sentiment",
    "model_agrees_star": "model_agrees_star",
    # helpfulness
    "helpfulness_ratio": "helpfulness_ratio",
    "helpfulness_votes": "helpfulness_votes",
    # dates
    "review_date":       "review_date",
    "review_year":       "review_year",
    "review_month":      "review_month",
    "review_month_name": "review_month_name",
    "review_quarter":    "review_quarter",
    "year_month":        "year_month",
    # text metrics
    "word_count":         "word_count",
    "char_count":         "char_count",
    "cleaned_word_count": "cleaned_word_count",
}


def select_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    present = {k: v for k, v in FINAL_COLUMNS.items() if k in df.columns}
    return df[list(present.keys())].rename(columns=present)


# ── 8. Quality report ────────────────────────────────────────────────────────

def quality_report(df: pd.DataFrame) -> None:
    log.info("=" * 60)
    log.info("PIPELINE QUALITY REPORT")
    log.info("=" * 60)
    log.info("Total reviews          : %s", f"{len(df):,}")
    log.info("Date range             : %s → %s",
             df["review_date"].min().date(), df["review_date"].max().date())

    dist = df["sentiment_label"].value_counts()
    for label in ["Positive", "Neutral", "Negative"]:
        count = dist.get(label, 0)
        pct   = count / len(df) * 100
        log.info("  %-10s : %7s  (%5.1f%%)", label, f"{count:,}", pct)

    acc = df["model_agrees_star"].mean() * 100
    log.info("Model vs star accuracy : %.1f%%", acc)
    log.info("Avg VADER compound     : %.4f", df["vader_compound"].mean())
    log.info("Avg subjectivity       : %.4f", df["subjectivity"].mean())
    log.info("Null counts            :\n%s", df.isnull().sum()[df.isnull().sum() > 0])
    log.info("=" * 60)


# ── 9. Export dimension tables (optional but great for Power BI) ─────────────

def export_dim_tables(df: pd.DataFrame, output_dir: str) -> None:
    """
    Export DimDate and DimProduct as separate CSVs.
    These feed the star schema relationships in Power BI.
    """

    # DimDate — one row per unique year_month
    dim_date = (
        df[["review_date", "review_year", "review_month",
            "review_month_name", "review_quarter", "year_month"]]
        .drop_duplicates(subset=["year_month"])
        .sort_values("review_date")
        .reset_index(drop=True)
    )
    dim_date_path = os.path.join(output_dir, "dim_date.csv")
    dim_date.to_csv(dim_date_path, index=False)
    log.info("DimDate exported       : %s rows → %s", f"{len(dim_date):,}", dim_date_path)

    # DimProduct — one row per unique product_id
    dim_product = (
        df.groupby("product_id")
          .agg(
              total_reviews   = ("review_id",      "count"),
              avg_star_rating = ("star_rating",     "mean"),
              avg_sentiment   = ("vader_compound",  "mean"),
          )
          .round(4)
          .reset_index()
    )
    dim_product_path = os.path.join(output_dir, "dim_product.csv")
    dim_product.to_csv(dim_product_path, index=False)
    log.info("DimProduct exported    : %s rows → %s", f"{len(dim_product):,}", dim_product_path)

    # DimSentiment — small lookup table for slicer labels
    dim_sentiment = pd.DataFrame({
        "sentiment_label": ["Positive", "Neutral", "Negative"],
        "sort_order":      [1,          2,          3],
        "color_hex":       ["#2ECC71",  "#F39C12",  "#E74C3C"],
    })
    dim_sentiment_path = os.path.join(output_dir, "dim_sentiment.csv")
    dim_sentiment.to_csv(dim_sentiment_path, index=False)
    log.info("DimSentiment exported  : %s rows → %s", f"{len(dim_sentiment):,}", dim_sentiment_path)


# ── 10. Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    # Validate input
    if not os.path.exists(INPUT_CSV):
        log.error("Input file not found: %s", INPUT_CSV)
        sys.exit(1)

    # Load
    df = load_data(INPUT_CSV, SAMPLE_SIZE, RANDOM_SEED)

    # Feature engineering
    df = engineer_features(df)

    # Select final columns
    df_final = select_and_rename(df)

    # Quality report
    quality_report(df_final)

    # Export fact table
    df_final.to_csv(OUTPUT_CSV, index=False, date_format="%Y-%m-%d")
    log.info("FactReviews exported   : %s rows → %s", f"{len(df_final):,}", OUTPUT_CSV)

    # Export dimension tables (same folder as output CSV)
    output_dir = os.path.dirname(os.path.abspath(OUTPUT_CSV)) or "."
    export_dim_tables(df_final, output_dir)

    elapsed = time.time() - t0
    log.info("Pipeline complete in   : %.1f seconds", elapsed)
    log.info("")
    log.info("FILES TO IMPORT INTO POWER BI")
    log.info("  Fact table : %s", OUTPUT_CSV)
    log.info("  Dimensions : dim_date.csv  |  dim_product.csv  |  dim_sentiment.csv")
    log.info("")
    log.info("POWER BI RELATIONSHIPS TO CREATE")
    log.info("  FactReviews[year_month]      → DimDate[year_month]        (Many-to-One)")
    log.info("  FactReviews[product_id]      → DimProduct[product_id]     (Many-to-One)")
    log.info("  FactReviews[sentiment_label] → DimSentiment[sentiment_label] (Many-to-One)")


if __name__ == "__main__":
    main()

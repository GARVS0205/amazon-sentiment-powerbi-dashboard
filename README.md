# Amazon Fine Food Reviews — Sentiment Dashboard

## Project Overview
An end-to-end data science + business intelligence project combining 
NLP-based sentiment analysis with an advanced Power BI dashboard.

## Business Problem
Amazon receives millions of product reviews. How do we understand 
customer sentiment at scale — across products, time periods, and 
review types — without reading every review?

## Solution Architecture
Raw Data (568K reviews) → Python NLP Pipeline → 4 structured CSVs → Power BI Star Schema → 3-page Interactive Dashboard
## ML Pipeline (Python)
- **Dataset:** Amazon Fine Food Reviews (Kaggle) — 568K reviews, 1999–2012
- **Sample size:** 100,000 reviews (stratified)
- **Text preprocessing:** Lowercasing, HTML stripping, stopword removal, lemmatization (NLTK)
- **Model:** VADER Sentiment Analyzer (vaderSentiment)
- **Accuracy:** 79.4% agreement with customer star ratings (ground truth validation)
- **Output:** 27 engineered features per review including polarity scores, sentiment labels, date parts, helpfulness ratio

## Power BI Dashboard

### Data Model
Star schema with 4 tables:
- **FactReviews** — 100K rows, 27 columns
- **DimDate** — 125 rows (year-month grain)
- **DimProduct** — 31,698 unique products
- **DimSentiment** — 3 rows (lookup with sort order and color codes)

### Pages
**Page 1 — Executive Overview**
KPI cards, sentiment trend line with 7-day rolling average, sentiment distribution donut, review volume by year, avg sentiment by year

**Page 2 — Product Deep Dive**
Pareto 80/20 analysis (top 20 products), staircase bar chart with dynamic conditional formatting, product sentiment rankings table, sentiment summary by star rating with gradient formatting

**Page 3 — Review Explorer**
AI Decomposition Tree, Key Influencers, dynamic metric line chart (Field Parameters), sentiment distribution histogram, review browser with data bars and conditional formatting

### Advanced Power BI Features
| Feature | Implementation |
|---|---|
| Field Parameters | Dynamic metric switcher — one chart shows any selected KPI |
| Decomposition Tree | AI-powered drill path finding sentiment drivers |
| Key Influencers | ML-powered analysis of what drives positive/negative labels |
| Drill Through | Product-level detail page with back navigation |
| Bookmarks + Buttons | App-like navigation — All/Positive/Negative views |
| Page Navigator | Three-page app navigation |
| Sync Slicers | Year and sentiment filters persist across all pages |
| Row Level Security | Role-based data access (Positive Analyst / Negative Analyst) |
| Custom Tooltip Page | Hover on trend line shows yearly breakdown popup |
| Pareto Analysis | 80/20 rule — top N products by review volume with cumulative % line |
| Running Total DAX | Cumulative review growth measure |
| Conditional Formatting | Data bars, background color rules, gradient bars |
| What-If Parameter | Top N products slider |
| Performance Optimization | All visuals load under 530ms on 100K rows |
| Star Schema | Proper dimensional model with 3 Many-to-One relationships |

## Key Insights
- **91.4%** of reviews are Positive sentiment
- **79.4%** model accuracy — VADER independently agrees with star ratings 4 in 5 times
- **Top 15 products** account for 80% of all review volume (Pareto principle confirmed)
- **Helpfulness ratio** is the strongest predictor of negative sentiment (7.39x increase in likelihood)
- Avg sentiment score remained stable at **0.71–0.74** across the entire 12-year period

## Tech Stack
Python · pandas · NLTK · vaderSentiment · Power BI Desktop · DAX · Power Query

## Files
| File | Description |
|---|---|
| `sentiment_pipeline.py` | Complete Python NLP pipeline |
| `dim_date.csv` | Date dimension table |
| `dim_product.csv` | Product dimension table |
| `dim_sentiment.csv` | Sentiment lookup table |
| `/screenshots` | Dashboard page screenshots |

## Dataset
Amazon Fine Food Reviews — [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

# Executive Briefing Bot

**An AI-powered morning intelligence system that automatically detects business anomalies, retrieves historical context, and generates a plain-English executive briefing — every day.**

🚀 **[Live Demo](https://executive-briefing-bot-euuvdb6qs8zeilkcjv2udk.streamlit.app/)**

---

## What It Does

Every morning, the system:

1. **Pulls the latest business data** — sales, profit, orders, discounts
2. **Runs ML anomaly detection** — flags what changed overnight vs the 30-day rolling average
3. **Retrieves historical context** — searches past briefings for similar patterns ("this happened on March 14th too")
4. **Generates a plain-English briefing** — Claude writes a confident, concise narrative with one concrete recommendation
5. **Answers follow-up questions** — ask anything about today's data in natural language

An executive can open the dashboard, read a 3-paragraph briefing, and understand the state of their business in under 2 minutes.

---

## Demo

![Dashboard Screenshot](screenshot.png)

**Example briefing output:**

> Sales came in at $993.90 yesterday, running 46% below the 30-day average with only one order processed. While this looks concerning at first glance, similar low-volume days appeared in November 2015 and July 2015 — suggesting this may be a recurring seasonal pattern rather than a structural issue. Recommend monitoring the next 3 days closely before escalating.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data pipeline | Python, pandas, APScheduler |
| ML anomaly detection | Scikit-learn (Isolation Forest) |
| Trend analysis | 30-day rolling window comparison |
| Memory / RAG | ChromaDB + Sentence Transformers (all-MiniLM-L6-v2) |
| LLM briefing writer | Anthropic Claude API (claude-sonnet-4-5) |
| Q&A layer | Claude API with conversation memory |
| Dashboard | Streamlit + Plotly |
| Deployment | Streamlit Cloud |

---

## Architecture

```
Raw Data (CSV / API)
        │
        ▼
Daily Aggregation (pandas)
        │
        ▼
Isolation Forest ──────► Anomaly Report (JSON)
        │                       │
        ▼                       ▼
30-day Rolling Avg      ChromaDB Vector Store
                                │
                                ▼
                        RAG Retrieval (similar past days)
                                │
                                ▼
                        Claude API ──► Plain-English Briefing
                                │
                                ▼
                    Streamlit Dashboard + Q&A Chat
```

---

## How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/supriyah26/executive-briefing-bot.git
cd executive-briefing-bot
```

### 2. Create a virtual environment
```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API key
```bash
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

### 5. Seed the memory and run the pipeline
```bash
python pipeline.py
```

### 6. Launch the dashboard
```bash
streamlit run app.py
```

---

## Project Structure

```
executive-briefing-bot/
├── app.py                     # Streamlit dashboard
├── pipeline.py                # Data pipeline + ML + RAG + LLM
├── requirements.txt
├── Sample - Superstore.csv    # Dataset (Kaggle Superstore)
└── .gitignore
```

---

## Key Design Decisions

**Why Isolation Forest?**
Unsupervised anomaly detection — no labelled data needed. Works well on multivariate daily metrics (sales, profit, orders, discount) and adapts as new data arrives.

**Why RAG over a simple lookup?**
Semantic similarity search means the system finds days that *felt* similar, not just days with matching numbers. A -46% sales day in November retrieves other low-volume November days — capturing seasonal patterns automatically.

**Why a narrative briefing instead of a dashboard summary?**
Executives don't read dashboards — they read memos. The LLM layer translates numbers into decisions. The Q&A layer means they never have to dig into the data themselves.

---

## Coming Soon

- 🔊 Voice output — spoken briefing via ElevenLabs TTS
- 📧 Email delivery — briefing sent to inbox at 7am automatically
- 📊 Multi-region breakdown — anomaly detection per region/category
- 🔗 Live data connectors — Salesforce, Google Analytics, Snowflake

---

## Dataset

[Superstore Sales Dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final) — Kaggle. 4 years of retail orders across regions, categories, and products.

---

## Author

**Supriya Hinge** — Data Analyst  
[LinkedIn](https://www.linkedin.com/in/supriya-hinge-82335311b/) · [GitHub](https://github.com/supriyah26)
import pandas as pd
import numpy as np
from apscheduler.schedulers.blocking import BlockingScheduler
from sklearn.ensemble import IsolationForest
import chromadb
from chromadb.utils import embedding_functions
import anthropic
from dotenv import load_dotenv
import os
import json
from datetime import datetime

load_dotenv()

# ── Load & prep dataset ──────────────────────────────────────────────────────
file_path = "/Users/supriyah/executive-briefing-bot/Sample - Superstore.csv"

df = pd.read_csv(file_path, encoding='latin1')
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

# ── Daily aggregation ────────────────────────────────────────────────────────
daily = (
    df.groupby('Order Date')
    .agg(
        total_sales=('Sales', 'sum'),
        total_profit=('Profit', 'sum'),
        order_count=('Order ID', 'nunique'),
        avg_discount=('Discount', 'mean')
    )
    .reset_index()
    .sort_values('Order Date')
)

# ── Train Isolation Forest ───────────────────────────────────────────────────
features = ['total_sales', 'total_profit', 'order_count', 'avg_discount']
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(daily[features])

# ── ChromaDB setup ───────────────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = chroma_client.get_or_create_collection(
    name="briefing_history",
    embedding_function=embedding_fn
)

# ── Anthropic client ─────────────────────────────────────────────────────────
claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── Phase 1: Get one day's raw orders ────────────────────────────────────────
def get_todays_data(date):
    result = df[df['Order Date'] == date]
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Pulled {len(result)} orders for {date}")
    return result

# ── Phase 2: Anomaly detection ───────────────────────────────────────────────
def detect_anomalies(date):
    day_metrics = daily[daily['Order Date'] == pd.to_datetime(date)]

    if day_metrics.empty:
        return None

    score = iso_forest.predict(day_metrics[features])[0]
    anomaly_score = iso_forest.decision_function(day_metrics[features])[0]
    row = day_metrics.iloc[0]

    day_index = daily[daily['Order Date'] == pd.to_datetime(date)].index[0]
    window_start = max(0, day_index - 30)
    rolling_avg = daily.iloc[window_start:day_index][features].mean()

    report = {
        "date": date,
        "is_anomaly": score == -1,
        "anomaly_score": round(float(anomaly_score), 4),
        "metrics": {
            "total_sales": round(float(row['total_sales']), 2),
            "total_profit": round(float(row['total_profit']), 2),
            "order_count": int(row['order_count']),
            "avg_discount": round(float(row['avg_discount']), 4)
        },
        "vs_30day_avg": {
            "sales_diff_pct": round((row['total_sales'] - rolling_avg['total_sales']) / rolling_avg['total_sales'] * 100, 1) if rolling_avg['total_sales'] > 0 else 0,
            "profit_diff_pct": round((row['total_profit'] - rolling_avg['total_profit']) / rolling_avg['total_profit'] * 100, 1) if rolling_avg['total_profit'] != 0 else 0,
            "orders_diff_pct": round((row['order_count'] - rolling_avg['order_count']) / rolling_avg['order_count'] * 100, 1) if rolling_avg['order_count'] > 0 else 0,
        }
    }

    status = "🚨 ANOMALY DETECTED" if report['is_anomaly'] else "✅ Normal day"
    print(f"\n  {status} — {date}")
    print(f"  Sales:  ${report['metrics']['total_sales']:,.2f}  ({report['vs_30day_avg']['sales_diff_pct']:+.1f}% vs 30-day avg)")
    print(f"  Profit: ${report['metrics']['total_profit']:,.2f}  ({report['vs_30day_avg']['profit_diff_pct']:+.1f}% vs 30-day avg)")
    print(f"  Orders: {report['metrics']['order_count']}  ({report['vs_30day_avg']['orders_diff_pct']:+.1f}% vs 30-day avg)")
    print(f"  Anomaly score: {report['anomaly_score']}")

    return report

# ── Phase 3: RAG — save report to ChromaDB ───────────────────────────────────
def save_to_memory(report):
    if report is None:
        return

    status = "anomaly" if report['is_anomaly'] else "normal day"
    text = (
        f"On {report['date']}, it was a {status}. "
        f"Sales were ${report['metrics']['total_sales']:,.2f} "
        f"({report['vs_30day_avg']['sales_diff_pct']:+.1f}% vs 30-day avg). "
        f"Profit was ${report['metrics']['total_profit']:,.2f} "
        f"({report['vs_30day_avg']['profit_diff_pct']:+.1f}% vs 30-day avg). "
        f"Order count was {report['metrics']['order_count']} "
        f"({report['vs_30day_avg']['orders_diff_pct']:+.1f}% vs 30-day avg). "
        f"Anomaly score: {report['anomaly_score']}."
    )

    collection.upsert(
        documents=[text],
        ids=[report['date']],
        metadatas=[{"date": report['date'], "is_anomaly": str(report['is_anomaly'])}]
    )
    print(f"  💾 Saved to memory: {report['date']}")

# ── Phase 3: RAG — retrieve similar past days ────────────────────────────────
def retrieve_similar(report, n_results=3):
    if report is None:
        return []

    count = collection.count()
    if count == 0:
        print("  📭 Memory is empty — no history to retrieve yet")
        return []

    query = (
        f"Sales were {report['vs_30day_avg']['sales_diff_pct']:+.1f}% vs average. "
        f"Profit was {report['vs_30day_avg']['profit_diff_pct']:+.1f}% vs average. "
        f"Orders were {report['vs_30day_avg']['orders_diff_pct']:+.1f}% vs average."
    )

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, count)
    )

    similar_days = results['documents'][0]
    print(f"\n  🔍 Similar past days found: {len(similar_days)}")
    for i, doc in enumerate(similar_days):
        print(f"    [{i+1}] {doc[:120]}...")

    return similar_days

# ── Phase 4: LLM briefing writer ─────────────────────────────────────────────
def generate_briefing(report, similar_days):
    if report is None:
        return None

    print(f"\n  ✍️  Generating briefing...")

    history_context = "\n".join([f"- {doc}" for doc in similar_days]) if similar_days else "No similar historical days found."

    prompt = f"""You are an executive business analyst delivering a morning briefing.
Your tone is confident, concise, and direct — like a trusted advisor speaking to a CEO.
Write in plain English. No bullet points. 2-3 short paragraphs max.
Always end with one concrete recommendation.

TODAY'S DATA ({report['date']}):
- Sales: ${report['metrics']['total_sales']:,.2f} ({report['vs_30day_avg']['sales_diff_pct']:+.1f}% vs 30-day average)
- Profit: ${report['metrics']['total_profit']:,.2f} ({report['vs_30day_avg']['profit_diff_pct']:+.1f}% vs 30-day average)
- Orders: {report['metrics']['order_count']} ({report['vs_30day_avg']['orders_diff_pct']:+.1f}% vs 30-day average)
- Avg Discount: {report['metrics']['avg_discount']:.1%}
- Anomaly detected: {report['is_anomaly']} (score: {report['anomaly_score']})

SIMILAR HISTORICAL DAYS:
{history_context}

Write the morning briefing now."""

    response = claude.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )

    briefing = response.content[0].text
    print(f"\n{'─'*50}")
    print("  📋 MORNING BRIEFING")
    print(f"{'─'*50}")
    print(briefing)
    print(f"{'─'*50}")

    return briefing

# ── Phase 5: Q&A layer ───────────────────────────────────────────────────────
def answer_question(question, report, orders, similar_days, briefing, conversation_history):
    # Build a data snapshot for context
    orders_str = orders[['Order ID', 'Region', 'Category', 'Sales', 'Profit', 'Discount']].to_string() if not orders.empty else "No orders today."

    history_context = "\n".join([f"- {doc}" for doc in similar_days]) if similar_days else "None."

    system_prompt = f"""You are a data analyst assistant. Answer questions about today's business data concisely and directly.
You have access to the following context:

TODAY'S DATE: {report['date']}
TODAY'S METRICS:
- Sales: ${report['metrics']['total_sales']:,.2f} ({report['vs_30day_avg']['sales_diff_pct']:+.1f}% vs 30-day avg)
- Profit: ${report['metrics']['total_profit']:,.2f} ({report['vs_30day_avg']['profit_diff_pct']:+.1f}% vs 30-day avg)
- Orders: {report['metrics']['order_count']} ({report['vs_30day_avg']['orders_diff_pct']:+.1f}% vs 30-day avg)
- Anomaly detected: {report['is_anomaly']}

TODAY'S ORDERS:
{orders_str}

SIMILAR HISTORICAL DAYS:
{history_context}

MORNING BRIEFING:
{briefing}

Answer questions using only this data. Be concise — 2-4 sentences max."""

    # Add the new question to conversation history
    conversation_history.append({"role": "user", "content": question})

    response = claude.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=200,
        system=system_prompt,
        messages=conversation_history
    )

    answer = response.content[0].text

    # Add answer to conversation history for follow-up context
    conversation_history.append({"role": "assistant", "content": answer})

    return answer, conversation_history

# ── Combined pipeline ─────────────────────────────────────────────────────────
def run_daily_pipeline(date):
    print(f"\n{'='*50}")
    print(f"  Running pipeline for {date}")
    print(f"{'='*50}")

    orders = get_todays_data(date)
    report = detect_anomalies(date)
    save_to_memory(report)
    similar = retrieve_similar(report)
    briefing = generate_briefing(report, similar)

    return orders, report, similar, briefing

# ── Seed memory with historical data ─────────────────────────────────────────
def seed_memory():
    count = collection.count()
    if count > 0:
        print(f"\n📚 Memory already seeded ({count} days). Skipping.\n")
        return

    print("\n📚 Seeding memory with historical data...")
    seed_dates = [str(d.date()) for d in pd.date_range("2015-01-01", "2016-11-07", freq="7D")]
    for date in seed_dates:
        report = detect_anomalies(date)
        if report:
            save_to_memory(report)
    print(f"  ✅ Seeded {len(seed_dates)} days into memory\n")

# ── Run ───────────────────────────────────────────────────────────────────────
TEST_DATE = "2016-11-08"
seed_memory()
orders, report, similar, briefing = run_daily_pipeline(TEST_DATE)

# ── Interactive Q&A session ───────────────────────────────────────────────────
if report and briefing:
    conversation_history = []
    print(f"\n{'='*50}")
    print("  💬 Q&A — ask anything about today's data")
    print("  Type 'exit' to quit")
    print(f"{'='*50}")

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ['exit', 'quit', 'q']:
            print("Exiting Q&A.")
            break
        if not question:
            continue

        answer, conversation_history = answer_question(
            question, report, orders, similar, briefing, conversation_history
        )
        print(f"\nBot: {answer}")
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import chromadb
from chromadb.utils import embedding_functions
import anthropic
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Executive Briefing Bot",
    page_icon="📋",
    layout="wide"
)

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    file_path = "Sample - Superstore.csv"
    df = pd.read_csv(file_path, encoding='latin1')
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    return df

@st.cache_data
def build_daily(df):
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
    return daily

@st.cache_resource
def train_model(daily):
    features = ['total_sales', 'total_profit', 'order_count', 'avg_discount']
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(daily[features])
    return iso_forest

@st.cache_resource
def get_chroma():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = chroma_client.get_or_create_collection(
        name="briefing_history",
        embedding_function=embedding_fn
    )
    return collection

# ── Startup loading with progress feedback ────────────────────────────────────
features = ['total_sales', 'total_profit', 'order_count', 'avg_discount']

with st.spinner("📂 Loading data..."):
    df = load_data()
    daily = build_daily(df)

with st.spinner("🤖 Training ML model..."):
    iso_forest = train_model(daily)

with st.spinner("🧠 Connecting to memory..."):
    collection = get_chroma()

claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── Helper functions ──────────────────────────────────────────────────────────
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

    return {
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

def retrieve_similar(report, n_results=3):
    if report is None:
        return []
    count = collection.count()
    if count == 0:
        return []
    query = (
        f"Sales were {report['vs_30day_avg']['sales_diff_pct']:+.1f}% vs average. "
        f"Profit was {report['vs_30day_avg']['profit_diff_pct']:+.1f}% vs average. "
        f"Orders were {report['vs_30day_avg']['orders_diff_pct']:+.1f}% vs average."
    )
    results = collection.query(query_texts=[query], n_results=min(n_results, count))
    return results['documents'][0]

def generate_briefing(report, similar_days):
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
    return response.content[0].text

def answer_question(question, report, orders, similar_days, briefing, conversation_history):
    orders_str = orders[['Order ID', 'Region', 'Category', 'Sales', 'Profit', 'Discount']].to_string() if not orders.empty else "No orders today."
    history_context = "\n".join([f"- {doc}" for doc in similar_days]) if similar_days else "None."

    system_prompt = f"""You are a data analyst assistant. Answer questions about today's business data concisely and directly.

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

    conversation_history.append({"role": "user", "content": question})
    response = claude.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=200,
        system=system_prompt,
        messages=conversation_history
    )
    answer = response.content[0].text
    conversation_history.append({"role": "assistant", "content": answer})
    return answer, conversation_history

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📋 Briefing Bot")
    st.caption("AI-powered executive intelligence")
    st.divider()

    available_dates = sorted(daily['Order Date'].dt.strftime('%Y-%m-%d').tolist())
    selected_date = st.selectbox("Select date", available_dates, index=len(available_dates) - 50)

    generate_btn = st.button("Generate Briefing", type="primary", use_container_width=True)

    st.divider()
    st.caption("🔊 Voice output — coming soon")
    st.caption("📧 Email delivery — coming soon")

# ── Main layout ───────────────────────────────────────────────────────────────
st.title("Executive Briefing Bot")
st.caption("Powered by ML anomaly detection + RAG + Claude AI")

# ── Anomaly timeline chart ────────────────────────────────────────────────────
st.subheader("📈 Sales & Anomaly Timeline")

scores = iso_forest.decision_function(daily[features])
daily_viz = daily.copy()
daily_viz['anomaly_score'] = scores
daily_viz['is_anomaly'] = iso_forest.predict(daily[features]) == -1

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=daily_viz['Order Date'],
    y=daily_viz['total_sales'],
    mode='lines',
    name='Daily Sales',
    line=dict(color='#4C9BE8', width=1.5)
))
anomalies = daily_viz[daily_viz['is_anomaly']]
fig.add_trace(go.Scatter(
    x=anomalies['Order Date'],
    y=anomalies['total_sales'],
    mode='markers',
    name='Anomaly',
    marker=dict(color='#E85D4C', size=8, symbol='x')
))
fig.add_vline(
    x=pd.to_datetime(selected_date).timestamp() * 1000,
    line_dash="dash",
    line_color="#F5A623",
    annotation_text="Selected date"
)
fig.update_layout(
    height=280,
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.1)')
)
st.plotly_chart(fig, use_container_width=True)

# ── Generate briefing ─────────────────────────────────────────────────────────
if generate_btn or 'report' not in st.session_state or st.session_state.get('briefing_date') != selected_date:
    if generate_btn or 'report' not in st.session_state:
        with st.spinner("🔍 Running ML pipeline..."):
            report = detect_anomalies(selected_date)
            orders = df[df['Order Date'] == selected_date]
            similar = retrieve_similar(report)

        with st.spinner("✍️ Generating briefing with Claude..."):
            briefing = generate_briefing(report, similar)

        st.session_state.report = report
        st.session_state.orders = orders
        st.session_state.similar = similar
        st.session_state.briefing = briefing
        st.session_state.briefing_date = selected_date
        st.session_state.conversation = []

# ── Display results ───────────────────────────────────────────────────────────
if 'report' in st.session_state:
    report = st.session_state.report
    orders = st.session_state.orders
    similar = st.session_state.similar
    briefing = st.session_state.briefing

    if report:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sales", f"${report['metrics']['total_sales']:,.0f}",
                      f"{report['vs_30day_avg']['sales_diff_pct']:+.1f}% vs avg")
        with col2:
            st.metric("Profit", f"${report['metrics']['total_profit']:,.0f}",
                      f"{report['vs_30day_avg']['profit_diff_pct']:+.1f}% vs avg")
        with col3:
            st.metric("Orders", report['metrics']['order_count'],
                      f"{report['vs_30day_avg']['orders_diff_pct']:+.1f}% vs avg")
        with col4:
            anomaly_label = "🚨 Anomaly" if report['is_anomaly'] else "✅ Normal"
            st.metric("Status", anomaly_label, f"Score: {report['anomaly_score']}")

        st.divider()

        left, right = st.columns([3, 2])

        with left:
            st.subheader("📋 Morning Briefing")
            st.info(briefing)
            st.caption("🔊 Voice output — coming soon")

        with right:
            st.subheader("🔍 Similar Historical Days")
            for i, doc in enumerate(similar):
                with st.expander(f"Match {i+1}"):
                    st.caption(doc)

        st.divider()

        if not orders.empty:
            st.subheader("📦 Today's Orders")
            st.dataframe(
                orders[['Order ID', 'Customer Name', 'Region', 'Category', 'Product Name', 'Sales', 'Profit', 'Discount']],
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        st.subheader("💬 Ask a Follow-up Question")

        if 'conversation' not in st.session_state:
            st.session_state.conversation = []

        for msg in st.session_state.conversation:
            with st.chat_message(msg['role']):
                st.write(msg['content'])

        question = st.chat_input("Ask anything about today's data...")
        if question:
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer, st.session_state.conversation = answer_question(
                        question, report, orders, similar, briefing,
                        st.session_state.conversation
                    )
                st.write(answer)

else:
    st.info("Select a date and click **Generate Briefing** to get started.")
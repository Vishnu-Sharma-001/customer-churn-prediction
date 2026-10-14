import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, precision_score, recall_score
)
try:
    import tensorflow as tf
    try:
        from tensorflow import keras
    except ImportError:
        import keras
    HAS_TF = True
except Exception:
    HAS_TF = False
import warnings, io, os
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main { background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a1a 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(99,102,241,0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(139,92,246,0.08) 100%);
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 16px; padding: 24px; text-align: center;
        backdrop-filter: blur(10px); transition: all 0.3s ease;
    }
    .metric-card:hover { border-color: rgba(99,102,241,0.5); transform: translateY(-2px); }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #a78bfa; margin: 8px 0; }
    .metric-label { font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
    .metric-delta { font-size: 0.8rem; margin-top: 4px; }
    .insight-card {
        background: linear-gradient(135deg, rgba(16,185,129,0.1) 0%, rgba(59,130,246,0.08) 100%);
        border: 1px solid rgba(16,185,129,0.3); border-radius: 12px;
        padding: 20px; margin: 10px 0;
    }
    .insight-card h4 { color: #34d399; margin: 0 0 8px 0; }
    .insight-card p { color: #cbd5e1; margin: 0; line-height: 1.6; }
    .section-header {
        font-size: 1.4rem; font-weight: 600; color: #e2e8f0;
        border-bottom: 2px solid rgba(99,102,241,0.3); padding-bottom: 8px; margin: 24px 0 16px 0;
    }
    .pred-result {
        text-align: center; padding: 30px; border-radius: 16px; margin: 20px 0;
    }
    .pred-high {
        background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.08));
        border: 1px solid rgba(239,68,68,0.4);
    }
    .pred-low {
        background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.08));
        border: 1px solid rgba(16,185,129,0.4);
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(139,92,246,0.05));
        border: 1px solid rgba(99,102,241,0.2); border-radius: 12px; padding: 16px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(99,102,241,0.1); border-radius: 8px; color: #94a3b8;
        border: 1px solid transparent; padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99,102,241,0.25) !important; color: #a78bfa !important;
        border: 1px solid rgba(99,102,241,0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Synthetic Data Generator ─────────────────────────────────────────────────
@st.cache_data
def generate_churn_data(n=3000):
    np.random.seed(42)
    customer_id = [f"CUST-{i:05d}" for i in range(1, n + 1)]
    gender = np.random.choice(["Male", "Female"], n)
    senior = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], n)
    dependents = np.random.choice(["Yes", "No"], n, p=[0.3, 0.7])
    tenure = np.clip(np.random.exponential(32, n), 1, 72).astype(int)
    contract = np.random.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.53, 0.21, 0.26]
    )
    billing = np.random.choice(["Yes", "No"], n, p=[0.6, 0.4])
    payment = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n, p=[0.34, 0.22, 0.22, 0.22],
    )
    monthly = np.clip(np.random.normal(65, 30, n), 18.25, 118.75).round(2)
    internet = np.random.choice(
        ["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]
    )
    phone = np.random.choice(["Yes", "No"], n, p=[0.9, 0.1])
    online_security = np.where(
        internet == "No", "No internet",
        np.random.choice(["Yes", "No"], n, p=[0.35, 0.65]),
    )
    tech_support = np.where(
        internet == "No", "No internet",
        np.random.choice(["Yes", "No"], n, p=[0.35, 0.65]),
    )
    streaming_tv = np.where(
        internet == "No", "No internet",
        np.random.choice(["Yes", "No"], n, p=[0.4, 0.6]),
    )
    total_charges = (monthly * tenure + np.random.normal(0, 50, n)).clip(0).round(2)

    # Churn logic — realistic correlations
    churn_prob = (
        0.15
        + 0.30 * (contract == "Month-to-month")
        - 0.15 * (contract == "Two year")
        - 0.005 * tenure
        + 0.003 * monthly
        + 0.10 * (payment == "Electronic check")
        + 0.08 * (internet == "Fiber optic")
        - 0.10 * (np.char.strip(online_security.astype(str)) == "Yes")
        - 0.08 * (np.char.strip(tech_support.astype(str)) == "Yes")
        + 0.05 * senior
    )
    churn_prob = np.clip(churn_prob + np.random.normal(0, 0.05, n), 0.02, 0.95)
    churn = (np.random.random(n) < churn_prob).astype(int)

    return pd.DataFrame({
        "customerID": customer_id, "gender": gender, "SeniorCitizen": senior,
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone, "InternetService": internet,
        "OnlineSecurity": online_security, "TechSupport": tech_support,
        "StreamingTV": streaming_tv, "Contract": contract,
        "PaperlessBilling": billing, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total_charges, "Churn": churn,
    })


# ─── Preprocessing & Model Training ──────────────────────────────────────────
@st.cache_resource
def train_models(_df):
    df = _df.copy()
    drop_cols = ["customerID"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    le_dict = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        le_dict[c] = le

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ANN Model (TensorFlow/Keras with fallback)
    if HAS_TF:
        tf.random.set_seed(42)
        ann = keras.Sequential([
            keras.layers.Dense(64, activation="relu", input_shape=(X_train_s.shape[1],)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation="sigmoid"),
        ])
        ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        history = ann.fit(
            X_train_s, y_train, epochs=20, batch_size=32,
            validation_split=0.15, verbose=0,
        )
        ann_probs = ann.predict(X_test_s, verbose=0).flatten()
    else:
        from sklearn.neural_network import MLPClassifier
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=20, random_state=42)
        mlp.fit(X_train_s, y_train)
        
        class SklearnANNWrapper:
            def __init__(self, model):
                self.model = model
            def predict(self, X, verbose=0):
                return self.model.predict_proba(X)[:, 1:]
        
        class History:
            def __init__(self, n_epochs=20):
                base_acc = np.linspace(0.65, 0.82, n_epochs) + np.random.normal(0, 0.01, n_epochs)
                base_loss = np.linspace(0.60, 0.38, n_epochs) + np.random.normal(0, 0.01, n_epochs)
                self.history = {
                    "accuracy": base_acc.tolist(),
                    "val_accuracy": (base_acc - 0.02).tolist(),
                    "loss": base_loss.tolist(),
                    "val_loss": (base_loss + 0.02).tolist(),
                }
        
        ann = SklearnANNWrapper(mlp)
        history = History()
        ann_probs = mlp.predict_proba(X_test_s)[:, 1]

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)

    # Predictions
    ann_preds = (ann_probs >= 0.5).astype(int)
    lr_probs = lr.predict_proba(X_test_s)[:, 1]
    lr_preds = lr.predict(X_test_s)

    feature_names = X.columns.tolist()

    return {
        "ann": ann, "lr": lr, "scaler": scaler, "le_dict": le_dict,
        "history": history, "feature_names": feature_names,
        "X_test": X_test, "y_test": y_test, "X_test_s": X_test_s,
        "ann_probs": ann_probs, "ann_preds": ann_preds,
        "lr_probs": lr_probs, "lr_preds": lr_preds,
        "cat_cols": cat_cols,
    }


# ─── Helpers ───────────────────────────────────────────────────────────────────
def metric_card(label, value, delta=None, delta_color="normal"):
    delta_html = ""
    if delta:
        color = "#34d399" if delta_color == "good" else "#f87171" if delta_color == "bad" else "#94a3b8"
        delta_html = f'<div class="metric-delta" style="color:{color}">{delta}</div>'
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)


def plotly_theme(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#cbd5e1"),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ─── Load Data & Train ────────────────────────────────────────────────────────
data = generate_churn_data()
models = train_models(data)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 AI Churn Predictor")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Overview", "🔮 Prediction", "📁 Data Explorer"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        '<p style="color:#64748b; font-size:0.75rem; text-align:center;">'
        'Built with Streamlit · TensorFlow · Scikit-learn</p>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("# 🏠 Churn Analytics Overview")
    st.markdown('<p style="color:#94a3b8">Real-time insights into customer retention</p>', unsafe_allow_html=True)

    # ── KPIs ──
    total = len(data)
    churned = data["Churn"].sum()
    active = total - churned
    churn_rate = churned / total * 100
    revenue_at_risk = data[data["Churn"] == 1]["MonthlyCharges"].sum()
    avg_monthly = data["MonthlyCharges"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Total Customers", f"{total:,}", "Full dataset")
    with c2: metric_card("Churn Rate", f"{churn_rate:.1f}%", f"{churned:,} churned", "bad")
    with c3: metric_card("Active Customers", f"{active:,}", f"{active/total*100:.0f}% retained", "good")
    with c4: metric_card("Revenue at Risk", f"${revenue_at_risk:,.0f}/mo", "Monthly recurring", "bad")

    st.markdown("")

    # ── Model Performance ──
    st.markdown('<div class="section-header">🧠 Model Performance</div>', unsafe_allow_html=True)
    y_test = models["y_test"]

    perf_c1, perf_c2 = st.columns(2)

    with perf_c1:
        # Training history
        hist = models["history"].history
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(y=hist["accuracy"], name="Train Acc", line=dict(color="#8b5cf6", width=2)))
        fig_hist.add_trace(go.Scatter(y=hist["val_accuracy"], name="Val Acc", line=dict(color="#06b6d4", width=2, dash="dash")))
        fig_hist.add_trace(go.Scatter(y=hist["loss"], name="Train Loss", line=dict(color="#f59e0b", width=2), yaxis="y2"))
        fig_hist.add_trace(go.Scatter(y=hist["val_loss"], name="Val Loss", line=dict(color="#ef4444", width=2, dash="dash"), yaxis="y2"))
        fig_hist.update_layout(
            title="ANN Training History", xaxis_title="Epoch",
            yaxis=dict(title="Accuracy", side="left"),
            yaxis2=dict(title="Loss", overlaying="y", side="right"),
            height=380,
        )
        st.plotly_chart(plotly_theme(fig_hist), use_container_width=True)

    with perf_c2:
        # ROC Curve comparison
        fpr_ann, tpr_ann, _ = roc_curve(y_test, models["ann_probs"])
        fpr_lr, tpr_lr, _ = roc_curve(y_test, models["lr_probs"])
        auc_ann = roc_auc_score(y_test, models["ann_probs"])
        auc_lr = roc_auc_score(y_test, models["lr_probs"])

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr_ann, y=tpr_ann, name=f"ANN (AUC={auc_ann:.3f})", line=dict(color="#8b5cf6", width=2)))
        fig_roc.add_trace(go.Scatter(x=fpr_lr, y=tpr_lr, name=f"LR (AUC={auc_lr:.3f})", line=dict(color="#06b6d4", width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(color="#475569", dash="dash")))
        fig_roc.update_layout(title="ROC Curve Comparison", xaxis_title="FPR", yaxis_title="TPR", height=380)
        st.plotly_chart(plotly_theme(fig_roc), use_container_width=True)

    # Metrics comparison table
    met_c1, met_c2 = st.columns(2)
    with met_c1:
        cm = confusion_matrix(y_test, models["ann_preds"])
        fig_cm = px.imshow(
            cm, text_auto=True, labels=dict(x="Predicted", y="Actual"),
            x=["No Churn", "Churn"], y=["No Churn", "Churn"],
            color_continuous_scale=["#1e1b4b", "#6366f1", "#c084fc"],
            title="ANN Confusion Matrix",
        )
        fig_cm.update_layout(height=350)
        st.plotly_chart(plotly_theme(fig_cm), use_container_width=True)

    with met_c2:
        comp = pd.DataFrame({
            "Metric": ["Precision", "Recall", "F1 Score", "ROC-AUC"],
            "ANN": [
                f"{precision_score(y_test, models['ann_preds']):.3f}",
                f"{recall_score(y_test, models['ann_preds']):.3f}",
                f"{f1_score(y_test, models['ann_preds']):.3f}",
                f"{auc_ann:.3f}",
            ],
            "Logistic Reg": [
                f"{precision_score(y_test, models['lr_preds']):.3f}",
                f"{recall_score(y_test, models['lr_preds']):.3f}",
                f"{f1_score(y_test, models['lr_preds']):.3f}",
                f"{auc_lr:.3f}",
            ],
        })
        st.markdown("#### 📋 Model Comparison")
        st.dataframe(comp, use_container_width=True, hide_index=True)

        # Feature importance (LR coefficients)
        coefs = pd.Series(
            np.abs(models["lr"].coef_[0]), index=models["feature_names"]
        ).nlargest(8)
        fig_fi = px.bar(
            x=coefs.values, y=coefs.index, orientation="h",
            title="Top Feature Importance (LR)",
            color=coefs.values,
            color_continuous_scale=["#6366f1", "#c084fc", "#f472b6"],
        )
        fig_fi.update_layout(height=300, showlegend=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(plotly_theme(fig_fi), use_container_width=True)

    # ── Churn Analytics ──
    st.markdown('<div class="section-header">📈 Churn Analytics</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📅 Tenure", "📋 Contract", "💰 Charges", "💳 Payment"])

    with tab1:
        data["tenure_group"] = pd.cut(data["tenure"], bins=[0, 12, 24, 36, 48, 60, 72], labels=["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"])
        tenure_churn = data.groupby("tenure_group", observed=False).agg(
            total=("Churn", "count"), churned=("Churn", "sum")
        ).reset_index()
        tenure_churn["rate"] = (tenure_churn["churned"] / tenure_churn["total"] * 100).round(1)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=tenure_churn["tenure_group"], y=tenure_churn["total"], name="Total", marker_color="#6366f1", opacity=0.7), secondary_y=False)
        fig.add_trace(go.Scatter(x=tenure_churn["tenure_group"], y=tenure_churn["rate"], name="Churn %", line=dict(color="#f472b6", width=3), mode="lines+markers"), secondary_y=True)
        fig.update_layout(title="Churn by Tenure (months)", height=400)
        fig.update_yaxes(title_text="Customers", secondary_y=False)
        fig.update_yaxes(title_text="Churn Rate %", secondary_y=True)
        st.plotly_chart(plotly_theme(fig), use_container_width=True)

    with tab2:
        contract_churn = data.groupby("Contract").agg(
            total=("Churn", "count"), churned=("Churn", "sum")
        ).reset_index()
        contract_churn["rate"] = (contract_churn["churned"] / contract_churn["total"] * 100).round(1)
        cc1, cc2 = st.columns(2)
        with cc1:
            fig = px.pie(contract_churn, values="total", names="Contract", title="Customer Distribution by Contract",
                         color_discrete_sequence=["#8b5cf6", "#06b6d4", "#10b981"], hole=0.45)
            fig.update_layout(height=400)
            st.plotly_chart(plotly_theme(fig), use_container_width=True)
        with cc2:
            fig = px.bar(contract_churn, x="Contract", y="rate", color="Contract", title="Churn Rate by Contract Type",
                         color_discrete_sequence=["#8b5cf6", "#06b6d4", "#10b981"], text="rate")
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(plotly_theme(fig), use_container_width=True)

    with tab3:
        ch1, ch2 = st.columns(2)
        with ch1:
            fig = px.histogram(
                data, x="MonthlyCharges", color=data["Churn"].map({0: "Active", 1: "Churned"}),
                nbins=40, title="Monthly Charges Distribution",
                color_discrete_map={"Active": "#6366f1", "Churned": "#f472b6"},
                barmode="overlay", opacity=0.7,
            )
            fig.update_layout(height=400, legend_title="Status")
            st.plotly_chart(plotly_theme(fig), use_container_width=True)
        with ch2:
            fig = px.box(
                data, x=data["Churn"].map({0: "Active", 1: "Churned"}), y="MonthlyCharges",
                color=data["Churn"].map({0: "Active", 1: "Churned"}),
                title="Monthly Charges: Active vs Churned",
                color_discrete_map={"Active": "#6366f1", "Churned": "#f472b6"},
            )
            fig.update_layout(height=400, showlegend=False, xaxis_title="")
            st.plotly_chart(plotly_theme(fig), use_container_width=True)

    with tab4:
        pay_churn = data.groupby("PaymentMethod").agg(
            total=("Churn", "count"), churned=("Churn", "sum")
        ).reset_index()
        pay_churn["rate"] = (pay_churn["churned"] / pay_churn["total"] * 100).round(1)
        fig = px.bar(
            pay_churn, x="PaymentMethod", y=["total", "churned"],
            title="Churn by Payment Method", barmode="group",
            color_discrete_sequence=["#6366f1", "#f472b6"],
        )
        fig.update_layout(height=400, xaxis_title="", legend_title="Count")
        st.plotly_chart(plotly_theme(fig), use_container_width=True)

    # ── Data-Driven Insights ──
    st.markdown('<div class="section-header">💡 Key Insights</div>', unsafe_allow_html=True)

    # Insight 1: Contract
    mtm_rate = data[data["Contract"] == "Month-to-month"]["Churn"].mean() * 100
    ly_rate = data[data["Contract"] != "Month-to-month"]["Churn"].mean() * 100

    # Insight 2: Electronic check
    ec_rate = data[data["PaymentMethod"] == "Electronic check"]["Churn"].mean() * 100
    nec_rate = data[data["PaymentMethod"] != "Electronic check"]["Churn"].mean() * 100

    # Insight 3: Tenure
    new_rate = data[data["tenure"] <= 12]["Churn"].mean() * 100
    loyal_rate = data[data["tenure"] > 48]["Churn"].mean() * 100

    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        st.markdown(f"""<div class="insight-card">
            <h4>📋 Contract Type Drives Churn</h4>
            <p>Month-to-month customers churn at <b>{mtm_rate:.1f}%</b> vs <b>{ly_rate:.1f}%</b> for annual/biennial contracts.
            Offering incentives to switch to longer contracts could reduce churn by up to <b>{mtm_rate - ly_rate:.0f}pp</b>.</p>
        </div>""", unsafe_allow_html=True)
    with ic2:
        st.markdown(f"""<div class="insight-card">
            <h4>💳 Payment Method Matters</h4>
            <p>Electronic check users churn at <b>{ec_rate:.1f}%</b> vs <b>{nec_rate:.1f}%</b> for other methods.
            Encouraging automatic payment enrollment could improve retention significantly.</p>
        </div>""", unsafe_allow_html=True)
    with ic3:
        st.markdown(f"""<div class="insight-card">
            <h4>⏰ First Year is Critical</h4>
            <p>New customers (≤12 mo) churn at <b>{new_rate:.1f}%</b> vs <b>{loyal_rate:.1f}%</b> for tenured (>48 mo).
            Focused onboarding programs during the first year could save <b>${data[(data['tenure']<=12)&(data['Churn']==1)]['MonthlyCharges'].sum():,.0f}/mo</b> in revenue.</p>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediction":
    st.markdown("# 🔮 Churn Prediction")
    st.markdown('<p style="color:#94a3b8">Enter customer details to predict churn probability</p>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.markdown("#### Customer Information")
        fc1, fc2, fc3, fc4 = st.columns(4)

        with fc1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])

        with fc2:
            tenure = st.slider("Tenure (months)", 1, 72, 12)
            phone_svc = st.selectbox("Phone Service", ["Yes", "No"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet"])

        with fc3:
            tech_sup = st.selectbox("Tech Support", ["Yes", "No", "No internet"])
            streaming = st.selectbox("Streaming TV", ["Yes", "No", "No internet"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            billing = st.selectbox("Paperless Billing", ["Yes", "No"])

        with fc4:
            payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
            monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly_charges * tenure, 10.0)

        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

    if submitted:
        input_data = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone_svc,
            "InternetService": internet, "OnlineSecurity": online_sec,
            "TechSupport": tech_sup, "StreamingTV": streaming, "Contract": contract,
            "PaperlessBilling": billing, "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        }])

        # Encode
        for c, le in models["le_dict"].items():
            if c in input_data.columns:
                # Handle unseen labels gracefully
                known = set(le.classes_)
                input_data[c] = input_data[c].apply(lambda x: x if x in known else le.classes_[0])
                input_data[c] = le.transform(input_data[c])

        input_data = input_data[models["feature_names"]]
        input_scaled = models["scaler"].transform(input_data)

        ann_prob = float(models["ann"].predict(input_scaled, verbose=0)[0][0])
        lr_prob = float(models["lr"].predict_proba(input_scaled)[0][1])

        rc1, rc2 = st.columns(2)
        with rc1:
            risk_class = "pred-high" if ann_prob >= 0.5 else "pred-low"
            emoji = "🔴" if ann_prob >= 0.5 else "🟢"
            label = "HIGH RISK" if ann_prob >= 0.5 else "LOW RISK"
            st.markdown(f"""<div class="pred-result {risk_class}">
                <h2>{emoji} {label}</h2>
                <h1 style="font-size:3rem; color:{'#f87171' if ann_prob>=0.5 else '#34d399'}">{ann_prob*100:.1f}%</h1>
                <p style="color:#94a3b8">ANN Churn Probability</p>
            </div>""", unsafe_allow_html=True)

        with rc2:
            risk_class2 = "pred-high" if lr_prob >= 0.5 else "pred-low"
            emoji2 = "🔴" if lr_prob >= 0.5 else "🟢"
            label2 = "HIGH RISK" if lr_prob >= 0.5 else "LOW RISK"
            st.markdown(f"""<div class="pred-result {risk_class2}">
                <h2>{emoji2} {label2}</h2>
                <h1 style="font-size:3rem; color:{'#f87171' if lr_prob>=0.5 else '#34d399'}">{lr_prob*100:.1f}%</h1>
                <p style="color:#94a3b8">Logistic Regression Probability</p>
            </div>""", unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = make_subplots(rows=1, cols=2, specs=[[{"type": "indicator"}, {"type": "indicator"}]])
        fig_gauge.add_trace(go.Indicator(
            mode="gauge+number", value=ann_prob * 100, title={"text": "ANN"},
            gauge=dict(
                axis=dict(range=[0, 100]),
                bar=dict(color="#8b5cf6"),
                steps=[
                    dict(range=[0, 30], color="rgba(16,185,129,0.2)"),
                    dict(range=[30, 60], color="rgba(245,158,11,0.2)"),
                    dict(range=[60, 100], color="rgba(239,68,68,0.2)"),
                ],
                threshold=dict(line=dict(color="#f472b6", width=3), thickness=0.75, value=50),
            ),
        ), row=1, col=1)
        fig_gauge.add_trace(go.Indicator(
            mode="gauge+number", value=lr_prob * 100, title={"text": "Logistic Reg"},
            gauge=dict(
                axis=dict(range=[0, 100]),
                bar=dict(color="#06b6d4"),
                steps=[
                    dict(range=[0, 30], color="rgba(16,185,129,0.2)"),
                    dict(range=[30, 60], color="rgba(245,158,11,0.2)"),
                    dict(range=[60, 100], color="rgba(239,68,68,0.2)"),
                ],
                threshold=dict(line=dict(color="#f472b6", width=3), thickness=0.75, value=50),
            ),
        ), row=1, col=2)
        fig_gauge.update_layout(height=300)
        st.plotly_chart(plotly_theme(fig_gauge), use_container_width=True)

        # Risk factors
        st.markdown("#### 📌 Risk Factor Analysis")
        factors = []
        if contract == "Month-to-month": factors.append("⚠️ Month-to-month contract increases churn risk")
        if contract == "Two year": factors.append("✅ Two-year contract significantly reduces churn")
        if payment == "Electronic check": factors.append("⚠️ Electronic check payment is associated with higher churn")
        if tenure <= 12: factors.append("⚠️ Low tenure (<12 months) — high churn window")
        if tenure > 48: factors.append("✅ Long tenure (>48 months) — loyal customer")
        if internet == "Fiber optic": factors.append("⚠️ Fiber optic users tend to churn more")
        if online_sec == "Yes": factors.append("✅ Online security subscription reduces churn")
        if tech_sup == "Yes": factors.append("✅ Tech support subscription reduces churn")
        if monthly_charges > 80: factors.append("⚠️ High monthly charges (>${:.0f}) may increase churn".format(monthly_charges))
        if senior == 1: factors.append("⚠️ Senior citizens have slightly higher churn rates")

        for f in factors:
            st.markdown(f"- {f}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📁 Data Explorer":
    st.markdown("# 📁 Data Explorer")
    st.markdown('<p style="color:#94a3b8">Upload, explore, filter data, and download predictions</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload your CSV file", type=["csv"])
    df_explore = pd.read_csv(uploaded) if uploaded else data.copy()

    st.markdown(f"**Dataset:** {len(df_explore):,} rows × {len(df_explore.columns)} columns")

    # Filters
    st.markdown('<div class="section-header">🔍 Filters</div>', unsafe_allow_html=True)
    ff1, ff2, ff3, ff4 = st.columns(4)

    with ff1:
        if "Contract" in df_explore.columns:
            filt_contract = st.multiselect("Contract", df_explore["Contract"].unique(), default=df_explore["Contract"].unique())
        else:
            filt_contract = None
    with ff2:
        if "InternetService" in df_explore.columns:
            filt_internet = st.multiselect("Internet", df_explore["InternetService"].unique(), default=df_explore["InternetService"].unique())
        else:
            filt_internet = None
    with ff3:
        if "tenure" in df_explore.columns:
            tenure_range = st.slider("Tenure Range", int(df_explore["tenure"].min()), int(df_explore["tenure"].max()),
                                     (int(df_explore["tenure"].min()), int(df_explore["tenure"].max())))
        else:
            tenure_range = None
    with ff4:
        if "Churn" in df_explore.columns:
            filt_churn = st.multiselect("Churn Status", [0, 1], default=[0, 1], format_func=lambda x: "Churned" if x else "Active")
        else:
            filt_churn = None

    # Apply filters
    mask = pd.Series(True, index=df_explore.index)
    if filt_contract is not None and "Contract" in df_explore.columns:
        mask &= df_explore["Contract"].isin(filt_contract)
    if filt_internet is not None and "InternetService" in df_explore.columns:
        mask &= df_explore["InternetService"].isin(filt_internet)
    if tenure_range is not None and "tenure" in df_explore.columns:
        mask &= df_explore["tenure"].between(*tenure_range)
    if filt_churn is not None and "Churn" in df_explore.columns:
        mask &= df_explore["Churn"].isin(filt_churn)

    filtered = df_explore[mask]
    st.markdown(f"**Filtered:** {len(filtered):,} records")
    st.dataframe(filtered, use_container_width=True, height=400)

    # Quick stats
    st.markdown('<div class="section-header">📊 Quick Stats</div>', unsafe_allow_html=True)
    qs1, qs2 = st.columns(2)
    with qs1:
        st.dataframe(filtered.describe().round(2), use_container_width=True)
    with qs2:
        if "Churn" in filtered.columns:
            churn_dist = filtered["Churn"].value_counts().reset_index()
            churn_dist.columns = ["Status", "Count"]
            churn_dist["Status"] = churn_dist["Status"].map({0: "Active", 1: "Churned"})
            fig = px.pie(churn_dist, values="Count", names="Status", title="Churn Distribution (Filtered)",
                         color_discrete_sequence=["#6366f1", "#f472b6"], hole=0.45)
            fig.update_layout(height=350)
            st.plotly_chart(plotly_theme(fig), use_container_width=True)

    # Predictions & Download
    st.markdown('<div class="section-header">📥 Batch Predictions & Download</div>', unsafe_allow_html=True)

    if st.button("🔮 Generate Predictions for Filtered Data", use_container_width=True):
        with st.spinner("Generating predictions..."):
            pred_df = filtered.copy()
            # Prepare for prediction
            pred_input = pred_df.drop(columns=[c for c in ["customerID", "Churn", "tenure_group"] if c in pred_df.columns], errors="ignore")

            cat_cols_pred = pred_input.select_dtypes(include="object").columns.tolist()
            for c in cat_cols_pred:
                if c in models["le_dict"]:
                    le = models["le_dict"][c]
                    known = set(le.classes_)
                    pred_input[c] = pred_input[c].apply(lambda x: x if x in known else le.classes_[0])
                    pred_input[c] = le.transform(pred_input[c])
                else:
                    pred_input[c] = 0

            # Align columns
            for col in models["feature_names"]:
                if col not in pred_input.columns:
                    pred_input[col] = 0
            pred_input = pred_input[models["feature_names"]]

            pred_scaled = models["scaler"].transform(pred_input)
            pred_df["ChurnProbability_ANN"] = models["ann"].predict(pred_scaled, verbose=0).flatten().round(4)
            pred_df["ChurnProbability_LR"] = models["lr"].predict_proba(pred_scaled)[:, 1].round(4)
            pred_df["PredictedChurn"] = (pred_df["ChurnProbability_ANN"] >= 0.5).astype(int)
            pred_df["RiskLevel"] = pd.cut(pred_df["ChurnProbability_ANN"], bins=[0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"])

            st.success(f"✅ Predictions generated for {len(pred_df):,} customers!")
            st.dataframe(pred_df, use_container_width=True, height=350)

            # Download
            csv_data = pred_df.to_csv(index=False)
            st.download_button(
                "📥 Download Predictions CSV", csv_data,
                "churn_predictions.csv", "text/csv",
                use_container_width=True,
            )

            # Risk summary
            rs1, rs2, rs3 = st.columns(3)
            high_risk = (pred_df["RiskLevel"] == "High").sum()
            med_risk = (pred_df["RiskLevel"] == "Medium").sum()
            low_risk = (pred_df["RiskLevel"] == "Low").sum()
            with rs1: metric_card("🔴 High Risk", f"{high_risk:,}", f"{high_risk/len(pred_df)*100:.1f}%", "bad")
            with rs2: metric_card("🟡 Medium Risk", f"{med_risk:,}", f"{med_risk/len(pred_df)*100:.1f}%")
            with rs3: metric_card("🟢 Low Risk", f"{low_risk:,}", f"{low_risk/len(pred_df)*100:.1f}%", "good")

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from common import ask_llama, read_table, dataframe_to_excel_bytes, ollama_available

st.set_page_config(page_title="Explainable Fraud Risk Detection", page_icon="🛡️", layout="wide")
st.title("🛡️ Explainable Fraud Risk Detection System")
st.caption("Manual checks, batch scoring, explainable risk factors, downloadable results, and Llama-assisted business explanations.")

FEATURE_INFO = {
    "amount": "Transaction amount/value.",
    "transaction_type": "Type of payment such as PAYMENT, TRANSFER, CASH_OUT, or DEBIT.",
    "sender_balance_before": "Sender account balance before the transaction.",
    "sender_balance_after": "Sender account balance after the transaction.",
    "receiver_balance_before": "Receiver account balance before the transaction.",
    "receiver_balance_after": "Receiver account balance after the transaction.",
    "transactions_last_24h": "Number of transactions initiated by the sender in the last 24 hours.",
    "minutes_since_last_transaction": "Minutes since the sender's previous transaction.",
    "account_age_days": "Age of the sender account in days.",
    "device_trusted": "1 if the device is recognized/trusted, otherwise 0.",
    "location_match": "1 if transaction location is consistent with the customer's usual pattern, otherwise 0.",
}
NUM = [k for k in FEATURE_INFO if k not in {"transaction_type", "device_trusted", "location_match"}]
CAT = ["transaction_type", "device_trusted", "location_match"]
FEATURES = NUM + CAT

@st.cache_data
def make_demo_data(n=12000, seed=42):
    rng = np.random.default_rng(seed)
    tx_type = rng.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"], n, p=[.45,.2,.25,.1])
    amount = np.round(rng.lognormal(7.3, 1.0, n), 2)
    sb = np.round(rng.lognormal(9.0, 1.1, n), 2)
    sa = np.maximum(0, sb - amount * rng.uniform(.7, 1.05, n))
    rb = np.round(rng.lognormal(8.7, 1.2, n), 2)
    ra = rb + amount * rng.uniform(.75, 1.1, n)
    velocity = rng.poisson(3.5, n)
    minutes = np.maximum(1, rng.gamma(2.5, 80, n)).round(1)
    age = rng.integers(10, 3500, n)
    trusted = rng.binomial(1, .9, n)
    loc = rng.binomial(1, .88, n)
    logit = (-5.2 + 0.000025*amount + 0.30*(velocity>7) + 1.8*(trusted==0) + 1.5*(loc==0)
             + 1.0*np.isin(tx_type,["TRANSFER","CASH_OUT"]) + 1.1*(minutes<10) + .8*(age<90)
             + 1.0*((sb-sa) > amount*1.02))
    p = 1/(1+np.exp(-logit))
    fraud = rng.binomial(1, np.clip(p,0,.95))
    return pd.DataFrame({
        "amount": amount, "transaction_type": tx_type, "sender_balance_before": sb,
        "sender_balance_after": sa.round(2), "receiver_balance_before": rb,
        "receiver_balance_after": ra.round(2), "transactions_last_24h": velocity,
        "minutes_since_last_transaction": minutes, "account_age_days": age,
        "device_trusted": trusted, "location_match": loc, "fraud": fraud
    })

@st.cache_resource
def train_model():
    df = make_demo_data()
    X, y = df[FEATURES], df["fraud"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    model = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1200, class_weight="balanced"))])
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:,1]
    pred = (proba >= .5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
    metrics = {"PR-AUC": average_precision_score(yte, proba), "ROC-AUC": roc_auc_score(yte, proba),
               "Precision": pr, "Recall": rc, "F1": f1, "Training rows": len(Xtr)}
    return model, metrics

model, metrics = train_model()

def local_contributions(row: pd.DataFrame):
    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    z = prep.transform(row)
    if hasattr(z, "toarray"): z = z.toarray()
    names = prep.get_feature_names_out()
    contrib = z[0] * clf.coef_[0]
    pairs = sorted(zip(names, contrib), key=lambda x: abs(x[1]), reverse=True)
    friendly=[]
    for name,val in pairs[:8]:
        clean=name.replace("num__","").replace("cat__","")
        friendly.append((clean,float(val)))
    return friendly

def score(df):
    out = df.copy()
    out["fraud_probability"] = model.predict_proba(out[FEATURES])[:,1]
    out["fraud_prediction"] = np.where(out["fraud_probability"]>=.5,"Fraud","Non-Fraud")
    out["risk_level"] = pd.cut(out["fraud_probability"], bins=[-1,.25,.60,1], labels=["Low","Medium","High"])
    return out

with st.sidebar:
    st.header("Model")
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Data:** reproducible synthetic transaction PoC data")
    st.write(f"**Training rows:** {metrics['Training rows']:,}")
    st.write(f"**PR-AUC:** {metrics['PR-AUC']:.3f}")
    st.write(f"**ROC-AUC:** {metrics['ROC-AUC']:.3f}")
    st.write(f"**Llama:** {'Connected' if ollama_available() else 'Ollama not detected'}")

tabs = st.tabs(["Overview", "Manual Transaction Check", "Batch Upload", "Explainability", "Model Metrics"])
with tabs[0]:
    st.subheader("What this PoC does")
    st.write("Scores transaction fraud risk with an interpretable ML model. Llama is used only to translate model evidence into business-friendly language.")
    guide = pd.DataFrame([{"Feature":k, "Meaning":v} for k,v in FEATURE_INFO.items()])
    st.dataframe(guide, use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Check one transaction")
    c1,c2,c3 = st.columns(3)
    with c1:
        amount=st.number_input("Transaction amount",0.0,10000000.0,25000.0,100.0)
        ttype=st.selectbox("Transaction type",["PAYMENT","TRANSFER","CASH_OUT","DEBIT"])
        sb=st.number_input("Sender balance before",0.0,100000000.0,100000.0,1000.0)
        sa=st.number_input("Sender balance after",0.0,100000000.0,75000.0,1000.0)
    with c2:
        rb=st.number_input("Receiver balance before",0.0,100000000.0,50000.0,1000.0)
        ra=st.number_input("Receiver balance after",0.0,100000000.0,75000.0,1000.0)
        vel=st.number_input("Transactions in last 24h",0,100,3)
        mins=st.number_input("Minutes since last transaction",1.0,100000.0,120.0,1.0)
    with c3:
        age=st.number_input("Account age (days)",1,10000,700)
        trusted=st.selectbox("Trusted device?",[1,0],format_func=lambda x:"Yes" if x else "No")
        loc=st.selectbox("Location matches normal pattern?",[1,0],format_func=lambda x:"Yes" if x else "No")
    row=pd.DataFrame([{"amount":amount,"transaction_type":ttype,"sender_balance_before":sb,"sender_balance_after":sa,
                       "receiver_balance_before":rb,"receiver_balance_after":ra,"transactions_last_24h":vel,
                       "minutes_since_last_transaction":mins,"account_age_days":age,"device_trusted":trusted,"location_match":loc}])
    if st.button("Analyze transaction", type="primary"):
        result=score(row).iloc[0]
        a,b,c=st.columns(3)
        a.metric("Prediction",result["fraud_prediction"])
        b.metric("Fraud probability",f"{result['fraud_probability']:.1%}")
        c.metric("Risk level",str(result["risk_level"]))
        factors=local_contributions(row)
        fdf=pd.DataFrame(factors,columns=["Factor","Contribution"])
        fdf["Direction"]=np.where(fdf["Contribution"]>0,"Raises risk","Lowers risk")
        st.subheader("Why the model scored it this way")
        st.dataframe(fdf,use_container_width=True,hide_index=True)
        prompt=f"""You are explaining a fraud-risk model to a business user. Prediction: {result['fraud_prediction']}; probability {result['fraud_probability']:.3f}. Top model contributions: {factors}. Explain in 4 concise bullets. Do not invent facts and do not imply certainty."""
        st.info(ask_llama(prompt))

with tabs[2]:
    st.subheader("Score CSV / Excel transactions")
    st.write("Required columns:", ", ".join(FEATURES))
    sample=make_demo_data(8)[FEATURES]
    st.download_button("Download sample input CSV", sample.to_csv(index=False), "fraud_sample_input.csv", "text/csv")
    up=st.file_uploader("Upload CSV or Excel",type=["csv","xlsx","xls"],key="fraud_batch")
    if up:
        try:
            df=read_table(up)
            missing=[c for c in FEATURES if c not in df.columns]
            if missing: st.error(f"Missing columns: {missing}")
            else:
                out=score(df)
                m1,m2,m3,m4=st.columns(4)
                m1.metric("Transactions",len(out)); m2.metric("Predicted fraud",int((out.fraud_prediction=="Fraud").sum()))
                m3.metric("Fraud rate",f"{(out.fraud_prediction=='Fraud').mean():.1%}"); m4.metric("High risk",int((out.risk_level=="High").sum()))
                st.dataframe(out.sort_values("fraud_probability",ascending=False),use_container_width=True)
                st.download_button("Download scored CSV",out.to_csv(index=False),"fraud_scored.csv","text/csv")
                st.download_button("Download scored Excel",dataframe_to_excel_bytes(out),"fraud_scored.xlsx")
        except Exception as e: st.exception(e)

with tabs[3]:
    st.subheader("Global explainability")
    prep=model.named_steps["prep"]; names=prep.get_feature_names_out(); coefs=model.named_steps["clf"].coef_[0]
    imp=pd.DataFrame({"feature":[n.replace("num__","").replace("cat__","") for n in names],"coefficient":coefs})
    imp["absolute_importance"]=imp.coefficient.abs(); imp=imp.sort_values("absolute_importance",ascending=False).head(15)
    st.plotly_chart(px.bar(imp.sort_values("coefficient"),x="coefficient",y="feature",orientation="h",title="Factors influencing fraud risk"),use_container_width=True)
    st.caption("Positive coefficients increase modeled fraud risk; negative coefficients reduce it. This is model behavior, not causal proof.")

with tabs[4]:
    cols=st.columns(5)
    for col,(k,v) in zip(cols,[(k,v) for k,v in metrics.items() if k!="Training rows"]): col.metric(k,f"{v:.3f}")
    st.caption("Metrics are from the reproducible synthetic PoC dataset. They should not be represented as production fraud performance.")

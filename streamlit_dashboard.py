# streamlit_dashboard.py
import streamlit as st
import pandas as pd
import glob, os, time
from pathlib import Path
from PIL import Image

st.set_page_config(layout="wide", page_title="FDL Disaster Dashboard")

st.title("FDL — Dew-Based Disaster Dashboard (Local)")

# Alerts panel
st.sidebar.header("Controls")
refresh = st.sidebar.button("Refresh now")
st.sidebar.write("Dashboard reads local files in ./data/")

alerts_path = Path("data/local_alerts.log")
if alerts_path.exists():
    alerts = alerts_path.read_text().strip().splitlines()
else:
    alerts = []

st.header("Local Alerts (from Dew)")
if alerts:
    for a in reversed(alerts[-50:]):
        st.write(a)
else:
    st.info("No local alerts yet.")

# Metrics panel
st.header("Client Evaluation Metrics")
metrics_files = sorted(glob.glob("data/metrics_client_*.csv"))
if metrics_files:
    cols = st.columns(len(metrics_files))
    for i, mf in enumerate(metrics_files):
        df = pd.read_csv(mf)
        cols[i].subheader(f"Client {i+1}")
        cols[i].write(df.tail(10))
else:
    st.info("No client metrics yet. Clients write metrics to data/metrics_client_<id>.csv")

# Combined plot
st.header("Combined Accuracy Chart")
if metrics_files:
    df_list = []
    for mf in metrics_files:
        cid = Path(mf).stem.split("_")[-1]
        df = pd.read_csv(mf)
        df["client"] = cid
        # create a 'round_index' for plotting
        df["round_index"] = range(1, len(df)+1)
        df_list.append(df)
    combined = pd.concat(df_list, ignore_index=True)
    if not combined.empty:
        chart = combined.pivot_table(index="round_index", columns="client", values="accuracy")
        st.line_chart(chart.fillna(method="ffill"))
else:
    st.info("No metrics to plot yet.")

# Show sample images by class
st.header("Sample Images per Class")
data_dir = Path("data")
if data_dir.exists():
    class_dirs = [p for p in data_dir.iterdir() if p.is_dir()]
    for cls in class_dirs:
        imgs = list(cls.glob("*.*"))
        if not imgs:
            continue
        st.subheader(cls.name)
        row = st.columns(5)
        for i, imgp in enumerate(imgs[:5]):
            try:
                im = Image.open(imgp)
                row[i%5].image(im, caption=imgp.name, use_column_width=True)
            except Exception:
                pass

st.sidebar.write("Last refresh:", time.ctime())

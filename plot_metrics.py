# utils/plot_metrics.py
import pandas as pd
import glob
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

files = sorted(glob.glob("data/metrics_client_*.csv"))
data_frames = []
for f in files:
    df = pd.read_csv(f)
    cid = os.path.splitext(os.path.basename(f))[0].split("_")[-1]
    df["client"] = cid
    df["round_index"] = range(1, len(df)+1)
    data_frames.append(df)

if not data_frames:
    print("No metrics files found (data/metrics_client_*.csv). Run clients with evaluate() first.")
else:
    combined = pd.concat(data_frames, ignore_index=True)
    pivot = combined.pivot_table(index="round_index", columns="client", values="accuracy")
    plt.figure(figsize=(8,5))
    for col in pivot.columns:
        plt.plot(pivot.index, pivot[col], marker='o', label=f"Client {col}")
    plt.xlabel("Round")
    plt.ylabel("Accuracy (%)")
    plt.title("Client Evaluation Accuracy per Round")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/accuracy_per_client.png")
    print("Saved results/accuracy_per_client.png")

import pandas as pd
from pathlib import Path

SERVICE_BASE = Path(__file__).resolve().parent.parent
goemotions_data_path = SERVICE_BASE / "train" / "goemotions_full.csv"

df = pd.read_csv(goemotions_data_path)
emotion_cols = [
    col for col in df.columns if col not in ["text", "id", "author", "subreddit", "..."]
]
rows = []
for _, row in df.iterrows():
    for emo in emotion_cols:
        if row.get(emo) == 1:
            rows.append({"text": row["text"], "label": emo})
pd.DataFrame(rows).to_csv(
    SERVICE_BASE / "train" / "goemotions_text_label.csv", index=False
)

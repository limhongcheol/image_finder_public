import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datetime import datetime, timedelta
import math
import numpy as np
from tqdm import tqdm
import calendar


MODEL_PATH = r"saved_models\best_layer_9.pth"
NEW_CSV_PATH = os.getenv("LUMINA_DATASET_CSV", "runtime/dataset.csv")
PT_DIR = os.getenv("LUMINA_EMBEDDINGS_DIR", "runtime/embeddings")
OUTPUT_SAVE_PATH = r"analysis_result\inference_results_layer9_new.csv"
LAYER_IDX = 9
YEAR = 2026

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DAY_TO_MONTH = 1 / (365 / 12)
MINUTE_TO_HOUR = 1 / 60

def inverse_angular_time(theta, phi):
    month_d = ((theta + math.pi) / (2 * math.pi)) * 12
    month = max(1, min(12, int(month_d) + 1))
    day_fraction = (month_d - int(month_d)) / DAY_TO_MONTH
    last_day = calendar.monthrange(YEAR, month)[1]
    day = max(1, min(last_day, int(day_fraction) + 1))
    hour_d = ((phi + math.pi) / (2 * math.pi)) * 24
    hour = int(hour_d) % 24
    minute = int((hour_d - int(hour_d)) * 60) % 60
    return datetime(YEAR, month, day, hour, minute, 0)

def calculate_time_diff(actual_dt, pred_dt):
    try:
        actual_2026 = actual_dt.replace(year=YEAR, second=0, microsecond=0)
    except ValueError:
        actual_2026 = actual_dt.replace(year=YEAR, month=2, day=28, second=0, microsecond=0)

    diff = abs((actual_2026 - pred_dt).total_seconds())
    if diff > 15768000: diff = 31536000 - diff
    return diff

class InferenceDataset(Dataset):
    def __init__(self, pt_dir, csv_path, layer_idx=9):
        self.pt_dir = pt_dir
        self.layer_key = f"hidden_states{layer_idx}"

        df = pd.read_csv(csv_path)
        df = df[df['Status'].astype(str) != '0'].copy()

        self.data_list = []
        print(f"Parsing CSV and loading data (Layer {layer_idx})...")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing CSV"):
            img_name = str(row['File Name']).strip()
            if not img_name or img_name == 'nan':
                img_name = os.path.basename(row['Full Path'])

            pt_path = os.path.join(pt_dir, f"{img_name}.pt")

            if os.path.exists(pt_path):
                try:
                    dt_str = str(row['DateTime']).strip()
                    dt = pd.to_datetime(dt_str.replace(':', '-', 2)) if ':' in dt_str[:11] else pd.to_datetime(dt_str)

                    self.data_list.append({
                        'pt_path': pt_path,
                        'actual_dt': dt,
                        'row_data': row.to_dict()
                    })
                except Exception as e:
                    continue
        print(f" {len(self.data_list)} items loaded successfully.")

    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx):
        item = self.data_list[idx]
        data = torch.load(item['pt_path'], map_location='cpu')
        return data[self.layer_key].to(torch.float32), item['actual_dt'], item['row_data']

def collate_fn(batch):
    embeddings, actual_dts, rows = zip(*batch)
    padded = pad_sequence(embeddings, batch_first=True)
    mask = torch.arange(padded.size(1))[None, :] < torch.tensor([e.size(0) for e in embeddings])[:, None]
    return padded, actual_dts, mask, rows

class SequenceProbeModel(nn.Module):
    def __init__(self, input_dim=4096):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(input_dim, 512), nn.Tanh(), nn.Linear(512, 1))
        self.head = nn.Sequential(nn.Linear(input_dim, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 2))
    def forward(self, x, mask):
        attn_scores = self.attention(x).squeeze(-1).masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        return self.head(torch.sum(attn_weights * x, dim=1))

def run_inference(model_path=MODEL_PATH, new_csv_path=NEW_CSV_PATH, pt_dir=PT_DIR,
                  output_save_path=OUTPUT_SAVE_PATH, layer_idx=LAYER_IDX):
    model = SequenceProbeModel().to(DEVICE)
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    dataset = InferenceDataset(pt_dir, new_csv_path, layer_idx=layer_idx)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    results = []

    print(f"Inference started (Device: {DEVICE})...")
    with torch.no_grad():
        for h, a_dts, m, rows in tqdm(loader, desc="Inference"):
            h, m = h.to(DEVICE), m.to(DEVICE)
            pred = model(h, m).cpu().numpy()[0]

            p_dt = inverse_angular_time(pred[0], pred[1])
            actual_dt = a_dts[0]
            row = rows[0]

            diff_sec = calculate_time_diff(actual_dt, p_dt)

            res_entry = row.copy()
            res_entry['Actual_Time'] = actual_dt.strftime('%Y-%m-%d %H:%M')
            res_entry['Predicted_Time'] = p_dt.strftime('%m-%d %H:%M')
            res_entry['Error_Format'] = str(timedelta(seconds=int(diff_sec)))
            results.append(res_entry)

    os.makedirs(os.path.dirname(output_save_path), exist_ok=True)
    pd.DataFrame(results).to_csv(output_save_path, index=False, encoding='utf-8-sig')
    print(f"Finished! Results saved to: {output_save_path}")

if __name__ == "__main__":
    run_inference()

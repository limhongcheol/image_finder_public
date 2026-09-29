import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from datetime import datetime, timedelta
import math
import numpy as np
from tqdm import tqdm
import calendar

RESULT_SAVE_DIR = "analysis_result"
MODEL_SAVE_DIR = "saved_models"
YEAR = 2026

DAY_TO_MONTH = 1 / (365 / 12)
MINUTE_TO_HOUR = 1 / 60

def angular_time_representation(T):
    month, day, hour, minute = torch.chunk(T.float(), 4, dim=1)
    month_d = (month - 1) + (day - 1) * DAY_TO_MONTH
    hour_d = hour + minute * MINUTE_TO_HOUR
    theta = 2 * math.pi * month_d / 12 - math.pi
    phi = 2 * math.pi * hour_d / 24 - math.pi
    return torch.cat((theta, phi), dim=1)

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

def calculate_time_diff(actual_str, pred_dt):
    """Calculate error in seconds between actual datetime string and predicted datetime object."""
    actual_dt = pd.to_datetime(actual_str)
    try:
        actual_2026 = actual_dt.replace(year=YEAR, second=0, microsecond=0)
    except ValueError:
        actual_2026 = actual_dt.replace(year=YEAR, month=2, day=28, second=0, microsecond=0)

    diff = abs((actual_2026 - pred_dt).total_seconds())
    if diff > 15768000: diff = 31536000 - diff
    return diff

def cyclic_loss(pred, target):
    return (1 - torch.cos(pred[:, 0] - target[:, 0]) + 1 - torch.cos(pred[:, 1] - target[:, 1])).mean()

class FullSequenceDataset(Dataset):
    def __init__(self, pt_dir, csv_path, layer_idx):
        self.pt_dir = pt_dir
        self.layer_key = f"hidden_states{layer_idx}"
        df = pd.read_csv(csv_path)
        df = df[df['Status'].astype(str) != '0'].copy()
        self.data_list = []
        for _, row in df.iterrows():
            img_name = str(row['File Name']).strip()
            pt_path = os.path.join(pt_dir, f"{img_name}.pt")
            if os.path.exists(pt_path):
                try:
                    dt = pd.to_datetime(str(row['DateTime']).strip())
                    self.data_list.append({'pt_path': pt_path, 'time_vec': [dt.month, dt.day, dt.hour, dt.minute], 'row_data': row.to_dict()})
                except: continue
    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx):
        item = self.data_list[idx]
        data = torch.load(item['pt_path'], map_location='cpu')
        return data[self.layer_key].to(torch.float32), torch.tensor(item['time_vec'], dtype=torch.float32), item['row_data']

def collate_fn(batch):
    embeddings, times, rows = zip(*batch)
    padded = pad_sequence(embeddings, batch_first=True)
    mask = torch.arange(padded.size(1))[None, :] < torch.tensor([e.size(0) for e in embeddings])[:, None]
    return padded, torch.stack(times), mask, rows

class SequenceProbeModel(nn.Module):
    def __init__(self, input_dim=4096):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(input_dim, 512), nn.Tanh(), nn.Linear(512, 1))
        self.head = nn.Sequential(nn.Linear(input_dim, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 2))
    def forward(self, x, mask):
        attn_scores = self.attention(x).squeeze(-1).masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        return self.head(torch.sum(attn_weights * x, dim=1))

def run_integrated_analysis(pt_dir, csv_path, layers=None, weights=None,
                            result_dir=RESULT_SAVE_DIR, model_save_dir=MODEL_SAVE_DIR,
                            epochs=10, batch_size=8, lr=1e-4, input_dim=4096):
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if layers is None:
        layers = [9]                          # best_layer_9.pth 로 통일
    if weights is None:
        weights = {9: 1.0}
    ensemble_data = {}

    for layer_idx in layers:
        print(f"\n{'='*20}Layer {layer_idx} Training & Validation Start {'='*20}")
        dataset = FullSequenceDataset(pt_dir, csv_path, layer_idx)
        train_len = int(0.8 * len(dataset))
        val_len = len(dataset) - train_len
        train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=1, collate_fn=collate_fn)

        model = SequenceProbeModel(input_dim=input_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model_path = os.path.join(model_save_dir, f"best_layer_{layer_idx}.pth")

        best_val_error = float('inf')

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for h, t, m, _ in tqdm(train_loader, desc=f"L{layer_idx} Epoch {epoch+1} [Train]", leave=False):
                h, t, m = h.to(device), t.to(device), m.to(device)
                optimizer.zero_grad()
                loss = cyclic_loss(model(h, m), angular_time_representation(t))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0
            total_error_sec = 0
            with torch.no_grad():
                for h, t, m, rows in val_loader:
                    h, t, m = h.to(device), t.to(device), m.to(device)
                    pred = model(h, m)
                    val_loss += cyclic_loss(pred, angular_time_representation(t)).item()

                    pred_np = pred.cpu().numpy()[0]
                    p_dt = inverse_angular_time(pred_np[0], pred_np[1])
                    total_error_sec += calculate_time_diff(rows[0]['DateTime'], p_dt)

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_val_error_sec = total_error_sec / len(val_loader)

            error_msg = str(timedelta(seconds=int(avg_val_error_sec)))

            print(f"Epoch {epoch+1:2d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Avg Error: {error_msg}")

            if avg_val_error_sec < best_val_error:
                best_val_error = avg_val_error_sec
                torch.save(model.state_dict(), model_path)
                print(f"   Best Model saved! (Error: {error_msg})")

        print(f"Layer {layer_idx} Final Report Generation...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        layer_results = []
        with torch.no_grad():
            for h, t, m, rows in tqdm(val_loader, desc=f"L{layer_idx} Final Inference", leave=False):
                h, m = h.to(device), m.to(device)
                pred = model(h, m).cpu().numpy()[0]
                row = rows[0]
                f_name = row['File Name']

                if f_name not in ensemble_data: ensemble_data[f_name] = {'theta': 0, 'phi': 0, 'row': row}
                ensemble_data[f_name]['theta'] += pred[0] * weights[layer_idx]
                ensemble_data[f_name]['phi'] += pred[1] * weights[layer_idx]

                p_dt = inverse_angular_time(pred[0], pred[1])
                layer_results.append({
                    'File Name': f_name,
                    'Actual': row['DateTime'],
                    'Predicted': p_dt.strftime('%m-%d %H:%M'),
                    'Error': str(timedelta(seconds=int(calculate_time_diff(row['DateTime'], p_dt))))
                })
        pd.DataFrame(layer_results).to_csv(os.path.join(result_dir, f"individual_layer_{layer_idx}.csv"), index=False, encoding='utf-8-sig')

    print(f"\nGenerating Final Ensemble Report...")
    ens_results = []
    for f_name, data in ensemble_data.items():
        p_dt = inverse_angular_time(data['theta'], data['phi'])
        diff_sec = calculate_time_diff(data['row']['DateTime'], p_dt)
        res = data['row'].copy()
        res.update({
            'Actual_Cycle': pd.to_datetime(data['row']['DateTime']).strftime('%m-%d %H:%M'),
            'Predicted_Cycle': p_dt.strftime('%m-%d %H:%M'),
            'Error_Format': str(timedelta(seconds=int(diff_sec)))
        })
        ens_results.append(res)

    pd.DataFrame(ens_results).to_csv(os.path.join(result_dir, "final_ensemble_report.csv"), index=False, encoding='utf-8-sig')
    print(f"Complete! All results saved in {result_dir}.")

if __name__ == "__main__":
    PT_DIR = os.getenv("LUMINA_EMBEDDINGS_DIR", "runtime/embeddings")
    CSV_PATH = os.getenv("LUMINA_DATASET_CSV", "runtime/dataset.csv")
    run_integrated_analysis(PT_DIR, CSV_PATH)

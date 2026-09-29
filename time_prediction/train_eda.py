import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def perform_detailed_eda(csv_path):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    df['Error_Total_Seconds'] = pd.to_timedelta(df['Error']).dt.total_seconds()

    def parse_actual(x):
        try:
            return pd.to_datetime(x)
        except:
            return None

    def parse_predicted(x):
        try:
            return datetime.strptime(f"2026-{x}", "%Y-%m-%d %H:%M")
        except:
            return None

    df['Actual_DT'] = df['Actual'].apply(parse_actual)
    df['Pred_DT'] = df['Predicted'].apply(parse_predicted)

    df = df.dropna(subset=['Actual_DT', 'Pred_DT'])

    df['Actual_Month'] = df['Actual_DT'].dt.month
    df['Actual_Hour'] = df['Actual_DT'].dt.hour
    df['Error_Hours'] = df['Error_Total_Seconds'] / 3600
    df['Error_Days'] = df['Error_Hours'] / 24

    print("\n=== Layer-wise Prediction Error Summary Statistics ===")
    stats = df['Error_Total_Seconds'].describe()
    print(f"Average error: {timedelta(seconds=int(stats['mean']))}")
    print(f"Median error: {timedelta(seconds=int(stats['50%']))}")
    print("="*30)

    fig = plt.figure(figsize=(18, 12))

    plt.subplot(2, 2, 1)
    sns.histplot(df['Error_Days'], bins=30, kde=True, color='skyblue')
    plt.title('Error Distribution (Unit: Days)')

    plt.subplot(2, 2, 2)

    def to_2026(dt):
        try:
            return dt.replace(year=2026)
        except ValueError:
            return dt.replace(year=2026, month=2, day=28)

    df['Actual_2026'] = df['Actual_DT'].apply(to_2026)
    df['Actual_DayOfYear'] = df['Actual_2026'].dt.dayofyear
    df['Pred_DayOfYear'] = df['Pred_DT'].dt.dayofyear

    plt.scatter(df['Actual_DayOfYear'], df['Pred_DayOfYear'], alpha=0.5, c=df['Error_Days'], cmap='viridis')
    plt.plot([0, 366], [0, 366], 'r--', alpha=0.7)
    plt.title('Actual Date vs Predicted Date (Cyclic Comparison)')
    plt.xlabel('Actual Date (Day of Year)')
    plt.ylabel('Predicted Date (Day of Year)')
    plt.colorbar(label='Error (Days)')

    plt.subplot(2, 2, 3)
    sns.barplot(data=df, x='Actual_Month', y='Error_Days', palette='coolwarm')
    plt.title('Average Prediction Error by Shooting Month')

    plt.subplot(2, 2, 4)
    sns.lineplot(data=df, x='Actual_Hour', y='Error_Hours', marker='o', color='orange')
    plt.xticks(range(0, 24))
    plt.grid(True, alpha=0.3)
    plt.title('Average Prediction Error by Hour')

    plt.tight_layout()
    plt.savefig('layer_analysis_plots_fixed.png')
    plt.show()

    print("\n=== Best Predictions (Top 5) ===")
    print(df.sort_values('Error_Total_Seconds').head(5)[['File Name', 'Actual', 'Predicted', 'Error']])
    print("\n=== Worst Predictions (Top 5) ===")
    print(df.sort_values('Error_Total_Seconds', ascending=False).head(5)[['File Name', 'Actual', 'Predicted', 'Error']])
if __name__ == "__main__":
    path = os.getenv("LUMINA_TRAIN_RESULTS", "analysis_result/individual_layer_9.csv")
    perform_detailed_eda(path)

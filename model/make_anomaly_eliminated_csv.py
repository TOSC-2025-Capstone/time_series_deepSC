import os
import pandas as pd
import numpy as np

SRC_DIR = "data_handling/merged"
DST_DIR = "data_handling/merged_anomaly_eliminated"
FEATURE_COLS = [
    'Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load', 'Time'
]

os.makedirs(DST_DIR, exist_ok=True)

def correct_outliers_with_interpolation(df, feature_cols):
    df = df.copy()
    for col in feature_cols:
        series = df[col]
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_mask = (series < lower) | (series > upper)
        if outlier_mask.any():
            series[outlier_mask] = np.nan
            series = series.interpolate(method='linear', limit_direction='both')
            series = series.fillna(method='bfill').fillna(method='ffill')
            df[col] = series
    return df

def main():
    files = [f for f in os.listdir(SRC_DIR) if f.endswith('.csv')]
    for fname in files:
        src_path = os.path.join(SRC_DIR, fname)
        dst_path = os.path.join(DST_DIR, fname)
        try:
            df = pd.read_csv(src_path)
            df_corr = correct_outliers_with_interpolation(df, FEATURE_COLS)
            df_corr.to_csv(dst_path, index=False)
            print(f"[OK] {fname} → {dst_path}")
        except Exception as e:
            print(f"[FAIL] {fname}: {e}")

if __name__ == "__main__":
    main() 
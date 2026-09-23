import os
import numpy as np
import pandas as pd

PROCESSED_DIR = "data/processed"

path_imgw = os.path.join(PROCESSED_DIR, "era5_imgw.csv")
path_lcs = os.path.join(PROCESSED_DIR, "krakow_lcs.csv")

# 1. Wczytanie IMGW
df_imgw = pd.read_csv(path_imgw)
df_imgw['timestamp'] = pd.to_datetime(df_imgw['timestamp'])
df_imgw.rename(columns={'temp_imgw': 'temp_ground'}, inplace=True)
df_imgw['source_type'] = 'IMGW'
df_imgw['station'] = np.where(df_imgw['is_urban'] == 1, 'Obserwatorium', 'Balice')
df_imgw['lat'] = np.where(df_imgw['is_urban'] == 1, 50.067, 50.077)
df_imgw['lon'] = np.where(df_imgw['is_urban'] == 1, 19.959, 19.784)

# 2. Wczytanie LCS
df_lcs = pd.read_csv(path_lcs)
df_lcs['timestamp'] = pd.to_datetime(df_lcs['timestamp'])
df_lcs['source_type'] = 'LCS'

# 3. Złożenie tabel
common_cols = ['timestamp', 'station', 'source_type', 'lat', 'lon', 'altitude', 'land_cover', 'is_urban', 'temp_era5', 'temp_ground']

df_master = pd.concat([df_imgw[common_cols], df_lcs[common_cols]], ignore_index=True)
df_master.dropna(subset=['temp_era5', 'temp_ground'], inplace=True)

# 4. Cechy cykliczne czasu
df_master['hour'] = df_master['timestamp'].dt.hour
df_master['month'] = df_master['timestamp'].dt.month
df_master['day_of_year'] = df_master['timestamp'].dt.dayofyear

df_master['hour_sin'] = np.sin(2 * np.pi * df_master['hour'] / 24)
df_master['hour_cos'] = np.cos(2 * np.pi * df_master['hour'] / 24)
df_master['month_sin'] = np.sin(2 * np.pi * df_master['month'] / 12)
df_master['month_cos'] = np.cos(2 * np.pi * df_master['month'] / 12)

df_master.sort_values(by=['timestamp', 'station'], inplace=True)
df_master.reset_index(drop=True, inplace=True)

# 5. Zapis
out_parquet = os.path.join(PROCESSED_DIR, "krakow_final_dataset.parquet")
out_csv = os.path.join(PROCESSED_DIR, "krakow_final_dataset.csv")

df_master.to_parquet(out_parquet, index=False)
df_master.to_csv(out_csv, index=False)

print("\n" + "="*50)
print(f"OSTATECZNY DATASET ZAPISANY: {out_parquet}")
print(f"Liczba wierszy: {len(df_master):,}")
print(f"Zakres dat: {df_master['timestamp'].min()} do {df_master['timestamp'].max()}")
print("\nPodział próbek na źródła:")
print(df_master['source_type'].value_counts())
print("\nLiczba obserwacji dla stacji:")
print(df_master['station'].value_counts())
print(df_master.head(5))
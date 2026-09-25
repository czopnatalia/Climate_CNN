import os
from pathlib import Path
import pandas as pd
import numpy as np
from climatology import add_guminski_seasons_harmonic

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

imgw_path = PROCESSED_DIR / "imgw_dataset.parquet"
lcs_path = PROCESSED_DIR / "lcs_dataset.parquet"

# Rezerwowe ścieżki do CSV, jeśli parquet nie byłby dostępny
if not imgw_path.exists():
    imgw_path = PROCESSED_DIR / "imgw_dataset.csv"
if not lcs_path.exists():
    lcs_path = PROCESSED_DIR / "lcs_dataset.csv"

print("1. Wczytywanie przetworzonych zbiorów IMGW i LCS...")
df_imgw = pd.read_parquet(imgw_path) if str(imgw_path).endswith('.parquet') else pd.read_csv(imgw_path)
df_lcs = pd.read_parquet(lcs_path) if str(lcs_path).endswith('.parquet') else pd.read_csv(lcs_path)

# Dodanie jednoznacznej flagi pochodzenia danych
df_imgw['data_source'] = 'imgw'
df_lcs['data_source'] = 'lcs'

# Standaryzacja kolumn bazowych przed wyliczeniem pór roku
base_columns = [
    'timestamp', 'station', 'data_source', 'lat', 'lon', 
    'altitude', 'urban_fraction_500m',
    'sin_hour', 'cos_hour', 'sin_doy', 'cos_doy', 
    'temp_era5', 'temp_ground'
]

df_merged = pd.concat([df_imgw[base_columns], df_lcs[base_columns]], ignore_index=True)
df_merged['timestamp'] = pd.to_datetime(df_merged['timestamp'])
df_merged.sort_values(by=['timestamp', 'station'], inplace=True)
df_merged.reset_index(drop=True, inplace=True)

# 2. Wyznaczenie pór roku Gumińskiego metodą harmoniczną
print("2. Klasyfikacja termicznych pór roku Gumińskiego metodą fali harmonicznej...")
df_merged = add_guminski_seasons_harmonic(
    df_merged, 
    timestamp_col='timestamp', 
    temp_col='temp_era5'
)

# 3. Uporządkowanie kolumn finalnych
target_columns = [
    'timestamp', 'station', 'data_source', 'lat', 'lon', 
    'altitude', 'urban_fraction_500m',
    'sin_hour', 'cos_hour', 'sin_doy', 'cos_doy', 
    'guminski_season', 'temp_era5', 'temp_ground'
]
df_merged = df_merged[target_columns].copy()

# 4. Sprawdzenie spójności i braków danych
print("\n--- RAPORT JAKOŚCI DANYCH PO POŁĄCZENIU ---")
print(f"Łączna liczba wierszy: {len(df_merged):,}")
print(f"Liczba unikalnych stacji: {df_merged['station'].nunique()}")
print(f"Stacje IMGW: {list(df_merged[df_merged['data_source'] == 'imgw']['station'].unique())}")
print(f"Stacje LCS:  {list(df_merged[df_merged['data_source'] == 'lcs']['station'].unique())}")

null_counts = df_merged.isna().sum()
if null_counts.sum() > 0:
    print("\nOstrzeżenie: Wykryto braki danych (NaN):")
    print(null_counts[null_counts > 0])
    df_merged.dropna(inplace=True)
    df_merged.reset_index(drop=True, inplace=True)
else:
    print("Brak wartości NaN we wszystkich kolumnach (100% spójności).")

# 5. Zapis pełnego zbioru (2000-2025)
out_parquet = PROCESSED_DIR / "final_dataset.parquet"
out_csv = PROCESSED_DIR / "final_dataset.csv"
df_merged.to_parquet(out_parquet, index=False)
df_merged.to_csv(out_csv, index=False)

# 6. Zapis wspólnego zbioru dla roku 2022 (porównanie LCS vs IMGW)
df_2022 = df_merged[df_merged['timestamp'].dt.year == 2022].copy().reset_index(drop=True)
out_parquet_2022 = PROCESSED_DIR / "final_dataset_2022.parquet"
out_csv_2022 = PROCESSED_DIR / "final_dataset_2022.csv"
df_2022.to_parquet(out_parquet_2022, index=False)
df_2022.to_csv(out_csv_2022, index=False)

print("\n" + "="*60)
print(f"✅ ZAPISANO FINALNE ZBIORY:")
print(f"1. Pełny zbiór:      {out_parquet} ({len(df_merged):,} wierszy)")
print(f"2. Zbiór 2022 roku:  {out_parquet_2022} ({len(df_2022):,} wierszy)")
print("="*60)
print("\nRozkład pór roku w roku 2022:")
print(df_2022['guminski_season'].value_counts())
print("\nPodgląd pierwszych rekordów:")
print(df_2022[['timestamp', 'station', 'data_source', 'altitude', 'urban_fraction_500m', 'guminski_season', 'temp_era5', 'temp_ground']].head(3))
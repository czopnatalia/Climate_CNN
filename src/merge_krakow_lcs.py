import os
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

excel_path = os.path.join(RAW_DIR, "lcs/lokalizacja_nazwy_miast_LCS.xlsx")
parquet_path = os.path.join(RAW_DIR, "lcs/dane_z_lokalizacja_godzinne.parquet")
era5_nc_path = os.path.join(RAW_DIR, "krakow_era5_2000_2025.nc")

# 1. Odczyt stacji z Krakowa i okolic Balic
print("1. Wczytywanie stacji z Excela...")
df_krk = pd.read_excel(excel_path, sheet_name="KRK")
df_all = pd.read_excel(excel_path, sheet_name="ALL_sensors")

df_balice_area = df_all[df_all['Localization'].isin(['Szczyglice', 'Aleksandrowice'])].copy()
selected_stations = pd.concat([df_krk, df_balice_area], ignore_index=True)

# 2. POPRAWNE PRZELICZENIE WSPÓŁRZĘDNYCH: UTM strefa 34N (EPSG:32634) -> WGS84 (EPSG:4326)
transformer = Transformer.from_crs("EPSG:32634", "EPSG:4326", always_xy=True)
selected_stations['lon'], selected_stations['lat'] = transformer.transform(
    selected_stations['X'].values, 
    selected_stations['Y'].values
)

selected_stations['X_round'] = selected_stations['X'].round(0).astype(int)
selected_stations['Y_round'] = selected_stations['Y'].round(0).astype(int)
selected_stations.rename(columns={'LCS altitude [m]': 'altitude', 'Localization': 'station'}, inplace=True)

# Cechy środowiskowe
selected_stations['is_urban'] = np.where(selected_stations['City'] == 'Krakow', 1, 0)
selected_stations['land_cover'] = np.where(selected_stations['is_urban'] == 1, 50, 30)

meta_clean = selected_stations[['station', 'City', 'lat', 'lon', 'altitude', 'land_cover', 'is_urban', 'X_round', 'Y_round']].copy()

print("Skorygowane współrzędne stacji:")
print(meta_clean[['station', 'lat', 'lon', 'altitude', 'is_urban']])

# 3. Wczytanie pomiarów z Parquet
print("\n2. Wczytywanie pomiarów naziemnych...")
df_measurements = pd.read_parquet(parquet_path)
df_measurements['X_round'] = df_measurements['X'].round(0).astype(int)
df_measurements['Y_round'] = df_measurements['Y'].round(0).astype(int)

# Łączenie po współrzędnych
df_merged = pd.merge(df_measurements, meta_clean, on=['X_round', 'Y_round'], how='inner')

df_merged['timestamp'] = pd.to_datetime(df_merged['date'].astype(str) + ' ' + df_merged['hour'].astype(str) + ':00:00')
df_merged.rename(columns={'temperature_2m': 'temp_ground'}, inplace=True)

# 4. Pobieranie temp_era5 z pliku NetCDF
print("\n3. Ekstrakcja odpowiadających temperatur z ERA5-Land...")
ds_era5 = xr.open_dataset(era5_nc_path)
time_col = 'valid_time' if 'valid_time' in ds_era5.coords else 'time'

unique_pts = meta_clean[['station', 'lat', 'lon']].drop_duplicates()
era5_records = []

for _, row in unique_pts.iterrows():
    pt = ds_era5['t2m'].sel(latitude=row['lat'], longitude=row['lon'], method='nearest').to_dataframe().reset_index()
    pt['temp_era5'] = pt['t2m'] - 273.15
    pt['timestamp'] = pd.to_datetime(pt[time_col])
    pt['station'] = row['station']
    era5_records.append(pt[['timestamp', 'station', 'temp_era5']])

df_era5_lcs = pd.concat(era5_records, ignore_index=True)

# Łączymy po czasie i stacji
df_lcs_final = pd.merge(df_merged, df_era5_lcs, on=['timestamp', 'station'], how='inner')

# 5. Wybór czystych kolumn
final_cols = ['timestamp', 'station', 'City', 'lat', 'lon', 'altitude', 'land_cover', 'is_urban', 'temp_era5', 'temp_ground']
df_lcs_final = df_lcs_final[final_cols].dropna(subset=['temp_era5', 'temp_ground']).copy()

df_lcs_final.sort_values(by=['timestamp', 'station'], inplace=True)
df_lcs_final.reset_index(drop=True, inplace=True)

# 6. Zapis pliku
out_csv = os.path.join(PROCESSED_DIR, "krakow_lcs.csv")

df_lcs_final.to_csv(out_csv, index=False)

print("\n" + "="*50)
print(f"Wygenerowano plik: {out_csv}")
print(f"Liczba rekordów: {len(df_lcs_final):,}")
print(f"Współrzędne sprawdzone: lat ~ {df_lcs_final['lat'].mean():.4f}, lon ~ {df_lcs_final['lon'].mean():.4f}")
print(df_lcs_final.head(3))
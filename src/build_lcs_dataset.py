import os
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point, Polygon, mapping
from pyproj import Transformer
from scipy.spatial import cKDTree

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

excel_path = os.path.join(RAW_DIR, "lcs/lokalizacja_nazwy_miast_LCS.xlsx")
parquet_path = os.path.join(RAW_DIR, "lcs/dane_z_lokalizacja_godzinne.parquet")
era5_nc_path = os.path.join(RAW_DIR, "krakow_era5_2000_2025.nc")
dem_tif_path = os.path.join(RAW_DIR, "krakow_dem.tif")
lc_tif_path = os.path.join(RAW_DIR, "krakow_land_cover.tif")

# Granice siatki ERA5 dla regionu krakowskiego
LON_MIN, LON_MAX = 19.7, 20.2
LAT_MIN, LAT_MAX = 49.9, 50.2

# ==========================================
# 1. FUNKCJE GIS DO PRÓBKOWANIA RASTRÓW
# ==========================================
def sample_dem(raster_path, lats, lons):
    with rasterio.open(raster_path) as src:
        crs = src.crs
        nodata = src.nodata
        if crs and crs.to_string() != "EPSG:4326":
            tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            xs, ys = tr.transform(lons, lats)
        else:
            xs, ys = lons, lats
        coords = list(zip(xs, ys))
        sampled = [val[0] for val in src.sample(coords)]
        return np.array([np.nan if (v == nodata or v <= 0) else float(v) for v in sampled])

def compute_urban_fraction(raster_path, lats, lons, radius_m=500.0):
    with rasterio.open(raster_path) as src:
        crs = src.crs
        nodata = src.nodata
        
        # Detekcja klas: CORINE (100-142) vs ESA WorldCover (50)
        sample_win = src.read(1, window=((0, 100), (0, 100)))
        urban_classes = list(range(100, 143)) if np.nanmax(sample_win) > 60 else [50]
        
        to_metric = Transformer.from_crs("EPSG:4326", "EPSG:32634", always_xy=True)
        to_raster = Transformer.from_crs("EPSG:32634", crs, always_xy=True)
        
        fractions = []
        for lat, lon in zip(lats, lons):
            xm, ym = to_metric.transform(lon, lat)
            circle = Point(xm, ym).buffer(radius_m)
            poly_coords = [to_raster.transform(x, y) for x, y in circle.exterior.coords]
            geom = Polygon(poly_coords)
            try:
                out_img, _ = mask(src, [mapping(geom)], crop=True)
                px = out_img[0].flatten()
                valid = px[px != nodata] if nodata is not None else px[px > 0]
                if len(valid) == 0:
                    fractions.append(np.nan)
                else:
                    fractions.append(round(float(np.isin(valid, urban_classes).sum() / len(valid)), 4))
            except Exception:
                fractions.append(np.nan)
        return np.array(fractions)

# ==========================================
# 2. WCZYTANIE I GEOMETRIA STACJI
# ==========================================
print("1. Przygotowywanie metadanych stacji...")
xl = pd.ExcelFile(excel_path)
meta_dfs = [pd.read_excel(excel_path, sheet_name=s) for s in ['KRK', 'ALL_sensors'] if s in xl.sheet_names]
stations_df = pd.concat(meta_dfs, ignore_index=True).dropna(subset=['X', 'Y', 'Localization'])
stations_df.rename(columns={'Localization': 'station'}, inplace=True)
stations_df = stations_df.drop_duplicates(subset=['station']).copy()

# UTM 34N -> WGS84
tr_utm = Transformer.from_crs("EPSG:32634", "EPSG:4326", always_xy=True)
stations_df['lon'], stations_df['lat'] = tr_utm.transform(stations_df['X'].values, stations_df['Y'].values)

# Filtracja przestrzenna (BBOX ERA5)
stations_df = stations_df[
    (stations_df['lon'] >= LON_MIN) & (stations_df['lon'] <= LON_MAX) &
    (stations_df['lat'] >= LAT_MIN) & (stations_df['lat'] <= LAT_MAX)
].copy()

# Próbkowanie cech środowiskowych
print("2. Próbkowanie DEM i liczenie Urban Fraction (500m)...")
stations_df['altitude'] = sample_dem(dem_tif_path, stations_df['lat'].values, stations_df['lon'].values)
stations_df['urban_fraction_500m'] = compute_urban_fraction(lc_tif_path, stations_df['lat'].values, stations_df['lon'].values)

# Usunięcie stacji poza zasięgiem DEM (gdzie altitude = NaN lub <= 0)
stations_df = stations_df.dropna(subset=['altitude', 'urban_fraction_500m']).copy()
stations_df = stations_df[stations_df['altitude'] > 0].reset_index(drop=True)
print(f"Stacje zakwalifikowane ({len(stations_df)}): {list(stations_df['station'].unique())}")

# ==========================================
# 3. DOPASOWANIE POMIARÓW LCS Z PARQUET
# ==========================================
print("3. Wczytywanie pomiarów i łączenie przestrzenne (KDTree)...")
df_raw = pd.read_parquet(parquet_path)
unique_pq = df_raw[['X', 'Y']].drop_duplicates().copy().reset_index(drop=True)

tree = cKDTree(stations_df[['X', 'Y']].values)
dists, indices = tree.query(unique_pq[['X', 'Y']].values)
unique_pq['station_idx'] = indices
unique_pq['dist_m'] = dists

# Akceptujemy tylko punkty w promieniu 100 m
valid_pq = unique_pq[unique_pq['dist_m'] <= 100.0].copy()
for col in ['station', 'lat', 'lon', 'altitude', 'urban_fraction_500m']:
    valid_pq[col] = stations_df.iloc[valid_pq['station_idx']][col].values

df_lcs = pd.merge(df_raw, valid_pq.drop(columns=['station_idx']), on=['X', 'Y'], how='inner')

# Konwersja czasu do UTC
print("4. Synchronizacja czasu do UTC...")
df_lcs['dt_local'] = pd.to_datetime(df_lcs['date'].astype(str) + ' ' + df_lcs['hour'].astype(str) + ':00:00')
df_lcs['timestamp'] = df_lcs['dt_local'].dt.tz_localize(
    'Europe/Warsaw', ambiguous='NaT', nonexistent='shift_forward'
).dt.tz_convert('UTC').dt.tz_localize(None)

df_lcs.dropna(subset=['timestamp'], inplace=True)
df_lcs.rename(columns={'temperature_2m': 'temp_ground'}, inplace=True)

# ==========================================
# 4. DOŁĄCZENIE TEMPERATURY Z ERA5-LAND
# ==========================================
print("5. Próbkowanie czasoprzestrzenne ERA5-Land (NetCDF)...")
ds_era5 = xr.open_dataset(era5_nc_path)
time_col = 'valid_time' if 'valid_time' in ds_era5.coords else 'time'

era5_pts = []
for _, r in stations_df[['station', 'lat', 'lon']].drop_duplicates().iterrows():
    pt = ds_era5['t2m'].sel(latitude=r['lat'], longitude=r['lon'], method='nearest').to_dataframe().reset_index()
    pt['temp_era5'] = pt['t2m'] - 273.15
    pt['timestamp'] = pd.to_datetime(pt[time_col])
    pt['station'] = r['station']
    era5_pts.append(pt[['timestamp', 'station', 'temp_era5']])

df_era5_all = pd.concat(era5_pts, ignore_index=True)
df_final = pd.merge(df_lcs, df_era5_all, on=['timestamp', 'station'], how='inner')


# ==========================================
# 6. CECHY CYKLICZNE CZASU I ZAPIS KOŃCOWY
# ==========================================
print("7. Dodawanie cech cyklicznych...")
df_final['sin_hour'] = np.sin(2 * np.pi * df_final['timestamp'].dt.hour / 24.0)
df_final['cos_hour'] = np.cos(2 * np.pi * df_final['timestamp'].dt.hour / 24.0)
df_final['sin_doy'] = np.sin(2 * np.pi * df_final['timestamp'].dt.dayofyear / 365.25)
df_final['cos_doy'] = np.cos(2 * np.pi * df_final['timestamp'].dt.dayofyear / 365.25)

final_cols = [
    'timestamp', 'station', 'lat', 'lon', 'altitude', 'urban_fraction_500m',
    'sin_hour', 'cos_hour', 'sin_doy', 'cos_doy',
    'temp_era5', 'temp_ground'
]

df_final = df_final[final_cols].dropna().sort_values(by=['timestamp', 'station']).reset_index(drop=True)

out_parquet = os.path.join(PROCESSED_DIR, "lcs_dataset.parquet")
out_csv = os.path.join(PROCESSED_DIR, "lcs_dataset.csv")
df_final.to_parquet(out_parquet, index=False)
df_final.to_csv(out_csv, index=False)

print("\n" + "="*60)
print(f"✅ GOTOWY ZBIÓR DANYCH ZAPISANY: {out_parquet}")
print(f"Liczba wierszy: {len(df_final):,}")
print("\nPodgląd pierwszych rekordów:")
print(df_final.head(3))
print("="*60)
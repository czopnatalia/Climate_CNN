import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point, Polygon, mapping
from pyproj import Transformer

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Ścieżki do rastrów i reanalizy
era5_nc_path = os.path.join(RAW_DIR, "krakow_era5_2000_2025.nc")
dem_tif_path = os.path.join(RAW_DIR, "krakow_dem.tif")
lc_tif_path = os.path.join(RAW_DIR, "krakow_land_cover.tif")

# Ścieżki do plików IMGW
balice_path = os.path.join(RAW_DIR, "krakow_balice_imgw_2000_2025.csv")
obs_path = os.path.join(RAW_DIR, "krakow_obserwatorium_imgw_2000_2025.csv")


if not balice_path or not obs_path:
    raise FileNotFoundError(
        f"Nie znaleziono plików IMGW!"
    )

# 1. Definicja metadanych stacji (współrzędne WGS84)
imgw_metadata = pd.DataFrame([
    {
        'station': 'Krakow-Balice',
        'lat': 50.0777,
        'lon': 19.7848,
        'file_path': balice_path
    },
    {
        'station': 'Krakow-Obserwatorium',
        'lat': 50.0614,
        'lon': 19.9594,
        'file_path': obs_path
    }
])

# ==========================================
# FUNKCJE GIS: DEM i URBAN FRACTION (500m)
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

print("1. Próbkowanie cech fizycznych z krakow_dem.tif i krakow_land_cover.tif...")
imgw_metadata['altitude'] = sample_dem(dem_tif_path, imgw_metadata['lat'].values, imgw_metadata['lon'].values)
imgw_metadata['urban_fraction_500m'] = compute_urban_fraction(lc_tif_path, imgw_metadata['lat'].values, imgw_metadata['lon'].values)

print("\n--- METADANE STACJI IMGW Z RASTRÓW ---")
print(imgw_metadata[['station', 'lat', 'lon', 'altitude', 'urban_fraction_500m']])

# ==========================================
# 2. WCZYTANIE I STANDARYZACJA PLIKÓW CSV
# ==========================================
print("\n2. Wczytywanie plików CSV i standaryzacja kolumn...")
records = []

for _, meta in imgw_metadata.iterrows():
    df_st = pd.read_csv(meta['file_path'])
    
    # 1. Obsługa kolumny czasu
    if 'dt' in df_st.columns:
        # Parsujemy kolumnę dt jako czas lokalny i konwertujemy do UTC
        dt_series = pd.to_datetime(df_st['dt'])
        if dt_series.dt.tz is None:
            df_st['timestamp'] = dt_series.dt.tz_localize(
                'Europe/Warsaw', ambiguous='NaT', nonexistent='shift_forward'
            ).dt.tz_convert('UTC').dt.tz_localize(None)
        else:
            df_st['timestamp'] = dt_series.dt.tz_convert('UTC').dt.tz_localize(None)
    elif 'timestamp' in df_st.columns:
        df_st['timestamp'] = pd.to_datetime(df_st['timestamp'])
    elif {'year', 'month', 'day', 'hour'}.issubset(df_st.columns):
        df_st['dt_local'] = pd.to_datetime(
            df_st['year'].astype(str) + '-' +
            df_st['month'].astype(str).str.zfill(2) + '-' +
            df_st['day'].astype(str).str.zfill(2) + ' ' +
            df_st['hour'].astype(str).str.zfill(2) + ':00:00'
        )
        df_st['timestamp'] = df_st['dt_local'].dt.tz_localize(
            'Europe/Warsaw', ambiguous='NaT', nonexistent='shift_forward'
        ).dt.tz_convert('UTC').dt.tz_localize(None)
    else:
        raise KeyError(f"Nie znaleziono kolumny czasu w {meta['file_path']}")

    # 2. Obsługa kolumny temperatury
    if 'temp' in df_st.columns:
        df_st.rename(columns={'temp': 'temp_ground'}, inplace=True)
    elif 'temp_ground' not in df_st.columns:
        temp_candidates = ['Temperatura powietrza [°C]', 't2m', 'temperatura', 't2m_ground']
        matched_temp = next((c for c in temp_candidates if c in df_st.columns), None)
        if matched_temp:
            df_st.rename(columns={matched_temp: 'temp_ground'}, inplace=True)
        else:
            raise KeyError(f"Nie znaleziono kolumny temperatury w {meta['file_path']}")

    df_st['station'] = meta['station']
    df_st['lat'] = meta['lat']
    df_st['lon'] = meta['lon']
    df_st['altitude'] = meta['altitude']
    df_st['urban_fraction_500m'] = meta['urban_fraction_500m']
    
    clean_sub = df_st[['timestamp', 'station', 'lat', 'lon', 'altitude', 'urban_fraction_500m', 'temp_ground']].dropna()
    records.append(clean_sub)

df_all_imgw = pd.concat(records, ignore_index=True)
print(f"Łączna liczba rekordów ze stacji IMGW: {len(df_all_imgw):,}")

# ==========================================
# 3. DOŁĄCZENIE TEMPERATURY Z ERA5-LAND
# ==========================================
print("\n3. Ekstrakcja temperatur temp_era5 z siatki NetCDF...")
ds_era5 = xr.open_dataset(era5_nc_path)
time_col = 'valid_time' if 'valid_time' in ds_era5.coords else 'time'

era5_pts = []
for _, r in imgw_metadata.iterrows():
    pt = ds_era5['t2m'].sel(latitude=r['lat'], longitude=r['lon'], method='nearest').to_dataframe().reset_index()
    pt['temp_era5'] = pt['t2m'] - 273.15
    pt['timestamp'] = pd.to_datetime(pt[time_col])
    pt['station'] = r['station']
    era5_pts.append(pt[['timestamp', 'station', 'temp_era5']])

df_era5_all = pd.concat(era5_pts, ignore_index=True)

# Łączenie IMGW z ERA5 po zunifikowanym czasie UTC i nazwie stacji
df_final = pd.merge(df_all_imgw, df_era5_all, on=['timestamp', 'station'], how='inner')


# ==========================================
# 5. CECHY CYKLICZNE I ZAPIS KOŃCOWY
# ==========================================
print("\n5. Obliczanie cech cyklicznych czasu...")
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

out_parquet = os.path.join(PROCESSED_DIR, "imgw_dataset.parquet")
out_csv = os.path.join(PROCESSED_DIR, "imgw_dataset.csv")

df_final.to_parquet(out_parquet, index=False)
df_final.to_csv(out_csv, index=False)

print("\n" + "="*60)
print(f"✅ ZAPISANO ZUNIFIKOWANY ZBIÓR IMGW: {out_parquet}")
print(f"Liczba wierszy: {len(df_final):,}")
print("\nStatystyki cech statycznych stacji:")
print(df_final[['station', 'altitude', 'urban_fraction_500m']].drop_duplicates())
print("\nPodgląd pierwszych 3 rekordów:")
print(df_final.head(3))
print("="*60)
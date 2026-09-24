import os
import numpy as np
import pandas as pd

PROCESSED_DIR = "data/processed"
input_path = os.path.join(PROCESSED_DIR, "krakow_final_dataset.parquet")

print("1. Wczytywanie krakow_final_dataset.parquet...")
df = pd.read_parquet(input_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = pd.to_datetime(df['timestamp'].dt.date)

# 2. Obliczenie średniej dobowej temperatury tła z ERA5 dla całego obszaru
print("2. Wyliczanie średniej dobowej makroklimatycznej z ERA5...")
daily_era5 = df.groupby('date')['temp_era5'].mean().reset_index()
daily_era5.sort_values('date', inplace=True)

# 7-dniowe wygładzenie średnią kroczącą (standard eliminujący jednodniowe anomalie)
daily_era5['temp_smooth'] = daily_era5['temp_era5'].rolling(window=7, center=True, min_periods=1).mean()

# 3. Klasyfikacja termicznych pór roku Gumińskiego (1948)
def assign_guminski(row):
    t = row['temp_smooth']
    doy = row['date'].dayofyear  # dzień w roku (1-365/366)
    
    if t < 0:
        return 'Zima'
    elif 0 <= t < 5:
        # Przedwiośnie w pierwszej połowie roku, Przedzimie pod koniec roku
        return 'Przedwiosnie' if doy < 180 else 'Przedzimie'
    elif 5 <= t < 15:
        # Wiosna przy wzroście temperatur, Jesień przy spadku
        return 'Wiosna' if doy < 200 else 'Jesien'
    else:
        return 'Lato'

daily_era5['season'] = daily_era5.apply(assign_guminski, axis=1)

# 4. Dołączenie kolumny 'season' do głównego zbioru danych godzinowych
print("3. Przypisywanie pory roku do każdego wiersza...")
season_mapping = dict(zip(daily_era5['date'], daily_era5['season']))
df['season'] = df['date'].map(season_mapping)

# Usunięcie tymczasowej kolumny 'date'
df.drop(columns=['date'], inplace=True)

# 5. Zapis zbioru ze zdefiniowanymi porami roku
out_parquet = os.path.join(PROCESSED_DIR, "krakow_dataset_with_seasons.parquet")
out_csv = os.path.join(PROCESSED_DIR, "krakow_dataset_with_seasons.csv")

df.to_parquet(out_parquet, index=False)
df.to_csv(out_csv, index=False)

print("\n" + "="*55)
print(f"ZAPISANO PLIK: {out_parquet}")
print("="*55)
print("Rozkład obserwacji w porach roku Gumińskiego (cały zbiór):")
print(df['season'].value_counts())
print("\nPrzykładowe wiersze:")
print(df[['timestamp', 'station', 'temp_era5', 'temp_ground', 'season']].head())
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

input_path = PROCESSED_DIR / "final_dataset_2022.parquet"
if not input_path.exists():
    input_path = PROCESSED_DIR / "final_dataset_2022.csv"

print(f"1. Wczytywanie: {input_path}...")
df = pd.read_parquet(input_path) if str(input_path).endswith('.parquet') else pd.read_csv(input_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date

# 2. Definicja tygodni testowych:
# Dla zimy bierzemy ostatnie 7 dni części zimowej z początku roku (luty)!
# Grudzień (27-31.12) w całości zasila zbiór treningowy.

test_days = []
print("\n--- WYBRANE TYGODNIE TESTOWE DLA KAŻDEJ PORY ROKU ---")

for season in ['zima', 'przedwiosnie', 'wiosna', 'lato', 'jesien', 'przedzimie']:
    season_df = df[df['guminski_season'] == season]
    
    if season == 'zima':
        # Filtrujemy tylko zimę z pierwszej połowy roku (styczeń-luty)
        zima_early = season_df[season_df['timestamp'].dt.month <= 3]
        days = sorted(zima_early['date'].unique())
        last_7 = days[-7:]
        print(f"{'Zima (luty)':<18}: od {last_7[0]} do {last_7[-1]} (7 dni - koniec zimy 2021/22)")
    else:
        days = sorted(season_df['date'].unique())
        last_7 = days[-7:]
        print(f"{season.capitalize():<18}: od {last_7[0]} do {last_7[-1]} (7 dni)")
        
    test_days.extend(last_7)

# 3. Podział zbioru
is_test = df['date'].isin(test_days)
df_train = df[~is_test].drop(columns=['date']).copy().reset_index(drop=True)
df_test = df[is_test].drop(columns=['date']).copy().reset_index(drop=True)

print("\n--- BILANS PODZIAŁU DANYCH ---")
print(f"Łączna liczba wierszy: {len(df):,}")
print(f"Zbiór TRENINGOWY:      {len(df_train):,} wierszy ({len(df_train)/len(df)*100:.1f}%)")
print(f"Zbiór TESTOWY:         {len(df_test):,} wierszy ({len(df_test)/len(df)*100:.1f}%)")

# Zapis do plików
train_path = PROCESSED_DIR / "train_2022.parquet"
test_path = PROCESSED_DIR / "test_2022.parquet"
df_train.to_parquet(train_path, index=False)
df_test.to_parquet(test_path, index=False)
print(f"\n✅ Zapisano zbiór treningowy: {train_path}")
print(f"✅ Zapisano zbiór testowy:    {test_path}")

# 4. Obliczenie Baseline 0 (Błąd surowego ERA5 na nowym zbiorze testowym)
y_true = df_test['temp_ground']
y_era5 = df_test['temp_era5']

mae = mean_absolute_error(y_true, y_era5)
rmse = np.sqrt(mean_squared_error(y_true, y_era5))
r2 = r2_score(y_true, y_era5)
bias = (y_era5 - y_true).mean()

print("\n" + "="*60)
print("📊 OFICJALNY BASELINE 0 (Surowe ERA5-Land na 42 dniach testowych)")
print("="*60)
print(f"MAE:   {mae:.3f} °C")
print(f"RMSE:  {rmse:.3f} °C")
print(f"R²:    {r2:.4f}")
print(f"BIAS:  {bias:.3f} °C")
print("="*60)
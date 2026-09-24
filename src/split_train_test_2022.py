import os
import pandas as pd

PROCESSED_DIR = "data/processed"
input_path = os.path.join(PROCESSED_DIR, "krakow_dataset_with_seasons.parquet")

print("1. Wczytywanie zbioru z porami roku i filtrowanie do 2022 r...")
df = pd.read_parquet(input_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Wybór roku referencyjnego 2022
df_2022 = df[df['timestamp'].dt.year == 2022].copy()
df_2022['date'] = pd.to_datetime(df_2022['timestamp'].dt.date)

# 2. Kalendarz pór roku w 2022 r. i wyznaczenie ostatnich 7 dni na test
print("2. Wyznaczanie ostatnich 7 dni każdej pory roku na zbiór testowy...")

# Tabela unikalnych dni i ich pór roku
daily_schedule = df_2022[['date', 'season']].drop_duplicates().sort_values('date')
daily_schedule['split'] = 'train'

# Kolejność pór roku w cyklu rocznym
season_order = ['Zima', 'Przedwiosnie', 'Wiosna', 'Lato', 'Jesien', 'Przedzimie']

print("\nZakresy czasowe pór roku w 2022 r. i podział:")
for season_name in season_order:
    season_days = daily_schedule[daily_schedule['season'] == season_name]
    if not season_days.empty:
        start_d = season_days['date'].min().strftime('%Y-%m-%d')
        end_d = season_days['date'].max().strftime('%Y-%m-%d')
        total_days = len(season_days)
        
        # Ostatnie 7 dni danej pory roku trafia do testu
        test_dates = season_days.tail(7)['date']
        daily_schedule.loc[daily_schedule['date'].isin(test_dates), 'split'] = 'test'
        
        test_start = test_dates.min().strftime('%Y-%m-%d')
        test_end = test_dates.max().strftime('%Y-%m-%d')
        print(f" - {season_name:13}: {start_d} do {end_d} ({total_days:2d} dni) | Test: {test_start} do {test_end} (7 dni)")

# 3. Przypisanie etykiety 'split' ('train' / 'test') do wszystkich pomiarów godzinowych
split_mapping = dict(zip(daily_schedule['date'], daily_schedule['split']))
df_2022['split'] = df_2022['date'].map(split_mapping)

df_2022.drop(columns=['date'], inplace=True)

# 4. Zapis gotowej bazy modelowej
out_parquet = os.path.join(PROCESSED_DIR, "krakow_2022_train_test.parquet")
out_csv = os.path.join(PROCESSED_DIR, "krakow_2022_train_test.csv")

df_2022.to_parquet(out_parquet, index=False)
df_2022.to_csv(out_csv, index=False)

print("\n" + "="*55)
print(f"GOTOWY ZBIÓR TRENINGOWO-TESTOWY: {out_parquet}")
print("="*55)
print(f"Liczba wierszy w 2022 roku: {len(df_2022):,}")
print("\nRozkład próbek (godzinowych) wg pory roku i zbioru:")
print(pd.crosstab(df_2022['season'], df_2022['split'], margins=True))
print("\nPodgląd kolumn:")
print(df_2022[['timestamp', 'station', 'temp_era5', 'temp_ground', 'season', 'split']].head(6))
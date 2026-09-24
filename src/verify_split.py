import os
import pandas as pd

PROCESSED_DIR = "data/processed"
file_path = os.path.join(PROCESSED_DIR, "krakow_2022_train_test.parquet")

if not os.path.exists(file_path):
    # Jesli plik zostal zapisany jako CSV
    file_path = os.path.join(PROCESSED_DIR, "krakow_2022_train_test.csv")

print(f"Weryfikacja pliku: {file_path}\n")
df = pd.read_parquet(file_path) if file_path.endswith('.parquet') else pd.read_csv(file_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['date'] = df['timestamp'].dt.date

# ---------------------------------------------------------
# TEST 1: Zakres roku i ogólna liczba próbek
# ---------------------------------------------------------
years = df['timestamp'].dt.year.unique()
print("=== TEST 1: Zakres czasowy ===")
print(f"Lata w zbiorze: {list(years)} (Powinno być wyłącznie: [2022])")
print(f"Liczba wszystkich wierszy: {len(df):,}")
assert list(years) == [2022], "UWAGA: W zbiorze znajdują się lata inne niż 2022!"

# ---------------------------------------------------------
# TEST 2: Liczba dni testowych (kalendarzowych) na porę roku
# ---------------------------------------------------------
print("\n=== TEST 2: Liczba dni w zbiorze TEST dla każdej pory roku ===")
test_days_per_season = df[df['split'] == 'test'].groupby('season')['date'].nunique()
print("Liczba unikalnych dni w teście (powinno być dokładnie 7 dla każdej pory):")
print(test_days_per_season)

# ---------------------------------------------------------
# TEST 3: Liczba godzin na stację w zbiorze testowym (7 dni * 24h = 168h)
# ---------------------------------------------------------
print("\n=== TEST 3: Liczba godzin w TEST per stacja i pora roku ===")
hours_per_station_test = df[df['split'] == 'test'].groupby(['season', 'station'])['timestamp'].count().unstack(fill_value=0)
print("Oczekiwana wartość dla pełnej stacji: 168 godzin (7 dni * 24h)")
print(hours_per_station_test)

# ---------------------------------------------------------
# TEST 4: Sprawdzenie wycieku danych (Data Leakage)
# ---------------------------------------------------------
print("\n=== TEST 4: Weryfikacja rozłączności dat między Train i Test ===")
train_dates = set(df[df['split'] == 'train']['date'].unique())
test_dates = set(df[df['split'] == 'test']['date'].unique())
leakage = train_dates.intersection(test_dates)

if len(leakage) == 0:
    print("✅ BRAK WYCIEKU DANYCH: Żaden dzień nie występuje jednocześnie w train i test.")
else:
    print(f"❌ WYKRYTO BŁĄD: Następujące dni są jednocześnie w train i test: {leakage}")

# ---------------------------------------------------------
# TEST 5: Zestawienie podziału
# ---------------------------------------------------------
print("\n=== PODSUMOWANIE PROCENTOWE PODZIAŁU ===")
summary = pd.crosstab(df['season'], df['split'], margins=True)
summary['% testu'] = (summary['test'] / summary['All'] * 100).round(1)
print(summary)
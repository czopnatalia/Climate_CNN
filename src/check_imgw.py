import pandas as pd
from pathlib import Path

balice_dir = Path("data/raw/imgw/balice")
obs_file = Path("data/raw/krakow_obserwatorium_imgw_2000_2025.csv")

# 1. Weryfikacja Balic
balice_files = sorted(list(balice_dir.glob("KRAKOW_BALICE_*.csv")))
print(f"Balice: znaleziono {len(balice_files)} plików rocznych.")
if balice_files:
    df_b = pd.read_csv(balice_files[0])
    print(f"   Przykładowy rok ({balice_files[0].name}): {len(df_b)} wierszy")

# 2. Weryfikacja Obserwatorium
if obs_file.exists():
    df_o = pd.read_csv(obs_file)
    print(f"\nObserwatorium: plik istnieje, liczba wierszy: {len(df_o):,}")
    print(df_o.head(2))
else:
    print(f"\nObserwatorium: brak pliku w {obs_file}. Sprawdź, czy nie leży w folderze 'dataset/imgw/obserwatorium/'.")
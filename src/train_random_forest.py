import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Wczytanie przygotowanych zbiorów
print("1. Wczytywanie zbiorów treningowego i testowego...")
df_train = pd.read_parquet(PROCESSED_DIR / "train_2022.parquet")
df_test = pd.read_parquet(PROCESSED_DIR / "test_2022.parquet")

# 2. Definicja cech wejściowych (X) i zmiennej celu (y)
features = [
    'temp_era5',
    'altitude',
    'urban_fraction_500m',
    'sin_hour',
    'cos_hour',
    'sin_doy',
    'cos_doy'
]
target = 'temp_ground'

X_train, y_train = df_train[features], df_train[target]
X_test, y_test = df_test[features], df_test[target]

print(f"Cechy wejściowe ({len(features)}): {features}")
print(f"Wymiary X_train: {X_train.shape} | X_test: {X_test.shape}")

# 3. Trening Random Forest
print("\n2. Trenowanie modelu Random Forest (n_estimators=100)...")
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=16,
    min_samples_leaf=4,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

# 4. Predykcja i ewaluacja
y_pred_rf = rf.predict(X_test)
y_pred_era5 = df_test['temp_era5']

# Metryki
mae_era5 = mean_absolute_error(y_test, y_pred_era5)
rmse_era5 = np.sqrt(mean_squared_error(y_test, y_pred_era5))
r2_era5 = r2_score(y_test, y_pred_era5)
bias_era5 = (y_pred_era5 - y_test).mean()

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)
bias_rf = (y_pred_rf - y_test).mean()

# 5. Zestawienie wyników
print("\n" + "="*65)
print("📊 WYNIKI NA ZBIORZE TESTOWYM (42 dni - końcówki pór roku)")
print("="*65)
results = pd.DataFrame([
    {
        'Model': 'Baseline 0 (Surowe ERA5)',
        'MAE [°C]': round(mae_era5, 3),
        'RMSE [°C]': round(rmse_era5, 3),
        'R²': round(r2_era5, 4),
        'BIAS [°C]': round(bias_era5, 3)
    },
    {
        'Model': 'Random Forest Regressor',
        'MAE [°C]': round(mae_rf, 3),
        'RMSE [°C]': round(rmse_rf, 3),
        'R²': round(r2_rf, 4),
        'BIAS [°C]': round(bias_rf, 3)
    }
])
print(results.to_string(index=False))

# 6. Ważność cech (Feature Importance)
print("\n--- WAŻNOŚĆ CECH (FEATURE IMPORTANCE) ---")
importances = pd.DataFrame({
    'Cecha': features,
    'Waga': rf.feature_importances_
}).sort_values('Waga', ascending=False)
print(importances.to_string(index=False))

# 7. Zapis wytrenowanego modelu
model_path = MODELS_DIR / "random_forest_baseline.joblib"
joblib.dump(rf, model_path)
print(f"\n✅ Zapisano wytrenowany model do: {model_path}")
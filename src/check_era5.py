import xarray as xr
from pathlib import Path

era5_dir = Path("data/raw/era5")
nc_files = sorted(list(era5_dir.glob("krakow_temp_*.nc")))

print(f"Liczba znalezionych plików NetCDF: {len(nc_files)}")

if nc_files:
    print(f"Pierwszy plik: {nc_files[0].name}")
    print(f"Ostatni plik: {nc_files[-1].name}")
    
    # Otwarcie przykładowego pliku
    ds = xr.open_dataset(nc_files[0])
    print("\nStruktura danych ERA5-Land w pliku:")
    print(ds)
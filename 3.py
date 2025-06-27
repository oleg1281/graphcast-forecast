from datetime import datetime, timedelta
from urllib.parse import urlencode
import requests
import xarray as xr
import numpy as np
import pandas as pd

# Чтобы z скачать одну дату и одно время с НОАА

# Получаем два времени
times = datetime(2025, 5, 30, 18, 0, 0)
#print(times)
#times = times.strftime('%Y-%m-%d %H:%M')

def prepare_graphcast_coordinates(ds, base_datetime_str):
    # ✅ Добавляем ось batch, если её ещё нет
    if "batch" not in ds.dims:
        ds = ds.expand_dims(dim={"batch": [0]})

    # 🎯 Преобразуем time в timedelta64[h]
    if not np.issubdtype(ds["time"].dtype, np.timedelta64):
        ds = ds.assign_coords(time=ds["time"].astype("timedelta64[h]"))

    # 🎯 datetime = base_time + time
    base_time = np.datetime64(base_datetime_str)
    datetime = ds["time"].values + base_time

    ds["datetime"] = (("batch", "time"), np.expand_dims(datetime, axis=0))
    ds = ds.set_coords("datetime")

    return ds

# Параметры
# = datetime.date(2025, 4, 29)
date = times.date()
run_hour = times.strftime('%H')  # Преобразуем в строку вида "06"   Запуск прогноза: 00, 06, 12, 18
forecast_hour = "f000"  # Анализ (без прогноза)

# Список уровней
levels = [
    'lev_1000_mb', 'lev_925_mb', 'lev_850_mb', 'lev_700_mb', 'lev_600_mb',
    'lev_500_mb', 'lev_400_mb', 'lev_300_mb', 'lev_250_mb', 'lev_200_mb',
    'lev_150_mb', 'lev_100_mb', 'lev_50_mb',
    'lev_surface', 'lev_2_m_above_ground', 'lev_10_m_above_ground',
    'lev_mean_sea_level'
]

# Список переменных (в формате, который понимает NOMADS)
variables = [
    'var_TMP', 'var_HGT', 'var_UGRD', 'var_VGRD',
    'var_VVEL', 'var_SPFH', 'var_PRATE',
    'var_DSWRF', 'var_LAND', 'var_PRMSL'
]

# Регион
region = {
    'subregion': '',
    'leftlon': 0,
    'rightlon': 359,
    'toplat': 90,
    'bottomlat': -90,
}

# Параметры запроса
params = {
    'file': f"gfs.t{run_hour}z.pgrb2.1p00.{forecast_hour}",
    **{lvl: "on" for lvl in levels},
    **{var: "on" for var in variables},
    **region,
    'dir': f"/gfs.{date.strftime('%Y%m%d')}/{run_hour}/atmos"
}
print('параметры = ', params)
# Генерация ссылки
base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_1p00.pl?"
download_url = base_url + urlencode(params)

print(download_url)
#--------------------------------------------------------------------------------------------------

# 🔗 Ссылка, сгенерированная ранее (сюда подставь свою)
#url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_1p00.pl?file=gfs.t00z.pgrb2.1p00.f000&lev_1000_mb=on&var_TMP=on&subregion=&leftlon=0&rightlon=359&toplat=90&bottomlat=-90&dir=%2Fgfs.20250420%2F00%2Fatmos"
url = download_url

# 📁 Название файла для сохранения
output_filename = "c:/Users/obedenok/PycharmProjects/Graphcast/temp/gfs_1p00_f000_lll.grib2"

# 📥 Скачиваем
print("📥 Скачивание началось...")
response = requests.get(url)

# 🧾 Сохраняем
with open(output_filename, "wb") as f:
    f.write(response.content)

print(f"✅ Файл сохранён: {output_filename}")
#---------------------------------------------------------------------------------------------------

grib_path = "c:/Users/obedenok/PycharmProjects/Graphcast/temp/gfs_1p00_f000_lll.grib2"

# Функция для безопасной загрузки с фильтром
def load_dataset(typeOfLevel, var_suffix):
    try:
        ds = xr.open_dataset(
            grib_path,
            engine="cfgrib",
            decode_timedelta=False,
            backend_kwargs = {"indexpath": ""},
            filter_by_keys={'typeOfLevel': typeOfLevel}
        )
        # Переименуем переменные с суффиксом, чтобы избежать конфликтов
        return ds.rename({k: f"{k}_{var_suffix}" for k in ds.data_vars})
    except Exception as e:
        print(f"⚠️ Ошибка при загрузке {typeOfLevel}: {e}")
        return None


# Функция для безопасной загрузки с фильтром
def load_dataset_uv(typeOfLevel, var_suffix):
    try:
        ds = xr.open_dataset(
            grib_path,
            engine="cfgrib",
            decode_timedelta=False,
            backend_kwargs = {"indexpath": ""},
            filter_by_keys={'typeOfLevel': typeOfLevel, 'level': 10}
        )
        # Переименуем переменные с суффиксом, чтобы избежать конфликтов
        return ds.rename({k: f"{k}_{var_suffix}" for k in ds.data_vars})
    except Exception as e:
        print(f"⚠️ Ошибка при загрузке {typeOfLevel}: {e}")
        return None


def load_dataset_t(typeOfLevel, var_suffix):
    try:
        ds = xr.open_dataset(
            grib_path,
            engine="cfgrib",
            decode_timedelta=False,
            backend_kwargs = {"indexpath": ""},
            filter_by_keys={'typeOfLevel': typeOfLevel, 'level': 2}
        )
        # Переименуем переменные с суффиксом, чтобы избежать конфликтов
        return ds.rename({k: f"{k}_{var_suffix}" for k in ds.data_vars})
    except Exception as e:
        print(f"⚠️ Ошибка при загрузке {typeOfLevel}: {e}")
        return None



# Загружаем по уровням
ds_pressure = load_dataset("isobaricInhPa", "press")
ds_surface = load_dataset("surface", "surf")
ds_heightAboveGround = load_dataset_uv("heightAboveGround", "height")
ds_t = load_dataset_t("heightAboveGround", "t")
ds_mean_sea = load_dataset("meanSea", "msl")

# Объединяем
datasets = [ds for ds in [ds_pressure, ds_surface, ds_heightAboveGround, ds_t['t2m_t'], ds_mean_sea["prmsl_msl"]] if ds is not None]
ds_all = xr.merge(datasets, compat='override')

# Добавим размерности
ds_all = ds_all.expand_dims({"batch": [0], "time": [0]})

base_datetime_str = f"{date.strftime('%Y-%m-%d')}T{run_hour}:00"
ds_all = prepare_graphcast_coordinates(ds_all, base_datetime_str)

ds_all["orog_surf"] = ds_all["orog_surf"].squeeze(dim=["batch", "time"])
ds_all["lsm_surf"] = ds_all["lsm_surf"].squeeze(dim=["batch", "time"])

#ds_all["v10_height"] = ds_all["v10_height"].expand_dims({"batch": [0], "time": [0]})


print("📦 Переменные в ds_all:")
print(list(ds_all.data_vars))
print(list(ds_all.coords))

if "orog_surf" in ds_all:
    ds_all["orog_surf"] = ds_all["orog_surf"] * 9.80665

if "gh_press" in ds_all:
    ds_all["gh_press"] = ds_all["gh_press"] * 9.80665

# Переименование переменных под GraphCast или свои нужды
rename_map = {
    "gh_press": "geopotential",
    "t_press": "temperature",
    "q_press": "specific_humidity",
    "w_press": "vertical_velocity",
    "u_press": "u_component_of_wind",
    "v_press": "v_component_of_wind",
    "t_surf": "surface_temperature",
    "prate_surf": "total_precipitation_6hr",
    "lsm_surf": "land_sea_mask",
    "orog_surf": "geopotential_at_surface",  # если нужно
    "isobaricInhPa": "level",
    "latitude": "lat",
    "longitude": "lon",
    "u10_height": "10m_u_component_of_wind",
    "v10_height": "10m_v_component_of_wind",
    "t2m_t": "2m_temperature",
    "prmsl_msl": "mean_sea_level_pressure"
}

# Применяем переименование
ds_all = ds_all.rename(rename_map)

# Удалим ненужные переменные
ds_all = ds_all.drop_vars(["batch", "heightAboveGround", "meanSea", "step", "surface", "surface_temperature", "valid_time"])


ds_all = ds_all.sortby("level", ascending=True)


# Приводим lat/lon к float32
ds_all = ds_all.assign_coords({
    "lat": ds_all["lat"].astype(np.float32),
    "lon": ds_all["lon"].astype(np.float32)
})

# Удаляем batch из координат (делаем как в оригинале)
if "batch" in ds_all.coords:
    ds_all = ds_all.swap_dims({"batch": "batch"})  # сохраняем размерность
    ds_all = ds_all.drop_vars("batch")  # убираем как координату

    # Убираем time и batch из geopotential_at_surface и land_sea_mask
if set(ds_all["geopotential_at_surface"].dims) == {"batch", "time", "lat", "lon"}:
    ds_all["geopotential_at_surface"] = ds_all["geopotential_at_surface"].isel(batch=0, time=0)
if set(ds_all["land_sea_mask"].dims) == {"batch", "time", "lat", "lon"}:
    ds_all["land_sea_mask"] = ds_all["land_sea_mask"].isel(batch=0, time=0)

# Сохраняем
out_path = f"c:/Users/obedenok/PycharmProjects/Graphcast/temp/test_{times.strftime('%Y-%m-%d_%H-%M')}.nc"
ds_all.to_netcdf(out_path)
print(f"✅ Всё успешно объединено и сохранено в {out_path}")


#-----------------------------------------------------------------------------------------------
'''
# Загружаем оба файла
ds1 = xr.open_dataset("c:/Users/obedenok/PycharmProjects/graphcast_clean/dataset/gfs_1p00_f000_all1.nc")
ds2 = xr.open_dataset("c:/Users/obedenok/PycharmProjects/graphcast_clean/dataset/gfs_1p00_f000_all2.nc")

# Объединяем по оси time
combined = xr.concat([ds1, ds2], dim="time")


# 🎯 Восстановим datetime и синхронизируем time
if "datetime" in ds1 and "datetime" in ds2:
    print("🛠 Восстанавливаем datetime и time...")

    # Берём datetime как (batch, time)
    dt1 = ds1["datetime"]
    dt2 = ds2["datetime"]

    # Проверим размеры
    assert dt1.dims == ("batch", "time")
    assert dt2.dims == ("batch", "time")

    # Объединяем datetime по оси времени (axis=1)
    dt_combined = np.concatenate([dt1.values, dt2.values], axis=1)

    # Назначаем datetime как координату
    combined["datetime"] = (("batch", "time"), dt_combined)
    combined = combined.set_coords("datetime")

    # Восстановим time как timedelta от первого datetime
    base_datetime = dt_combined[0, 0]
    time_deltas = (dt_combined[0] - base_datetime) / np.timedelta64(1, "h")
    time_values = np.array(time_deltas, dtype="timedelta64[h]")

    # Заменим координату time
    combined = combined.assign_coords(time=("time", time_values))

    print("✅ datetime и time синхронизированы")
else:
    print("⚠️ В одном из файлов нет координаты 'datetime'")



# Удаляем time из статичных переменных
for var in ["geopotential_at_surface", "land_sea_mask"]:
    if var in combined and "time" in combined[var].dims:
        print(f"⚙️ Удаляем ось 'time' из {var}")
        combined[var] = combined[var].isel(time=0, drop=True)

# Исправляем порядок осей
def fix_dims(da):
    dims = da.dims
    if "level" in dims:
        return da.transpose("batch", "time", "level", "lat", "lon")
    elif "time" in dims:
        return da.transpose("batch", "time", "lat", "lon")
    else:
        return da.transpose("lat", "lon")

# Применяем ко всем переменным
for var in combined.data_vars:
    combined[var] = fix_dims(combined[var])

# Убираем лишние координаты
for coord in list(combined.coords):
    if coord not in combined.dims and coord != "datetime":
        combined = combined.drop_vars(coord)

# Устанавливаем желаемый порядок переменных (пример из эталонного файла)
desired_order = [
    'geopotential_at_surface',
    'land_sea_mask',
    '2m_temperature',
    'mean_sea_level_pressure',
    '10m_v_component_of_wind',
    '10m_u_component_of_wind',
    'total_precipitation_6hr',
    'temperature',
    'geopotential',
    'u_component_of_wind',
    'v_component_of_wind',
    'vertical_velocity',
    'specific_humidity'
]

# Собираем датасет заново в нужном порядке
combined_ordered = xr.Dataset({var: combined[var] for var in desired_order if var in combined.data_vars}, coords=combined.coords)

# Сохраняем
combined_ordered.to_netcdf("c:/Users/obedenok/PycharmProjects/graphcast_clean/dataset/combined_gfs.nc")
print("✅ Файл сохранён в правильном формате как combined_gfs.nc")

#-------------------------------------------------------------------------------------------------

# Загружаем исходный файл
ds = xr.open_dataset("c:/Users/obedenok/PycharmProjects/graphcast_clean/dataset/combined_gfs.nc", decode_timedelta=True)

# Получим шаг времени (timedelta)
time_step = ds.time[1].item() - ds.time[0].item()
datetime_step = pd.to_datetime(ds.datetime.values[0, 1]) - pd.to_datetime(ds.datetime.values[0, 0])

# Сколько новых шагов добавим
n_new = 50

# Новые значения времени и даты
new_time = ds.time.values[-1] + time_step * np.arange(1, n_new + 1)
new_datetime = pd.to_datetime(ds.datetime.values[0, -1]) + datetime_step * np.arange(1, n_new + 1)

# Создадим переменные с NaN только для переменных, у которых есть измерение "time"
new_data_vars = {}
for var in ds.data_vars:
    da = ds[var]
    if "time" in da.dims:
        dims = da.dims
        shape = list(da.shape)
        shape[dims.index("time")] = n_new  # заменим размер оси времени

        # координаты, кроме "datetime", иначе будет ошибка при concat
        coords = {k: v for k, v in da.coords.items() if k in dims and k != "datetime"}
        coords["time"] = new_time

        new_da = xr.DataArray(
            data=np.full(shape, np.nan, dtype=da.dtype),
            dims=dims,
            coords=coords
        )

        # Убираем координаты для безопасного слияния
        if "datetime" in da.coords:
            da = da.reset_coords(names="datetime", drop=True)

        # Объединяем
        new_data_vars[var] = xr.concat([da, new_da], dim="time")
    else:
        new_data_vars[var] = da  # без изменений

# Объединяем time и datetime координаты
new_time_full = np.concatenate([ds.time.values, new_time])
new_datetime_full = np.concatenate([ds.datetime.values[0], new_datetime])
datetime_expanded = np.expand_dims(new_datetime_full, axis=0)

# Создаём итоговый Dataset
ds_out = xr.Dataset(new_data_vars)
ds_out = ds_out.assign_coords(time=("time", new_time_full))
ds_out = ds_out.assign_coords(datetime=(("batch", "time"), datetime_expanded))

# если batch была координатой
if "batch" in ds.coords:
    ds_out = ds_out.assign_coords(batch=ds.coords["batch"])

# Сохраняем результат
ds_out.to_netcdf(f"w:/Postprocesing/Oleh Bedenok/GRAPHCAST/NOAA/dataset_NOAA1/NOAA_{datt}.nc", format="NETCDF3_CLASSIC")
print(f"✅ Готово: данные расширены и сохранены в NOAA_{datt}.nc")'''

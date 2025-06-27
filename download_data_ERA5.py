from datetime import datetime, timedelta
import cdsapi
import numpy as np
import xarray as xr
import os
import pandas as pd
from collections import defaultdict


c = cdsapi.Client(
    url='https://cds.climate.copernicus.eu/api',
    key='d1c02362-b7bf-453b-9da4-ce9ddad97925'
)

def download_surface_part1(filename, day, month, hour):
    """
    Скачивает surface-поля ERA5 за указанные datetime (шаг 6 часов),
    затем удаляет лишние временные метки.
    """
    print('📥 Скачивание surface-полей ERA5...')

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                'geopotential',
                'land_sea_mask',
                '2m_temperature',
                'mean_sea_level_pressure',
                '10m_v_component_of_wind',
                '10m_u_component_of_wind',
            ],
            'year': '2025',
            'month': month,
            'day': day,
            'time': hour,
            'format': 'netcdf',
            'grid': [1.0, 1.0],
            #'area': [66.5, 5.0, 48.5, 35.0],  # [North, West, South, East]
        },
        filename
    )

    print(f'✅ Скачивание surface_part1 завершено → {filename}')

def change_part1(filename):
    ds = xr.load_dataset(filename, decode_timedelta=True)

    # Переименовываем координаты
    ds = ds.rename({
        "valid_time": "time",
        "latitude": "lat",
        "longitude": "lon",
        "z": "geopotential_at_surface",
        "lsm": "land_sea_mask",
        "t2m": "2m_temperature",
        "msl": "mean_sea_level_pressure",
        "v10": "10m_v_component_of_wind",
        "u10": "10m_u_component_of_wind",
    })

    # Добавляем ось batch
    ds = ds.expand_dims(dim={"batch": [0]})

    # Преобразуем time в timedelta64[ns]
    base_time = ds.time.values[0]
    time_delta = ds.time.values - base_time
    ds = ds.assign_coords(time=time_delta)

    # Добавляем datetime и делаем его координатой
    datetime = ds.time.values + base_time
    ds["datetime"] = (("batch", "time"), np.expand_dims(datetime, axis=0))
    ds = ds.set_coords("datetime")

    # Удаляем лишние переменные (если есть)
    ds = ds.drop_vars(["number", "expver"], errors="ignore")

    # Меняем порядок осей у переменной
    ds["geopotential_at_surface"] = ds["geopotential_at_surface"].transpose("batch", "time", "lat", "lon")
    ds["land_sea_mask"] = ds["land_sea_mask"].transpose("batch", "time", "lat", "lon")
    ds["2m_temperature"] = ds["2m_temperature"].transpose("batch", "time", "lat", "lon")
    ds["mean_sea_level_pressure"] = ds["mean_sea_level_pressure"].transpose("batch", "time", "lat", "lon")
    ds["10m_v_component_of_wind"] = ds["10m_v_component_of_wind"].transpose("batch", "time", "lat", "lon")
    ds["10m_u_component_of_wind"] = ds["10m_u_component_of_wind"].transpose("batch", "time", "lat", "lon")

    # Сортируем по широте
    ds = ds.sortby("lat")

    # Приводим lat/lon к float32
    ds = ds.assign_coords({
        "lat": ds["lat"].astype(np.float32),
        "lon": ds["lon"].astype(np.float32)
    })

    # Обрезаем до первых 6 временных шагов
    # ds = ds.isel(time=slice(0, 42))

    # Удаляем batch из координат (делаем как в оригинале)
    if "batch" in ds.coords:
        ds = ds.swap_dims({"batch": "batch"})  # сохраняем размерность
        ds = ds.drop_vars("batch")  # убираем как координату

    # Убираем time и batch из geopotential_at_surface и land_sea_mask
    if set(ds["geopotential_at_surface"].dims) == {"batch", "time", "lat", "lon"}:
        ds["geopotential_at_surface"] = ds["geopotential_at_surface"].isel(batch=0, time=0)
    if set(ds["land_sea_mask"].dims) == {"batch", "time", "lat", "lon"}:
        ds["land_sea_mask"] = ds["land_sea_mask"].isel(batch=0, time=0)

    print('Обработка ds_surface_part1 завершена')

    # Сохраняем
    ds.to_netcdf(filename, format="NETCDF4_CLASSIC")




def download_surface_part2(filename, day, month, hour):
    """
        Скачивает surface-поля ERA5 за указанные datetime (шаг 6 часов)
        """
    print('📥 Скачивание поверхностных данных (single-levels_part2)...')

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                'total_precipitation',
                'toa_incident_solar_radiation'
            ],
            'year': '2025',
            'month': month,
            'day': day,
            'time': hour,
            'format': 'netcdf',
            'grid': [1.0, 1.0],
            #'area': [66.5, 5.0, 48.5, 35.0],  # [North, West, South, East]
        },
        filename
    )
    print(f'✅ Скачивание surface_part2 завершено → {filename}')

def change_part2(filename):
    ds = xr.load_dataset(filename, decode_timedelta=True)

    # Переименовываем координаты
    ds = ds.rename({
        "valid_time": "time",
        "tp": "total_precipitation_6hr",
        "latitude": "lat",
        "longitude": "lon",
        "tisr": "toa_incident_solar_radiation"
    })

    # Добавляем ось batch
    ds = ds.expand_dims(dim={"batch": [0]})

    # Преобразуем time в timedelta64[ns]
    base_time = ds.time.values[0]
    time_delta = ds.time.values - base_time
    ds = ds.assign_coords(time=time_delta)

    # Добавляем datetime и делаем его координатой
    datetime = ds.time.values + base_time
    ds["datetime"] = (("batch", "time"), np.expand_dims(datetime, axis=0))
    ds = ds.set_coords("datetime")

    # Удаляем лишние переменные (если есть)
    ds = ds.drop_vars(["number", "expver"], errors="ignore")

    # Меняем порядок осей у переменной
    ds["toa_incident_solar_radiation"] = ds["toa_incident_solar_radiation"].transpose("batch", "time", "lat", "lon")
    ds["total_precipitation_6hr"] = ds["total_precipitation_6hr"].transpose("batch", "time", "lat", "lon")

    # Сортируем по широте
    ds = ds.sortby("lat")

    # Приводим lat/lon к float32
    ds = ds.assign_coords({
        "lat": ds["lat"].astype(np.float32),
        "lon": ds["lon"].astype(np.float32)
    })

    # Обрезаем до первых 6 временных шагов
    # ds = ds.isel(time=slice(0, 42))

    # Удаляем batch из координат (делаем как в оригинале)
    if "batch" in ds.coords:
        ds = ds.swap_dims({"batch": "batch"})  # сохраняем размерность
        ds = ds.drop_vars("batch")  # убираем как координату

    print('Обработка ds_surface_part2 завершена')

    # Сохраняем
    ds.to_netcdf(filename, format="NETCDF4_CLASSIC")


def download_pressure_part1(filename, day, month, hour):
    """
            Скачивает surface-поля ERA5 за указанные datetime (шаг 6 часов)
            """
    print('📥 Скачивание поверхностных данных (pressure-levels_part1)...')

    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                'temperature',
                'geopotential',
                'u_component_of_wind',
                'v_component_of_wind',
                'vertical_velocity',
                'specific_humidity',
            ],
            'pressure_level': [
                '50', '100', '150', '200', '250', '300', '400', '500',
                '600', '700', '850', '925', '1000',
            ],
            'year': '2025',
            'month': month,
            'day': day,
            'time': hour,
            'format': 'netcdf',
            'grid': [1.0, 1.0],
            #'area': [66.5, 5.0, 48.5, 35.0],  # [North, West, South, East]     'area': [90, 0, -90, 359],
        },
        filename
    )
    print(f'✅ Скачивание pressure_part1 завершено → {filename}')

def change_pressure_part1(filename):
    ds = xr.load_dataset(filename, decode_timedelta=True)

    # Переименовываем координаты
    ds = ds.rename({
        "z": "geopotential",
        "w": "vertical_velocity",
        "latitude": "lat",
        "longitude": "lon",
        "valid_time": "time",
        "v": "v_component_of_wind",
        "u": "u_component_of_wind",
        "t": "temperature",
        "q": "specific_humidity",
        "pressure_level": "level"
    })

    # Добавляем ось batch
    ds = ds.expand_dims(dim={"batch": [0]})

    # Преобразуем time в timedelta64[ns]
    base_time = ds.time.values[0]
    time_delta = ds.time.values - base_time
    ds = ds.assign_coords(time=time_delta)

    # Добавляем datetime и делаем его координатой
    datetime = ds.time.values + base_time
    ds["datetime"] = (("batch", "time"), np.expand_dims(datetime, axis=0))
    ds = ds.set_coords("datetime")

    # Удаляем лишние переменные (если есть)
    ds = ds.drop_vars(["number", "expver"], errors="ignore")

    # Меняем порядок осей у переменной
    ds["u_component_of_wind"] = ds["u_component_of_wind"].transpose("batch", "time", "level", "lat", "lon")
    # ds["total_precipitation_6hr"] = ds["total_precipitation_6hr"].transpose("batch", "time", "lat", "lon")

    # Сортируем по широте
    ds = ds.sortby("lat")

    # Oтсортировать уровень давления по возрастанию
    ds = ds.sortby("level")

    # Приводим lat/lon к float32
    ds = ds.assign_coords({
        "lat": ds["lat"].astype(np.float32),
        "lon": ds["lon"].astype(np.float32)
    })

    # Обрезаем до первых 6 временных шагов
    # ds = ds.isel(time=slice(0, 42))

    # Удаляем batch из координат (делаем как в оригинале)
    if "batch" in ds.coords:
        ds = ds.swap_dims({"batch": "batch"})  # сохраняем размерность
        ds = ds.drop_vars("batch")  # убираем как координату

    print('Обработка era5_pressure_part1.nc завершена')

    # Сохраняем
    ds.to_netcdf(filename, format="NETCDF4_CLASSIC")
    print('сохранили файл ')


def join_files(date):
    # Загружаем файлы
    surface1 = xr.open_dataset(f"temp/era5_surface_part1.nc", decode_timedelta=True)
    surface2 = xr.open_dataset(f"temp/era5_surface_part2.nc", decode_timedelta=True)
    pressure = xr.open_dataset(f"temp/era5_pressure_part1.nc", decode_timedelta=True)

    # Преобразуем переменные с фиксированными измерениями (если они есть)
    # for ds in [surface1, surface2, pressure]:
    #    for var in ["geopotential_at_surface", "land_sea_mask"]:
    #        if var in ds:
    #            ds[var] = ds[var].isel(batch=0, time=0)

    # Объединяем в один Dataset
    ds_merged = xr.merge([surface1, surface2, pressure], compat="override")

    # Удаляем лишние служебные переменные, если есть
    ds_merged = ds_merged.drop_vars(["number", "expver", "toa_incident_solar_radiation"],
                                    errors="ignore")  # , "toa_incident_solar_radiation"

    print(f"✅ Объединение surface1, surface2, pressure1 завершено. Сохраняем в файл")
    # Сохраняем в файл
    output_file = f"w:/Postprocesing/Oleh Bedenok/GRAPHCAST/NOAA/TEST_Graphcast_14,05,2025-31,05,2025/data_ERA5/era5_{date}.nc"
    ds_merged.to_netcdf(output_file, format="NETCDF3_CLASSIC")

    print(f"✅ Объединение завершено. Сохранено в файл: {output_file}")


def add_targets(date):
    # Загружаем исходный файл
    ds = xr.open_dataset("temp/out_file.nc", decode_timedelta=True)

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
    ds_out.to_netcdf(f"w:/Postprocesing/Oleh Bedenok/GRAPHCAST/data_dla_tests/out_file_for_graphcast_{date}.nc", format="NETCDF3_CLASSIC")
    print(f"✅ Готово: данные расширены и сохранены в w:/Postprocesing/Oleh Bedenok/GRAPHCAST/data_dla_tests/out_file_for_graphcast_{date}.nc")


#dict_keys = ['23/01_18', '24/01_06', '24/01_18', '25/01_06', '25/01_18', '26/01_06', '26/01_18', '27/01_06', '27/01_18', '28/01_06', '28/01_18', '29/01_06', '29/01_18', '30/01_06', '30/01_18', '31/01_06', '31/01_18', '01/02_06', '01/02_18', '02/02_06', '02/02_18', '03/02_06', '03/02_18', '04/02_06', '04/02_18', '05/02_06', '05/02_18', '06/02_06', '06/02_18', '07/02_06', '07/02_18', '08/02_06', '08/02_18', '09/02_06', '09/02_18', '10/02_06', '10/02_18', '11/02_06', '11/02_18', '12/02_06', '12/02_18', '13/02_06', '13/02_18', '14/02_06', '14/02_18', '15/02_06', '15/02_18', '16/02_06', '16/02_18', '17/02_06', '17/02_18', '18/02_06', '18/02_18', '19/02_06', '19/02_18', '20/02_06', '20/02_18', '21/02_06', '21/02_18', '22/02_06', '22/02_18', '23/02_06', '23/02_18', '24/02_06', '24/02_18', '25/02_06', '25/02_18', '26/02_06', '26/02_18', '27/02_06', '27/02_18', '28/02_06', '28/02_18', '01/03_06', '01/03_18', '02/03_06', '02/03_18', '03/03_06', '03/03_18', '04/03_06', '04/03_18', '05/03_06', '05/03_18', '06/03_06', '06/03_18', '07/03_06', '07/03_18', '08/03_06', '08/03_18', '09/03_06', '09/03_18', '10/03_06', '10/03_18', '11/03_06', '11/03_18', '12/03_06', '12/03_18', '13/03_06', '13/03_18', '14/03_06', '14/03_18', '15/03_06', '15/03_18', '16/03_06', '16/03_18', '17/03_06', '17/03_18', '18/03_06', '18/03_18', '19/03_06', '19/03_18', '20/03_06', '20/03_18', '21/03_06', '21/03_18', '22/03_06', '22/03_18', '23/03_06', '23/03_18', '24/03_06', '24/03_18', '25/03_06', '25/03_18', '26/03_06', '26/03_18', '27/03_06', '27/03_18', '28/03_06', '28/03_18', '29/03_06', '29/03_18', '30/03_06', '30/03_18', '31/03_06', '31/03_18', '01/04_06', '01/04_18', '02/04_06', '02/04_18', '03/04_06', '03/04_18', '04/04_06', '04/04_18', '05/04_06', '05/04_18', '06/04_06', '06/04_18', '07/04_06', '07/04_18', '08/04_06', '08/04_18', '09/04_06', '09/04_18', '10/04_06', '10/04_18', '11/04_06']
#dict_keys = ['23/01_18', '24/01_06', '24/01_18', '25/01_06', '25/01_18', '26/01_06', '26/01_18', '27/01_06', '27/01_18', '28/01_06', '28/01_18', '29/01_06', '29/01_18', '30/01_06', '30/01_18', '31/01_06', '31/01_18', '01/02_06', '01/02_18', '02/02_06', '02/02_18', '03/02_06', '03/02_18', '04/02_06', '04/02_18', '05/02_06', '05/02_18', '06/02_06', '06/02_18', '07/02_06', '07/02_18', '08/02_06', '08/02_18', '09/02_06', '09/02_18', '10/02_06', '10/02_18', '11/02_06', '11/02_18', '12/02_06', '12/02_18', '13/02_06', '13/02_18', '14/02_06', '14/02_18', '15/02_06', '15/02_18', '16/02_06', '16/02_18', '17/02_06', '17/02_18', '18/02_06', '18/02_18', '19/02_06', '19/02_18', '20/02_06', '20/02_18', '21/02_06', '21/02_18', '22/02_06', '22/02_18', '23/02_06', '23/02_18', '24/02_06', '24/02_18', '25/02_06', '25/02_18', '26/02_06', '26/02_18', '27/02_06', '27/02_18', '28/02_06', '28/02_18', '01/03_06', '01/03_18', '02/03_06', '02/03_18', '03/03_06', '03/03_18', '04/03_06', '04/03_18', '05/03_06', '05/03_18', '06/03_06', '06/03_18', '07/03_06', '07/03_18', '08/03_06', '08/03_18', '09/03_06', '09/03_18', '10/03_06', '10/03_18', '11/03_06', '11/03_18', '12/03_06', '12/03_18', '13/03_06', '13/03_18', '14/03_06', '14/03_18', '15/03_06', '15/03_18', '16/03_06', '16/03_18', '17/03_06', '17/03_18', '18/03_06', '18/03_18', '19/03_06', '19/03_18', '20/03_06', '20/03_18', '21/03_06', '21/03_18', '22/03_06', '22/03_18', '23/03_06', '23/03_18', '24/03_06', '24/03_18', '25/03_06', '25/03_18', '26/03_06', '26/03_18', '27/03_06', '27/03_18', '28/03_06', '28/03_18', '29/03_06', '29/03_18', '30/03_06', '30/03_18', '31/03_06', '31/03_18', '01/04_06', '01/04_18', '02/04_06', '02/04_18', '03/04_06', '03/04_18', '04/04_06', '04/04_18', '05/04_06', '05/04_18', '06/04_06', '06/04_18', '07/04_06', '07/04_18', '08/04_06', '08/04_18', '09/04_06', '09/04_18', '10/04_06', '10/04_18', '11/04_06']
dict_keys = ['25/05_06', '25/05_18', '26/05_06', '26/05_18', '27/05_06', '27/05_18', '28/05_06', '28/05_18', '29/05_06', '29/05_18', '30/05_06', '30/05_18', '31/05_06', '31/05_18']

#dict_keys = [k for k in dict_keys if k.endswith('_06')]

for date in dict_keys:
    # Разделяем
    date_part, hour = date.split('_')
    day, month = map(int, date_part.split('/'))
    hour = int(hour)

    # Кол-во дней
    num_days = 13

    # Создаём объект datetime (год можно указать любой, например, 2025)
    dt = datetime(year=2025, month=month, day=day)

    # Словарь, где ключ — месяц, значение — список дней
    result = defaultdict(list)

    for i in range(num_days):
        current = dt + timedelta(days=i)
        month = current.strftime('%m')  # '01', '02', ...
        day = current.strftime('%d')  # '28', '29', ...
        result[month].append(day)

    # Вывод
    # заменить / на _
    date = date.replace('/', '_')
    x = 1
    for month in result:
        print(f"month ['{month}'] day {result[month]}")
        day_str = result[month]
        month_str = month
        times_str = ['00:00', '06:00', '12:00', '18:00']

        download_surface_part1("temp/era5_surface_part1.nc", day_str, month_str, times_str)
        change_part1("temp/era5_surface_part1.nc")

        download_surface_part2("temp/era5_surface_part2.nc", day_str, month_str, times_str)
        change_part2("temp/era5_surface_part2.nc")

        download_pressure_part1("temp/era5_pressure_part1.nc", day_str, month_str, times_str)
        change_pressure_part1("temp/era5_pressure_part1.nc")

        join_files(f'{date}_{x}')

        #add_targets(f'{date}_{x}')

        x += 1
def load_for_dataset():
    import cdsapi
    import datetime
    import numpy as np
    import xarray as xr
    import os
    import pandas as pd

    c = cdsapi.Client(
        url='https://cds.climate.copernicus.eu/api',
        key='d1c02362-b7bf-453b-9da4-ce9ddad97925'
    )

    filename = "era5_surface_part1.nc"

    def download_surface_part1(filename):
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
                'month': '03',
                'day': ['01'],
                'time': ['00:00', '06:00'],
                'format': 'netcdf',
                'grid': [1.0, 1.0],
            },
            filename
        )

        print(f'✅ Скачивание surface_part1 завершено → {filename}')

    def change_part1():
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
        ds.to_netcdf("delete/era5_surface_part1.nc", format="NETCDF4_CLASSIC")

    download_surface_part1(filename)
    change_part1()

    filename = "era5_surface_part2.nc"

    def download_surface_part2(filename):
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
                'month': '03',
                'day': ['01'],
                'time': ['00:00', '06:00'],
                'format': 'netcdf',
                'grid': [1.0, 1.0],

            },
            filename
        )
        print(f'✅ Скачивание surface_part2 завершено → {filename}')

    def change_part2():
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
        ds.to_netcdf("delete/era5_surface_part2.nc", format="NETCDF4_CLASSIC")

    download_surface_part2(filename)
    change_part2()

    filename = "era5_pressure_part1.nc"

    def download_pressure_part1(filename):
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
                'month': '03',
                'day': ['01'],
                'time': ['00:00', '06:00'],
                'format': 'netcdf',
                'grid': [1.0, 1.0],
                'area': [90, 0, -90, 359],
            },
            filename
        )
        print(f'✅ Скачивание pressure_part1 завершено → {filename}')

    def change_pressure_part1():
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
        ds.to_netcdf("delete/era5_pressure_part1.nc", format="NETCDF4_CLASSIC")

    download_pressure_part1(filename)
    change_pressure_part1()

    # 4 4 4 4 4 4 4

    # Загружаем файлы
    surface1 = xr.open_dataset(f"delete/era5_surface_part1.nc", decode_timedelta=True)
    surface2 = xr.open_dataset(f"delete/era5_surface_part2.nc", decode_timedelta=True)
    pressure = xr.open_dataset(f"delete/era5_pressure_part1.nc", decode_timedelta=True)

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

    # Сохраняем в файл
    output_file = f"delete/out_file.nc"
    ds_merged.to_netcdf(output_file, format="NETCDF3_CLASSIC")

    print(f"✅ Объединение завершено. Сохранено в файл: {output_file}")

    # 5 5 5 5 5 5 5 5

    # Загружаем исходный файл
    ds = xr.open_dataset("delete/out_file.nc", decode_timedelta=True)

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
    ds_out.to_netcdf("datasets/NOAA_05052025_00_06.nc", format="NETCDF3_CLASSIC")
    print("✅ Готово: данные расширены и сохранены в datasets/NOAA_05052025_00_06.nc")

load_for_dataset()

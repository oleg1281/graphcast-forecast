import os
import xarray as xr

folder_delete = r"c:\Users\obedenok\PycharmProjects\Graphcast\delete"
folder_predict = r"c:\Users\obedenok\PycharmProjects\Graphcast\predictions"
file = r"pred_NOAA_2025-05-14_06h00m_2025-05-14_12h00m.nc"

out_delete = []
out_predict = []

for file_name in os.listdir(folder_delete):
    if file_name.endswith(".nc") and file_name.startswith("NOAA_"):
        file_path = os.path.join(folder_delete, file_name)
        try:
            ds = xr.open_dataset(file_path, decode_timedelta=True)

            # Первая временная точка
            temp = (ds["2m_temperature"].isel(time=0))#- 273.15

            # Значение в точке lat=55, lon=17 (с ближайшими координатами)
            value = round((temp.sel(lat=55, lon=17, method="nearest").item()), 1)

            out_delete.append(value)

            #print(f"{file_name}: {value:.1f} С")

        except Exception as e:
            print(f"❌ Ошибка в файле {file_name}: {e}")

print('проб данных = ',len(out_delete), "\n", out_delete[2:])


file_path = os.path.join(folder_predict, file)
ds = xr.open_dataset(file_path, decode_timedelta=True)

temp = ds["2m_temperature"]- 273.15

# Выбор значений в точке lat=55, lon=17 по всем time
values = (temp.sel(lat=55, lon=17, method="nearest").values)
values = values.flatten().round(1)  # Преобразуем в 1D-массив
print(f"Температура в точке (55N, 17E) на всех {len(values)} временных шагах:")

print(values)
from os.path import exists

import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Открываем файлы
ds_predict = xr.open_dataset(f"w:/Postprocesing/Oleh Bedenok/GRAPHCAST/NOAA/predict_noaa/pred_NOAA_2025-05-14_00h00m_2025-05-14_06h00m.nc", decode_timedelta=True)
ds_origin = xr.open_dataset(f"w:/Postprocesing/Oleh Bedenok/GRAPHCAST/NOAA/data_era5/era5_06_05_06_1.nc", decode_timedelta=True)
#ds_origin2 = xr.open_dataset(f"w:/Postprocesing/Oleh Bedenok/GRAPHCAST/data_dla_tests_025_13/out_file_01_02_18_2.nc", decode_timedelta=True)
df_purchase = pd.read_excel("w:/Postprocesing/Oleh Bedenok/GRAPHCAST/NOAA/raports/report_tables.xlsx", sheet_name=f"06_05_06")


# Преобразуем температуру в градусы Цельсия
ds_predict["2m_temperature_celsius"] = (ds_predict["2m_temperature"] - 273.15).round(1)
ds_origin["2m_temperature_celsius"] = (ds_origin["2m_temperature"] - 273.15).round(1)
#ds_origin2["2m_temperature_celsius"] = (ds_origin2["2m_temperature"] - 273.15).round(1)

# Извлекаем данные для точки (55, 17)
temp_c_predict = ds_predict["2m_temperature_celsius"].sel(lat=55, lon=17, method="nearest").values.flatten()
#temp_k_predict = ds_predict["2m_temperature"].sel(lat=55, lon=17, method="nearest").values.flatten()
temp_c_origin = ds_origin["2m_temperature_celsius"].sel(lat=55, lon=17, method="nearest").values.flatten()
#temp_k_origin = ds_origin["2m_temperature"].sel(lat=55, lon=17, method="nearest").values.flatten()
#temp_c_origin2 = ds_origin2["2m_temperature_celsius"].sel(lat=55, lon=17, method="nearest").values.flatten()
#temp_k_origin2 = ds_origin2["2m_temperature"].sel(lat=55, lon=17, method="nearest").values.flatten()
temp_c_purchase = df_purchase['T2m'].values.flatten()

# Удаляем первые 3 значения из первого набора
temp_c_origin = temp_c_origin[2:]
#temp_k_origin = temp_k_origin[3:]
temp_c_purchase = temp_c_purchase[1:]

# Объединяем с данными из второго файла
#temp_c_origin = np.concatenate([temp_c_origin, temp_c_origin2])
#temp_k_origin = np.concatenate([temp_k_origin, temp_k_origin2])

# 🛠 Безопасное дополнение (или обрезка) массивов до одинаковой длины
def safe_pad(array, target_len):
    current_len = len(array)
    if current_len < target_len:
        return np.pad(array, (0, target_len - current_len), constant_values=np.nan)
    else:
        return array[:target_len]  # 🟠 обрезаем, если длиннее

# Определяем максимальную длину
max_len = max(
    len(temp_c_predict),
    len(temp_c_origin),
    len(temp_c_purchase)  # 🛠 добавлено, чтобы точно не было ошибок
)

# Применяем безопасную функцию
temp_c_predict = safe_pad(temp_c_predict, max_len)
#temp_k_predict = safe_pad(temp_k_predict, max_len)
temp_c_origin  = safe_pad(temp_c_origin, max_len)
#temp_k_origin  = safe_pad(temp_k_origin, max_len)
temp_c_purchase = safe_pad(temp_c_purchase.astype(float), max_len)


# Создаем даты: начиная с 2025-01-23 18:00 каждые 6 часов
start_time = datetime(2025, 5, 6, 12)
dates = [start_time + timedelta(hours=6 * i) for i in range(max_len)]

# Формируем DataFrame
df = pd.DataFrame({
    "Data": [dt.strftime("%d.%m.%Y %H:%M") for dt in dates],
    "Pred (°C)": temp_c_predict,
    "Orig (°C)": temp_c_origin,
    "Purchase (°C)": temp_c_purchase
})
print(temp_c_origin)
print(df["Orig (°C)"])
df.set_index("Data", inplace=True)

# Вывод таблицы
print(df)

# Построение графика
ax = df.plot(
    y=["Pred (°C)", "Orig (°C)", "Purchase (°C)"],  # <== добавь третью колонку
    figsize=(12, 6),
    title="Температура в точке (55°N, 17°E)"
)

# Настройка сетки и подписей
plt.grid()
plt.ylabel("Температура (°C)")
plt.xticks(rotation=90)

# ✅ Отображаем все метки по оси X
ax.set_xticks(range(len(df.index)))  # Каждая дата
ax.set_xticklabels(df.index, rotation=90)

# ✅ Устанавливаем целочисленные деления по оси Y
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# ✅ Заливка зелёным первых 20 интервалов
for i in range(19):  # 20 интервалов — это 19 промежутков
    ax.axvspan(df.index[i], df.index[i + 1], color="green", alpha=0.15)

# ✅ Устанавливаем границы и шаги по оси Y
ax.set_ylim(-5, 10)
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

plt.tight_layout()
plt.show()

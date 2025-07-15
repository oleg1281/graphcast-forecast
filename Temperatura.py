from os.path import exists
import os
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def noaa_data():
    folder = r"W:\Postprocesing\Oleh Bedenok\GRAPHCAST\NOAA\TEST_Graphcast_14,05,2025-31,05,2025\dataset_NOAA"
    data_NOAA_C = []
    data_NOAA_K = []

    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".nc"):
            filepath = os.path.join(folder, filename)
            try:
                ds = xr.open_dataset(filepath, decode_timedelta=True)
                temp = ds["2m_temperature"].isel(time=0).sel(lat=55, lon=17, method="nearest")
                temp_value_K = round(float(temp.values.item()), 1)
                temp_value = round(float(temp.values.item() - 273.15), 1)
                data_NOAA_C.append(temp_value)
                data_NOAA_K.append(temp_value_K)
            except Exception as e:
                print(f"❌ Помилка у файлі {filename}: {e}")

    #print(f"✅ Значення температури у точці (55, 17) для кожного файлу: {len(data_NOAA_C)}")

    dates = [
        '14.05.2025 18:00', '15.05.2025 00:00', '15.05.2025 06:00', '15.05.2025 12:00', '15.05.2025 18:00',
        '16.05.2025 00:00',
        '16.05.2025 06:00', '16.05.2025 12:00', '16.05.2025 18:00', '17.05.2025 00:00', '17.05.2025 06:00',
        '17.05.2025 12:00',
        '17.05.2025 18:00', '18.05.2025 00:00', '18.05.2025 06:00', '18.05.2025 12:00', '18.05.2025 18:00',
        '19.05.2025 00:00',
        '19.05.2025 06:00', '19.05.2025 12:00', '19.05.2025 18:00', '20.05.2025 00:00', '20.05.2025 06:00',
        '20.05.2025 12:00',
        '20.05.2025 18:00', '21.05.2025 00:00', '21.05.2025 06:00', '21.05.2025 12:00', '21.05.2025 18:00',
        '22.05.2025 00:00',
        '22.05.2025 06:00', '22.05.2025 12:00', '22.05.2025 18:00', '23.05.2025 00:00', '23.05.2025 06:00',
        '23.05.2025 12:00',
        '23.05.2025 18:00', '24.05.2025 00:00', '24.05.2025 06:00', '24.05.2025 12:00', '24.05.2025 18:00',
        '25.05.2025 00:00',
        '25.05.2025 06:00', '25.05.2025 12:00', '25.05.2025 18:00', '26.05.2025 00:00', '26.05.2025 06:00',
        '26.05.2025 12:00',
        '26.05.2025 18:00', '27.05.2025 00:00', '27.05.2025 06:00', '27.05.2025 12:00', '27.05.2025 18:00',
        '28.05.2025 00:00',
        '28.05.2025 06:00', '28.05.2025 12:00', '28.05.2025 18:00', '29.05.2025 00:00', '29.05.2025 06:00',
        '29.05.2025 12:00',
        '29.05.2025 18:00', '30.05.2025 00:00', '30.05.2025 06:00', '30.05.2025 12:00', '30.05.2025 18:00',
        '31.05.2025 00:00',
        '31.05.2025 06:00', '31.05.2025 12:00', '31.05.2025 18:00', '01.06.2025 00:00', '01.06.2025 06:00',
        '01.06.2025 12:00',
        '01.06.2025 18:00', '02.06.2025 00:00', '02.06.2025 06:00', '02.06.2025 12:00', '02.06.2025 18:00',
        '03.06.2025 00:00',
        '03.06.2025 06:00', '03.06.2025 12:00', '03.06.2025 18:00', '04.06.2025 00:00', '04.06.2025 06:00',
        '04.06.2025 12:00',
        '04.06.2025 18:00', '05.06.2025 00:00', '05.06.2025 06:00', '05.06.2025 12:00', '05.06.2025 18:00',
        '06.06.2025 00:00',
        '06.06.2025 06:00', '06.06.2025 12:00', '06.06.2025 18:00', '07.06.2025 00:00', '07.06.2025 06:00',
        '07.06.2025 12:00',
        '07.06.2025 18:00', '08.06.2025 00:00', '08.06.2025 06:00', '08.06.2025 12:00', '08.06.2025 18:00',
        '09.06.2025 00:00',
        '09.06.2025 06:00', '09.06.2025 12:00', '09.06.2025 18:00', '10.06.2025 00:00', '10.06.2025 06:00',
        '10.06.2025 12:00',
        '10.06.2025 18:00'
    ]

    index = pd.to_datetime(dates, dayfirst=True)

    while len(data_NOAA_C) < len(index):
        data_NOAA_C.append(None)

    # Усечение NOAA при необходимости
    data_NOAA_C = data_NOAA_C[:len(index)]
    noa_data = pd.DataFrame({'NOAA': data_NOAA_C}, index=index)
    return noa_data

noaa_date = noaa_data()
print(noaa_date)


date = '14_05_18'

x = 0

def pred(date):
    global x

    # Открываем файлы
    ds_predict_NOAA = xr.open_dataset(
        f"w:/Postprocesing/Oleh Bedenok/GRAPHCAST/NOAA/TEST_Graphcast_14,05,2025-31,05,2025/predict_noaa/pred_NOAA_{date}.nc",
        decode_timedelta=True)
    #print(ds_predict_NOAA["2m_temperature"])

    ds_predict = xr.open_dataset(f"w:/Postprocesing/Oleh Bedenok/GRAPHCAST/NOAA/TEST_Graphcast_14,05,2025-31,05,2025/predict_ERA5/pred_out_file_14_05_18.nc",  #{date}_1.nc
                                 decode_timedelta=True)
    ds_origin = xr.open_dataset(f"w:/Postprocesing/Oleh Bedenok/GRAPHCAST/NOAA/TEST_Graphcast_14,05,2025-31,05,2025/data_ERA5/era5_{date}_1.nc",
                                decode_timedelta=True)

    if os.path.exists(f"w:/Postprocesing/Oleh Bedenok/GRAPHCAST/NOAA/TEST_Graphcast_14,05,2025-31,05,2025/data_ERA5/era5_{date}_2.nc"):
        ds_origin2 = xr.open_dataset(f"w:/Postprocesing/Oleh Bedenok/GRAPHCAST/NOAA/TEST_Graphcast_14,05,2025-31,05,2025/data_ERA5/era5_{date}_2.nc",
                                     decode_timedelta=True)
        ds_origin2["2m_temperature_celsius"] = (ds_origin2["2m_temperature"] - 273.15).round(1)
        temp_c_origin2 = ds_origin2["2m_temperature_celsius"].sel(lat=55, lon=17, method="nearest").values.flatten()
        temp_k_origin2 = ds_origin2["2m_temperature"].sel(lat=55, lon=17, method="nearest").values.flatten()

    df_purchase = pd.read_excel(f"w:/Postprocesing/Oleh Bedenok/GRAPHCAST/NOAA/TEST_Graphcast_14,05,2025-31,05,2025/report_tables.xlsx", sheet_name=f"{date}")

    # Преобразуем температуру в градусы Цельсия
    ds_predict["2m_temperature_celsius"] = (ds_predict["2m_temperature"] - 273.15).round(1)
    ds_origin["2m_temperature_celsius"] = (ds_origin["2m_temperature"] - 273.15).round(1)
    ds_predict_NOAA["2m_temperature_celsius"] = (ds_predict_NOAA["2m_temperature"] - 273.15).round(1)

    # Извлекаем данные для точки (55, 17)
    temp_c_predict = ds_predict["2m_temperature_celsius"].sel(lat=55, lon=17, method="nearest").values.flatten()
    temp_k_predict = ds_predict["2m_temperature"].sel(lat=55, lon=17, method="nearest").values.flatten()
    temp_c_origin = ds_origin["2m_temperature_celsius"].sel(lat=55, lon=17, method="nearest").values.flatten()
    temp_k_origin = ds_origin["2m_temperature"].sel(lat=55, lon=17, method="nearest").values.flatten()

    temp_c_predict_NOAA = ds_predict_NOAA["2m_temperature_celsius"].sel(lat=55, lon=17, method="nearest").values.flatten()

    temp_c_purchase = df_purchase['T2m'].values.flatten()

    # Удаляем первые 3 значения из первого набора
    temp_c_origin = temp_c_origin[3:]
    temp_k_origin = temp_k_origin[3:]

    # Объединяем с данными из второго файла
    if os.path.exists(f"w:/Postprocesing/Oleh Bedenok/GRAPHCAST/NOAA/TEST_Graphcast_14,05,2025-31,05,2025/data_ERA5/era5_{date}_2.nc"):
        temp_c_origin = np.concatenate([temp_c_origin, temp_c_origin2])
        temp_k_origin = np.concatenate([temp_k_origin, temp_k_origin2])

    # Определяем максимальную длину
    max_len = max(len(temp_c_predict), len(temp_c_origin))

    # Дополняем короткие массивы NaN до максимальной длины
    temp_c_predict = np.pad(temp_c_predict, (0, max_len - len(temp_c_predict)), constant_values=np.nan)
    temp_k_predict = np.pad(temp_k_predict, (0, max_len - len(temp_k_predict)), constant_values=np.nan)
    temp_c_origin = np.pad(temp_c_origin, (0, max_len - len(temp_c_origin)), constant_values=np.nan)
    temp_k_origin = np.pad(temp_k_origin, (0, max_len - len(temp_k_origin)), constant_values=np.nan)
    temp_c_purchase = np.pad(temp_c_purchase.astype(float), (0, max_len - len(temp_c_purchase)), constant_values=np.nan)
    temp_c_predict_NOAA = np.pad(temp_c_predict_NOAA, (0, max_len - len(temp_c_predict_NOAA)), constant_values=np.nan)

    # Преобразуем строку '22_02_18' в datetime-объект
    start_time = datetime.strptime(date, "%d_%m_%H").replace(year=2025)

    # Генерируем список дат каждые 6 часов
    dates = [start_time + timedelta(hours=6 * i) for i in range(max_len)]
    #print(len(noaa_date[:108]))

    noaa_d = noaa_date["NOAA"][x:50 + x].tolist()
    noaa_d = noaa_d + [np.nan] * (50 - len(noaa_d))  # дополнение до длины 50
    print(len(temp_c_purchase), len(temp_c_predict_NOAA), len(noaa_d))

    # Формируем DataFrame
    df = pd.DataFrame({
        "Data": [dt.strftime("%d.%m.%Y %H:%M") for dt in dates],
        "Prognoza MEWO (°C)": temp_c_predict,
        "dane rzeczywiste ERA5 (°C)": temp_c_origin,
        "Prognoza MEWO (K)": temp_k_predict,
        "dane rzeczywiste ERA5 (K)": temp_k_origin,
        "prognoza kupowana od StormGeo (°C)": temp_c_purchase,
        "Prognoza MEWO NOAA (°C)": temp_c_predict_NOAA,
        "data_NOAA (°C)": noaa_d

    })

    x+=2

    df.set_index("Data", inplace=True)

    # Вывод таблицы
    #print(df)

    dt_minus_6 = dates[0] - timedelta(hours=6)

    # Построение графика
    fig, ax = plt.subplots(figsize=(12, 6))

    # Линии с индивидуальными стилями
    ax.plot(df.index, df["prognoza kupowana od StormGeo (°C)"], label="prognoza kupowana od StormGeo (°C)",
            linestyle='--', linewidth=1.5)
    ax.plot(df.index, df["Prognoza MEWO NOAA (°C)"], label="Prognoza MEWO (°C)", linestyle='--', linewidth=1.5)
    ax.plot(df.index, df["data_NOAA (°C)"], label="dane rzeczywiste NOAA (°C)", linestyle='-', linewidth=2.5)

    # Заголовок и подписи
    ax.set_title(f"Temperatura w punkcie (55°N, 17°E) prognoza zrobiona {dt_minus_6} siatka 1,0")
    ax.set_ylabel("Temperatura (°C)")

    # Настройка сетки
    ax.grid(True)

    # Метки по оси X
    ax.set_xticks(range(len(df.index)))
    ax.set_xticklabels(df.index, rotation=90)

    # Заливка зелёным первых 20 интервалов
    for i in range(min(20, len(df.index) - 1)):
        ax.axvspan(df.index[i], df.index[i + 1], color="green", alpha=0.15)

    # Настройка оси Y
    ax.set_ylim(0, 15)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    # Легенда и компоновка
    ax.legend()
    plt.tight_layout()

    return fig


dict_keys = ['14/05_18', '15/05_06', '15/05_18', '16/05_06', '16/05_18', '17/05_06', '17/05_18', '18/05_06', '18/05_18', '19/05_06', '19/05_18', '20/05_06', '20/05_18', '21/05_06', '21/05_18', '22/05_06', '22/05_18', '23/05_06', '23/05_18', '24/05_06', '24/05_18', '25/05_06', '25/05_18', '26/05_06', '26/05_18', '27/05_06', '27/05_18', '28/05_06', '28/05_18', '29/05_06', '29/05_18', '30/05_06', '30/05_18', '31/05_06', '31/05_18']
#dict_keys = [k for k in dict_keys if k.endswith('_18')]


from matplotlib.backends.backend_pdf import PdfPages
import os

# 🎯 Подготовка ключей
#dict_keys = [k for k in dict_keys if k.endswith('_18')]
dict_keys = [k.replace('/', '_') for k in dict_keys]

# 📄 Куда сохранить итоговый PDF
output_pdf_path = "w:/Postprocesing/Oleh Bedenok/GRAPHCAST/NOAA/TEST_Graphcast_14,05,2025-31,05,2025/raport/temperature_report_all.pdf"
os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)

# 🖼️ Генерация всех графиков и сохранение в PDF
with PdfPages(output_pdf_path) as pdf:
    for date in dict_keys:
        print(f"Обработка {date}...")
        try:
            fig = pred(date)
            if date not in ['14_05_18', '15_05_06', '15_05_18', '16_05_06']:
                pdf.savefig(fig)
            plt.close(fig)
        except Exception as e:
            print(f"❌ Ошибка при обработке {date}: {e}")

print(f"✅ Все графики сохранены в: {output_pdf_path}")


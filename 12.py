from os.path import exists
import os
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#date = '23_01_18'

def pred():
    # Открываем файлы
    ds_predict = xr.open_dataset(f"w:/Postprocesing/Oleh Bedenok/GRAPHCAST/NOAA/predict_noaa/pred_NOAA_2025-05-14_00h00m_2025-05-14_06h00m.nc",
                                 decode_timedelta=True)
    ds_origin = xr.open_dataset(f"w:/Postprocesing/Oleh Bedenok/GRAPHCAST/data_dla_tests/out_file_{date}_1.nc",
                                decode_timedelta=True)

    if os.path.exists(f"w:/Postprocesing/Oleh Bedenok/GRAPHCAST/data_dla_tests/out_file_{date}_2.nc"):
        ds_origin2 = xr.open_dataset(f"w:/Postprocesing/Oleh Bedenok/GRAPHCAST/data_dla_tests/out_file_{date}_2.nc",
                                     decode_timedelta=True)
        ds_origin2["2m_temperature_celsius"] = (ds_origin2["2m_temperature"] - 273.15).round(1)
        temp_c_origin2 = ds_origin2["2m_temperature_celsius"].sel(lat=55, lon=17, method="nearest").values.flatten()
        temp_k_origin2 = ds_origin2["2m_temperature"].sel(lat=55, lon=17, method="nearest").values.flatten()

    df_purchase = pd.read_excel(f"w:/Postprocesing/Oleh Bedenok/GRAPHCAST/reports/report_tables.xlsx", sheet_name=f"{date}")

    # Преобразуем температуру в градусы Цельсия
    ds_predict["2m_temperature_celsius"] = (ds_predict["2m_temperature"] - 273.15).round(1)
    ds_origin["2m_temperature_celsius"] = (ds_origin["2m_temperature"] - 273.15).round(1)

    # Извлекаем данные для точки (55, 17)
    temp_c_predict = ds_predict["2m_temperature_celsius"].sel(lat=55, lon=17, method="nearest").values.flatten()
    temp_k_predict = ds_predict["2m_temperature"].sel(lat=55, lon=17, method="nearest").values.flatten()
    temp_c_origin = ds_origin["2m_temperature_celsius"].sel(lat=55, lon=17, method="nearest").values.flatten()
    temp_k_origin = ds_origin["2m_temperature"].sel(lat=55, lon=17, method="nearest").values.flatten()

    temp_c_purchase = df_purchase['T2m'].values.flatten()

    # Удаляем первые 3 значения из первого набора
    temp_c_origin = temp_c_origin[3:]
    temp_k_origin = temp_k_origin[3:]

    # Объединяем с данными из второго файла
    if os.path.exists(f"w:/Postprocesing/Oleh Bedenok/GRAPHCAST/data_dla_tests/out_file_{date}_2.nc"):
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

    # Преобразуем строку '22_02_18' в datetime-объект
    start_time = datetime.strptime(date, "%d_%m_%H").replace(year=2025)

    # Генерируем список дат каждые 6 часов
    dates = [start_time + timedelta(hours=6 * i) for i in range(max_len)]

    # Формируем DataFrame
    df = pd.DataFrame({
        "Data": [dt.strftime("%d.%m.%Y %H:%M") for dt in dates],
        "Prognoza MEWO (°C)": temp_c_predict,
        "dane rzeczywiste ERA5 (°C)": temp_c_origin,
        "Prognoza MEWO (K)": temp_k_predict,
        "dane rzeczywiste ERA5 (K)": temp_k_origin,
        "prognoza kupowana od StormGeo (°C)": temp_c_purchase
    })

    df.set_index("Data", inplace=True)

    # Вывод таблицы
    #print(df)

    dt_minus_6 = dates[0] - timedelta(hours=6)

    # Построение графика
    ax = df.plot(
        y=["Prognoza MEWO (°C)", "dane rzeczywiste ERA5 (°C)", "prognoza kupowana od StormGeo (°C)"],
        figsize=(12, 6),
        title=f"Temperatura w punkcie (55°N, 17°E) prognoza zrobiona {dt_minus_6} siatka 0,25 "
    )

    # Настройка сетки и подписей
    plt.grid()
    plt.ylabel("Temperatura (°C)")
    plt.xticks(rotation=90)

    # Отображаем все метки по оси X
    ax.set_xticks(range(len(df.index)))
    ax.set_xticklabels(df.index, rotation=90)

    # Целочисленные деления по оси Y
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Заливка зелёным первых 20 интервалов
    for i in range(19):
        ax.axvspan(df.index[i], df.index[i + 1], color="green", alpha=0.15)

    # Границы и шаги по оси Y
    ax.set_ylim(-5, 10)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    plt.tight_layout()

    # ✅ Показываем на 5 секунд и закрываем
    #plt.pause(10)
    #plt.close()

    plt.tight_layout()
    fig = plt.gcf()
    return fig


pred()

'''
#dict_keys = ['23/01_18', '24/01_06', '24/01_18', '25/01_06', '25/01_18', '26/01_06', '26/01_18', '27/01_06', '27/01_18', '28/01_06', '28/01_18', '29/01_06', '29/01_18', '30/01_06', '30/01_18', '31/01_06', '31/01_18', '01/02_06', '01/02_18', '02/02_06', '02/02_18', '03/02_06', '03/02_18', '04/02_06', '04/02_18', '05/02_06', '05/02_18', '06/02_06', '06/02_18', '07/02_06', '07/02_18', '08/02_06', '08/02_18', '09/02_06', '09/02_18', '10/02_06', '10/02_18', '11/02_06', '11/02_18', '12/02_06', '12/02_18', '13/02_06', '13/02_18', '14/02_06', '14/02_18', '15/02_06', '15/02_18', '16/02_06', '16/02_18', '17/02_06', '17/02_18', '18/02_06', '18/02_18', '19/02_06', '19/02_18', '20/02_06', '20/02_18', '21/02_06', '21/02_18', '22/02_06', '22/02_18', '23/02_06', '23/02_18', '24/02_06', '24/02_18', '25/02_06', '25/02_18', '26/02_06', '26/02_18', '27/02_06', '27/02_18', '28/02_06', '28/02_18', '01/03_06', '01/03_18', '02/03_06', '02/03_18', '03/03_06', '03/03_18', '04/03_06', '04/03_18', '05/03_06', '05/03_18', '06/03_06', '06/03_18', '07/03_06', '07/03_18', '08/03_06', '08/03_18', '09/03_06', '09/03_18', '10/03_06', '10/03_18', '11/03_06', '11/03_18', '12/03_06', '12/03_18', '13/03_06', '13/03_18', '14/03_06', '14/03_18', '15/03_06', '15/03_18', '16/03_06', '16/03_18', '17/03_06', '17/03_18', '18/03_06', '18/03_18', '19/03_06', '19/03_18', '20/03_06', '20/03_18', '21/03_06', '21/03_18', '22/03_06', '22/03_18', '23/03_06', '23/03_18', '24/03_06', '24/03_18', '25/03_06', '25/03_18', '26/03_06', '26/03_18', '27/03_06', '27/03_18', '28/03_06', '28/03_18', '29/03_06', '29/03_18', '30/03_06', '30/03_18', '31/03_06', '31/03_18', '01/04_06', '01/04_18', '02/04_06', '02/04_18', '03/04_06', '03/04_18', '04/04_06', '04/04_18', '05/04_06', '05/04_18', '06/04_06', '06/04_18', '07/04_06', '07/04_18', '08/04_06', '08/04_18', '09/04_06', '09/04_18', '10/04_06', '10/04_18', '11/04_06']
#dict_keys = [k for k in dict_keys if k.endswith('_18')]



from matplotlib.backends.backend_pdf import PdfPages
import os

# 🎯 Подготовка ключей
dict_keys = [k for k in dict_keys if k.endswith('_18')]
dict_keys = [k.replace('/', '_') for k in dict_keys]

# 📄 Куда сохранить итоговый PDF
output_pdf_path = "w:/Postprocesing/Oleh Bedenok/GRAPHCAST/reports/temperature_report_025.pdf"
os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)

# 🖼️ Генерация всех графиков и сохранение в PDF
with PdfPages(output_pdf_path) as pdf:
    for date in dict_keys:
        print(f"Обработка {date}...")
        try:
            fig = pred(date)       # вызываем функцию
            pdf.savefig(fig)       # сохраняем график в PDF
            plt.close(fig)         # закрываем его, чтобы не показывался
        except Exception as e:
            print(f"❌ Ошибка при обработке {date}: {e}")

print(f"✅ Все графики сохранены в: {output_pdf_path}")'''


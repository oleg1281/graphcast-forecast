from os.path import exists
import os
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

-------------------------------------------------  нерабочий ----------------------------------------

def noaa_data():
    folder = r"W:\Postprocesing\Oleh Bedenok\GRAPHCAST\NOAA\TEST_Graphcast_14,05,2025-31,05,2025\dataset_NOAA"
    data_NOAA_knots = []


    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".nc"):
            filepath = os.path.join(folder, filename)
            try:
                ds = xr.open_dataset(filepath, decode_timedelta=True)
                temp_u = ds["10m_u_component_of_wind"].isel(time=0).sel(lat=55, lon=17, method="nearest").values.flatten()
                temp_v = ds["10m_v_component_of_wind"].isel(time=0).sel(lat=55, lon=17, method="nearest").values.flatten()

                # Рассчитываем скорость
                ws10_o = np.sqrt(temp_u ** 2 + temp_v ** 2)
                ws10_knots_o = ws10_o * 1.94384
                #print(f'temp_u={temp_u}, temp_v={temp_v}, ws10_knots_o={ws10_knots_o}')

                data_NOAA_knots.append(ws10_knots_o.round(1))

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

    while len(data_NOAA_knots) < len(index):
        data_NOAA_knots.append(None)

    # Усечение NOAA при необходимости
    data_NOAA_C = data_NOAA_knots[:len(index)]
    noa_data = pd.DataFrame({'NOAA_original': data_NOAA_C}, index=index)
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

    # Извлекаем компоненты ветра
    u10_o = ds_predict_NOAA["10m_u_component_of_wind"].sel(lat=55, lon=17, method="nearest").values.flatten()
    v10_o = ds_predict_NOAA["10m_v_component_of_wind"].sel(lat=55, lon=17, method="nearest").values.flatten()
    # Рассчитываем скорость
    ws10_o = np.sqrt(u10_o ** 2 + v10_o ** 2)
    ws10_knots_o = ws10_o * 1.94384
    #print(f"ws10_knots_o={ws10_knots_o}")  # 50 список
    print(f'noaa_date= {noaa_date}')  # 109 pandas


    df_purchase = pd.read_excel(f"w:/Postprocesing/Oleh Bedenok/GRAPHCAST/NOAA/TEST_Graphcast_14,05,2025-31,05,2025/report_tables.xlsx", sheet_name=f"{date}")

    Ws10m_purchase = df_purchase['Ws10m'].values.flatten()
    #print(f"Ws10m_purchase= {Ws10m_purchase}") # 24 список

    # Преобразуем списки к длине index
    ws10_knots_o_full = ws10_knots_o + [None] * (109 - len(ws10_knots_o))
    Ws10m_purchase_full = Ws10m_purchase + [None] * (109 - len(Ws10m_purchase))

    # Создаем новый DataFrame на основе существующего
    noaa_date["Ws10m_predict"] = ws10_knots_o_full
    noaa_date["Ws10m_purchase"] = Ws10m_purchase_full

    # Печать
    print(noaa_date)


    # Определяем максимальную длину
    max_len = max(len(noaa_date), len(ws10_knots_o))


    # Преобразуем строку '22_02_18' в datetime-объект
    start_time = datetime.strptime(date, "%d_%m_%H").replace(year=2025)

    # Генерируем список дат каждые 6 часов
    dates = [start_time + timedelta(hours=6 * i) for i in range(max_len)]

    noaa_d = noaa_date["NOAA"][x:50 + x].tolist()

    # Формируем DataFrame
    df = pd.DataFrame({
        "Data": [dt.strftime("%d.%m.%Y %H:%M") for dt in dates],
        "ws10 MEWO (knots)": ws10_knots_p,
        "dane rzeczywiste ERA5 (knots)": ws10_knots_o,
        "ws10 prognoza kupowana od StormGeo (knots)": Ws10m_purchase
    })

    x += 4

    df.set_index("Data", inplace=True)

    # Вывод таблицы
    #print(df)

    dt_minus_6 = dates[0] - timedelta(hours=6)

    # Построение графика
    ax = df.plot(
        y=["ws10 MEWO (knots)", "dane rzeczywiste ERA5 (knots)", "ws10 prognoza kupowana od StormGeo (knots)"],
        figsize=(12, 6),
        title=f"Wiatr w punkce (55°N, 17°E) prognoza zrobiona {dt_minus_6}"
    )

    # Настройка сетки и подписей
    plt.grid()
    plt.ylabel("Wiatr (knots)")
    plt.xticks(rotation=90)

    # Горизонтальная линия на уровне 10 m/s ≈ 19.44 узлов
    #ax.axhline(y=19.44, color="red", linestyle="--", linewidth=1.5, label="10 m/s")

    # Отображаем все метки по оси X
    ax.set_xticks(range(len(df.index)))
    ax.set_xticklabels(df.index, rotation=90)

    # Целочисленные деления по оси Y
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Заливка зелёным первых 20 интервалов
    for i in range(19):
        ax.axvspan(df.index[i], df.index[i + 1], color="green", alpha=0.15)

    # Границы и шаги по оси Y
    ax.set_ylim(-10, 30)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

    plt.tight_layout()

    # ✅ Показываем на 5 секунд и закрываем
    # plt.pause(10)
    # plt.close()

    plt.tight_layout()
    fig = plt.gcf()
    return fig


dict_keys = ['14/05_18', '15/05_06', '15/05_18', '16/05_06', '16/05_18', '17/05_06', '17/05_18', '18/05_06', '18/05_18', '19/05_06', '19/05_18', '20/05_06', '20/05_18', '21/05_06', '21/05_18', '22/05_06', '22/05_18', '23/05_06', '23/05_18', '24/05_06', '24/05_18', '25/05_06', '25/05_18', '26/05_06', '26/05_18', '27/05_06', '27/05_18', '28/05_06', '28/05_18', '29/05_06', '29/05_18', '30/05_06', '30/05_18', '31/05_06', '31/05_18']
#dict_keys = [k for k in dict_keys if k.endswith('_18')]

from matplotlib.backends.backend_pdf import PdfPages
import os

# 🎯 Подготовка ключей
dict_keys = [k for k in dict_keys if k.endswith('_18')]
dict_keys = [k.replace('/', '_') for k in dict_keys]

# 📄 Куда сохранить итоговый PDF
output_pdf_path = "w:/Postprocesing/Oleh Bedenok/GRAPHCAST/NOAA/TEST_Graphcast_14,05,2025-31,05,2025/raport/winds_report.pdf"
os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)

# 🖼️ Генерация всех графиков и сохранение в PDF
with PdfPages(output_pdf_path) as pdf:
    for date in dict_keys:
        print(f"Обработка {date}...")
        try:
            #pass
            fig = pred(date)       # вызываем функцию
            pdf.savefig(fig)       # сохраняем график в PDF
            plt.close(fig)         # закрываем его, чтобы не показывался
        except Exception as e:
            print(f"❌ Ошибка при обработке {date}: {e}")

print(f"✅ Все графики сохранены в: {output_pdf_path}")


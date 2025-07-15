import time
import argparse
import os


def main_run(date):
    '''parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()
    input_file = args.input'''
    # GraphCast

    # The model weights, normalization statistics, and example inputs are available on [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/dm_graphcast).

    '''# @title Pip install graphcast and dependencies

    import subprocess
    subprocess.run(["pip", "install", "--upgrade", "https://github.com/deepmind/graphcast/archive/master.zip"], check=True)

    # @title Workaround for cartopy crashes

    # Workaround for cartopy crashes due to the shapely installed by default in
    # google colab kernel (https://github.com/anitagraser/movingpandas/issues/81):
    !pip uninstall -y shapely
    !pip install shapely --no-binary shapely'''
    import time

    start = time.time()  # Засекаем время перед выполнением кода

    # @title Imports
    # from for_data import load_for_dataset
    import dataclasses
    import datetime
    import functools
    import math
    import re
    from typing import Optional

    import cartopy.crs as ccrs
    from google.cloud import storage
    from graphcast import autoregressive
    from graphcast import casting
    from graphcast import checkpoint
    from graphcast import data_utils
    from graphcast import graphcast
    from graphcast import normalization
    from graphcast import rollout
    from graphcast import xarray_jax
    from graphcast import xarray_tree
    from IPython.display import HTML
    import ipywidgets as widgets
    import haiku as hk
    import jax
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import animation
    import numpy as np
    import xarray

    '''import os
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Отключает предвыделение VRAM
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # Разрешает использовать RAM
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"  # Использовать только 70% VRAM'''

    # jax.config.update("jax_platform_name", "gpu")
    # jax.config.update("jax_enable_x64", True)  # Включаем 64-битные вычисления
    # print(jax.devices())  # Должен показать GPU

    print('1. Модули импортированы')

    # @title Authenticate with Google Cloud Storage

    # Создаём анонимного клиента Google Cloud Storage
    gcs_client = storage.Client.create_anonymous_client()
    # Подключаемся к бакету с моделями GraphCast
    gcs_bucket = gcs_client.get_bucket("dm_graphcast")
    # Путь к файлам
    dir_prefix = "graphcast/"
    print('2. Получили доступ к папке graphcast на Google Cloud ')

    # load_for_dataset()
    # print('Загрузили и обработали датасет')
    # @title Plotting functions

    def select(
            data: xarray.Dataset,
            variable: str,
            level: Optional[int] = None,
            max_steps: Optional[int] = None
    ) -> xarray.Dataset:
        data = data[variable]
        if "batch" in data.dims:
            data = data.isel(batch=0)
        if max_steps is not None and "time" in data.sizes and max_steps < data.sizes["time"]:
            data = data.isel(time=range(0, max_steps))
        if level is not None and "level" in data.coords:
            data = data.sel(level=level)
        return data

    def scale(
            data: xarray.Dataset,
            center: Optional[float] = None,
            robust: bool = False,
    ) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:
        vmin = np.nanpercentile(data, (2 if robust else 0))
        vmax = np.nanpercentile(data, (98 if robust else 100))
        if center is not None:
            diff = max(vmax - center, center - vmin)
            vmin = center - diff
            vmax = center + diff
        return (data, matplotlib.colors.Normalize(vmin, vmax),
                ("RdBu_r" if center is not None else "viridis"))

    def plot_data(
            data: dict[str, xarray.Dataset],
            fig_title: str,
            plot_size: float = 5,
            robust: bool = False,
            cols: int = 4
    ) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:

        first_data = next(iter(data.values()))[0]
        max_steps = first_data.sizes.get("time", 1)
        assert all(max_steps == d.sizes.get("time", 1) for d, _, _ in data.values())

        cols = min(cols, len(data))
        rows = math.ceil(len(data) / cols)
        figure = plt.figure(figsize=(plot_size * 2 * cols,
                                     plot_size * rows))
        figure.suptitle(fig_title, fontsize=16)
        figure.subplots_adjust(wspace=0, hspace=0)
        figure.tight_layout()

        images = []
        for i, (title, (plot_data, norm, cmap)) in enumerate(data.items()):
            ax = figure.add_subplot(rows, cols, i + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title)
            im = ax.imshow(
                plot_data.isel(time=0, missing_dims="ignore"), norm=norm,
                origin="lower", cmap=cmap)
            plt.colorbar(
                mappable=im,
                ax=ax,
                orientation="vertical",
                pad=0.02,
                aspect=16,
                shrink=0.75,
                cmap=cmap,
                extend=("both" if robust else "neither"))
            images.append(im)

        def update(frame):
            if "time" in first_data.dims:
                td = datetime.timedelta(microseconds=first_data["time"][frame].item() / 1000)
                figure.suptitle(f"{fig_title}, {td}", fontsize=16)
            else:
                figure.suptitle(fig_title, fontsize=16)
            for im, (plot_data, norm, cmap) in zip(images, data.values()):
                im.set_data(plot_data.isel(time=frame, missing_dims="ignore"))

        # Создаём и сохраняем анимацию
        ani = animation.FuncAnimation(
            fig=figure, func=update, frames=max_steps, interval=250)

        # 👇 ЭТО ВАЖНО: сохраняем анимацию (теперь сохранится файл)
        # ani.save("animation.mp4", writer="ffmpeg", fps=10)
        ani.save("animation.gif", writer="pillow", fps=10)

        print("Анимация сохранена в animation.mp4")

    '''Загрузка параметров модели
    Выберите один из двух способов получения параметров модели:

    random (случайный):

    Вы получите случайные предсказания, но сможете изменять архитектуру модели.
    Это может работать быстрее или занимать меньше памяти на вашем устройстве.
    checkpoint (контрольная точка):

    Вы получите адекватные предсказания, но будете ограничены архитектурой, с которой модель была обучена.
    Эта архитектура может не поместиться на ваше устройство.
    В частности, генерация градиентов требует очень много памяти (нужно минимум 25 ГБ RAM, например, TPUv4 или A100).
    Ключевые различия между моделями
    Контрольные точки различаются по следующим параметрам:

    Размер сетки (mesh size):

    Определяет, как внутренне представляется земной шар в графовой модели.
    Меньшие сетки работают быстрее, но дают хуже качество предсказаний.
    Размер сетки не влияет на количество параметров модели.
    Разрешение и количество уровней давления:

    Должны соответствовать данным.
    Меньшее разрешение и меньшее число уровней давления ускоряют работу модели.
    Разрешение данных влияет только на кодировщик/декодировщик модели.
    Использование осадков:

    Все модели предсказывают осадки.
    ERA5 включает осадки, а HRES их не включает.
    Модели, обученные на ERA5, принимают осадки на вход.
    Модели, обученные на ERA5-HRES, не принимают осадки на вход и обучены на HRES-fc0.
    Предоставленные предобученные модели
    GraphCast (высокое разрешение):

    Использовалась в статье о GraphCast.
    Разрешение 0.25° (градуса).
    37 уровней давления.
    Обучена на данных ERA5 за 1979–2017 годы.
    GraphCast_small (меньшая версия с низким разрешением):

    Разрешение 1°.
    13 уровней давления.
    Упрощённая сетка, требует меньше памяти и вычислительных ресурсов.
    Обучена на ERA5 за 1979–2015 годы.
    GraphCast_operational (операционная версия):

    Высокое разрешение 0.25°.
    13 уровней давления.
    Обучена на ERA5 (1979–2017), затем дообучена на HRES (2016–2021).
    Может работать с HRES-данными (не требует входных данных по осадкам).'''

    # @title Choose the model

    params_file_options = [
        name for blob in gcs_bucket.list_blobs(prefix=dir_prefix + "params/")
        if (name := blob.name.removeprefix(dir_prefix + "params/"))]  # Drop empty string.

    random_mesh_size = widgets.IntSlider(
        value=4, min=4, max=6, description="Mesh size:")
    random_gnn_msg_steps = widgets.IntSlider(
        value=4, min=1, max=32, description="GNN message steps:")
    random_latent_size = widgets.Dropdown(
        options=[int(2 ** i) for i in range(4, 10)], value=32, description="Latent size:")
    random_levels = widgets.Dropdown(
        options=[13, 37], value=13, description="Pressure levels:")

    params_file = widgets.Dropdown(
        options=params_file_options,
        description="Params file:",
        layout={"width": "max-content"})

    source_tab = widgets.Tab([
        widgets.VBox([
            random_mesh_size,
            random_gnn_msg_steps,
            random_latent_size,
            random_levels,
        ]),
        params_file,
    ])
    source_tab.set_title(0, "Random")
    source_tab.set_title(1, "Checkpoint")
    widgets.VBox([
        source_tab,
        widgets.Label(value="Run the next cell to load the model. Rerunning this cell clears your selection.")
    ])

    # @title Load the model

    # source = source_tab.get_title(source_tab.selected_index)               изменено
    source = "Checkpoint"  # Явно указываем, что загружаем чекпоинт         изменено

    if source == "Random":
        params = None  # Filled in below
        state = {}
        model_config = graphcast.ModelConfig(
            resolution=0,
            mesh_size=random_mesh_size.value,
            latent_size=random_latent_size.value,
            gnn_msg_steps=random_gnn_msg_steps.value,
            hidden_layers=1,
            radius_query_fraction_edge_length=0.6)
        task_config = graphcast.TaskConfig(
            input_variables=graphcast.TASK.input_variables,
            target_variables=graphcast.TASK.target_variables,
            forcing_variables=graphcast.TASK.forcing_variables,
            pressure_levels=graphcast.PRESSURE_LEVELS[random_levels.value],
            input_duration=graphcast.TASK.input_duration,
        )
    else:
        assert source == "Checkpoint"
        params_filename1 = "GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz"
        params_filename2 = "GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz"  # Укажи нужный файл
        params_filename3 = "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
        local_file_path = f"params/{params_filename3}"
        with open(local_file_path, "rb") as f:
            ckpt = checkpoint.load(f, graphcast.CheckPoint)

        params = ckpt.params
        state = {}

        model_config = ckpt.model_config
        task_config = ckpt.task_config
        print("Model description:\n", ckpt.description, "\n")
        print("Model license:\n", ckpt.license, "\n")

    print(model_config)

    ## Load the example data

    # Several example datasets are available, varying across a few axes:
    # - **Source**: fake, era5, hres
    # - **Resolution**: 0.25deg, 1deg, 6deg
    # - **Levels**: 13, 37
    # - **Steps**: How many timesteps are included

    # Not all combinations are available.
    # - Higher resolution is only available for fewer steps due to the memory requirements of loading them.
    # - HRES is only available in 0.25 deg, with 13 pressure levels.

    # The data resolution must match the model that is loaded.

    # Some transformations were done from the base datasets:
    # - We accumulated precipitation over 6 hours instead of the default 1 hour.
    # - For HRES data, each time step corresponds to the HRES forecast at leadtime 0, essentially providing an "initialisation" from HRES. See HRES-fc0 in the GraphCast paper for further description. Note that a 6h accumulation of precipitation is not available from HRES, so our model taking HRES inputs does not depend on precipitation. However, because our models predict precipitation, we include the ERA5 precipitation in the example data so it can serve as an illustrative example of ground truth.
    # - We include ERA5 `toa_incident_solar_radiation` in the data. Our model uses the radiation at -6h, 0h and +6h as a forcing term for each 1-step prediction. If the radiation is missing from the data (e.g. in an operational setting), it will be computed using a custom implementation that produces values similar to those in ERA5.

    # @title Get and filter the list of available example datasets

    dataset_file_options = [
        name for blob in gcs_bucket.list_blobs(prefix=dir_prefix + "dataset/")
        if (name := blob.name.removeprefix(dir_prefix + "dataset/"))]  # Drop empty string.

    def data_valid_for_model(
            file_name: str, model_config: graphcast.ModelConfig, task_config: graphcast.TaskConfig):
        file_parts = parse_file_parts(file_name.removesuffix(".nc"))
        return (
                model_config.resolution in (0, float(file_parts["res"])) and
                len(task_config.pressure_levels) == int(file_parts["levels"]) and
                (
                        ("total_precipitation_6hr" in task_config.input_variables and
                         file_parts["source"] in ("era5", "fake")) or
                        ("total_precipitation_6hr" not in task_config.input_variables and
                         file_parts["source"] in ("hres", "fake"))
                )
        )

    dataset_file = widgets.Dropdown(
        options=[
            (", ".join([f"{k}: {v}" for k, v in parse_file_parts(option.removesuffix(".nc")).items()]), option)
            for option in dataset_file_options
            if data_valid_for_model(option, model_config, task_config)
        ],
        description="Dataset file:",
        layout={"width": "max-content"})
    widgets.VBox([
        dataset_file,
        widgets.Label(
            value="Run the next cell to load the dataset. Rerunning this cell clears your selection and refilters the datasets that match your model.")
    ])

    # @title Load weather data

    if not data_valid_for_model(dataset_file.value, model_config, task_config):
        raise ValueError(
            "Invalid dataset file, rerun the cell above and choose a valid dataset file.")

    dataset_file_era5_1 = "source-era5_date-2022-01-01_res-0.25_levels-13_steps-01.nc"
    dataset_file_era5_2 = "source-era5_date-2022-01-01_res-0.25_levels-13_steps-04.nc"
    dataset_file_era5_3 = "source-era5_date-2022-01-01_res-0.25_levels-13_steps-12.nc"
    dataset_file_era5_4 = "source-era5_date-2022-01-01_res-0.25_levels-37_steps-01.nc"
    dataset_file_era5_5 = "source-era5_date-2022-01-01_res-0.25_levels-37_steps-04.nc"
    dataset_file_era5_6 = "source-era5_date-2022-01-01_res-0.25_levels-37_steps-12.nc"
    dataset_file_era5_7 = "source-era5_date-2022-01-01_res-1.0_levels-13_steps-01.nc"
    dataset_file_era5_8 = "source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc"
    dataset_file_era5_9 = "source-era5_date-2022-01-01_res-1.0_levels-13_steps-12.nc"
    dataset_file_era5_10 = "source-era5_date-2022-01-01_res-1.0_levels-13_steps-20.nc"
    dataset_file_era5_11 = "source-era5_date-2022-01-01_res-1.0_levels-13_steps-40.nc"
    dataset_file_era5_12 = "source-era5_date-2022-01-01_res-1.0_levels-37_steps-01.nc"
    dataset_file_era5_13 = "source-era5_date-2022-01-01_res-1.0_levels-37_steps-04.nc"
    dataset_file_era5_14 = "source-era5_date-2022-01-01_res-1.0_levels-37_steps-12.nc"
    dataset_file_era5_15 = "source-era5_date-2022-01-01_res-1.0_levels-37_steps-20.nc"
    dataset_file_era5_16 = file
    dataset_file_era5_17 = f"NOAA_.nc"

    # dataset_file_era5 = "source-era5_date-2022-01-01_res-0.25_levels-37_steps-12.nc"        #---------------------ERA5-------------------
    local_file_path = f"datasets/{dataset_file_era5_16}"
    example_batch = xarray.load_dataset(local_file_path, decode_timedelta=True).compute()

    # 👇 Показать все переменные и их размеры
    print("\n📦 Все переменные и их размерности:")
    for name, var in example_batch.data_vars.items():
        print(f"{name:30} shape: {var.shape}, dims: {var.dims}")

    # 👇 Попробовать собрать всё в массив и посмотреть его форму
    try:
        inputs_all = example_batch.to_array()
        print(f"\n✅ to_array() shape: {inputs_all.shape}")
        print("📋 Переменные, собранные в to_array():")
        print(list(inputs_all.coords["variable"].values))
        print("🔢 Всего переменных:", len(inputs_all.coords["variable"]))
    except Exception as e:
        print(f"\n⚠️ Ошибка при вызове to_array(): {e}")

    assert example_batch.sizes["time"] >= 3  # 2 for input, >=1 for targets

    print('Загрузили данные ERA5')

    print(", ".join([f"{k}: {v}" for k, v in parse_file_parts(dataset_file.value.removesuffix(".nc")).items()]))

    print(example_batch)

    # @title Choose data to plot

    plot_example_variable = widgets.Dropdown(
        options=example_batch.data_vars.keys(),
        value="2m_temperature",
        description="Variable")
    plot_example_level = widgets.Dropdown(
        options=example_batch.coords["level"].values,
        value=500,
        description="Level")
    plot_example_robust = widgets.Checkbox(value=True, description="Robust")
    plot_example_max_steps = widgets.IntSlider(
        min=1, max=example_batch.sizes["time"], value=example_batch.sizes["time"],
        description="Max steps")

    widgets.VBox([
        plot_example_variable,
        plot_example_level,
        plot_example_robust,
        plot_example_max_steps,
        widgets.Label(value="Run the next cell to plot the data. Rerunning this cell clears your selection.")
    ])

    # @title Plot example data

    plot_size = 7

    data = {
        " ": scale(
            select(example_batch, plot_example_variable.value, plot_example_level.value, plot_example_max_steps.value),
            robust=plot_example_robust.value),
    }
    fig_title = plot_example_variable.value
    if "level" in example_batch[plot_example_variable.value].coords:
        fig_title += f" at {plot_example_level.value} hPa"

    plot_data(data, fig_title, plot_size, plot_example_robust.value)

    # @title Choose training and eval data to extract
    train_steps = widgets.IntSlider(
        value=1, min=1, max=example_batch.sizes["time"] - 2, description="Train steps")
    eval_steps = widgets.IntSlider(
        value=example_batch.sizes["time"] - 2, min=1, max=example_batch.sizes["time"] - 2, description="Eval steps")

    widgets.VBox([
        train_steps,
        eval_steps,
        widgets.Label(value="Run the next cell to extract the data. Rerunning this cell clears your selection.")
    ])

    # @title Extract training and eval data

    train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
        example_batch, target_lead_times=slice("6h", f"{train_steps.value * 6}h"),
        **dataclasses.asdict(task_config))

    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        example_batch, target_lead_times=slice("6h", f"{eval_steps.value * 6}h"),
        **dataclasses.asdict(task_config))

    print("All Examples:  ", example_batch.sizes.mapping)
    print("Train Inputs:  ", train_inputs.sizes.mapping)
    print("Train Targets: ", train_targets.sizes.mapping)
    print("Train Forcings:", train_forcings.sizes.mapping)
    print("Eval Inputs:   ", eval_inputs.sizes.mapping)
    print("Eval Targets:  ", eval_targets.sizes.mapping)
    print("Eval Forcings: ", eval_forcings.sizes.mapping)

    # @title Load normalization data

    with gcs_bucket.blob(dir_prefix + "stats/diffs_stddev_by_level.nc").open("rb") as f:
        diffs_stddev_by_level = xarray.load_dataset(f).compute()
    with gcs_bucket.blob(dir_prefix + "stats/mean_by_level.nc").open("rb") as f:
        mean_by_level = xarray.load_dataset(f).compute()
    with gcs_bucket.blob(dir_prefix + "stats/stddev_by_level.nc").open("rb") as f:
        stddev_by_level = xarray.load_dataset(f).compute()

    # @title Build jitted functions, and possibly initialize random weights

    def construct_wrapped_graphcast(
            model_config: graphcast.ModelConfig,
            task_config: graphcast.TaskConfig):
        """Constructs and wraps the GraphCast Predictor."""
        # Deeper one-step predictor.
        predictor = graphcast.GraphCast(model_config, task_config)

        # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
        # from/to float32 to/from BFloat16.
        predictor = casting.Bfloat16Cast(predictor)

        # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
        # BFloat16 happens after applying normalization to the inputs/targets.
        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=diffs_stddev_by_level,
            mean_by_level=mean_by_level,
            stddev_by_level=stddev_by_level)

        # Wraps everything so the one-step model can produce trajectories.
        predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
        return predictor

    @hk.transform_with_state
    def run_forward(model_config, task_config, inputs, targets_template, forcings):
        predictor = construct_wrapped_graphcast(model_config, task_config)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    @hk.transform_with_state
    def loss_fn(model_config, task_config, inputs, targets, forcings):
        predictor = construct_wrapped_graphcast(model_config, task_config)
        loss, diagnostics = predictor.loss(inputs, targets, forcings)
        return xarray_tree.map_structure(
            lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
            (loss, diagnostics))

    def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
        def _aux(params, state, i, t, f):
            (loss, diagnostics), next_state = loss_fn.apply(
                params, state, jax.random.PRNGKey(0), model_config, task_config,
                i, t, f)
            return loss, (diagnostics, next_state)

        (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
            _aux, has_aux=True)(params, state, inputs, targets, forcings)
        return loss, diagnostics, next_state, grads

    # Jax doesn't seem to like passing configs as args through the jit. Passing it
    # in via partial (instead of capture by closure) forces jax to invalidate the
    # jit cache if you change configs.
    def with_configs(fn):
        return functools.partial(
            fn, model_config=model_config, task_config=task_config)

    # Always pass params and state, so the usage below are simpler
    def with_params(fn):
        return functools.partial(fn, params=params, state=state)

    # Our models aren't stateful, so the state is always empty, so just return the
    # predictions. This is requiredy by our rollout code, and generally simpler.
    def drop_state(fn):
        return lambda **kw: fn(**kw)[0]

    init_jitted = jax.jit(with_configs(run_forward.init))

    if params is None:
        params, state = init_jitted(
            rng=jax.random.PRNGKey(0),
            inputs=train_inputs,
            targets_template=train_targets,
            forcings=train_forcings)

    loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(loss_fn.apply))))
    grads_fn_jitted = with_params(jax.jit(with_configs(grads_fn)))
    run_forward_jitted = drop_state(with_params(jax.jit(with_configs(
        run_forward.apply))))

    # Run the model

    # Note that the cell below may take a while (possibly minutes) to run the first time you execute them, because this will include the time it takes for the code to compile. The second time running will be significantly faster.

    # This use the python loop to iterate over prediction steps, where the 1-step prediction is jitted. This has lower memory requirements than the training steps below, and should enable making prediction with the small GraphCast model on 1 deg resolution data for 4 steps.

    # @title Autoregressive rollout (loop in python)

    '''assert model_config.resolution in (0, 360. / eval_inputs.sizes["lon"]), (
      "Model resolution doesn't match the data resolution. You likely want to "
      "re-filter the dataset list, and download the correct data.")'''

    # Вычисляем реальное разрешение по долготе
    lon = eval_inputs["lon"].values
    if len(lon) > 1:
        actual_resolution = round(float(lon[1] - lon[0]), 5)
    else:
        actual_resolution = model_config.resolution  # fallback

    # Проверка соответствия
    assert model_config.resolution == actual_resolution, (
        f"Model resolution ({model_config.resolution}) doesn't match "
        f"data resolution ({actual_resolution})."
    )

    print("Inputs:  ", eval_inputs.sizes.mapping)
    print("Targets: ", eval_targets.sizes.mapping)
    print("Forcings:", eval_forcings.sizes.mapping)

    # num_forecast_steps = 2  # 40 шагов = 10 дней (по 6 часов)

    predictions = rollout.chunked_prediction(
        run_forward_jitted,
        rng=jax.random.PRNGKey(0),
        inputs=eval_inputs,
        targets_template=eval_targets * np.nan,
        forcings=eval_forcings)
    # num_steps=num_forecast_steps)
    # predictions

    # взять из полного пути только название файла
    #file_name = os.path.basename(input_file)

    # Сохранение результата в NetCDF (лучше всего подходит для метеоданных)
    output_filename = f"predictions/pred_{file}"
    predictions.to_netcdf(output_filename)
    print(f"Прогноз сохранён в {output_filename}")

    # @title Choose predictions to plot

    plot_pred_variable = widgets.Dropdown(
        options=predictions.data_vars.keys(),
        value="2m_temperature",
        description="Variable")
    plot_pred_level = widgets.Dropdown(
        options=predictions.coords["level"].values,
        value=500,
        description="Level")
    plot_pred_robust = widgets.Checkbox(value=True, description="Robust")
    plot_pred_max_steps = widgets.IntSlider(
        min=1,
        max=predictions.sizes["time"],
        value=predictions.sizes["time"],
        description="Max steps")

    widgets.VBox([
        plot_pred_variable,
        plot_pred_level,
        plot_pred_robust,
        plot_pred_max_steps,
        widgets.Label(value="Run the next cell to plot the predictions. Rerunning this cell clears your selection.")
    ])

    # @title Plot predictions

    plot_size = 5
    plot_max_steps = min(predictions.sizes["time"], plot_pred_max_steps.value)

    data = {
        "Targets": scale(select(eval_targets, plot_pred_variable.value, plot_pred_level.value, plot_max_steps),
                         robust=plot_pred_robust.value),
        "Predictions": scale(select(predictions, plot_pred_variable.value, plot_pred_level.value, plot_max_steps),
                             robust=plot_pred_robust.value),
        "Diff": scale((select(eval_targets, plot_pred_variable.value, plot_pred_level.value, plot_max_steps) -
                       select(predictions, plot_pred_variable.value, plot_pred_level.value, plot_max_steps)),
                      robust=plot_pred_robust.value, center=0),
    }
    fig_title = plot_pred_variable.value
    if "level" in predictions[plot_pred_variable.value].coords:
        fig_title += f" at {plot_pred_level.value} hPa"

    plot_data(data, fig_title, plot_size, plot_pred_robust.value)

    print('Прогноз готов, начинаем расчитывать потери!')
    # Train the model

    # The following operations require a large amount of memory and, depending on the accelerator being used, will only fit the very small "random" model on low resolution data. It uses the number of training steps selected above.

    # The first time executing the cell takes more time, as it include the time to jit the function.

    # @title Loss computation (autoregressive loss over multiple steps)
    '''loss, diagnostics = loss_fn_jitted(
        rng=jax.random.PRNGKey(0),
        inputs=train_inputs,
        targets=train_targets,
        forcings=train_forcings)
    print("Loss:", float(loss))

    # @title Gradient computation (backprop through time)
    loss, diagnostics, next_state, grads = grads_fn_jitted(
        inputs=train_inputs,
        targets=train_targets,
        forcings=train_forcings)
    mean_grad = np.mean(jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: np.abs(x).mean(), grads))[0])
    print(f"Loss: {loss:.4f}, Mean |grad|: {mean_grad:.6f}")

    # @title Autoregressive rollout (keep the loop in JAX)
    print("Inputs:  ", train_inputs.sizes.mapping)
    print("Targets: ", train_targets.sizes.mapping)
    print("Forcings:", train_forcings.sizes.mapping)

    predictions = run_forward_jitted(
        rng=jax.random.PRNGKey(0),
        inputs=train_inputs,
        targets_template=train_targets * np.nan,
        forcings=train_forcings)'''

    # Завершение
    end = time.time()
    elapsed_total = end - start
    print(f"Общее время: {int(elapsed_total // 60)} мин {elapsed_total % 60:.2f} сек")

    import gc

    # Очистка памяти вручную
    del example_batch, train_inputs, train_targets, train_forcings
    del eval_inputs, eval_targets, eval_forcings
    del predictions
    gc.collect()


dict_keys = ['14/05_18', '15/05_06', '15/05_18', '16/05_06', '16/05_18', '17/05_06', '17/05_18', '18/05_06', '18/05_18', '19/05_06', '19/05_18', '20/05_06', '20/05_18', '21/05_06', '21/05_18', '22/05_06', '22/05_18', '23/05_06', '23/05_18', '24/05_06', '24/05_18', '25/05_06', '25/05_18', '26/05_06', '26/05_18', '27/05_06', '27/05_18', '28/05_06', '28/05_18', '29/05_06', '29/05_18', '30/05_06', '30/05_18', '31/05_06', '31/05_18']
#dict_keys = [k for k in dict_keys if k.endswith('_18')]

for date in dict_keys:
    date_clean = date.replace('/', '_')  # '23/01_18' → '23_01_18'
    filename = f"ERA5_{date_clean}.nc"
    print(filename)
    # можешь использовать filename, например:
    # ds = xr.open_dataset(filename)

    main_run(date_clean)



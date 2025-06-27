# 📌 Базовый образ с поддержкой CUDA (Ubuntu + Python + CUDA 12)
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# 📌 Устанавливаем Python и pip
RUN apt-get update && apt-get install -y python3 python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python

# 📌 Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# 📌 Копируем файлы проекта в контейнер
COPY . /app

# 📌 Устанавливаем зависимости проекта (кроме JAX)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install netcdf4 h5netcdf

# 📌 Удаляем старый JAX и JAXLIB (если есть)
RUN pip uninstall -y jax jaxlib

# 📌 Устанавливаем JAX с поддержкой CUDA 12
RUN pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 📌 Проверяем, работает ли JAX на GPU
CMD ["python", "main.py"]

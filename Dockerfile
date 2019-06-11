FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    cmake \
    unzip \
    build-essential \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libjasper-dev \
    libavformat-dev \
    libpq-dev \
    libturbojpeg \
    software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install --no-cache-dir \
    numpy \
    pandas \
    PyYAML \
    cycler \
    dill \
    h5py \
    imgaug \
    matplotlib \
    opencv-contrib-python \
    Pillow \
    scikit-image \
    scikit-learn \
    scipy \
    setuptools \
    six \
    tqdm \
    ipython \
    ipdb \
    albumentations \
    click \
    jpeg4py \
    addict \
    colorama \
    torchvision \
    iterative-stratification

RUN pip install --upgrade --no-cache-dir cython && pip install --no-cache-dir pycocotools==2.0.0 mmcv==0.2.5

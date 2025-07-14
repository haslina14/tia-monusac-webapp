FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

#install dependencies
RUN apt-get update && apt-get install -y \
    wget bzip2 ca-certificates curl git && \
    rm -rf /var/lib/apt/lists/*

#install miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"

RUN conda init bash

# Install system libraries for GUI/OpenCV support
RUN apt-get update && \
    apt-get install -y \
    libopenjp2-7-dev \
    libopenjp2-tools \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN conda create -n tiagui-v2 python=3.10 -y

RUN conda run -n tiagui-v2 pip install --no-cache-dir \
    tiatoolbox==1.6.0 \
    opencv-python-headless==4.9.0.80 \
    scikit-image==0.24.0 \
    matplotlib==3.9.3 \
    pandas==2.2.3 \
    numpy==1.26.4 \
    Pillow==11.0.0 \
    scipy==1.14.1 \
    scikit-learn==1.5.2 \
    pyvips==2.2.3 \
    openslide-python==1.4.1

#set env default
ENV CONDA_DEFAULT_ENV=tiagui-v2
ENV PATH="/opt/conda/envs/tiagui-v2/bin:$PATH"

WORKDIR /app

#copy application files
COPY . .

EXPOSE 5000



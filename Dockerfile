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

# Install system libraries for GUI/OpenCV support
RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY tia-gui.yml /app

RUN conda env create -f tia-gui.yml

#activate env and make it a default (chech this!!!!!!!!)
RUN echo "conda activate tia-gui" >> ~/.bashrc
SHELL [ "/bin/bash", "--login", "-c" ]

#copy application files
COPY . .

EXPOSE 5000



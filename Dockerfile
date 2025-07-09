FROM continuumio/miniconda3

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

CMD ["conda", "run", "-n", "tia-gui", "python", "app.py"]


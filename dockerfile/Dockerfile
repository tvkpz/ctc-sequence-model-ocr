FROM tensorflow/tensorflow:latest-gpu-py3

ENV http_proxy=http://proxy.sin.sap.corp:8080 
ENV HTTP_PROXY=http://proxy.sin.sap.corp:8080 
ENV ftp_proxy=ftp://proxy.sin.sap.corp:8080
ENV FTP_PROXY=ftp://proxy.sin.sap.corp:8080
ENV https_proxy=https://proxy.sin.sap.corp:8080
ENV HTTPS_PROXY=https://proxy.sin.sap.corp:8080

COPY apt.conf /etc/apt/

RUN apt-get -y update && apt-get -y upgrade

# Pick up some TF dependencies
RUN apt-get install -y --no-install-recommends \
	htop \
        build-essential \
        curl \
        nano \
        pkg-config \
        rsync \
        software-properties-common \
        unzip \
        git \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3 get-pip.py

RUN pip3 --no-cache-dir install Cython

RUN pip3 --no-cache-dir install \
        Pillow \
        h5py \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        pandas \
        scipy \
        sklearn \
        keras \
        nltk \
	scikit-image \
        && \
    python3 -m ipykernel.kernelspec

# to install opencv
COPY install-opencv.sh /data/install-opencv.sh
CMD /data/install-opencv.sh

# For TensorBoard
# EXPOSE 6011
# For IPython
# EXPOSE 8811

# Clean up APT when done
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

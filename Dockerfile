FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]
ENV PATH=/home/$USER/miniconda3/bin:${PATH}

RUN apt update && apt -y upgrade
RUN apt install -y --no-install-recommends \
 vim \
 tmux \
 wget \
 git \
 curl \
 libgl1-mesa-dev

## Install Anaconda2
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/$USER/miniconda3 && \
    rm -r Miniconda3-latest-Linux-x86_64.sh && \
    echo “source /home/$USER/miniconda3/etc/profile.d/conda.sh” >> /home/$USER/.bashrc

RUN conda create -n ws python=3.7 cudnn=7.6.5 -y \
    &&  source activate ws \
    &&  pip install jupyterlab \
    && echo “source activate ws” >> /root/.bashrc
    
EXPOSE 8686

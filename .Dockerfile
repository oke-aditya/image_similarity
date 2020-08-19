# Choose the base image from to take.
# Using slim images is best practice
FROM ubuntu:latest

ARG PYTHON_VERSION=3.6

# This is one of the best practice. 
# This technique is known as “cache busting”.
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl

# add non-root user
RUN useradd --create-home --shell /bin/bash containeruser
USER containeruser
WORKDIR /home/containeruser

# install miniconda and python
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /home/containeruser/conda && \
    rm ~/miniconda.sh && \
    /home/containeruser/conda/bin/conda clean -ya && \
    /home/containeruser/conda/bin/conda install -y python=$PYTHON_VERSION 

# add conda to path
ENV PATH /home/containeruser/conda/bin:$PATH

# Now install this repo
# We need only the master branch not all branches

COPY requirements.txt  requirements.txt
COPY requirements-extra.txt requirements-extra.txt
RUN pip install -r requirements.txt && \
    pip install -r requirements-extra.txt &&


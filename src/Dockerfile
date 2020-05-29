# start from base -- builder
FROM continuumio/miniconda3 as builder
RUN pwd
# label(s)
LABEL maintainer="Mirko D'Andrea"
LABEL email="mirko.dandrea@cimafoundation.org"
LABEL version="1.0.0"
LABEL release_date="2020/05/28"

# change default shell (from sh to bash)
SHELL ["/bin/bash", "-c"]

# install system-wide deps 
RUN apt-get update
RUN apt-get install -y \
	build-essential \
	libpng-dev \
	libpng++-dev\
	libgeotiff-dev \ 
	libjpeg-dev \
	libz-dev \
	libc6-dev \
	bash-completion \
	nano

RUN useradd -m -p fp -s /bin/bash risico
WORKDIR /home/risico/

RUN conda install -c conda-forge eccodes
ADD ./requirements.txt .
RUN pip install -r requirements.txt

ADD ./risico.tgz .
ADD --chown=risico RISICO2015 .
RUN chmod +x RISICO2015

ADD ./adapter.tgz .
WORKDIR /home/risico/adapter

 
WORKDIR /home/risico/
ADD --chown=risico run_risico.sh .
RUN chmod +x run_risico.sh


WORKDIR /home/risico/
ENTRYPOINT ["/home/risico/run_risico.sh"]

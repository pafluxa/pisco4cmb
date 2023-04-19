FROM nvidia/cuda:12.1.0-devel-ubuntu20.04
# set timezone data
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
# install c and fortran compiler
RUN apt-get update
RUN apt-get -y install \
vim \
zlib1g-dev \
build-essential cmake
#
# Intel oneAPI
RUN mkdir -p /pisco4cmb/optional
ADD optional /pisco4cmb/optional
WORKDIR /pisco4cmb/optional
RUN sh l_BaseKit_p_2023.1.0.46401_offline.sh \
-a --silent \
--eula accept
#
# create directories to put dependencies and install stuff
RUN mkdir -p /pisco4cmb/data/
RUN mkdir -p /pisco4cmb/workdir/
ADD dependencies/ /pisco4cmb/deps/
#
WORKDIR /pisco4cmb/
WORKDIR deps/

# install SOFA
RUN tar -xzvf sofa_c-20210512.tar.gz
RUN chmod +x install_sofa.sh
RUN /bin/bash install_sofa.sh 

# install bpoint
#RUN tar -xzvf bpoint.tar.gz
#RUN /bin/bash install_bpoint.sh 

# install cfitsio
RUN tar -xzvf cfitsio-4.2.0.tar.gz
RUN /bin/bash install_cfitsio.sh

# install healpix
RUN tar -xzvf Healpix_3.82_2022Jul28.tar.gz
RUN /bin/bash install_healpix.sh

# nsight compute
RUN apt-get update -y && \
     DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         apt-transport-https \
         ca-certificates \
         gnupg \
         wget && \
     rm -rf /var/lib/apt/lists/*
RUN  echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
     wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
         apt-get update -y && \
     DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         nsight-compute-2020.1.0 && \
     rm -rf /var/lib/apt/lists/*

## create user pisco4cmb
RUN useradd -ms /bin/bash pisco4cmb
# change ownership of workind directory
RUN chown pisco4cmb.pisco4cmb /pisco4cmb/workdir
USER pisco4cmb
WORKDIR /pisco4cmb/workdir

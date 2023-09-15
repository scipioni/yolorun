# build image with:
# - cuda
# - opencv with cuda support and gstreamer
# - python 3.9
# - ultralytics
# - yolorun package


ARG CUDA_VERSION=11.5.2
ARG UBUNTU_VERSION=20.04

#### stage: nvidia
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${UBUNTU_VERSION} AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN \
  apt-get update && \
  apt-get install -y -q --no-install-recommends \
  wget \
  curl \
  build-essential 

#### stage: python 3.9
FROM builder AS python
#ARG NUMPY_VERSION=1.25.2
ARG PYTHON_VERSION=3.9
COPY ./deadsnakes/deadsnakes_ubuntu_ppa.gpg /etc/apt/trusted.gpg.d/deadsnakes_ubuntu_ppa.gpg
COPY ./deadsnakes/deadsnakes-ubuntu-ppa-focal.list /etc/apt/sources.list.d/deadsnakes-ubuntu-ppa-focal.list

RUN apt-get update && \
  apt-get install -y -q --no-install-recommends \
  python${PYTHON_VERSION} \
  python${PYTHON_VERSION}-dev \
  python${PYTHON_VERSION}-venv

RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

RUN python${PYTHON_VERSION} -m ensurepip
RUN python${PYTHON_VERSION} -m pip install --upgrade wheel
RUN python${PYTHON_VERSION} -m pip install numpy
#==${NUMPY_VERSION}

#### stage: build opencv library
FROM python AS opencv
ARG OPENCV_VERSION=4.8.0
ARG PYTHON_VERSION=3.9
ARG DEBIAN_FRONTEND=noninteractive

RUN \
  apt-get install -y -q --no-install-recommends \
  libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev \
  libgtk2.0-dev \
  pkg-config \
  cmake

WORKDIR /build

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
COPY opencv-build.sh .
#RUN ./opencv-build.sh ${OPENCV_VERSION} ${PYTHON_VERSION}


#### 3 stage: import opencv libray in smaller cuda image
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu${UBUNTU_VERSION} AS runtime
ARG PYTHON_VERSION=3.9
ENV DEBIAN_FRONTEND=noninteractive

COPY ./deadsnakes/deadsnakes_ubuntu_ppa.gpg /etc/apt/trusted.gpg.d/deadsnakes_ubuntu_ppa.gpg
COPY ./deadsnakes/deadsnakes-ubuntu-ppa-focal.list /etc/apt/sources.list.d/deadsnakes-ubuntu-ppa-focal.list

RUN \
  apt-get update && \
  apt-get install -y -q --no-install-recommends \
  python${PYTHON_VERSION} \
  python${PYTHON_VERSION}-dev \
  python${PYTHON_VERSION}-venv

RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1

RUN python${PYTHON_VERSION} -m ensurepip && \
  python${PYTHON_VERSION} -m pip install --upgrade wheel && \
  python${PYTHON_VERSION} -m pip install numpy && \
  update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3 1

RUN \
  apt-get install -y -q --no-install-recommends \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-libav \
  gstreamer1.0-tools \
  gstreamer1.0-x \
  libgtk2.0-0 \
  libcanberra-gtk-module && \
  rm -rf /var/lib/apt/lists/*

# opencv installa in /usr/lib/python${PYTHON_VERSION}/dist-packages
COPY --from=opencv /build/release/lib/ /usr/local/lib/
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
#ENV PYTHONPATH=/usr/local/lib/python${PYTHON_VERSION}/site-packages:$PYTHONPATH
RUN ldconfig


FROM runtime as pdm
ARG ENV=development

#ARG PYPI_URL=https://pypi.csgalileo.org/simple
# ARG PYPI_USERNAME=galileo
# ARG PYPI_PASSWORD

ENV \
  PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONHASHSEED=random \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  DEBIAN_FRONTEND=noninteractive \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8

# serve per cv2.imshow
RUN if [ "$ENV" = "development" ]; then \
  apt-get update && \
  apt-get install -y -q --no-install-recommends libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 && \
  rm -rf /var/lib/apt/lists/*; fi

RUN pip install setuptools wheel pdm && \
  pdm config python.use_venv false
# RUN pdm config pypi.extra.url ${PYPI_URL}
# RUN pdm config pypi.extra.username ${PYPI_USERNAME}
# RUN pdm config pypi.extra.password ${PYPI_PASSWORD}


FROM pdm as ultralitycs

# fake opencv before ultralytics
WORKDIR /app
COPY ./opencv-python ./opencv-python
RUN pip install -e opencv-python && \
  pip install ultralytics

FROM ultralitycs

WORKDIR /project
COPY pyproject.toml pdm.lock README.md ./

RUN pdm install --prod --no-lock --no-editable --no-isolation
RUN pdm install -G $ENV --no-lock --no-editable --no-isolation

WORKDIR /app
COPY ./yolorun ./yolorun
RUN mkdir -p /app/pkgs
RUN bash -c "cp -a /project/__pypackages__/3*/* /app/pkgs"

ENV PYTHONPATH=/app/pkgs/lib:/app:$PYTHONPATH
ENV PATH=/app/pkgs/bin:$PATH

ARG OPENCV_VERSION=4.8.0

FROM yololab/opencv:${OPENCV_VERSION} as pdm

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

RUN pip install setuptools wheel pdm
RUN pdm config python.use_venv false
# RUN pdm config pypi.extra.url ${PYPI_URL}
# RUN pdm config pypi.extra.username ${PYPI_USERNAME}
# RUN pdm config pypi.extra.password ${PYPI_PASSWORD}


FROM pdm

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

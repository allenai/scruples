# Usage: A containerized environment for scruples.

FROM nvidia/cuda:10.1-devel-ubuntu18.04


# Configure language settings.

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8


# Create the working directory.

ENV APP_DIR=/app

WORKDIR $APP_DIR


# Install the OS packages.

RUN apt-get update --fix-missing                  \
 && DEBIAN_FRONTEND=noninteractive                \
    apt-get install --no-install-recommends --yes \
      build-essential                             \
      curl                                        \
      git                                         \
      libbz2-dev                                  \
      libffi-dev                                  \
      liblzma-dev                                 \
      libmysqlclient-dev                          \
      libncurses5-dev                             \
      libreadline-dev                             \
      libsqlite3-dev                              \
      libssl-dev                                  \
      libxml2-dev                                 \
      libxmlsec1-dev                              \
      llvm                                        \
      make                                        \
      tk-dev                                      \
      wget                                        \
      xz-utils                                    \
      zlib1g-dev


# Install Python.

# environment variables for pyenv
ENV _PYTHON_VERSION="3.7.0"
ENV PYENV_ROOT="$APP_DIR/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PATH"

# install pyenv version 1.2.15
RUN git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT \
 && cd $PYENV_ROOT \
 && git checkout b5f69fe58a58f3d2d59511c4630c96aafe5ad2b0

# install the python version
RUN pyenv install $_PYTHON_VERSION

# place the new python version on the path
ENV PATH="$PYENV_ROOT/versions/$_PYTHON_VERSION/bin:$PATH"


# Install python package dependencies.

COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --requirement requirements.txt


# Download the spacy models.

RUN python -m spacy download en


# Copy over the source code

COPY . .


# Install the scruples package.

RUN pip install .


# Define the entrypoint and default command.

ENTRYPOINT [ "python" ]

CMD [ "bin/scruples", "demo", "scoracle", "--port", "8000", "--expose" ]

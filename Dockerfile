# Usage: A containerized environment for scruples.

FROM nvidia/cuda:10.1-devel-ubuntu18.04


# Configure language settings.

ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8


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

# Set environment variables for pyenv.

ENV PYTHON_VERSION="3.7.0" \
    PYENV_ROOT="$APP_DIR/.pyenv"

ENV PATH="$PYENV_ROOT/bin:$PATH"

# Install pyenv version 1.2.15.

RUN git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT \
 && cd $PYENV_ROOT \
 && git checkout b5f69fe58a58f3d2d59511c4630c96aafe5ad2b0

# Install the python version.

RUN pyenv install $PYTHON_VERSION

# Place the new python version on the path.

ENV PATH="$PYENV_ROOT/versions/$PYTHON_VERSION/bin:$PATH"


# Install python package dependencies.

COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --requirement requirements.txt


# Download the spacy models.

RUN python -m spacy download en


# Setup the norms demo.

# Download the models.

RUN mkdir --parents models/norms

RUN mkdir models/norms/actions \
 && wget \
      --quiet \
      --directory-prefix models/norms/actions \
      https://storage.googleapis.com/ai2-mosaic-public/projects/scruples/v1.0/demos/norms/actions/config.json \
      https://storage.googleapis.com/ai2-mosaic-public/projects/scruples/v1.0/demos/norms/actions/pytorch_model.bin

RUN mkdir models/norms/corpus \
 && wget \
      --quiet \
      --directory-prefix models/norms/corpus \
      https://storage.googleapis.com/ai2-mosaic-public/projects/scruples/v1.0/demos/norms/corpus/config.json \
      https://storage.googleapis.com/ai2-mosaic-public/projects/scruples/v1.0/demos/norms/corpus/pytorch_model.bin

# Set the environment variables.

ENV SCRUPLES_NORMS_ACTIONS_BASELINE="roberta" \
    SCRUPLES_NORMS_ACTIONS_MODEL="/app/models/norms/actions" \
    SCRUPLES_NORMS_CORPUS_BASELINE="roberta" \
    SCRUPLES_NORMS_CORPUS_MODEL="/app/models/norms/corpus" \
    SCRUPLES_NORMS_PREDICT_BATCH_SIZE="2" \
    SCRUPLES_NORMS_GPU_IDS=""


# Copy over the source code

COPY . .


# Install the scruples package.

RUN pip install .


# Define the entrypoint and default command.

ENTRYPOINT [ "scruples" ]

CMD [ "demo", "norms", "--port", "8000", "--expose" ]

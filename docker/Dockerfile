# This Dockrerfile is based on
# https://github.com/ankur-gupta/pyspark-playground/blob/master/Dockerfile
FROM ubuntu:focal

# This is the user that will execute most of the commands within the docker
# container.
ARG PLAYGROUND_USER="neo"
ARG PLAYGROUND_USER_PASSWORD="agentsmith"

# Install the things that need root access first.
USER root

# (1) Install Java 8. Note Java 9+ is not compatible with Spark 2.4.+.
# See https://stackoverflow.com/questions/51330840/why-apache-spark-does-not-work-with-java-10-we-get-illegal-reflective-then-java
# (2) We clean up apt cache to reduce image size as mentioned here:
# https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#run
# (3) One of the packages below installs tzdata which requires user input
# even with `apt-get install -y`. Setting DEBIAN_FRONTEND and setting localtime
# is needed. Setting DEBIAN_FRONTEND in-line is better than setting it using
# ENV because we won't have to "unset" it later (unsetting is not truly
# possible anyways).
# See https://serverfault.com/questions/949991/how-to-install-tzdata-on-a-ubuntu-docker-image
RUN ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime
RUN DEBIAN_FRONTEND="noninteractive" apt-get update \
    && apt-get install -y \
        sudo \
        unzip \
        nano \
        wget \
        man \
        tree \
        vim-tiny \
        iputils-ping \
        ssh \
        openjdk-8-jdk \
        python3.8 \
        python3-dev \
        python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Print Python and Java version
RUN echo java -version
RUN echo python3 --version

# Create $PLAYGROUND_USER non-interactively and add it to sudo group.
# See
# (1) https://stackoverflow.com/questions/25845538/how-to-use-sudo-inside-a-docker-container
# (2) https://askubuntu.com/questions/7477/how-can-i-add-a-new-user-as-sudoer-using-the-command-line
RUN useradd -m $PLAYGROUND_USER \
    && adduser $PLAYGROUND_USER sudo \
    && echo $PLAYGROUND_USER:$PLAYGROUND_USER_PASSWORD | chpasswd

# Set the default values
ENV PYSPARK_PYTHON=/usr/bin/python \
    PYSPARK_DRIVER_PYTHON=/usr/bin/python

# We don't have `python` executable. Since some spark scripts have hardcoded
# `python`, we will symlink a `python` but we aim to use the symlinked
# `python` sparingly.
RUN cd /usr/bin && ln -s /usr/bin/python3 /usr/bin/python

# Setup tini. Tini operates as a process subreaper for jupyter. This prevents
# kernel crashes. Based on https://jupyter-notebook.readthedocs.io/en/stable/public_server.html#docker-cmd
# See https://github.com/krallin/tini/issues/8 for a detailed understanding of
# what tini does.
ENV TINI_VERSION="v0.18.0"
ADD https://github.com/krallin/tini/releases/download/"${TINI_VERSION}"/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0"]

# We will setup environment variables and python packages for the
# $PLAYGROUND_USER instead of root.
USER $PLAYGROUND_USER

# Note that there is no `pip` executable; use `pip3`.
# Install the common packages we may need. Don't install pyspark using pip3
# because we installed it from source already. We should be able to install
# more packages by running `pip3 install --user <package-name>` within the
# container later on, if needed.
# We remove pip cache so docker can store the layer for later reuse.
RUN pip3 install --user \
    numpy \
    pandas \
    six \
    ipython \
    jupyter \
    matplotlib \
    seaborn \
    scipy \
    scikit-learn \
    pyspark \
    flicker \
  && rm -rf /home/$PLAYGROUND_USER/.cache/pip

# Augment path so we can call ipython and jupyter
# Using $HOME would just use the root user. $HOME works with the RUN directive
# which uses the userid of the user in the relevant USER directive. But ENV
# doesn't seem to use this. See https://stackoverflow.com/questions/57226929/dockerfile-docker-directive-to-switch-home-directory
# This is probably why variables set by ENV directive are available to all
# users as mentioned in https://stackoverflow.com/questions/32574429/dockerfile-create-env-variable-that-a-user-can-see
ENV PATH=$PATH:/home/$PLAYGROUND_USER/.local/bin

# Set the working directory as the home directory of $PLAYGROUND_USER
# Using $HOME would not work and is not a recommended way.
# See https://stackoverflow.com/questions/57226929/dockerfile-docker-directive-to-switch-home-directory
WORKDIR /home/$PLAYGROUND_USER

# FIXME: Should we build matplotlib cache? (from https://registry.hub.docker.com/r/jupyter/scipy-notebook/dockerfile):
# # Import matplotlib the first time to build the font cache.
# ENV XDG_CACHE_HOME /home/$NB_USER/.cache/
# RUN MPLBACKEND=Agg python -c "import matplotlib.pyplot" && \
#    fix-permissions /home/$NB_USER

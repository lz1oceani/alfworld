FROM gitlab-registry.nrp-nautilus.io/lz1oceani/docker:nlp_11.7

ARG USER_NAME
ARG USER_PASSWORD
ARG USER_ID
ARG USER_GID

RUN apt-get update
RUN apt install -y sudo
RUN useradd -ms /bin/bash $USER_NAME --no-log-init
RUN usermod -aG sudo $USER_NAME
RUN echo "$USER_NAME:$USER_PASSWORD" | chpasswd

# Set UID and GID to match those outside the container
RUN usermod -u $USER_ID $USER_NAME
RUN groupmod -g $USER_GID $USER_NAME

# Work directory
WORKDIR /home/$USER_NAME

# Install system dependencies
COPY ./docker/install_deps.sh /tmp/install_deps.sh
RUN yes "Y" | /tmp/install_deps.sh

COPY ./docker/install_nvidia.sh /tmp/install_nvidia.sh
RUN yes "Y" | /tmp/install_nvidia.sh

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /home/$USER_NAME/miniconda && \
    rm /miniconda.sh

ENV PATH="/home/$USER_NAME/miniconda/bin:${PATH}"

# Setup Python environment with Miniconda
RUN conda create -n alfworld_env python=3.6
ENV PATH="/home/$USER_NAME/miniconda/envs/alfworld_env/bin:${PATH}"
ENV VIRTUAL_ENV="/home/$USER_NAME/miniconda/envs/alfworld_env"

# Install Python requirements
COPY ./requirements.txt /tmp/requirements.txt
RUN conda activate alfworld_env && \
    pip install --upgrade pip==19.3.1 && \
    pip install -U setuptools && \
    pip install -r /tmp/requirements.txt

# Install GLX-Gears (for debugging)
RUN apt-get update && apt-get install -y \
   mesa-utils && \
   rm -rf /var/lib/apt/lists/*

# Change ownership of everything to our user
RUN mkdir /home/$USER_NAME/alfworld
RUN chown $USER_NAME:$USER_NAME -R /home/$USER_NAME

USER $USER_NAME
ENTRYPOINT bash -c "export ALFRED_ROOT=~/alfworld && /bin/bash"
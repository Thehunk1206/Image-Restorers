ARG TAG=2.10.0

FROM tensorflow/tensorflow:${TAG}

LABEL   name="tensorflow-dev-container" \
        maintainer="Tauhid Khan" \
        version="0.0.1" \
        description="Tensorflow development container for developing and testing tensorflow models"

ARG USERNAME="vscode"
ARG DEVELOPER_HOME="/home/${USERNAME}"

# Upgrade and update packages
RUN apt-get update -y && apt-get upgrade -y

# install basic packages for dev environment
RUN apt-get install -y --no-install-recommends \
    git  \
    wget \
    tree \
    curl \ 
    zip  \
    xz-utils \
    libglu1-mesa \
    unzip \
    gnupg \
    zsh \                   
    apt-utils \
    zsh-autosuggestions \
    htop \
    nano \
    sudo

RUN apt-get clean

# install oh-my-zsh
# RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Set up new user
RUN useradd -ms /usr/bin/zsh ${USERNAME}
USER ${USERNAME}
WORKDIR ${DEVELOPER_HOME}

# Copy .zshrc
COPY ./.devcontainer/docker.zshrc.sh ${DEVELOPER_HOME}/.zshrc

# Install python packages from requirements.txt
COPY ./.devcontainer/requirements.txt ${DEVELOPER_HOME}/workspace/
RUN pip install --upgrade pip
RUN pip install --user -r ${DEVELOPER_HOME}/workspace/requirements.txt

# Setup user sudo permissions
USER root
RUN apt-get install nano sudo;
RUN chown ${USERNAME}:${USERNAME} ${DEVELOPER_HOME}/.zshrc
RUN echo ${USERNAME}:${USERNAME} | chpasswd;
RUN adduser ${USERNAME} sudo

USER ${USERNAME}
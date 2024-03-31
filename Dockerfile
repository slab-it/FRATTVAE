FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04

# create a new user
RUN useradd -u $uid -o -m jupyter-user

# suppress interactive messages
ENV DEBIAN_FRONTEND=noninteractive

# update and install packages
RUN echo "installing packages..."
RUN apt-get update -y \
	&& apt-get install -y \
	git vim wget make sudo psmisc\
	build-essential libssl-dev zlib1g-dev llvm \
	libbz2-dev libreadline-dev libsqlite3-dev curl \
	libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# install pyenv
USER jupyter-user
ENV HOME=/home/jupyter-user
ENV PYENV_ROOT=$HOME/.pyenv
ENV PATH=$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH
ENV PATH=$PATH:$HOME/bin:/sbin:/usr/sbin:/usr/local/bin

RUN echo "installing pyenv..."
RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv

RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc \
	&& eval "$(pyenv init -)"

# install python
WORKDIR ${HOME}
RUN echo "installing python..."
RUN	pyenv install -v 3.10.8 \
	&& pyenv global 3.10.8
RUN eval "$(pyenv init -)"

# update python packages
RUN pip install --upgrade pip setuptools

# install python packages
RUN pip install jupyter \
				jupyter-contrib-nbextensions \
				pipdeptree \
				numpy \
				pandas==1.5.2 \
				pyyaml \
				joblib \
				matplotlib \
				ordered-set \
				umap-learn \
				scikit-learn==1.1.3 \
				rdkit==2023.3.1 \
				molvs \
				timeout-decorator \
				scipy==1.11.3

RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 \
	&& pip install dgl==1.1.1+cu113 -f https://data.dgl.ai/wheels/cu113/repo.html
RUN pip install fcd-torch guacamol
ENV DGLBACKEND="pytorch"
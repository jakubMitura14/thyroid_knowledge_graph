# FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
# FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
FROM ubuntu:jammy-20240405


################################################################################
# Prevent apt-get from prompting for keyboard choice
#  https://superuser.com/questions/1356914/how-to-install-xserver-xorg-in-unattended-mode
ENV DEBIAN_FRONTEND=noninteractive

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    build-essential \
    wget\
    manpages-dev\
    g++\
    gcc\
    nodejs\
    libssl-dev\
    unzip\
    libidn11-dev\
    libglu1-mesa-dev \
    # libpangox-1.0-dev \
    # libpangoxft-1.0-0 \
    #cuda-11.3\
    #nvidia-cuda-toolkit-11-3\
    && rm -rf /var/lib/apt/lists/*


# ENV PATH="/opt/cmake/bin:${PATH}"
ENV NB_USER sliceruser
ENV NB_UID 1000
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.10 && \
    apt-get install -y  python3.10-dev



RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    cmake \
    g++ \
    gfortran \
    git \
    vim \
    #   hdf5-tools \
    libmkl-full-dev \ 
    libboost-dev \
    libfftw3-dev \
    libnfft3-dev \
    libgfortran5 \
    libgmp-dev \
    #   libhdf5-dev \
    libopenmpi-dev \
    cython3 \
    openmpi-bin \
    openmpi-common \
    openmpi-doc \
    python3-dev \
    python3-mako \
    python3-numpy \
    python3-scipy \
    python3-matplotlib \
    python3-mpi4py \
    python3-pip \
    python3-sphinx \
    python3-skimage \ 
    python3-nbsphinx \
    python3-notebook \
    libpython3-dev \
    git \
    ca-certificates \
    openssl \
    rsync \
    libomp-dev \
    nano \
    less \
    # Jupyter lab packages
    nodejs \
    #   texlive \
    #   dvipng \
    #   texlive-latex-extra \
    #   texlive-fonts-recommended \
    vim net-tools curl \
    libgl1-mesa-glx \
    xserver-xorg-video-dummy \
    libxrender1 \
    libpulse0 \
    libpulse-mainloop-glib0  \
    libnss3  \
    libxcomposite1 \
    libxcursor1 \
    libfontconfig1 \
    libxrandr2 \
    libasound2 \
    libglu1 \
    x11vnc \
    awesome \
    jq \
    git \
    nautilus \
    jupyter-core \
    unzip \
    default-libmysqlclient-dev


#   apt-get autoremove --purge -y && \
#   apt-get autoclean -y && \
#   rm -rf /var/cache/apt/* /var/lib/apt/lists/*


# Install cmake v3.13.2
RUN apt-get purge -y cmake && \
    mkdir /root/temp && \
    cd /root/temp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.13.2/cmake-3.13.2.tar.gz && \
    tar -xzvf cmake-3.13.2.tar.gz && \
    cd cmake-3.13.2 && \
    bash ./bootstrap && \
    make && \
    make install && \
    cmake --version && \
    rm -rf /root/temp


RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py


RUN python3 -m pip install torch==2.0.1 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 
# RUN python3 -m pip install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2--index-url https://download.pytorch.org/whl/cu121


RUN python3 -m pip  --no-cache-dir install install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    # torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 \
    pandas \
    grpcio \
    mock \
    portpicker \
    requests \
    h5py\
    #itk \
    #SimpleITK>=2.1.1.2 \
    # pandas \
    pydicom-seg \
    numpy\
    # numpy==1.17.4\
    opencv-python \
    pydicom \
    more-itertools \
    # SimpleITK\
    # nnunet==1.7.0 \
    multiprocess \
    six \
    einops seaborn \
    optuna \
    torchio \
    # connected-components-3d \
    ninja\
    cython\
    decorator \
    jupyter \
    ipywidgets \
    pytest \
    pandas \
    shapely \
    monty \
    # mpi4py \
    pymatgen \
    ase \ 
    Jinja2\
    Mako \
    Pandoc \
    # Jupyter lab packages
    jupyterlab \
    # scikit-image \
    # itk \
    # vtk \
    # monai[nibabel,itk,tqdm] \
    nibabel \
    llvmlite \
    numba \
    charset-normalizer\
    click \
    docker-pycreds \
    filelock \
    flashtool \
    gitdb \
    GitPython \
    idna \
    inflect \
    joblib \
    langchain langchain-community langchain-openai langchain-experimental neo4j tiktoken \
    evaluate seqeval graphdatascience dill gensim ipymarkup more_itertools peft sentence_transformers tqdm transformers \
    habanero doi2pdf paperscraper \
    spacy \
    scispacy \
    PyPDF2 \
    semantic-text-splitter \
    PyPaperBot \
    git+https://github.com/titipata/scipdf_parser \
    git+https://github.com/networkx/networkx.git \
    networkx \
    sentencepiece \
    click \
    loguru \
    python-dateutil \
    pytz\
    scipy \
    tqdm \
    typer \
    typing-extensions \
    tzdata \
    git+https://github.com/HTian1997/getarticle.git \
    langchain-anthropic \
    # https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz \
    # https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz \
    faiss-cpu \
    python-liquid \
    pyserini \
    datasets \
    neo4j-driver graphistry[all] protobuf \
    huggingface_hub instructor sentencepiece \
    click \
    loguru \
    numpy \
    pandas \
    python-dateutil \
    pytz \
    scipy \
    tqdm \
    typer \
    typing-extensions \
    tzdata \
    # git+https://github.com/HTian1997/getarticle.git \
    langchain-anthropic \
    # https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz \
    # https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz \
    python-liquid \
    pyserini \
    datasets \
    neo4j-driver graphistry[all] protobuf huggingface_hub chromadb 





RUN python3 -m pip  --no-cache-dir install faiss-cpu \
beautifulsoup4 \
# psycopg2 \
pyodbc \
pymssql \
medcat \
plotly \
llama_index

RUN python3 -m pip  --no-cache-dir install  ollama yachalk gspread


    # python3 -m pip  --no-cache-dir install install --no-cache-dir  git+https://github.com/NLPatVCU/PaperScraper.git


# spacy \

# python3 -m pip install selenium

# RUN python3 -m pip install -U spacy[cuda118] \
#     scispacy \
#     PyPDF2 \
#     semantic-text-splitter \
#     PyPaperBot \
#     git+https://github.com/titipata/scipdf_parser



RUN python3 -m spacy download en_core_web_lg
RUN python3 -m spacy download en_core_web_sm
RUN python3 -m spacy download en

# RUN python3 -m pip  --no-cache-dir install install --no-cache-dir neo4j-driver graphistry[all] protobuf MedRAG


# RUN python3 -m pip install habanero==1.2.6 doi2pdf paperscraper 



# RUN git clone https://github.com/kermitt2/grobid_client_python &\
# cd grobid_client_python &\
# python3 setup.py install

# RUN python3 -m pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
# RUN python3 -m pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz
# RUN python3 -m pip install git+https://github.com/HTian1997/getarticle.git
# # RUN python3 -m pip install doi2pdf paperscraper 

# RUN python3 -m pip install langchain-anthropic

# COPY /workspaces/thyroid_knowledge_graph/.devcontainer/download_for_paper_scrapper.py /
# RUN python3 download_for_paper_scrapper.py
# RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

# RUN python3 -m pip install charset-normalizer==2.0.12 \
#                                     click==8.1.3 \
#                                     docker-pycreds==0.4.0 \
#                                     filelock==3.7.1 \
#                                     flashtool==0.0.10 \
#                                     gitdb==4.0.9 \
#                                     GitPython==3.1.27 \
#                                     idna==3.3 \
#                                     inflect==5.6.0 \
#                                     joblib==1.1.0 

# RUN python3 -m pip install --upgrade --quiet  langchain langchain-community langchain-openai langchain-experimental neo4j tiktoken                                   
# RUN python3 -m pip install evaluate seqeval graphdatascience dill gensim ipymarkup more_itertools peft sentence_transformers tqdm transformers
# RUN python3 -m pip uninstall -y numpy

# RUN python3 -m pip install protobuf 

# ARG CONDA_VER=latest
# ARG OS_TYPE=x86_64
# ARG PY_VER=3.11
# ARG PANDAS_VER=1.3

# # System packages 
# RUN apt-get update && apt-get install -yq curl wget jq vim

# # Use the above args 
# ARG CONDA_VER
# ARG OS_TYPE
# # Install miniconda to /miniconda
# RUN curl -LO "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh"
# RUN bash Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh -p /miniconda -b
# RUN rm Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh
# ENV PATH=/miniconda/bin:${PATH}
# RUN conda update -y conda
# RUN conda init

# ARG PY_VER
# ARG PANDAS_VER
# # Install packages from conda 
# RUN conda install -c anaconda -y python=${PY_VER}
# RUN conda install -c milagraph -c conda-forge graphvite cudatoolkit=$(nvcc -V | grep -Po "(?<=V)\d+.\d+")






# COPY docker-entrypoint.sh /
# RUN chmod +x /docker-entrypoint.sh
# ENTRYPOINT [ "/docker-entrypoint.sh" ]
# CMD ["sleep", "infinity" ,"sh", "-c", "./Slicer/bin/PythonSlicer -m jupyter notebook --port=$JUPYTERPORT --ip=0.0.0.0 --no-browser --NotebookApp.default_url=/lab/"]
# CMD ["sleep", "infinity"]
################################################################################
# Install Slicer application startup script

# COPY .slicerrc.py .

#mkdir ~/.config/gspread

#cp /workspaces/Hausdorff_morphological/mituratest-161fba324ffd.json ~/.config/gspread/service_account.json

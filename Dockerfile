FROM python:3.11


# Supposedly the various commands represent different layers,
# => freq. changed files should be moved as far down as possible

# all the deps for keras and c tetris (like ncruses)
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    hdf5-tools \
    libhdf5-serial-dev \
    libncurses-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt . 
RUN pip install -r requirements.txt

#RUN sudo apt update && sudo apt install -y 

# provisioning
COPY src src
RUN mkdir res 
RUN mkdir logs
RUN mkdir models
COPY models models
COPY res/precollected-memory/memory.pkl res/precollected-memory/memory.pkl
# copy trained model to the docker container. 
#COPY res/collected_logs/782b2eb93b11/models/model.pt models/model.pt
WORKDIR /app/src
RUN chmod +x running-script.sh


CMD ["./running-script.sh"]
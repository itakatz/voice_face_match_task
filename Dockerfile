FROM ubuntu:latest

RUN apt update && apt install -y \
    python3 \
    python3-pip \
    git

RUN ln -s /usr/bin/python3 /usr/bin/python
WORKDIR /home
RUN git clone https://github.com/itakatz/voice_face_match_task.git
WORKDIR /home/voice_face_match_task
RUN mkdir data
COPY data/data_num_neg_pp_2.pickle data
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

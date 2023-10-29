FROM docker.io/tensorflow/tensorflow:latest-gpu

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY dockerreq.txt .

RUN pip --default-timeout=100 install -r dockerreq.txt
#RUN pip install gymnasium shimmy[gym-v21] gym[classic_control] tf-agents

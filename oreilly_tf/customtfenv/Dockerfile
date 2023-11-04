FROM docker.io/tensorflow/tensorflow:2.13.0-gpu

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 git build-essential cmake -y

COPY dockerreq.txt .

RUN pip --default-timeout=100 install -r dockerreq.txt

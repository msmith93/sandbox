#!/bin/bash

docker build . -t mytensor
docker run -u $(id -u ${USER}):$(id -g ${USER}) -v $(pwd):/app --gpus all -it --rm docker.io/library/mytensor bash -c "pushd /app && ./internal_script.sh"
#docker run -v $(pwd)/script.py:/script.py --gpus all -it --rm docker.io/tensorflow/tensorflow:latest-gpu python "script.py"

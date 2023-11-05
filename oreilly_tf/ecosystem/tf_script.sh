#!/bin/bash
set -e

docker build . -t mytensor
docker run --oom-score-adj -1000 --memory=20g -u $(id -u ${USER}):$(id -g ${USER}) -v $(pwd):/app --gpus all -it --rm docker.io/library/mytensor bash -c "pushd /app && ./internal_script.sh"
#docker run -v $(pwd)/script.py:/script.py --gpus all -it --rm docker.io/tensorflow/tensorflow:latest-gpu python "script.py"

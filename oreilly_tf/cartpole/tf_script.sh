#!/bin/bash
docker run -v $(pwd):/app --gpus all -it --rm docker.io/tensorflow/tensorflow:latest-gpu bash -c "pushd /app && ./internal_script.sh"
#docker run -v $(pwd)/script.py:/script.py --gpus all -it --rm docker.io/tensorflow/tensorflow:latest-gpu python "script.py"

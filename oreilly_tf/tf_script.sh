docker run  --gpus all -it --rm docker.io/tensorflow/tensorflow:latest-gpu    python -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))"

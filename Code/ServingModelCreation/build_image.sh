#!/bin/bash
WORKDIR=$(pwd)
nvidia-docker build -f Dockerfile_gpu -t glomerulia_mrcnn .
nvidia-docker run --rm -p 8500:8500 -p 8501:8501 -t glomerulia_mrcnn --model_config_file=/models/model_config.tf &
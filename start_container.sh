#!/bin/bash

# docker run [OPTIONS] IMAGE [COMMAND] [ARG...]
docker run  -it \
            -v /Users/akshay/Desktop/PhD/courses/Deep-Learning-CSI-5140/myProject_1/dnn-cifar-10:/workspace  \
            -w /workspace   \
            --network=host  \
            dnn-cifar10
            

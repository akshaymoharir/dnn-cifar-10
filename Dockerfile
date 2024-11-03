

# Use a base image with PyTorch installed
FROM pytorch/pytorch:latest

# Install pip and required packages
# Packages: matplotlib
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    pip3 install matplotlib

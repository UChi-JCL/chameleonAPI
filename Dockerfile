# Use the NVIDIA CUDA image with cuDNN runtime on UBI 8 as the base image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel 
# Set the environment variables

# Install necessary packages
RUN apt update -y && \
    apt-get install -y wget bzip2 unzip  git

# Set the working directory to /workspace
WORKDIR /workspace

# Copy the current working directory contents into the container at /workspace
COPY . /workspace

# Install PyTorch, torchvision, and torchaudio
RUN pip install -e nlp/transformers && \
    pip install accelerate==0.29.3 && \
    pip install datasets==2.19.0 && \
    pip install evaluate==0.4.1 && \
    pip install numpy==1.26.3 && \
    pip install pandas==2.2.2 && \
    pip install pillow==10.2.0 && \
    pip install protobuf==4.25.3 && \
    pip install pyarrow==16.0.0 && \
    pip install randaugment==1.0.2 && \
    pip install requests==2.28.1 && \
    pip install scikit-learn==1.4.2 && \
    pip install scipy==1.13.0 && \
    pip install safetensors==0.4.3 && \
    pip install six==1.16.0 && \
    pip install tqdm && \
    pip install matplotlib && \
    pip install pycocotools

# Copy the current working directory contents into the container at /workspace

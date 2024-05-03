To install the packages with Docker, you should follow the steps below:

### Install docker 

First please make sure docker is installed. We provide a **reference** installation guide in ```scripts/install_docker.sh```, while you may need to adjust some of the steps to make this run. 

### Pull and run docker image 
```
sudo docker pull yuhanliuchi/chameleonapi

sudo docker run --gpus all -it yuhanliuchi/chameleonapi bash
```

### Git clone

Make sure you git clone the latest GitHub code, don't use the code under ```/workspace``` !
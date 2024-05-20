This is the readme for running the training script for ChameleonAPI. 

First, activate the docker image with (Note that you **need** to set --shm-size=10gb):

```
sudo docker run --shm-size=10gb  --gpus all -it yuhanliuchi/chameleonapi 
```


To run the training code for one example app, first clone the code repo with ```git clone https://github.com/UChi-JCL/chameleonAPI.git```. If you have already cloned the repo before, please do ``` git pull```.

Then, please go into ```chameleonAPI/image_classification```.

Then run the following code:

```
bash train.sh
```

To install necessary packages required to run ChameleonAPI, please install the conda environment with ```env.yml```. 

### Install miniconda

If miniconda (or anaconda) is not installed on your local machine, install it with 

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

Follow the instructions to finish installing miniconda. 

### Install ChameleonAPI's required packages 

```
conda env create -f env.yml 
```
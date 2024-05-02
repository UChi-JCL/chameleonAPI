# Overview 

This is the artifact for ChamleonAPI: Automatic and Efficient Customization of Neural Networks for ML Applications (to appear at OSDI 24). 

This artifact contains the code and trained model weights required to reproduce the key results in the paper (you may find it at the HotCRP website). 

The installation of required packages can be found at install.md, and for reproducing the results, you may find it at scripts/eval.md. 

#  Structure of the repo

```scripts/```: This contains the scripts to plot Table 3. 

```image_classification/ ```: This contains the scripts to run evaluation for the image classification apps. 

```object_cc/ ```: This contains the scripts to run evaluation for the object detection apps. 

```nlp/ ```: This contains the scripts to run evaluation for the text topic classification apps. 

## Installation 
Go to [install.md](install.md)

## Kick-the-tire experiments

Please check if you are able to download ChameleonAPI (and baseline) models in the links below: 

``` wget https://s3-us-east-2.amazonaws.com/chameleonapi/image_classification_models.zip ```

``` wget https://s3-us-east-2.amazonaws.com/chameleonapi/object_detection_models.zip ```

``` wget https://s3-us-east-2.amazonaws.com/chameleonapi/nlp_models.zip ```

To run a "hello-world" experiment, first, make sure the environment is installed [install.md](install.md).

Then, run the following code:

```
bash scripts/run.sh
```

## Reproducing the main results
Details in [here](scripts/eval.md)

## Contact

Yuhan Liu (yuhanl@uchicago.edu)


Before running the code, please do ```git pull ``` first. 

## Image classification applications
To run image classification apps, go to path ``` chameleonAPI/image_classification ```. 

First, download the trained models and evaluation datasets from s3 by: 

``` wget https://s3-us-east-2.amazonaws.com/chameleonapi/image_classification_models.zip ```

Then, unzip it (put under the path```chameleonAPI/image_classification ```).

Now, reproduce the accuracy in the paper by running:

#### If you installed with docker images
``` bash run_docker.sh ```

#### If you installed with conda
``` bash run_conda.sh ```

## Object detection applications

To run object detection apps, go to path ``` chameleonAPI/object_detection ```

First, download the trained models and evaluation datasets from s3 by: ``` wget https://s3-us-east-2.amazonaws.com/chameleonapi/object_detection_models.zip ```

Then, unzip it (put under the path ```chameleonAPI/object_detection ```).

Now, reproduce the accuracy of the applications by running: 
#### If you installed with docker images
``` bash run_docker.sh ```

#### If you installed with conda
``` bash run_conda.sh ```

## Text topic classification applications
To run object detection apps, go to path ``` chameleonAPI/nlp/transformers/examples/pytorch/text-classification/ ```

First, download the trained models and evaluation datasets from s3 by: ``` wget https://s3-us-east-2.amazonaws.com/chameleonapi/nlp_models.zip ```

Then, unzip it (put under the path ```chameleonAPI/nlp/transformers/examples/pytorch/text-classification/ ```).


Now, reproduce the accuracy of the applications by running: 

#### If you installed with docker images
``` bash run_docker.sh ```

#### If you installed with conda
``` bash run_conda.sh ```


## Getting Table 3
After running the above commands, you can get the summary by running ```python scripts/draw_table.py ``` under the root path ``` <YOUR PATH TO THE GITHUB REPO>/chameleonAPI/ ```

You will see the output ![Table 3](results.png "Results")
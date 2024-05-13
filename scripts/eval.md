
Before running the code, please do ```git pull ``` first. 

## Image classification applications
To run image classification apps, go to path ``` chameleonAPI/image_classification ```. 

Now, reproduce the error rate in the paper by running:

#### If you installed with docker images
``` bash run_docker.sh ```

#### If you installed with conda
``` bash run_conda.sh ```

## Object detection applications

To run object detection apps, go to path ``` chameleonAPI/object_detection ```

Now, reproduce the error rate of the applications by running: 
#### If you installed with docker images
``` bash run_docker.sh ```

#### If you installed with conda
``` bash run_conda.sh ```

## Text topic classification applications
To run object detection apps, go to path ``` chameleonAPI/nlp/transformers/examples/pytorch/text-classification/ ```

Now, reproduce the error rate of the applications by running: 

#### If you installed with docker images
``` bash run_docker.sh ```

#### If you installed with conda
``` bash run_conda.sh ```


## Getting Table 3
After running the above commands, you can get the summary by running ```python scripts/draw_table.py ``` under the root path ``` <YOUR PATH TO THE GITHUB REPO>/chameleonAPI/ ```

You will see the output ![Table 3](results.png "Results")
# Knowledge-Assisted Multi-Graph Structure Learning for Multivariate Time Series Anomaly Detection in Multi-Stage Processes

Code implementation for : [Knowledge-Assisted Multi-Graph Structure Learning for Multivariate Time Series Anomaly Detection in Multi-Stage Processes].

The code is partly based on the implementation of ["Graph Neural Network-Based Anomaly Detection in Multivariate Time Series" (AAAI '21)](https://ojs.aaai.org/index.php/AAAI/article/view/16523).

# Installation
### Requirements
* Python >= 3.6
* cuda == 10.2
* [Pytorch==1.5.1](https://pytorch.org/)
* [PyG: torch-geometric==1.5.0](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

### Install packages
```
    # run after installing correct Pytorch package
    bash install.sh
```

### Run
Run to check if the environment is ready
```
    bash run.sh cpu <dataset>
    # or with gpu
    bash run.sh <gpu_id> <dataset>  
```

## Data Preparation
```
# put your dataset under data/ directory with the following structure of data/<dataset>/

data
 |-<dataset>
 | |-list.txt    # the feature names, one feature per line
 | |-train.csv   # training data
 | |-test.csv    # test data
 |-<dataset2>
 | ...

```

### Notices:
* The first column in .csv will be regarded as index column. 
* The column sequence in .csv don't need to match the sequence in list.txt, we will rearrange the data columns according to the sequence in list.txt.
* test.csv should have a column named "attack" which contains ground truth label(0/1) of being attacked or not(0: normal, 1: attacked)

## Run
```
    # using gpu
    bash run.sh <gpu_id> <dataset>

    # or using cpu
    bash run.sh cpu <dataset>
```
You can change running parameters in the run.sh.

# Others
SWaT and WADI datasets can be requested from [iTrust](https://itrust.sutd.edu.sg/itrust-labs_datasets/)


```

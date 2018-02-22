# Progressive Learning Technique for Multi-class Classification

This repository contains the python implementation of progressive multi-class classifier of our paper "A Novel Progressive Learning Technique for Multi-class Classification". The paper is available at https://arxiv.org/pdf/1609.00085.pdf 

__Folder Description__
1. src - Contains the python source codes
2. datasets - Contains the dataset csv file
3. results - Contains the results file generated during execution
4. logs - Contains the logs of each execution

__Dependencies__
This code is developed in Python 3.6 (Ubuntu 16.04) and has following package dependencies:
 - numpy 1.14.0
 - pandas 0.21.1
 - scikit-learn 0.19.1

__To run the code:__
Move to src folder from command line and
> $ python main.py <optional command line arguements>

__Arguements:__
 - Filename of the dataset
 
        -f FILENAME, --filename FILENAME    (default: ../datasets/iris_plt.csv)
 - Label location in the dataset
 
        -l LABEL_LOCATION, --label LABEL_LOCATION     [can take "last" or "first" or None](default: last) 
 - Scaling Type
 
        -s SCALE_TYPE, --scale SCALE_TYPE   [can take "minmax" or "std"](default: minmax)
 - Testing Ratio
 
        -t TEST_RATIO, --testratio TEST_RATIO     [can take values in range 0 to 1](default: 0.1)
 __Hyperparameters of the model:__
 - Number of Hidden layer neurons
        
        -n HIDDEN_NEURONS, --neurons HIDDEN_NEURONS     (default: 10)
 - Number of samples in initial block
 
        -i INIT_BLOCK_SIZE, --initial INIT_BLOCK_SIZE     (default=30)
 - Batch size for training
        
        -b BATCH_SIZE, --batch BATCH_SIZE   (default=1)

__Contact__
For queries, please email RAJA0046@e.ntu.edu.sg

__Citation__
 - Please consider the following paper for citing this work:
        
        @article{venkatesan2016novel,
        title={A novel progressive learning technique for multi-class classification},
        author={Venkatesan, Rajasekar and Er, Meng Joo},
        journal={Neurocomputing},
        volume={207},
        pages={310--321},
        year={2016},
        publisher={Elsevier}
        }

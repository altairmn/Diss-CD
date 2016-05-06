### Instructions

The dataset(s) must be present in the same folder as the program.
They should have the `.pkl.gz` extension.

To download MNIST and CIFAR10 (processed for this project), use the following links:
MNIST:
CIFAR10:


### Description

#### rbm.py

This file contains the core program. To view help on how to run, use the `--help` flag.

Example: 

    python rbm.py --help


Example:
    
    python rbm.py --dataset=mnist.pkl.gz --neg-dataset=cifar_disscd.pkl.gz --epochs=1 --lr=0.1 --k=1 --output-folder=verification --n-hidden=200 --batch-size=40 --pcd



#### mnist_digits_pcd.py

This file trains an RBM for each digit separately
using PCD.


Example:
    python mnist_digits_pcd.py


#### mnist_digits_disscd.py

This file trains an RBM for each digit separately
and untrains on the rest of the digits.


Example:
    python mnist_digits_disscd.py

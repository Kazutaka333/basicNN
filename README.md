# basicNN
Basic Neural Network

The simplest form of neural network to understand the basic concept.

- this code is based on a book, Make Your Own Neural Network.

### dependancies:
    
numpy

scipy


## The following command creates the environment to run this program

    pip install virtualenv

    virtualenv --system-site-packages [directory name]

    cp /path/to/basicNN/. [directory name]/

    pip3 install numpy

    pip3 install scipy


## main.py
This will train a neural network instance with training data sets (this 
process could take a while) and at the end it will check the performance 
with test data sets. 

## trained.py
This program loads pre-trained weights for neural net so you can check the 
performance without waiting for your own training. This performance should
show the same result as main.py 

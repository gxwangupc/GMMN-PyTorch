import pickle
import numpy as np
"""
Load the training images from the MNIST dataset.
"""
def loadMNIST():

    # Downloaded from http://deeplearning.net/data/mnist/mnist.pkl.gz
    with open('./data/mnist.pkl', 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
    train_data, val_data, test_data = u.load()
    train_x, train_y = train_data

    return train_x

"""
Load the training images from the cropped LFW dataset.
"""
def loadLFW():

    # 32x32 version of grayscale cropped LFW
    # Original dataset here: http://conradsanderson.id.au/lfwcrop/
    return np.load('lfw.npy')

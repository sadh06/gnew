import numpy as np
import valohai

# example of loading the mnist dataset
from tensorflow.keras.datasets import mnist

def main():
    # valohai.prepare enables us to update the valohai.yaml configuration file with
    # the Valohai command-line client by running `valohai yaml step preprocess_dataset.py`

    valohai.prepare (
        step='preprocess-dataset',
        image='tensorflow/tensorflow:1.13.1-py3',
        default_inputs = {
        'name': 'training-set-images',
        'training-set-images': 'https://github.com/sadh06/dataset/blob/main/train-images-idx3-ubyte.gz',
        'name': 'training-set-labels',
        'training-set-labels': 'https://github.com/sadh06/dataset/blob/main/train-labels-idx1-ubyte.gz',
        'name': 'test-set-images',
        'test-set-images': 'https://github.com/sadh06/dataset/blob/main/t10k-images-idx3-ubyte.gz',
        'name': 'test-set-labels',
        'test-set-labels': 'https://github.com/sadh06/dataset/blob/main/t10k-labels-idx1-ubyte.gz'
        }
    )

    # Read input files from Valohai inputs directory
    # This enables Valohai to version your training data
    # and cache the data for quick experimentation

    print('Loading data')
    #with np.load(valohai.inputs('dataset').path(), allow_pickle=True) as file:
        #x_train, y_train = file['x_train'], file['y_train']
        #x_test, y_test = file['x_test'], file['y_test']
    with np.load(valohai.inputs('training-set-images').path(), allow_pickle=True) as file:
        x_train = file['x_train']
        print(x_train)
    with np.load(valohai.inputs('training-set-labels').path(), allow_pickle=True) as file:
        y_train = file['y_train']
        print(y_train)
    with np.load(valohai.inputs('test-set-images').path(), allow_pickle=True) as file:
        x_test = file['x_test']
        print(x_test)
    with np.load(valohai.inputs('test-set-labels').path(), allow_pickle=True) as file:
        y_test = file['y_test']
        print(y_test)

    print('Preprocessing data')
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Write output files to Valohai outputs directory
    # This enables Valohai to version your data
    # and upload output it to the default data store

    print('Saving preprocessed data')
    path = valohai.outputs().path('preprocessed_mnist.npz')
    np.savez_compressed(path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


if __name__ == '__main__':
    main()

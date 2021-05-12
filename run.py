import argparse
import numpy as np
import tensorflow as tf
from skimage.util import random_noise 
from watermark import watermarking
from train import training



def data_loader(args):
    
    if args.dataset=="cifar10":
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    elif args.dataset=="cifar100":
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data(label_mode='coarse')
    else:
        print("data load error")

    if args.alpha != 0:
        x_train=train_images / 255.
        x_test=test_images / 255.
        x_train_wm=watermarking(train_images, args.alpha)
        x_test_wm=watermarking(test_images, args.alpha)
        
        ## Gaussian noise
#         x_train_wm=random_noise(train_images, mode="gaussian", var=args.alpha)
#         x_test_wm=watermarking(test_images, mode="gaussian", var=args.alpha)
        
        ## Salt-and-pepper noise
#         x_train_wm=random_noise(train_images, mode="s&p", amount=args.alpha)
#         x_test_wm=random_noise(test_images, mode="s&p", amount=args.alpha)
    else:
        x_train=train_images / 255.
        x_test=test_images / 255.
        x_train_wm=train_images / 255.
        x_test_wm=test_images / 255.

    x_train=x_train.astype("float32")
    x_train_wm=x_train_wm.astype("float32")
    x_test=x_test.astype("float32")
    x_test_wm=x_test_wm.astype("float32")

    return x_train, x_train_wm, train_labels, x_test, x_test_wm, test_labels


def run(args):
    
    print("=== Data Loading ===")
    x_train, x_train_wm, y_train, x_test, x_test_wm, y_test = data_loader(args)
    print("=== Modeling ===")
    training(args, x_train, x_train_wm, y_train, x_test, x_test_wm, y_test)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Watermark Out-of-Distribution Detection')
    
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--alpha', default='0', type=float)
    parser.add_argument('--model', default='resnet', type=str)
    
    args = parser.parse_args('')
    run(args)
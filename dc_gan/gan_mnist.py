from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
from deconv2d_layer import Deconv2DLayer
import matplotlib.pyplot as plt

# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [-1,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return ((data / np.float32(256))*2) - 1

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test



########################### Build the generator ######################################
# input_var : vector of 100 random variable z
# output : Tensor4 of size (nbBatch, num_channels, width, height) which
# contains generated samples.
# for reference at the network architecture take a look at the figure 1 in :
# https://arxiv.org/pdf/1511.06434.pdf

# the layer Deconv2DLayer is not included in lasagne, but taken from a Lasagne
# contributor in his DCGAN implementation :
# https://gist.github.com/f0k/738fa2eedd9666b78404ed1751336f56
# unlike the deconv layers provided by lasagne, it allow to have stride and pad at
# the same time.
def build_generator(input_var= None):

    # input: z = vector of 100 random numbers generated by a normal distribution.
    network = lasagne.layers.InputLayer(shape=(None, 100), input_var=input_var)

    # project z
    network = lasagne.layers.batch_norm(
                lasagne.layers.DenseLayer(network, 512 * 7 * 7)
    )
    # reshape z to have the shape for the convolution
    # n.b = the [0] keep the previous size at position 0 (unknown batch size)
    network = lasagne.layers.ReshapeLayer(network, ([0], 512, 7, 7))

    # Given that the MNIST dataset doesnt have the same image size (28 x 28) from the article
    # we need to adapt the architecture, each fractional-stride convolutions
    # with padding =2, strides = 2 and filter size = 5 give an images two
    # times bigger. With a size of 28 x 28 we are limited to 2 deconv 7 x 7 -> 14 x 14 -> 28 x 28
    # this is the reason why we don't have the same number of layers compared to the article.

    # We use the nonlinear function ReLU (default for the first deconv and tanh for the last deconv
    # to have a restricted interval

    # two fractional-stride convolutions
    network = lasagne.layers.batch_norm(
        Deconv2DLayer(network, num_filters=256, filter_size=5, stride=2, pad=2)
    )
    # we want one output feature to have grayscale images
    network = Deconv2DLayer(network,num_filters=1,filter_size= 5, stride=2, pad=2,
                          nonlinearity=lasagne.nonlinearities.tanh)

    return network

################################## Discriminator ###########################################
# The discriminator is a simple convolutional network which does binary classification,
# input: 28 x 28 MNIST image
# output: 1 for a real image, 0 for a fake image
# for the figure you can look at this paper (figure 3) :
# https://arxiv.org/pdf/1607.07539.pdf
def build_discriminator(input_var=None):

    # Leaky rectify at 0.2 like in the article
    leaky_rectify = lasagne.nonlinearities.LeakyRectify(0.2)

    # input size (batch size, channel size, width, height)
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)

    # First convolution 28 x 28 -> 14 -> 14
    network = lasagne.layers.batch_norm(
        lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=5, stride=2, pad=2,nonlinearity=leaky_rectify)
    )
    # Second convolution 14 x 14 -> 7 -> 7
    network = lasagne.layers.batch_norm(
        lasagne.layers.Conv2DLayer(network, num_filters=128, filter_size=5, stride=2, pad=2,nonlinearity=leaky_rectify)
    )

    # We would normally have a fully connected layer here but we'll skip it

    # Output layer (the article says that's better without fully connected layer before the output)
    network = lasagne.layers.DenseLayer(network, 1, nonlinearity=lasagne.nonlinearities.sigmoid)


    return network



# ############################# Batch iterator ###############################
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]



################## Main function ##############################
def main(network_type="mlp",num_epochs=500, learning_rate = 0.0002, batch_size = 126):

    print('Loading the dataset ...')
    # Load the dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    print('data loaded !')

    print('Prepare and construct the model ...')

    # prepare theano variables for inputs and targets

    # Generator inputs ( random z vectors)
    gen_inputs = T.matrix('gen_inputs')

    # Discriminator inputs ( can be generate or real images)
    disc_inputs = T.tensor4('disc_inputs')

    # We save the losses for further plotting
    hist_loss_gen = []
    hist_loss_disc = []

    # Create neural network model

    # Initialize the generator network and functions
    generator = build_generator(gen_inputs)

    # Function to generate image from the generator
    generate_sample = lasagne.layers.get_output(generator)

    # Initialize the discriminator network and functions
    discriminator = build_discriminator(disc_inputs)

    # Will give use the predicted labels for the images (real input images)
    disc_prediction_real = lasagne.layers.get_output(discriminator)
    # Same prediction but we now pass fake images from the generator ( generate sample will give type tensor4)
    # n.b: get_output function can take a second argument as input for the network
    disc_prediction_fake = lasagne.layers.get_output(discriminator, generate_sample)


    # Classical logistic regression loss function

    # Will evaluate if the discriminator evaluate well that the real images are real
    # we inverse fake and real 0 = real, 1 = fake-> tips from : https://github.com/soumith/ganhacks
    disc_loss_real = lasagne.objectives.binary_crossentropy(disc_prediction_real, 0)
    # Will evaluate if the discriminator evaluate well that the fake image are fake
    # ( the generator will try to maximise this and the generator minimize it)
    disc_loss_fake = lasagne.objectives.binary_crossentropy(disc_prediction_fake, 1)

    # Give the real loss (fake and real)
    disc_loss = (disc_loss_fake + disc_loss_real).mean()

    # Here is the trick why the generator will actually to train to make realistic
    # generated images, his objective is to maximize the fact the generator evaluate
    # the fake images real.
    gen_loss = lasagne.objectives.binary_crossentropy(disc_prediction_fake, 0)
    gen_loss = gen_loss.mean()

    # Get the parameters and define the updates
    gen_params = lasagne.layers.get_all_params(generator, trainable = True)
    disc_params = lasagne.layers.get_all_params(discriminator, trainable=True)

    # We use adam optimizer with beta1=0.5 because the article (DCGAN) tell use that it works well
    gen_updates = lasagne.updates.adam(gen_loss, gen_params, learning_rate = learning_rate, beta1=0.5)
    disc_updates = lasagne.updates.adam(disc_loss, disc_params, learning_rate = learning_rate, beta1=0.5)

    # SGD steps

    # a generator step
    train_gen = theano.function([gen_inputs], gen_loss, updates=gen_updates)

    # a discriminator step
    train_disc = theano.function([disc_inputs, gen_inputs], disc_loss, updates=disc_updates)

    # get the losses
    loss_disc_fn = theano.function([disc_inputs, gen_inputs], disc_loss)
    gen_disc_fn = theano.function([gen_inputs], gen_loss)

    # Evaluate the generated samples and give the images ( for visualisation)
    fn_generate_sample = theano.function(
        inputs = [gen_inputs],
        outputs =  generate_sample
    )

    print("Model created !")

    print("Training ...")
    for epoch in range(num_epochs):

        # In each epochm we do a full pass over the training data:
        train_err_gen = 0
        train_err_disc = 0
        train_batches = 0
        start_time = time.time()

        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):

            inputs, targets = batch
            #noise_batch = np.random.uniform(-1, 1, [batch_size, 100]).astype(theano.config.floatX)
            noise_batch = np.random.normal(0, 1, [batch_size, 100]).astype(theano.config.floatX)
            train_err_gen += train_gen(noise_batch)
            train_err_disc += train_disc(inputs, noise_batch)
            train_batches += 1
            print(train_batches)

        samples = fn_generate_sample(np.random.normal(0, 1, [100, 100]).astype(theano.config.floatX))

        # save the result of untrained generated images
        plt.imsave('output/untrained_generator_epoch'+str(epoch)+'.png',
                   (samples.reshape(10, 10, 28, 28)
                    .transpose(0, 2, 1, 3)
                    .reshape(10 * 28, 10 * 28)),
                   cmap='gray')


        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss generator:\t\t{:.6f}".format(train_err_gen / train_batches))
        print("  training loss discriminator:\t\t{:.6f}".format(train_err_disc / train_batches))

main()

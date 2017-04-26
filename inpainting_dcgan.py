import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
import lasagne
from deconv2d_layer import Deconv2DLayer
import PIL.Image as Image
import matplotlib.pyplot as plt
from read_data import get_nb_train, get_train_batch
from matplotlib.pyplot import ion

import scipy.misc
from tempfile import TemporaryFile

# Global variables
image_size = 32


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
                lasagne.layers.DenseLayer(network, 1024 * 4 * 4)
    )
    # reshape z to have the shape for the convolution
    # n.b = the [0] keep the previous size at position 0 (unknown batch size)
    network = lasagne.layers.ReshapeLayer(network, ([0], 1024, 4, 4))

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
    # two fractional-stride convolutions
    network = lasagne.layers.batch_norm(
        Deconv2DLayer(network, num_filters=128, filter_size=5, stride=2, pad=2)
    )

    # we want one output feature to have grayscale images
    network = Deconv2DLayer(network,num_filters=3,filter_size= 5, stride=2, pad=2,
                          nonlinearity=lasagne.nonlinearities.tanh)
    print(lasagne.layers.get_output_shape(network))
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
    network = lasagne.layers.InputLayer(shape=(None, 3, image_size, image_size), input_var=input_var)

    # First convolution 32 x 32 -> 16 -> 16
    network = lasagne.layers.batch_norm(
        lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=5, stride=2, pad=2,nonlinearity=leaky_rectify)
    )

    network = lasagne.layers.batch_norm(
        lasagne.layers.Conv2DLayer(network, num_filters=128, filter_size=5, stride=2, pad=2,nonlinearity=leaky_rectify)
    )

    network = lasagne.layers.batch_norm(
        lasagne.layers.Conv2DLayer(network, num_filters=256, filter_size=5, stride=2, pad=2,nonlinearity=leaky_rectify)
    )

    # We would normally have a fully connected layer here but we'll skip it

    # Output layer (the article says that's better without fully connected layer before the output)
    network = lasagne.layers.DenseLayer(network, 1, nonlinearity=lasagne.nonlinearities.sigmoid)


    return network




################## Main function ##############################
def main(network_type="mlp",num_epochs=500, learning_rate = 0.0002, batch_size = 100):

    #nb_train = get_nb_train()

    nb_train = 82610
    nb_train_batch = nb_train // batch_size

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

        # In each epoch we do a full pass over the training data:
        train_err_gen = 0
        train_err_disc = 0
        train_batches = 0
        start_time = time.time()

        for i in range(nb_train_batch):


            # create batch
            batch = get_train_batch(i, batch_size)

            [data_input, data_target] = get_train_batch(i, batch_size)
            print("batch :",i)

            inputs = data_target
            #noise_batch = np.random.uniform(-1, 1, [batch_size, 100]).astype(theano.config.floatX)
            noise_batch = np.random.normal(0, 1, [batch_size, 100]).astype(theano.config.floatX)
            train_err_gen = train_gen(noise_batch)
            train_err_disc = train_disc(inputs, noise_batch)
            train_batches += 1

            hist_loss_gen.append(train_err_gen)
            hist_loss_disc.append(train_err_disc)

            samples = fn_generate_sample(np.random.normal(0, 1, [100, 100]).astype(theano.config.floatX))

            # Some output after 10 iterations
            if train_batches % 200 == 0:

                print("Generator error",train_err_gen)
                print("Discriminator error", train_err_disc)

                # 10 x 10 img batch
                img_100 =  inputs.reshape(10, 10, 3, image_size, image_size).transpose(0, 3, 1, 4, 2).reshape(10 * image_size, 10 * image_size, 3)
                rgbArray = np.zeros((10 * image_size, 10 * image_size, 3), 'uint8')
                rgbArray[..., 0] = ((img_100[:,:,0] + 1) / 2) * 256
                rgbArray[..., 1] = ((img_100[:,:,1] + 1) / 2) * 256
                rgbArray[..., 2] = ((img_100[:,:,2] + 1) / 2) * 256
                img = Image.fromarray(rgbArray)
                img.save('inpainting_output/real_samples'+str(epoch)+'.png')

                # 10 x 10 img generated
                generated_100 =  samples.reshape(10, 10, 3, image_size, image_size).transpose(0, 3, 1, 4, 2).reshape(10 * image_size, 10 * image_size, 3)
                rgbArray = np.zeros((10 * image_size, 10 * image_size, 3), 'uint8')
                rgbArray[..., 0] = ((generated_100[:,:,0] + 1) / 2) * 256
                rgbArray[..., 1] = ((generated_100[:,:,1] + 1) / 2) * 256
                rgbArray[..., 2] = ((generated_100[:,:,2] + 1) / 2) * 256
                img = Image.fromarray(rgbArray)
                img.save('inpainting_output/generated_epoch_'+str(epoch)+'batch_'+str(train_batches)+'.png')

                # save model
                np.savez('generator'+str(epoch)+ 'model.npz', *lasagne.layers.get_all_param_values(generator))
                np.savez('discriminator' + str(epoch) + 'model.npz',*lasagne.layers.get_all_param_values(discriminator))
                # Plot the figures

                ## Compute the x and y coordinates for points on sine and cosine curves
                x = np.arange(len(hist_loss_gen))
                # Plot the points using matplotlib
                plt.plot(x, hist_loss_gen)
                plt.plot(x, hist_loss_disc)
                plt.xlabel('update')
                plt.ylabel('loss')
                plt.title('Erreur du générateur et du discriminant.')
                plt.legend(['Générateur', 'Discriminant'])
                plt.savefig("inpainting_output/figure_epoch"+str(epoch)+".png")
                plt.close()
        # Then we print the results for this epoch:
        #print("Epoch {} of {} took {:.3f}s".format(
        #    epoch + 1, num_epochs, time.time() - start_time))
        #print("  training loss generator:\t\t{:.6f}".format(train_err_gen ))
        #print("  training loss discriminator:\t\t{:.6f}".format(train_err_disc ))

main()

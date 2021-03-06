import sys
import os
import time

import numpy as np
import theano
import random
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
latent_space_size = 500
latent_text_space_size = 500
SEQ_LENGTH = 40
DICT_SIZE = 35629
def build_encoder(input_var = None):

    # Leaky rectify at 0.2 like in the article
    leaky_rectify = lasagne.nonlinearities.LeakyRectify(0.2)    # Leaky rectify at 0.2 like in the article

    # input: context image (image whitout center region)
    # rgb image of size 64 x 64
    network = lasagne.layers.InputLayer(shape=(None, 3, image_size *2, image_size*2), input_var=input_var)
    #network = lasagne.layers.DropoutLayer(network)
    network = lasagne.layers.batch_norm(
        lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=5, stride=2, pad=2,nonlinearity=leaky_rectify)
    )
    # First convolution 32 x 32 -> 16 -> 16
    network = lasagne.layers.batch_norm(
        lasagne.layers.Conv2DLayer(network, num_filters=128, filter_size=5, stride=2, pad=2,nonlinearity=leaky_rectify)
    )

    network = lasagne.layers.batch_norm(
        lasagne.layers.Conv2DLayer(network, num_filters=256, filter_size=5, stride=2, pad=2,nonlinearity=leaky_rectify)
    )

    network = lasagne.layers.batch_norm(
        lasagne.layers.Conv2DLayer(network, num_filters=512, filter_size=5, stride=2, pad=2,nonlinearity=leaky_rectify)
    )


    network = lasagne.layers.ReshapeLayer(network, ([0], 512 * 4 *4))

    network = lasagne.layers.DropoutLayer(network)
    # Channel wise fully connected
    network = lasagne.layers.DenseLayer(network, latent_space_size, nonlinearity=leaky_rectify)

    # Output layer (the article says that's better without fully connected layer before the output)
    return network

# Hybrid CNN-LSTM Word encoder
def build_text_encoder(input_var = None):

    leaky_rectify = lasagne.nonlinearities.LeakyRectify(0.2)

    # shape = (batch_size, num_input_channels, input_length)
    network = lasagne.layers.InputLayer(shape=(None,DICT_SIZE,SEQ_LENGTH), input_var=input_var)

    network = lasagne.layers.batch_norm(lasagne.layers.Conv1DLayer(network, num_filters=124, filter_size=3, nonlinearity=leaky_rectify))
    network = lasagne.layers.batch_norm(lasagne.layers.Conv1DLayer(network, num_filters=256, filter_size=2, nonlinearity=leaky_rectify))
    network = lasagne.layers.batch_norm(lasagne.layers.MaxPool1DLayer(network,pool_size=3))
    network = lasagne.layers.batch_norm(lasagne.layers.Conv1DLayer(network,num_filters=512, filter_size=2, nonlinearity=leaky_rectify))
    #print(lasagne.layers.get_output_shape(network))(None, 512, 11)
    # (batch size, SEQ_LENGTH, num_features)
    network = lasagne.layers.DimshuffleLayer(network,(0,2,1))

    network = lasagne.layers.batch_norm(lasagne.layers.LSTMLayer(network, num_units=8,nonlinearity=leaky_rectify))
    #network =lasagne.layers.ReshapeLayer(network,( -1,8))

    network = lasagne.layers.DropoutLayer(network)

    network = lasagne.layers.DenseLayer(network, latent_text_space_size, nonlinearity=leaky_rectify)

    # Network output size = 500 (latent space of the text part)
    return network

def build_concat_encoder(text_encoder, image_encoder):
    network = lasagne.layers.ConcatLayer(incomings=[text_encoder, image_encoder])
    print(lasagne.layers.get_output_shape(network))
    return network
########################### Build the generator ######################################
# in this approach the generator will also have the role of the decoder
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
    network = lasagne.layers.InputLayer(shape=(None, latent_space_size + latent_text_space_size), input_var=input_var)

    # project z
    network = lasagne.layers.batch_norm(
                lasagne.layers.DenseLayer(network, 1024 * 2 * 2)
    )
    # reshape z to have the shape for the convolution
    # n.b = the [0] keep the previous size at position 0 (unknown batch size)
    network = lasagne.layers.ReshapeLayer(network, ([0], 1024,2, 2))

    # We use the nonlinear function ReLU (default for the first deconv and tanh for the last deconv
    # to have a restricted interval

    # two fractional-stride convolutions
    network = lasagne.layers.batch_norm(
        Deconv2DLayer(network, num_filters=512, filter_size=5, stride=2, pad=2)
    )
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
def main(num_epochs=500, learning_rate = 0.00001, batch_size = 100,disc_save_model="discriminator9model.npz",gen_save_model="generator9model.npz",
         enc_save_model="encoder9model.npz", text_enc_save_model="textencoder9model.npz"):

    nb_train = get_nb_train()
    nb_val_batch = get_nb_train("../../data/inpainting/val2014/") // batch_size

    nb_train_batch = nb_train // batch_size

    print('Prepare and construct the model ...')

    # prepare theano variables for inputs and targets

    # Generator inputs ( random z vectors)
    gen_inputs = T.matrix('gen_inputs')

    # Discriminator inputs ( can be generate or real images)
    disc_inputs = T.tensor4('disc_inputs')

    # image encoder input
    encoder_inputs = T.tensor4('encoder_inputs')

    #text encoder input
    text_encoder_inputs = T.tensor3('text_encoder_inputs')


    # We save the losses for further plotting
    hist_loss_gen = []
    hist_loss_disc = []
    hist_loss_enc = []



    # Create neural network model

    # Build the text encoder
    text_encoder = build_text_encoder(text_encoder_inputs);

    if(text_enc_save_model != ""):
        with np.load(text_enc_save_model) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(text_encoder, param_values)

    # Build the encoder (image)
    encoder = build_encoder(encoder_inputs)
    #encode_inputs = lasagne.layers.get_output(encoder)

    if(enc_save_model != ""):
        with np.load(enc_save_model) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(encoder, param_values)

    final_encoder = build_concat_encoder(text_encoder, encoder)
    final_encoder_output = lasagne.layers.get_output(final_encoder)
    # Initialize the generator network and functions
    generator = build_generator(final_encoder_output)

    # Load previous generator parameters if necessary
    if(gen_save_model != ""):
        with np.load(gen_save_model) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(generator, param_values)

    # Function to generate image from the generator
    generate_sample = lasagne.layers.get_output(generator, final_encoder_output)

    # Initialize the discriminator network and functions
    discriminator = build_discriminator(disc_inputs)

    # Load previous discriminator parameters if necessary
    if(disc_save_model != ""):
        with np.load(disc_save_model) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(discriminator, param_values)



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

    # Reconstruction loss
    encoder_loss = lasagne.objectives.squared_error(generate_sample, disc_inputs).mean()

    overall_loss = 0.001*gen_loss + 0.999*encoder_loss

    # Get the parameters and define the updates
    gen_params = lasagne.layers.get_all_params(generator, trainable = True)
    disc_params = lasagne.layers.get_all_params(discriminator, trainable=True)
    encoder_params = lasagne.layers.get_all_params(encoder, trainable=True)
    text_encoder_params = lasagne.layers.get_all_params(text_encoder, trainable=True)

    # We use adam optimizer with beta1=0.5 because the article (DCGAN) tell use that it works well
    gen_updates = lasagne.updates.adam(gen_loss, gen_params, learning_rate = learning_rate* 2, beta1=0.5)
    disc_updates = lasagne.updates.adam(disc_loss, disc_params, learning_rate = learning_rate, beta1=0.5)
    encoder_updates = lasagne.updates.adam(overall_loss,encoder_params, learning_rate = learning_rate*10, beta1=0.5)
    decoder_updates = lasagne.updates.adam(overall_loss,gen_params, learning_rate = learning_rate*10, beta1=0.5)
    text_encoder_updates = lasagne.updates.rmsprop(overall_loss, text_encoder_params, learning_rate=0.0007)
    # SGD steps


    # a discriminator step
    train_disc = theano.function([disc_inputs, encoder_inputs,text_encoder_inputs], disc_loss, updates=disc_updates)

    # an encoder step
    train_encoder = theano.function([encoder_inputs,text_encoder_inputs,disc_inputs], overall_loss, updates = encoder_updates)
    train_decoder = theano.function([disc_inputs, encoder_inputs,text_encoder_inputs], overall_loss, updates = decoder_updates)
    train_text_encoder = theano.function([encoder_inputs,text_encoder_inputs, disc_inputs], overall_loss, updates=text_encoder_updates)
    # get the losses
    #loss_disc_fn = theano.function([disc_inputs, encoder_inputs], disc_loss)
    #gen_disc_fn = theano.function([encoder_inputs], gen_loss)


    # Evaluate the generated samples and give the images ( for visualisation)
    fn_generate_sample = theano.function(
        inputs = [encoder_inputs,text_encoder_inputs],
        outputs =  generate_sample
    )

    print("Model created !")

    print("Training ...")
    for epoch in range(num_epochs):

        # In each epoch we do a full pass over the training data:
        train_err_gen = 0
        train_err_disc = 0
        train_err_encoder = 0
        train_batches = 0
        start_time = time.time()

        for i in range(nb_train_batch):


            # create batch
            #batch = get_train_batch(i, batch_size)

            [data_input, data_target,caption_array] = get_train_batch(i, batch_size)
            print(caption_array.shape)
            print("batch :",i)


            train_err_disc = train_disc(data_target, data_input, caption_array)
            train_err_encoder = train_encoder(data_input,caption_array, data_target)
            train_err_text_encoder = train_text_encoder(data_input, caption_array, data_target)
            train_err_decoder = train_decoder(data_target,data_input,caption_array)
            train_batches += 1

            hist_loss_gen.append(train_err_gen)
            hist_loss_disc.append(train_err_disc)
            hist_loss_enc.append(train_err_text_encoder)


            # Some output after 10 iterations
            if train_batches % 100 == 0:
                # sample some generations
                rand_val_index = np.random.randint(nb_val_batch-10)
                print(rand_val_index)
                [data_input_example, data_target_example,caption_array_example] = get_train_batch(i, 100,active_shift=False, active_rotation=False)
                # generation from validation set
                [data_input_example_val, data_target_example_val,caption_array_val] = get_train_batch(rand_val_index, 100,data_path="../../data/inpainting/val2014/", active_shift=False,
                                                                            active_rotation=False)

                samples = fn_generate_sample(data_input_example,caption_array_example)
                samples_val = fn_generate_sample(data_input_example_val,caption_array_val)
                print("Generator error",train_err_gen)
                print("Discriminator error", train_err_disc)
                print("Encoder error", train_err_encoder)

                # 10 x 10 img real
                # Training
                input_print = np.copy(data_input_example)
                input_print[:, :, 16:48, 16:48] = data_target_example
                img_100 =  input_print.reshape(10, 10, 3, image_size* 2, image_size* 2).transpose(0, 3, 1, 4, 2).reshape(10 * image_size* 2, 10 * image_size* 2, 3)
                rgbArray = np.zeros((10 * image_size* 2, 10 * image_size* 2, 3), 'uint8')
                rgbArray[..., 0] = ((img_100[:,:,0] + 1) / 2) * 256
                rgbArray[..., 1] = ((img_100[:,:,1] + 1) / 2) * 256
                rgbArray[..., 2] = ((img_100[:,:,2] + 1) / 2) * 256
                img = Image.fromarray(rgbArray)
                img.save('inpainting_output/real_samples'+str(epoch)+'batch_'+str(train_batches)+'.png')

                # validation
                input_print = np.copy(data_input_example_val)
                input_print[:, :, 16:48, 16:48] = data_target_example_val
                img_100 =  input_print.reshape(10, 10, 3, image_size* 2, image_size* 2).transpose(0, 3, 1, 4, 2).reshape(10 * image_size* 2, 10 * image_size* 2, 3)
                rgbArray = np.zeros((10 * image_size* 2, 10 * image_size* 2, 3), 'uint8')
                rgbArray[..., 0] = ((img_100[:,:,0] + 1) / 2) * 256
                rgbArray[..., 1] = ((img_100[:,:,1] + 1) / 2) * 256
                rgbArray[..., 2] = ((img_100[:,:,2] + 1) / 2) * 256
                img = Image.fromarray(rgbArray)
                img.save('inpainting_output/real_samples_val'+str(epoch)+'batch_'+str(train_batches)+'.png')

                # 10 x 10 img generated
                #train
                data_input_example[:,:,16:48, 16:48] = samples
                img_100 = data_input_example.reshape(10, 10, 3, image_size * 2, image_size * 2).transpose(0, 3, 1, 4,2).reshape(10 * image_size * 2, 10 * image_size * 2, 3)
                rgbArray = np.zeros((10 * image_size * 2, 10 * image_size * 2, 3), 'uint8')

                rgbArray[..., 0] = ((img_100[:, :, 0] + 1) / 2) * 256
                rgbArray[..., 1] = ((img_100[:, :, 1] + 1) / 2) * 256
                rgbArray[..., 2] = ((img_100[:, :, 2] + 1) / 2) * 256
                img = Image.fromarray(rgbArray)
                img.save('inpainting_output/generated_epoch_'+str(epoch)+'batch_'+str(train_batches)+'.png')

                #validation
                data_input_example_val[:, :, 16:48, 16:48] = samples_val
                img_100 = data_input_example_val.reshape(10, 10, 3, image_size * 2, image_size * 2).transpose(0, 3, 1, 4,
                                                                                                          2).reshape(
                    10 * image_size * 2, 10 * image_size * 2, 3)
                rgbArray = np.zeros((10 * image_size * 2, 10 * image_size * 2, 3), 'uint8')

                rgbArray[..., 0] = ((img_100[:, :, 0] + 1) / 2) * 256
                rgbArray[..., 1] = ((img_100[:, :, 1] + 1) / 2) * 256
                rgbArray[..., 2] = ((img_100[:, :, 2] + 1) / 2) * 256
                img = Image.fromarray(rgbArray)
                img.save('inpainting_output/generated_val_epoch_' + str(epoch) + 'batch_' + str(train_batches) + '.png')

                # save model
                np.savez('generator'+str(epoch)+ 'model.npz', *lasagne.layers.get_all_param_values(generator))
                np.savez('discriminator' + str(epoch) + 'model.npz',*lasagne.layers.get_all_param_values(discriminator))
                np.savez('encoder' + str(epoch) + 'model.npz', *lasagne.layers.get_all_param_values(encoder))
                np.savez('textencoder' + str(epoch) + 'model.npz', *lasagne.layers.get_all_param_values(text_encoder))
                # Plot the figures

                ## Compute the x and y coordinates for points on sine and cosine curves
                x = np.arange(len(hist_loss_gen))
                # Plot the points using matplotlib
                #plt.plot(x, hist_loss_gen)
                plt.plot(x, hist_loss_disc)
                plt.plot(x, hist_loss_enc)
                plt.xlabel('update')
                plt.ylabel('loss')
                plt.title('Erreur du générateur et du discriminant.')
                plt.legend(['Discriminant','Encoder'])
                plt.savefig("inpainting_output/figure_epoch"+str(epoch)+".png")
                plt.close()
        # Then we print the results for this epoch:
        #print("Epoch {} of {} took {:.3f}s".format(
        #    epoch + 1, num_epochs, time.time() - start_time))
        #print("  training loss generator:\t\t{:.6f}".format(train_err_gen ))
        #print("  training loss discriminator:\t\t{:.6f}".format(train_err_disc ))


def generate(model=""):
    if(model != ""):
        with np.load(model) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(generator, param_values)
main()

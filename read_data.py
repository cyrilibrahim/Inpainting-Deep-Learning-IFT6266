import os, sys
import glob
import six.moves.cPickle as pkl
import numpy as np
import theano
import PIL.Image as Image



image_size = 64



def get_nb_train(data_path="../../data/inpainting/train2014/"):
    imgs = glob.glob(data_path + "/*.jpg")
    return len(imgs)

def get_train_batch(batch_idx, batch_size,data_path="../../data/inpainting/train2014/"):
    imgs = glob.glob(data_path + "/*.jpg")
    batch_imgs = imgs[batch_idx * batch_size:(batch_idx + 1) * batch_size]

    input_batch = np.empty((0,3, image_size, image_size), dtype=theano.config.floatX)
    target_batch = np.empty((0, 3, image_size//2, image_size//2), dtype=theano.config.floatX)

    for i, img_path in enumerate(batch_imgs):

        img = Image.open(img_path)
        img_array = np.array(img)
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        if len(img_array.shape) == 3:
            input = np.copy(img_array)
            input[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :] = 0
            target = img_array[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16, :]
            # transform size to fit our neural network
            input = input.transpose(2, 0, 1)
            input = input.reshape(1, 3, image_size, image_size)
            target = target.transpose(2, 0, 1)
            target = target.reshape(1, 3, image_size//2, image_size//2)

            # append to the minibatch
            input_batch = np.append(input, input_batch, axis=0)
            target_batch = np.append(target, target_batch, axis=0)
        else:
            input = np.copy(img_array)
            input[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16] = 0
            target = img_array[center[0] - 16:center[0] + 16, center[1] - 16:center[1] + 16]

            input = input.reshape(1,1,image_size,image_size)
            input = np.repeat(input, 3, axis=1)
            target = target.reshape(1,1,image_size//2,image_size//2)
            target = np.repeat(target, 3, axis=1)

            input_batch = np.append(input, input_batch, axis=0)
            target_batch = np.append(target, target_batch, axis=0)

    # We want input in the interval [ - 1,  1 ]
    return [(input_batch / 256) * 2 - 1,(target_batch / 256) * 2 - 1]


if __name__ == '__main__':
    #resize_mscoco()
    #show_examples(5, 10)
    #get_nb_train()
    [data_input, data_target] = get_train_batch(1,10)
    print(data_input.shape)
    print(data_target.shape)

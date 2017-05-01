import os, sys
import glob
import six.moves.cPickle as pkl
import numpy as np
import theano
from PIL import Image, ImageChops, ImageOps
import nltk
from nltk.tokenize import RegexpTokenizer



image_size = 64
# Maximum number of captions to use
SEQ_LENGTH = 40

def get_dict_correspondance(worddict="../../data/inpainting/worddict.pkl"):
    with open(worddict,'rb') as fd:
        dictionary = pkl.load(fd)

    word_to_ix = {word: i for i, word in enumerate(dictionary)}
    ix_to_word = {i: word for i, word in enumerate(dictionary)}

    #print(ix_to_word[200]) = plane
    #print(word_to_ix["plane"]) = 200

    return [word_to_ix, ix_to_word]

def get_nb_train(data_path="../../data/inpainting/train2014/"):
    imgs = glob.glob(data_path + "/*.jpg")
    return len(imgs)

def get_train_batch(batch_idx, batch_size,data_path="../../data/inpainting/train2014/",
                    caption_path="../../data/inpainting/dict_key_imgID_value_caps_train_and_valid.pkl",
                    active_shift=True,  active_rotation =True):
    imgs = glob.glob(data_path + "/*.jpg")
    batch_imgs = imgs[batch_idx * batch_size:(batch_idx + 1) * batch_size]

    input_batch = np.empty((0,3, image_size, image_size), dtype=theano.config.floatX)
    target_batch = np.empty((0, 3, image_size//2, image_size//2), dtype=theano.config.floatX)

    # Read the caption dictionnary (train + valid)
    with open(caption_path,'rb') as fd:
        caption_dict = pkl.load(fd)

    # Get the correspondance to create the captions array
    [word_to_index, index_to_word] = get_dict_correspondance()

    vocab_size = len(word_to_index);
    # Shape for a 1D-CNN (batch_size, nb_channel = vocab_size, height = SEQ_LENGTH)
    captions_array = np.zeros((batch_size,vocab_size, SEQ_LENGTH),dtype=theano.config.floatX)

    # Liste des mots disponibles
    #for x in dictionary:
    #    print(x)

    # Tokenizer wich remove punctuation
    tokenizer = RegexpTokenizer(r'\w+')

    for i, img_path in enumerate(batch_imgs):


        # treat the caption
        cap_id = os.path.basename(img_path)[:-4]

        caption = caption_dict[cap_id]
        tokenize_caption = []

        for j in range(len(caption)):
            tokenize_caption = tokenize_caption + tokenizer.tokenize(caption[j])

        len_caption = len(tokenize_caption)

        # Create the one hot vector for the current sentence
        for j in range(SEQ_LENGTH):
            # If the sentence is smaller than the sentence size we keep 0
            if j < len_caption:
                word = tokenize_caption[j]
                captions_array[i,word_to_index[word],j] = 1.

        #print(np.sum(captions_array[i])) # Give SEQ_LENGHT most of the time the processing seems correct
        img = Image.open(img_path)

        # Dynamic data augmentation

        # rotation aleatoire (dans un angle de 50 deg)
        if active_rotation:
            random_angle = np.random.uniform(-25, 25)
            img = img.rotate(random_angle)


        # shift aleatoire (de 20% de la taille de l'image maximum)
        if active_shift:
            random_y_shift = np.random.randint(-(image_size // 20),image_size // 20)
            random_x_shift = np.random.randint(-(image_size // 20),image_size // 20)
            img = ImageChops.offset(img, random_x_shift, random_y_shift)

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
    return [(input_batch / 256) * 2 - 1,(target_batch / 256) * 2 - 1, captions_array]


if __name__ == '__main__':
    #resize_mscoco()
    #show_examples(5, 10)
    #get_nb_train()
    [data_input, data_target, captions_array] = get_train_batch(1,10)

    #get_dict_correspondance()

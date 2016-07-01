import cv2
import numpy as np
import os
import keras
from keras.layers import Dense, Activation, Input, merge, Convolution2D, LeakyReLU, \
                         MaxPooling2D, BatchNormalization, Flatten, UpSampling2D, AveragePooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from test_making_blurred_images import show_img, make_and_save_images
from data_feeder import DataFeeder
from utils import normalize_pred_img_array


def make_models(input_shape):
    '''
    Creates 3 models. A disciminator, a generator, and a model that stacks these two.
    Some layers are shared in multiple models

    inputs:
        input_shape: single shape describing the blurry and clear image. In th dim_order
    '''


    n_filter = 16

    blurry_img = Input(shape=input_shape, name='blurry_img')

    # Generative model here
    gen_output = Convolution2D(3, 3, 3, border_mode='same', init='he_normal', activation='relu')(blurry_img)
    gen_model = Model(input=blurry_img, output=gen_output)

    # Discrim model here
    clear_img = Input(shape=input_shape, name='clear_img')
    discrim_layer = Convolution2D(n_filter, 4, 4, subsample=(2, 2), border_mode='same',
                                  init='glorot_uniform', activation='relu')
    layer_on_blurry = discrim_layer(blurry_img)
    layer_on_clear = discrim_layer(clear_img)
    disc_blurry_flattened = Flatten()(layer_on_blurry)
    disc_clear_flattened = Flatten()(layer_on_clear)
    disc_merged = merge([disc_blurry_flattened, disc_clear_flattened],
                   mode='concat',
                   concat_axis=1)

    disc_output_maker = Dense(1, activation='sigmoid', init='he_normal', name='y')
    disc_output = disc_output_maker(disc_merged)
    disc_model = Model(input=[blurry_img, clear_img],
                       output=disc_output)

    # Gen-disc stack here
    gen_disc_first_layer = discrim_layer(gen_output)
    gen_disc_clear_flattened = Flatten()(gen_disc_first_layer)
    gen_disc_merged = merge([disc_blurry_flattened, gen_disc_clear_flattened],
                            mode='concat',
                            concat_axis=1)
    gen_disc_output = disc_output_maker(gen_disc_merged)
    gen_disc_model = Model(input=[blurry_img], output=gen_disc_output)

    slower_adam = Adam(lr=0.003, beta_1=0.5)
    disc_model.compile(loss='binary_crossentropy', optimizer=slower_adam)
    gen_disc_model.compile(loss='binary_crossentropy', optimizer=slower_adam)
    return gen_model, disc_model, gen_disc_model



def train_disc(gen_model, disc_model, data_feeder, batch_size=4, nb_epochs=10, batches_per_epoch=4):
    '''train the discriminator model. data_feeder is a DataFeeder object that creates the batches'''
    for i in range(nb_epochs):
        for j in range(batches_per_epoch):
            loss = []
            blur, clear, is_real = data_feeder.get_batch(gen_model)
            X = [blur, clear]
            loss.append(disc_model.train_on_batch(X, is_real))
        print('Mean loss this epoch: ', np.mean(loss))
    return

def train_gen(gen_model, disc_model, data_feeder, batch_size=4, nb_epochs=1, batches_per_epoch=5):
    '''train the generator model. The loss function is -1 * discriminators loss'''
    pass

def save_predicted_images(gen_model, blurred_images):
    '''
    Creates predicted versions of clear image from blurred images.  Saves them
    in tmp/predicted.  Blurred images should be a numpy array of out-of-sample images
    '''
    pred_imgs = my_model.predict(blurred_val_images)
    n_pred_imgs = pred_imgs.shape[0]
    for i in range(n_pred_imgs):
        img = pred_imgs[i]
        out_fname = './tmp/predicted/' +  str(i) + '.jpg'
        img = normalize_pred_img_array(img)
        cv2.imwrite(out_fname, img)

if __name__ == "__main__":
    total_images = 120
    num_training_images = 100
    img_height = 112
    img_width = 64

    input_shape = (3, img_height, img_width)
    remake_images = True

    if remake_images:
        make_and_save_images(images_to_make=total_images,
                             frames_to_mix = 3,
                             img_height=img_height,
                             img_width=img_width)

    gen_model, disc_model, gen_disc_model = make_models(input_shape)
    data_feeder = DataFeeder()
    train_disc(gen_model, disc_model, data_feeder)

    #save_predicted_images(gen_model, blurred_val_images)

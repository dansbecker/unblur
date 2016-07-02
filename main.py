import cv2
import numpy as np
import os
from blurred_image_maker import show_img, make_and_save_images
from data_feeder import DataFeeder
from model import make_models
from utils import normalize_pred_img_array, get_blurred_img_array


def set_disc_trainability(gen_disc_model, trainable_mode):
    '''
    Sets trainable property of layers with names including 'disc' to value of trainable_mode
    Trainable_mode is either True or False
    '''
    relevant_layers = [layer for layer in gen_disc_model.layers if 'disc' in layer.name]
    for layer in relevant_layers:
        layer.trainable = trainable_mode
    return


def train_gen(gen_disc_model, data_feeder, batch_size=8, nb_epochs=1, batches_per_epoch=2):
    '''train the generator model. The loss function is -1 * discriminators loss'''
    set_disc_trainability(gen_disc_model, False)
    for i in range(nb_epochs):
        for j in range(batches_per_epoch):
            loss = []
            blur, is_real = data_feeder.get_gen_only_batch()
            loss.append(gen_disc_model.train_on_batch(blur, is_real))
    set_disc_trainability(gen_disc_model, True)
    return np.mean(loss)

def train_disc(gen_model, disc_model, data_feeder, batch_size=8, nb_epochs=1, batches_per_epoch=1):
    '''train the discriminator model. data_feeder is a DataFeeder object that creates the batches'''
    for i in range(nb_epochs):
        for j in range(batches_per_epoch):
            loss = []
            blur, clear, is_real = data_feeder.get_batch(gen_model)
            X = [blur, clear]
            loss.append(disc_model.train_on_batch(X, is_real))
    return np.mean(loss)

def save_predicted_images(gen_model, blurred_images):
    '''
    Creates predicted versions of clear image from blurred images.  Saves them
    in tmp/predicted.  Blurred images should be a numpy array of out-of-sample images
    '''
    pred_imgs = gen_model.predict(blurred_val_images)
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
    remake_images = False

    if remake_images:
        make_and_save_images(images_to_make=total_images,
                             frames_to_mix = 3,
                             img_height=img_height,
                             img_width=img_width)

    gen_model, disc_model, gen_disc_model = make_models(input_shape)
    data_feeder = DataFeeder()

    d_loss = train_disc(gen_model, disc_model, data_feeder)
    g_loss = train_gen(gen_disc_model, data_feeder)

    for i in range(5000):
        if abs(g_loss) < 0.5 :  # let generator catch up
            g_loss = train_gen(gen_disc_model, data_feeder)
        elif (abs(g_loss) > 0.9) and (d_loss > 1): # let discriminator catch up
            d_loss = train_disc(gen_model, disc_model, data_feeder)
        else:  # run both discrim and gen
            d_loss = train_disc(gen_model, disc_model, data_feeder)
            g_loss = train_gen(gen_disc_model, data_feeder)
        if i % 20 == 0:
            print('Num iterations: ', i)
            print('Generator fools discrim with probability: ', -1 * g_loss)
            print('Mean discriminator loss this epoch: ', d_loss)

    blurred_val_images = get_blurred_img_array(num_training_images, total_images)
    save_predicted_images(gen_model, blurred_val_images)

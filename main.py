import cv2
import numpy as np
import os
from blurred_image_maker import show_img, make_and_save_images
from data_feeder import DataFeeder
from model import make_models
from trainer import Trainer
from utils import normalize_pred_img_array, get_blurred_img_array



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
    total_images = 250
    num_training_images = 200
    img_height = 144
    img_width = 80

    input_shape = (3, img_height, img_width)
    remake_images = False

    if remake_images:
        make_and_save_images(images_to_make=total_images,
                             frames_to_mix = 3,
                             img_height=img_height,
                             img_width=img_width)

    gen_model, disc_model, gen_disc_model = make_models(input_shape)
    data_feeder = DataFeeder()
    trainer = Trainer(gen_model, disc_model, gen_disc_model, data_feeder, report_freq=20)
    trainer.train(n_steps=1000)
    gen_model, disc_model, gen_disc_model = trainer.get_models()
    blurred_val_images = get_blurred_img_array(num_training_images, total_images)
    save_predicted_images(gen_model, blurred_val_images)

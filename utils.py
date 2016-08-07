import cv2
import numpy as np
import os

pixel_norm_factor = 255

def read_and_transform_img(fpath):
    '''
    Reads image and transforms it for modeling. Inverse of normalize_pred_img_array fn.
    '''
    return tf_to_th_dim_order(cv2.imread(fpath)).astype(float) / pixel_norm_factor

def normalize_pred_img_array(img):
    '''
    Convert predicted values from generated model to properly formatted image
    Largely the inverse function of read_and_transform_img
    '''
    img_with_index = img[0,:,:,:]
    out = th_to_tf_dim_order(img_with_index) * pixel_norm_factor
    out = out.clip(0,255).astype('uint8')
    return out

def tf_to_th_dim_order(img):
    '''Moves channel from index position 2 to index position 0'''
    return img.swapaxes(1, 2).swapaxes(0, 1)


def th_to_tf_dim_order(img):
    '''Moves channel from index position 0 to index position 2'''
    return img.swapaxes(0, 1).swapaxes(1, 2)


def train_val_split(raw_vids_dir, num_training_images, images_per_video):
    vid_fnames = os.listdir(raw_vids_dir)
    train_fnames = []
    val_fnames = []
    for vid_fname_with_extension in vid_fnames:
        vid_name = vid_fname_with_extension.split('.')[0]
        train_fnames += [vid_name + '_' + str(i) + '.jpg' for i in range(num_training_images)]
        val_fnames += [vid_name + '_' + str(i) + '.jpg' for i in range(num_training_images, images_per_video)]
    return train_fnames, val_fnames


def save_predicted_images(gen_model, val_fnames):
    '''
    Creates predicted versions of clear image and saves it in tmp/predicted.
    '''
    for fname in val_fnames:
        blurred_img = read_and_transform_img('./tmp/blur/' + fname)
        blurred_img = blurred_img[np.newaxis, :, :, :]  # Image number is new index 0
        img = gen_model.predict(blurred_img)
        img = normalize_pred_img_array(img)
        out_fname = './tmp/predicted/' + fname
        cv2.imwrite(out_fname, img)

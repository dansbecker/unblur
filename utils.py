import cv2
import numpy as np

pixel_norm_factor = 255
def read_and_transform_image(fpath):
        return tf_to_th_dim_order(cv2.imread(fpath)).astype(float) / pixel_norm_factor

def normalize_pred_img_array(img):
    out = th_to_tf_dim_order(img) * pixel_norm_factor
    out = out.clip(0,255).astype('uint8')
    return out

def tf_to_th_dim_order(img):
    '''Moves channel from index position 2 to index position 0'''
    return img.swapaxes(1, 2).swapaxes(0, 1)


def th_to_tf_dim_order(img):
    '''Moves channel from index position 0 to index position 2'''
    return img.swapaxes(0, 1).swapaxes(1, 2)

def get_blurred_img_array(start_num, end_num):
    '''Creates training data. Filenaming convention uses integers. Args indicate which files to pull'''
    blurred_images = []
    for i in range(start_num, end_num):
        fpath = './tmp/blur/' + str(i) + '.jpg'
        blurred_images.append(read_and_transform_image(fpath))
    blurred_images = np.array(blurred_images)
    return blurred_images

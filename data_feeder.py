import numpy as np
from random import shuffle, choice
import os
from keras.utils.np_utils import to_categorical
from utils import read_and_transform_image

class DataFeeder(object):
    '''
    The DataFeeder object provides the get_batch and get_gen_only_batch methods
    The get_batch method is appropriate for training the discriminator.
    The get_gen_only_batch method is appropriate for training the generator model

    batch_size should be even, since disc data comes in pairs of real/generated obs.
    '''
    def __init__(self, train_data_dir='./tmp/', batch_size=8, gen_only_batch_size=8, fnames=[]):
        self.clean_dir = train_data_dir + 'clean/'
        self.blur_dir = train_data_dir + 'blur/'
        self.fnames = fnames

        shuffle(self.fnames)
        self.batch_size = batch_size
        self.gen_only_batch_size = gen_only_batch_size
        self.blur_images = []
        self.clear_images = []
        self.is_real = []

    def _add_true_image_pair_to_batch(self, fname):
        '''
        Adds generated image to accumulator, with fname being input to generator
        Also adds the input and is_real values to appropriate accumulators to
        keep the 3 accumulators in sync
        '''
        self.blur_images.append(read_and_transform_image(self.blur_dir + fname))
        self.clear_images.append(read_and_transform_image(self.clean_dir + fname))
        self.is_real.append(1)

    def _add_gen_image_pair_to_batch(self, fname, gen_model):
        '''
        Adds generated image to accumulator, with fname being input to generator
        Also adds the input and is_real values to appropriate accumulators to
        keep the 3 accumulators in sync
        '''

        blur_item = read_and_transform_image(self.blur_dir + fname)
        array_to_predict_on = np.array([blur_item])
        generated_image = gen_model.predict(array_to_predict_on)[0,:] #dim 0 is img index
        self.blur_images.append(blur_item)
        self.clear_images.append(generated_image)
        self.is_real.append(0)

    def _reset_accumulators(self):
        '''Clear accumulators to start a new batch'''
        self.blur_images = []
        self.clear_images = []
        self.is_real = []

    def get_batch(self, gen_model):
        '''
        Returns tuple of blurred images, clear images and indication of which "real."
        Blurred_images are from raw/read data.  Clear images can be from original data, or generated from
        gen_model.  is_real is 1 if the clear_image is real, 0 otherwise.
        '''
        self._reset_accumulators()
        while len(self.is_real) < self.batch_size:
            fname = choice(self.fnames)
            self._add_true_image_pair_to_batch(fname)
            self._add_gen_image_pair_to_batch(fname, gen_model)
        blur_out, clear_out, is_real = (np.array(x) for x in (self.blur_images, self.clear_images, self.is_real))
        return (blur_out, clear_out, is_real)

    def get_gen_only_batch(self):
        '''
        Returns a batch of (generated_image, is_real) pairs. is_real equals 0
        since the generated images aren't real.
        Pulls blurry images to generate from out of self.blur_dir
        '''

        self._reset_accumulators()
        while len(self.is_real) < self.gen_only_batch_size:
            fname = choice(self.fnames)
            self.blur_images.append(read_and_transform_image(self.blur_dir + fname))
            self.is_real.append(0)
        blur_out, is_real = (np.array(x) for x in (self.blur_images, self.is_real))
        return (blur_out, is_real)

import keras
from keras.layers import Dense, Activation, Input, merge, Convolution2D, LeakyReLU, \
                         MaxPooling2D, BatchNormalization, Flatten, UpSampling2D, AveragePooling2D
from keras.models import Model, Sequential
from keras.optimizers import SGD
import theano.tensor as T


def gen_disc_objective(y_true, y_pred):
    '''
    Objective function for optimizing the stacked generator-discriminator.
    We only train the generator layers, and we want the outcome probability reported
    by the generator to be high.

    Conventionally would use -1 * np.log(y_pred). Though the simpler -1 * y_pred
    may reduce risk of vanishing or exploding gradient.'''
    #epsilon = 1.0e-9    # for use with logged probability
    #y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    #
    return -1 * y_pred

def make_models(input_shape):
    '''
    Creates 3 models. A disciminator, a generator, and a model that stacks these two.
    Some layers are shared in multiple models

    inputs:
        input_shape: single shape describing the blurry and clear image. In th dim_order
    '''


    n_disc_filter = 16

    blurry_img = Input(shape=input_shape, name='blurry_img')

    # Generative model here
    gen_l1 = Convolution2D(32, 4, 4, border_mode='same', activation='relu')(blurry_img)
    gen_l1_squeeze = Convolution2D(16, 1, 1, border_mode='same', activation='relu')(gen_l1)
    gen_l2 = Convolution2D(64, 4, 4, border_mode='same', activation='relu')(gen_l1_squeeze)
    gen_l2_squeeze = Convolution2D(32, 1, 1, border_mode='same', activation='relu')(gen_l2)
    gen_output = Convolution2D(3, 4, 4, border_mode='same', activation='relu')(gen_l2_squeeze)
    gen_model = Model(input=blurry_img, output=gen_output)

    # Discrim model here
    clear_img = Input(shape=input_shape, name='clear_img')
    disc_layer = Convolution2D(n_disc_filter, 4, 4, subsample=(2, 2), border_mode='same',
                                  init='glorot_uniform', activation='relu', name='disc_layer_1')
    layer_on_blurry = disc_layer(blurry_img)
    layer_on_clear = disc_layer(clear_img)
    disc_blurry_flattened = Flatten()(layer_on_blurry)
    disc_clear_flattened = Flatten()(layer_on_clear)
    disc_merged = merge([disc_blurry_flattened, disc_clear_flattened],
                   mode='concat',
                   concat_axis=1)

    disc_output_maker = Dense(1, activation='sigmoid', init='he_normal', name='disc_output_maker')
    disc_output = disc_output_maker(disc_merged)
    disc_model = Model(input=[blurry_img, clear_img],
                       output=disc_output)

    # Gen-disc stack here
    gen_disc_first_layer = disc_layer(gen_output)
    gen_disc_clear_flattened = Flatten()(gen_disc_first_layer)
    gen_disc_merged = merge([disc_blurry_flattened, gen_disc_clear_flattened],
                            mode='concat',
                            concat_axis=1)
    gen_disc_output = disc_output_maker(gen_disc_merged)
    gen_disc_model = Model(input=[blurry_img], output=gen_disc_output)

    disc_optimizer = SGD(lr=0.001)
    disc_model.compile(loss='binary_crossentropy', optimizer=disc_optimizer)
    gen_disc_model.compile(loss=gen_disc_objective, optimizer='rmsprop')
    return gen_model, disc_model, gen_disc_model

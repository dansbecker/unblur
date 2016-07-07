import keras
from keras.layers import Dense, Activation, Input, merge, Convolution2D, LeakyReLU, \
                         MaxPooling2D, BatchNormalization, Flatten, UpSampling2D, AveragePooling2D, \
                         ELU

from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.backend import log, clip


def apply_res_block(block_input,
                    n_filters,
                    filter_size=3,
                    layers_in_res_blocks=3):
    '''
    inputs:
        block_input: input tensor to apply the resnet block to
        n_filters: number of convolutional filters in each layer
        filter_size: both the height and width of convolutional filters
        layers_in_res_blocks: number of conv layers on the convolutional path


    outputs:
        output: the tensor after applying the resnet to the input

    NOTE: Current implementation assumes dim_order is 'th'.  (channel, row, col)
    '''

    # convolution path
    conv_y = block_input

    for i in range(layers_in_res_blocks):
        conv_y = Convolution2D(n_filters, filter_size, filter_size, border_mode='same')(conv_y)
        conv_y = LeakyReLU(0.2)(conv_y)

    output = merge([block_input, conv_y], mode='sum')
    return output

def gen_disc_objective(y_true, y_pred):
    '''
    Objective function for optimizing the stacked generator-discriminator.
    We only train the generator layers, and we want the outcome probability reported
    by the generator to be high.

    Conventionally would use -1 * np.log(y_pred). Though the simpler -1 * y_pred
    may reduce risk of vanishing or exploding gradient.'''
    epsilon = 1.0e-9
    y_pred = clip(y_pred, epsilon, 1.0 - epsilon)

    return -1 * log(y_pred)

def make_disc_layers(input_shape, n_disc_filter):
    output = Sequential(name='disc_layer_stack')
    output.add(Convolution2D(n_disc_filter, 3, 3, border_mode='same',
                             name='disc_layer_1', input_shape=input_shape))
    output.add(LeakyReLU(0.2, name='disc_activation_1'))


    output.add(Convolution2D(n_disc_filter, 3, 3, subsample=(2, 2), border_mode='same',
                             name='disc_layer_2', input_shape=input_shape))
    output.add(LeakyReLU(0.2, name='disc_activation_2'))
    output.add(Convolution2D(int(n_disc_filter/2), 1, 1, name='disc_shrink_2'))
    output.add(LeakyReLU(0.2, name='disc_shrink_activation_2'))

    output.add(Convolution2D(n_disc_filter, 3, 3, subsample=(2,2), border_mode='same',
                             name='disc_layer_3'))
    output.add(LeakyReLU(0.2, name='disc_activation_3'))
    output.add(Convolution2D(int(n_disc_filter/2), 1, 1, name='disc_shrink_3'))
    output.add(LeakyReLU(0.2, name='disc_shrink_activation_3'))

    output.add(Convolution2D(n_disc_filter, 3, 3, subsample=(2,2), border_mode='same',
                             name='disc_layer_4'))
    output.add(LeakyReLU(0.2, name='disc_activation_4'))
    output.add(Convolution2D(int(n_disc_filter/2), 1, 1, name='disc_shrink_4'))
    output.add(LeakyReLU(0.2, name='disc_shrink_activation_4'))

    output.add(Convolution2D(n_disc_filter, 3, 3, subsample=(2,2), border_mode='same',
                             name='disc_layer_5'))
    output.add(LeakyReLU(0.2, name='disc_activation_5'))
    output.add(Convolution2D(int(n_disc_filter/2), 1, 1, name='disc_shrink_5'))
    output.add(LeakyReLU(0.2, name='disc_shrink_activation_5'))
    return output

def make_models(input_shape,
                n_res_blocks_in_gen=3,
                n_filters_in_res_blocks=64,
                gen_filter_size=3,
                layers_in_res_blocks=3,
                res_block_subsample=(2, 2),
                filters_in_deconv=32,
                deconv_filter_size=3,
                n_disc_filter=64):
    '''
    Creates 3 models. A disciminator, a generator, and a model that stacks these two.
    Some layers are shared in multiple models

    inputs:
        input_shape: single shape describing the blurry and clear image. In th dim_order
    '''


    blurry_img = Input(shape=input_shape, name='blurry_img')

    # GENERATIVE MODEL
    gen_x = Convolution2D(64, gen_filter_size, gen_filter_size, border_mode='same')(blurry_img)
    gen_x = LeakyReLU(0.2)(gen_x)

    for i in range(n_res_blocks_in_gen):

        gen_x = apply_res_block(gen_x,
                                n_filters=n_filters_in_res_blocks,
                                filter_size=gen_filter_size,
                                layers_in_res_blocks=layers_in_res_blocks)
        if res_block_subsample != (1,1):
            gen_x = AveragePooling2D(pool_size=res_block_subsample, border_mode='same')(gen_x)

    if res_block_subsample != (1,1):
        for deconv_layer in range(n_res_blocks_in_gen):
            gen_x = Convolution2D(int(filters_in_deconv/2), 1, 1, border_mode='same', activation='relu')(gen_x)
            gen_x = LeakyReLU(0.2)(gen_x)
            gen_x = Convolution2D(filters_in_deconv, deconv_filter_size, deconv_filter_size, border_mode='same')(gen_x)
            gen_x = LeakyReLU(0.2)(gen_x)
            gen_x = UpSampling2D(size=res_block_subsample)(gen_x)
            gen_x = Convolution2D(filters_in_deconv, deconv_filter_size, deconv_filter_size, border_mode='same', activation='relu')(gen_x)
            gen_x = LeakyReLU(0.2)(gen_x)

    # fix channels back to 3
    gen_output = Convolution2D(3, 1, 1, border_mode='same', activation='relu')(gen_x)
    gen_model = Model(input=blurry_img, output=gen_output)


    # Discrim model here
    clear_img = Input(shape=input_shape, name='clear_img')
    disc_layer = make_disc_layers(input_shape, n_disc_filter)
    layer_on_blurry = disc_layer(blurry_img)
    layer_on_clear = disc_layer(clear_img)
    disc_blurry_flattened = Flatten()(layer_on_blurry)
    disc_clear_flattened = Flatten()(layer_on_clear)
    # SHould probably concat on axis 0 and convolve the images
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

    disc_optimizer = Adam(0.0001, beta_1=0.5)
    gen_optimizer  = Adam(0.0001, beta_1=0.5)
    disc_model.compile(loss='binary_crossentropy', optimizer=disc_optimizer)
    gen_disc_model.compile(loss=gen_disc_objective, optimizer=gen_optimizer)
    for model in gen_model, disc_model, gen_disc_model:
        print('------------------------')
        print(model.summary())

    return gen_model, disc_model, gen_disc_model

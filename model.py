import keras
from keras.layers import Dense, Activation, Input, merge, Convolution2D, LeakyReLU, \
                         MaxPooling2D, BatchNormalization, Flatten, UpSampling2D, AveragePooling2D, \
                         ELU, Lambda, Reshape, Convolution3D, MaxPooling3D

from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.backend import log, clip


def res_block_2D(block_input,
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
    '''
    epsilon = 1.0e-9
    y_pred = clip(y_pred, epsilon, 1.0 - epsilon)

    return -1 * log(y_pred)

def make_disc_model_body(input_shape, n_disc_filters):

    def make_disc_encoder(input_shape, n_disc_filters):
        output = Sequential(name='disc_layer_stack')
        output.add(Convolution2D(n_disc_filters[0], 1, 1, border_mode='same', input_shape=input_shape))
        output.add(LeakyReLU(0.2))
        for n_filters_in_block in n_disc_filters:
            output.add(Convolution2D(n_filters_in_block, 3, 3, subsample=(2, 2), border_mode='same'))
            output.add(LeakyReLU(0.2))
            output.add(Convolution2D(n_filters_in_block, 1, 1))
            output.add(LeakyReLU(0.2))

        # Add an extra dimension on front of this output, to allow convolving images together
        curr_output_shape = output.output_shape
        new_shape = (1, curr_output_shape[1], curr_output_shape[2], curr_output_shape[3])
        output.add(Reshape(new_shape))
        return output

    disc_encoder = make_disc_encoder(input_shape, n_disc_filters)

    blurry_img = Input(shape=input_shape)
    blurry_branch = disc_encoder(blurry_img)
    clear_img = Input(shape=input_shape)
    clear_branch = disc_encoder(clear_img)

    # SHould probably concat on axis 0 and convolve the images
    disc_merged = merge([blurry_branch, clear_branch],
                        mode='concat',
                        concat_axis=1)

    for i in range(2):
        encoded_pair = Convolution3D(32, 1, 1, 1, border_mode='same')(disc_merged)
        encoded_pair = LeakyReLU(0.2)(encoded_pair)
        encoded_pair = Convolution3D(32, 1, 3, 3, border_mode='same')(encoded_pair)
        encoded_pair = LeakyReLU(0.2)(encoded_pair)
        encoded_pair = MaxPooling3D(pool_size=(1,2,2))(encoded_pair)

    encoded_pair = Flatten()(encoded_pair)
    disc_output = Dense(50, name='disc_output')(encoded_pair)
    disc_output = LeakyReLU(0.2)(disc_output)
    disc_output = Dense(1, activation='sigmoid', name='disc_output')(encoded_pair)
    disc_model_body = Model(input=[blurry_img, clear_img], output=disc_output, name='disc_model_body')
    return disc_model_body


def make_models(input_shape, n_filters_in_res_blocks, gen_filter_size,
                layers_in_res_blocks, res_block_subsample, filters_in_deconv,
                deconv_filter_size, n_disc_filters):
    '''
    Creates 3 models. A disciminator, a generator, and a model that stacks these two.
    Some layers are shared in multiple models

    inputs:
        input_shape: single shape describing the blurry and clear image. In th dim_order
    '''


    blurry_img = Input(shape=input_shape, name='blurry_img')

    # GENERATIVE MODEL
    gen_x = Convolution2D(n_filters_in_res_blocks[0], gen_filter_size, gen_filter_size, border_mode='same')(blurry_img)
    gen_x = LeakyReLU(0.2)(gen_x)

    for n_filters in n_filters_in_res_blocks:
        gen_x = res_block_2D(gen_x,
                                n_filters=n_filters,
                                filter_size=gen_filter_size,
                                layers_in_res_blocks=layers_in_res_blocks)
        if res_block_subsample != (1,1):
            gen_x = AveragePooling2D(pool_size=res_block_subsample, border_mode='same')(gen_x)

    if res_block_subsample != (1,1):
        for _ in n_filters_in_res_blocks:
            gen_x = Convolution2D(int(filters_in_deconv/2), 1, 1, border_mode='same')(gen_x)
            gen_x = LeakyReLU(0.2)(gen_x)
            gen_x = Convolution2D(filters_in_deconv, deconv_filter_size, deconv_filter_size, border_mode='same')(gen_x)
            gen_x = LeakyReLU(0.2)(gen_x)
            gen_x = UpSampling2D(size=res_block_subsample)(gen_x)
            gen_x = Convolution2D(filters_in_deconv, deconv_filter_size, deconv_filter_size, border_mode='same')(gen_x)
            gen_x = LeakyReLU(0.2)(gen_x)

    # fix channels back to 3
    gen_x = Convolution2D(3, 1, 1, border_mode='same')(gen_x)
    gen_x = LeakyReLU(0.2)(gen_x)
    gen_output = merge([blurry_img, gen_x], mode='sum')
    gen_model = Model(input=blurry_img, output=gen_output)


    # Discrim model here
    clear_img = Input(shape=input_shape, name='clear_img')
    disc_model_body = make_disc_model_body(input_shape, n_disc_filters)

    disc_model_out = disc_model_body([blurry_img, clear_img])
    gen_disc_model_out = disc_model_body([blurry_img, gen_output])

    disc_model = Model(input=[blurry_img, clear_img], output=disc_model_out)
    gen_disc_model = Model(input=blurry_img, output=gen_disc_model_out)

    disc_optimizer = Adam(4e-4, beta_1=0.5, beta_2=0.99)
    gen_optimizer  = Adam(4e-4, beta_1=0.5, beta_2=0.99)
    disc_model.compile(loss='binary_crossentropy', optimizer=disc_optimizer)
    gen_disc_model.compile(loss=gen_disc_objective, optimizer=gen_optimizer)

    print('--------------------------------------------------')
    print('GENERATIVE MODEL')
    print(gen_model.summary())
    print('--------------------------------------------------')
    print('DISCRIMINATIVE MODEL')
    print(disc_model_body.summary())



    return gen_model, disc_model, gen_disc_model

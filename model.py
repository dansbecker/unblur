import keras
from keras.layers import Dense, Activation, Input, merge, Convolution2D, LeakyReLU, \
                         MaxPooling2D, BatchNormalization, Flatten, UpSampling2D, AveragePooling2D, \
                         ELU, LeakyReLU

from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam
from keras.backend import log, clip


def apply_res_block(block_input,
                    n_filters,
                    filter_size=3,
                    layers_in_res_blocks=3,
                    subsample=(1, 1)):
    '''
    inputs:
        block_input: input tensor to apply the resnet block to
        n_filters: number of convolutional filters in each layer
        filter_size: both the height and width of convolutional filters
        layers_in_res_blocks: number of conv layers on the convolutional path
        subsample: tuple with stride for convolutional path. APPLIED ONLY ONCE, not in each conv
    outputs:
        output: the tensor after applying the resnet to the input

    NOTE: Current implementation assumes dim_order is 'th'.  (channel, row, col)
    '''

    # convolution path
    conv_y = block_input
    for i in range(layers_in_res_blocks):
        if i == 0:
            this_layer_subsample = subsample
        else:
            this_layer_subsample = (1,1)
        conv_y = Convolution2D(n_filters, filter_size, filter_size,
                               subsample=this_layer_subsample,
                               activation='relu', border_mode='same')(conv_y)

    # Keras doesn't currently support 'same' mode for AveragePooling2D with theano
    # otherwise would do:
    # shortcut = AveragePooling2D(pool_size=subsample, border_mode='same')(block_input)
    # in the meantime, we use identity for shortcut, and require no subsampling

    assert subsample==(1,1)
    shortcut = block_input

    output = merge([shortcut, conv_y], mode='sum')
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
    output.add(Convolution2D(n_disc_filter, 4, 4, subsample=(2, 2), border_mode='same',
                             name='disc_layer_1', input_shape=input_shape))
    output.add(ELU(1))
    output.add(Convolution2D(n_disc_filter, 3, 3, subsample=(2,2), border_mode='same',
                             name='disc_layer_2'))
    output.add(ELU(1))
    output.add(Convolution2D(n_disc_filter, 2, 2, border_mode='same',
                             name='disc_layer_3'))
    output.add(ELU(1))
    return output

def make_models(input_shape,
                n_res_blocks_in_gen=2,
                n_filters_in_res_blocks=64,
                gen_filter_size=4,
                layers_in_res_blocks=3,
                res_block_subsample=(1,1),
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
    gen_x = Convolution2D(32, gen_filter_size, gen_filter_size, border_mode='same', activation='relu')(blurry_img)
    gen_x = Convolution2D(n_filters_in_res_blocks, 1, 1, border_mode='same', activation='relu')(gen_x)
    for i in range(n_res_blocks_in_gen):
        gen_x = apply_res_block(gen_x,
                                n_filters=n_filters_in_res_blocks,
                                filter_size=gen_filter_size,
                                layers_in_res_blocks=layers_in_res_blocks,
                                subsample=res_block_subsample)
    if res_block_subsample != (1,1):
        for deconv_layer in range(n_res_blocks_in_gen):
            gen_x = Convolution2D(filters_in_deconv, deconv_filter_size, deconv_filter_size, border_mode='same', activation='relu')(gen_x)
            gen_x = UpSampling2D(size=res_block_subsample)(gen_x)
            gen_x = Convolution2D(filters_in_deconv, deconv_filter_size, deconv_filter_size, border_mode='same', activation='relu')(gen_x)

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
    gen_optimizer = SGD(lr=0.0003)
    disc_model.compile(loss='binary_crossentropy', optimizer=disc_optimizer)
    gen_disc_model.compile(loss=gen_disc_objective, optimizer=gen_optimizer)
    for model in gen_model, disc_model, gen_disc_model:
        print('------------------------')
        print(model.summary())

    return gen_model, disc_model, gen_disc_model

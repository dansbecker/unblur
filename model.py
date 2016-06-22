import cv2
import numpy as np
import os
import keras
from keras.layers import Dense, Activation, Input, merge, Convolution2D, LeakyReLU, \
                         MaxPooling2D, BatchNormalization, Flatten, UpSampling2D, AveragePooling2D
from keras.models import Model, Sequential
from test_making_blurred_images import show_img, make_and_save_images

def apply_res_block(block_input,
                    n_filters=24,
                    filter_size=3,
                    layers_in_res_blocks=3,
                    subsample=(2,2)):
    '''
    Applies a residual block to block_input and returns result.

    Inputs:
    -------
        block_input: input tensor to apply the resnet block to
        n_filters: number of convolutional filters in each layer
        filter_size: both the height and width of convolutional filters
        layers_in_res_blocks: number of conv layers on the convolutional path
        subsample: tuple with stride for convolutional path. Applied only once, not in each conv

    Outputs:
    --------
        output: the tensor after applying the resnet to the input

    NOTE: Current implementation assumes dim_order is 'th'.  (channel, row, col)
          Current implementation likely not appropriate for deconvolution
    '''

    # convolution path
    shortcut = AveragePooling2D(pool_size=subsample, border_mode='same')(block_input)
    conv_y = block_input
    for i in range(layers_in_res_blocks):
        if i == 0:
            this_layer_subsample = subsample
        else:
            this_layer_subsample = (1,1)
        conv_y = Convolution2D(n_filters, filter_size, filter_size,
                               subsample=this_layer_subsample,
                               activation='relu', border_mode='same')(conv_y)

    output = merge([shortcut, conv_y], mode='sum')
    #block = Model(input=block_input, output=block_output)
    return output


def make_models(input_shape):
    blurry_img = Input(shape=input_shape, name='blurry_img')

    gen_output = Convolution2D(3, 3, 3, border_mode='same', activation='relu')(blurry_img))
    gen_model = Model(input=blurry_img, output=gen_output)

    clear_img = Input(shape=input_shape, name='clear_img')
    discrim_layer = Convolution2D(n_filt_base, 4, 4, subsample=(2, 2), border_mode='same', activation='relu')
    layer_on_blurry = discrim_layer(blurry_img)
    layer_on_clear = discrim_layer(clear_img)
    disc_blurry_flattened = Flatten()(layer_on_blurry)
    disc_clear_flattened = Flatten()(layer_on_clear)
    disc_merged = merge([disc_blurry_flattened, disc_clear_flattened],
                   mode='concat',
                   concat_axis=1)
    disc_output_maker = Dense(1, activation='sigmoid', name='y')
    disc_output = disc_output_maker(merged)
    disc_model = Model(input=[blurry_img, clear_img],
                       output=disc_output)

    gen_disc_first_layer = discrim_layer(gen_output)
    gen_disc_clear_flattened = Flatten()(gen_disc_first_layer)
    gen_disc_merged = merge([disc_blurry_flattened, gen_disc_clear_flattened])
    gen_disc_output = disc_output_maker(gen_disc_merged)
    gen_disc_model = Model(input=[blurry_img], output=gen_disc_output)

    return gen_model, disc_model, gen_disc_model




def make_gen_model(input_shape,
                   num_res_blocks = 3,
                   layers_in_res_blocks=2,
                   filters_in_res_blocks=20,
                   filter_size=3,
                   block_subsample=(1,1),
                   filters_in_deconv=32):
    '''
    Returns compiled generator model that translates blurred to unblurred images

    Inputs:
    -------
        input_shape: tuple of (n_channels, img_height, img_width)
        num_res_blocks: Number of resnet blocks to apply in convolution stage
        layers_in_res_blocks: Number of convolutional layers in each res net block
        filters_in_res_blocks: Number of convolutional filters in the resnet blocks
        filter_size: Size of the kernel/filters in the resnets
        block_subsample: Tuple of stride length for resnet blocks
        filters_in_deconv: Number of filters to use in deconvolution stage
            note that deconvolution only happens if we are doing sampling in
            conv stage

    Outputs:
    --------
        Flattened version of image pixel intensities. This format simplifies calculation
            of loss function
    '''

    blurry_img = Input(shape=input_shape, name='blurry_img')
    # Convert num channels to match upcoming resnets
    x = Convolution2D(filters_in_res_blocks, 3, 3, border_mode='same', activation='relu')(blurry_img)
    for i in range(num_res_blocks):
        x = apply_res_block(x,
                            n_filters=filters_in_res_blocks,
                            layers_in_res_blocks=layers_in_res_blocks,
                            filter_size=filter_size,
                            subsample=block_subsample)

    # deconvolution to fix shrinking of image in previous blocks
    if block_subsample != (1,1):
        for deconv_layer in range(num_res_blocks):
            x = Convolution2D(filters_in_deconv, 3, 3, border_mode='same', activation='relu')(x)
            x = UpSampling2D(size=block_subsample)(x)
            x = Convolution2D(filters_in_deconv, 3, 3, border_mode='same', activation='relu')(x)

    # fix channels back to 3
    # DCGAN paper suggests last layer of generator be tanh. Didn't work for me in early experiment
    clear_img = Convolution2D(3, 2, 2, border_mode='same', activation='relu', name='clear_img')(x)
    model = Model(input=blurry_img, output=[blurry_img, clear_img])
    return model

def make_discrim_model(input_shape, n_filt_base = 16):
    '''
    Returns a model that discriminates between actual and generated (clear) images

    Inputs:
    -------
        input_shape: 3-dimensional tuple indicating size of image model will operate on
        n_filt_base: number of filters in smallest/first layer of network

    Output
    ------
        Compiled disciminative model
    '''
    blurry_img = Input(shape=input_shape, name='blurry_img')
    clear_img = Input(shape=input_shape, name='clear_img')
    my_layer = Convolution2D(n_filt_base, 4, 4, subsample=(2, 2), border_mode='same', activation='relu')
    layer_on_blurry = my_layer(blurry_img)
    layer_on_clear = my_layer(clear_img)
    layer_on_blurry = Flatten()(layer_on_blurry)
    layer_on_clear = Flatten()(layer_on_clear)
    merged = merge([layer_on_blurry, layer_on_clear],
                   mode='concat',
                   concat_axis=1)
    y = Dense(1, activation='sigmoid', name='y')(merged)
    my_model = Model(input=[blurry_img, clear_img],
                    output=y)
    return my_model

def make_gen_on_discrim(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    print(model.summary())
    return model

def tf_to_th_dim_order(img):
    '''Moves channel from index position 2 to index position 0'''
    return img.swapaxes(1, 2).swapaxes(0, 1)


def th_to_tf_dim_order(img):
    '''Moves channel from index position 0 to index position 2'''
    return img.swapaxes(0, 1).swapaxes(1, 2)

def normalize_pred_img_array(img):
    out = th_to_tf_dim_order(img)
    out = out.clip(0,255).astype('uint8')
    return out

def get_blurred_img_array(start_num=0, end_num=10):
    '''Creates training data. Filenaming convention uses integers. Args indicate which files to pull'''
    blurred_images = []
    for i in range(start_num, end_num):
        blurred_img = cv2.imread('./tmp/' + str(i) + '_blur.jpg')
        blurred_images.append(tf_to_th_dim_order(blurred_img))
    blurred_images = np.array(blurred_images)
    return blurred_images

def get_clear_img_array(start_num=0, end_num=10):
    '''Creates target data. Filenaming convention uses integers. Args indicate which files to pull'''
    clear_images = []
    for i in range(start_num, end_num):
        clear_img = cv2.imread('./tmp/' + str(i) + '_single.jpg')
        # ravel the clear_img to simplify evaluating loss in model
        clear_images.append(tf_to_th_dim_order(clear_img).ravel())
    clear_images = np.array(clear_images)
    return clear_images

def get_batches(train_data_dir='./tmp/', batch_size=2):
    clean_dir = train_data_dir + 'clean/'
    blur_dir = train_data_dir + 'blur/'
    clean_dir_list = os.listdir(clean_dir)
    blur_dir_list = os.listdir(blur_dir)
    assert clean_dir_list == blur_dir_list

    def read_and_transform(fpath):
        return tf_to_th_dim_order(cv2.imread(fpath))

    blur_images = []
    clear_images = []
    for fname in clean_dir_list:
        clear_images.append(read_and_transform(clean_dir + fname))
        blur_images.append(read_and_transform(blur_dir + fname))
        if len(clear_images) == batch_size:
            y = [1 for _ in clear_images]
            yield [np.array(image_list) for image_list in (blur_images, clear_images, y)]
            blur_images = []
            clear_images = []
    # in case end of files doesn't correspond to end of a batch
    yield [np.array(image_list) for image_list in (blur_images, clear_images)]


def train(discrim_model, gen_model, train_data_dir, batch_size=2, nb_pseudo_epochs=100):
    for i in range(nb_pseudo_epochs):
        for blur_images, clear_images, y in get_batches(train_data_dir, batch_size):
            X = [blur_images, clear_images]
            ### MAKE GENERATED/PREDICTED IMAGES HERE
            gen_on_discrim = make_gen_on_discrim(gen_model, discrim_model)
            gen_on_discrim.compile(loss='binary_crossentropy', optimizer='adam')
            g_loss = gen_on_discrim.train_on_batch(X, y)
            print('Loss after gen_model training: ', g_loss)
            discriminator.trainable = True
            discriminator.compile(loss='binary_crossentropy', optimizer='adam')
            g_loss = gen_on_discrim.train_on_batch(X, y)
            print('Loss after discrim_model training: ', d_loss)
    return discriminator, generator, gen_on_discrim


def save_predicted_images(gen_model, blurred_images):
    pred_imgs = my_model.predict(blurred_val_images)
    n_pred_imgs = pred_imgs.shape[0]
    for i in range(n_pred_imgs):
        img = pred_imgs[i]
        out_fname = './tmp/predicted/' +  str(i) + '.jpg'
        img = normalize_pred_img_array(img)
        cv2.imwrite(out_fname, img)

if __name__ == "__main__":
    total_images = 40
    num_training_images = 30
    img_height = 176
    img_width = 96
    input_shape = (3, img_height, img_width)
    remake_images = False

    if remake_images:
        make_and_save_images(images_to_make=total_images,
                             frames_to_mix = 3,
                             img_height=img_height,
                             img_width=img_width)

    discrim_model = make_discrim_model(input_shape)
    gen_model = make_gen_model(input_shape,
                               num_res_blocks=1,
                               layers_in_res_blocks=2,
                               filters_in_res_blocks=16,
                               filter_size=3,
                               block_subsample=(2,2),
                               filters_in_deconv=16)

    # Just for experimentation
    gen_on_discrim = make_gen_on_discrim(gen_model, discrim_model)

    discrim_model, gen_model, gen_on_discrim = train(discrim_model,
                                                     gen_model,
                                                     train_data_dir = './tmp/')

    blurred_val_images = get_blurred_img_array(num_training_images, total_images)
    save_predicted_images(gen_model, blurred_val_images)

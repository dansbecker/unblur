import cv2
import numpy as np
import keras
from keras.layers import Dense, Activation, Input, merge, Convolution2D, \
                         MaxPooling2D, BatchNormalization, Flatten, UpSampling2D, AveragePooling2D
from keras.models import Model
from test_making_blurred_images import show_img, make_and_save_images

def apply_res_block(block_input,
                    n_filters=24,
                    filter_size=3,
                    layers_in_res_blocks=3,
                    subsample=(2,2)):
    '''
    inputs:
        block_input: input tensor to apply the resnet block to
        n_filters: number of convolutional filters in each layer
        filter_size: both the height and width of convolutional filters
        layers_in_res_blocks: number of conv layers on the convolutional path
        subsample: tuple with stride for convolutional path. Applied only once, not in each conv

    outputs:
        output: the tensor after applying the resnet to the input
    NOTE: Current implementation assumes dim_order is 'th'.  (channel, row, col)
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


def make_gen_model(img_height, img_width,
                         num_res_blocks = 3,
                         layers_in_res_blocks=2,
                         filters_in_res_blocks=20,
                         filter_size=3,
                         block_subsample=(1,1),
                         filters_in_deconv=32):
    '''
    inputs:
        img_height and img_width: pixel height and width of inputs.
            input is assumed to be in 'th' dim order, with 3 channels
        num_res_blocks: Number of resnet blocks to apply in convolution stage
        layers_in_res_blocks: Number of convolutional layers in each res net block
        filters_in_res_blocks: Number of convolutional filters in the resnet blocks
        filter_size: Size of the kernel/filters in the resnets
        block_subsample: Tuple of stride length for resnet blocks
        filters_in_deconv: Number of filters to use in deconvolution stage
            note that deconvolution only happens if we are doing sampling in
            conv stage

    Outputs:
        Flattened version of image pixel intensities. This format simplifies calculation
            of loss function
    '''
    layer_input_shape = (3, img_height, img_width)
    model_input = Input(shape=layer_input_shape)
    # Convert num channels to match upcoming resnets
    x = Convolution2D(filters_in_res_blocks, 2, 2, border_mode='same', activation='relu')(model_input)
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
    channel_fixing_layer = Convolution2D(3, 2, 2, border_mode='same', activation='relu')(x)
    # DCGAN paper suggests last layer of generator be tanh. Didn't work for me in early experiment
    output = Flatten()(channel_fixing_layer)
    model = Model(input=model_input, output=output)
    print(model.summary())
    model.compile(optimizer='adam', loss='mse')
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


if __name__ == "__main__":
    total_images = 40
    num_training_images = 30
    img_height = 320
    img_width = 184  # 320:180 preserves aspect ratio. 320:184 avoids padding issue
    make_and_save_images(images_to_make=total_images,
                         img_height=img_height,
                         img_width=img_width)


    blurred_images = get_blurred_img_array(0, num_training_images)
    clear_images = get_clear_img_array(0, num_training_images)
    my_model = make_gen_model(img_height, img_width,
                              num_res_blocks=3,
                              layers_in_res_blocks=2,
                              filters_in_res_blocks=32,
                              filter_size=3,
                              block_subsample=(2,2),
                              filters_in_deconv=24)
    my_model.fit(blurred_images, clear_images, nb_epoch=500, batch_size=2)


    blurred_val_images = get_blurred_img_array(num_training_images, total_images)
    val_pred_imgs = my_model.predict(blurred_val_images).reshape([total_images - num_training_images, 3, img_height, img_width])
    for i, img in enumerate(val_pred_imgs):
        img_num = num_training_images + i
        out_fname = './tmp/' +  str(img_num) + '_predicted.jpg'
        img = normalize_pred_img_array(img)
        cv2.imwrite(out_fname, img)

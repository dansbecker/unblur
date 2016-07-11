from blurred_image_maker import show_img, make_and_save_images
from data_feeder import DataFeeder
from model import make_models
from trainer import Trainer
from utils import normalize_pred_img_array, get_blurred_img_array, save_predicted_images


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

    gen_model, disc_model, gen_disc_model = make_models(input_shape,
                                                        n_filters_in_res_blocks=[64 for _ in range(4)],
                                                        gen_filter_size=3,
                                                        layers_in_res_blocks=2,
                                                        res_block_subsample=(2, 2),
                                                        filters_in_deconv=[32 for _ in range(4)],
                                                        deconv_filter_size=3,
                                                        n_disc_filters=[64, 64])
    data_feeder = DataFeeder(batch_size=16, gen_only_batch_size=16)
    trainer = Trainer(gen_model, disc_model, gen_disc_model, data_feeder, report_freq=10)
    trainer.train(n_steps=1000)
    gen_model, disc_model, gen_disc_model = trainer.get_models()
    blurred_val_images = get_blurred_img_array(num_training_images, total_images)
    save_predicted_images(gen_model, blurred_val_images)

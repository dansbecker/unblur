from video_to_blurred_img_converter import VideoToBlurredImgConverter
from data_feeder import DataFeeder
from model import make_models
from trainer import Trainer
from utils import normalize_pred_img_array, save_predicted_images, train_val_split



if __name__ == "__main__":
    images_per_video = 250
    num_training_images = 200
    img_height = 288
    img_width = 160
    frames_to_mix = 3

    input_shape = (3, img_height, img_width)
    remake_images = False


    ImageMaker = VideoToBlurredImgConverter(images_per_video, frames_to_mix, img_height,
                                   img_width, vid_dir = './data/', rebuild_target_dir=False)
    if remake_images:
        ImageMaker.make_and_save_images()

    train_fnames, val_fnames = train_val_split('./data', num_training_images, images_per_video)
    data_feeder = DataFeeder(batch_size=20, gen_only_batch_size=20, fnames=train_fnames)

    gen_model, disc_model, gen_disc_model = make_models(input_shape,
                                                        n_filters_in_res_blocks=[64 for _ in range(3)],
                                                        gen_filter_size=3,
                                                        layers_in_res_blocks=2,
                                                        res_block_subsample=(2, 2),
                                                        filters_in_deconv=[32 for _ in range(3)],
                                                        deconv_filter_size=3,
                                                        n_disc_filters=[64, 32, 32])

    trainer = Trainer(gen_model, disc_model, gen_disc_model, data_feeder, report_freq=10)
    trainer.train(n_steps=2500)
    gen_model, disc_model, gen_disc_model = trainer.get_models()
    save_predicted_images(gen_model, val_fnames)

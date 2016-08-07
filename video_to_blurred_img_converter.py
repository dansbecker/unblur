import cv2
import numpy as np
import os
import shutil


class VideoToBlurredImgConverter(object):
    def __init__(self, images_to_make_per_vid, frames_to_mix, img_height, img_width,
                 vid_dir = './data/', rebuild_target_dir=False):
        self.images_to_make_per_vid = images_to_make_per_vid
        self.frames_to_mix = frames_to_mix
        self.img_height = img_height
        self.img_width = img_width
        self.vid_dir = vid_dir
        if rebuild_target_dir:
            shutil.rmtree('./tmp')
            os.makedirs('./tmp/clean', exist_ok=True)
            os.makedirs('./tmp/blur', exist_ok=True)
            os.makedirs('./tmp/predicted', exist_ok=True)

    def _resize(self, img, img_height, img_width):
        assert img.shape == (1920, 1080, 3)
        return cv2.resize(img, (img_width, img_height))

    def show_img(self, img, msec_to_show_for=500):
        window_name = ""
        cv2.imshow(window_name, img)
        cv2.waitKey(msec_to_show_for)
        cv2.destroyWindow(window_name)
        return

    def _get_n_frames(self, video_obj, n_frames, img_height, img_width):
        raw_frames = (video_obj.read()[1] for c in range(n_frames))
        output_frames = [self._resize(frame, img_height, img_width) for frame in raw_frames]
        return output_frames

    def _mix_frames(self, frame_list):
        frame_count = len(frame_list)
        new_frame = sum([frame / frame_count for frame in frame_list])
        new_frame = new_frame.astype('uint8')
        return new_frame

    def _set_random_starting_frame(self, vc, mixing_length):
        # not working as expected to due to a bug in vc.set
        # the bug has been fixed in master, but may not be in current
        # binary distribution.
        # test by setting to a specific frame, and then calling
        # vc.get(cv2.CAP_PROP_POS_FRAMES) to verify it's correct
        # The issue is likely this: https://github.com/Itseez/opencv/issues/4890
        vid_length = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        max_starting_frame = vid_length - mixing_length
        start_frame_num = np.random.randint(0, max_starting_frame)
        vc.set(cv2.CAP_PROP_POS_FRAMES, start_frame_num)
        return

    def _make_clean_blur_pair(self, vc, frames_to_mix, img_height, img_width):
        raw_frames = self._get_n_frames(vc, frames_to_mix, img_height, img_width)
        new_frame = self._mix_frames(raw_frames)
        return raw_frames[0], new_frame

    def make_and_save_images(self):
        vid_fnames = os.listdir(self.vid_dir)
        for vid_fname in vid_fnames:
            vid_path = os.path.join(self.vid_dir, vid_fname)
            vc = cv2.VideoCapture(vid_path)
            output = [self._make_clean_blur_pair(vc, self.frames_to_mix, self.img_height, self.img_width)
                        for i in range(self.images_to_make_per_vid)]
            vc.release()
            vid_name_without_extension = vid_fname.split('.')[0]
            for counter, img_pair in enumerate(output):
                cv2.imwrite('./tmp/clean/' + vid_name_without_extension + '_' +str(counter) + '.jpg', img_pair[0])
                cv2.imwrite('./tmp/blur/' + vid_name_without_extension + '_' + str(counter) + '.jpg', img_pair[1])

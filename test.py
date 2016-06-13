import cv2
import numpy as np
import os
import shutil

def my_resize(img):
    assert img.shape == (1920, 1080, 3)
    return cv2.resize(img, (162, 288))

def show_img(img, msec_to_show_for=500):
    window_name = ""
    cv2.imshow(window_name, img)
    cv2.waitKey(msec_to_show_for)
    cv2.destroyWindow(window_name)
    return

def tmp_setup():
    shutil.rmtree('./tmp')
    os.makedirs('./tmp', exist_ok=True)

def get_n_frames(video_obj, n_frames):
    raw_frames = (vc.read()[1] for c in range(n_frames))
    output_frames = [my_resize(frame) for frame in raw_frames]
    return output_frames

def mix_frames(frame_list):
    frame_count = len(frame_list)
    new_frame = sum([frame / frame_count for frame in frame_list])
    new_frame = new_frame.astype('uint8')
    return new_frame

def set_random_starting_frame(vc, mixing_length):
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

def get_pair_to_yield(vc, frames_to_mix):
    raw_frames = get_n_frames(vc, n_frames = frames_to_mix)
    new_frame = mix_frames(raw_frames)
    return raw_frames[0], new_frame

if __name__ == "__main__":
    tmp_setup()
    images_to_make = 20
    frames_to_mix = 3
    fname = './data/test.mp4'
    vc = cv2.VideoCapture(fname)
    # set_random_starting_frame(vc, frames_to_mix)
    output = [get_pair_to_yield(vc, frames_to_mix) for i in range(images_to_make)]
    vc.release()
    counter = 0
    for counter, img_pair in enumerate(output):
        cv2.imwrite('./tmp/' + str(counter) + '_single.jpg', img_pair[0])
        cv2.imwrite('./tmp/' + str(counter) + '_blur.jpg', img_pair[1])

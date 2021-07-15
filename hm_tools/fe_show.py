import os

import cv2
import numpy as np

cam_names = [
    "front_fisheye_camera_record",
    "left_fisheye_camera_record",
    "rear_fisheye_camera_record",
    "right_fisheye_camera_record"
]
APP_NAME = "Fisheye Player"
TRACKERBAR_NAME = "Position:"
Hz = None
delay = 0
canvas, cam_dirs, cam_lists = None, None, None
current_frame = 0

def load_and_resize(path, size=(1920 // 4, 1080 // 4)):
    img = cv2.imread(path)
    img = cv2.resize(img, size)
    return img


def make_grid(grids, grid_size):
    gw, gh = grid_size
    nw, nh = grids
    return np.zeros((nh * gh, nw * gw, 3), dtype=np.uint8)


def draw_grid(canvas, imgs, idx):
    assert (len(imgs) == len(idx))
    if len(imgs) == 0:
        return canvas
    ch, cw, _ = canvas.shape
    sh, sw, _ = imgs[0].shape
    for (h, w), img in zip(idx, imgs):
        canvas[h * sh: h * sh + sh, w * sw:w * sw + sw, :] = img[:, :, :]
    return canvas


def on_trackbar(val):
    global current_frame
    current_frame = val


def draw_frame(canvas, cam_dirs, cam_lists, i):
    paths = [os.path.join(root, cam_dirs[ci], cam_lists[ci][i]) for ci in range(len(cam_names))]
    imgs = [load_and_resize(path) for path in paths]
    front_img, left_img, rear_img, right_img = imgs

    if canvas is None:
        canvas = make_grid((3, 2), front_img.shape[:-1][::-1])
    draw_grid(canvas, imgs, [(0, 1), (0, 0), (1, 1), (0, 2)])
    return canvas


def play_from_current(canvas, cam_dirs, cam_lists):
    global delay
    global current_frame
    while current_frame < min(list(len(x) for x in cam_lists)):
        canvas = draw_frame(canvas, cam_dirs, cam_lists, current_frame)
        cv2.setTrackbarPos(TRACKERBAR_NAME, APP_NAME, current_frame)
        cv2.imshow(APP_NAME, canvas)
        ret = cv2.waitKey(delay)
        current_frame += 1
        if ret == 27:
            break
        if ret == 115:
            if delay == 0:
                delay = 0 if Hz is None else 1000 // Hz
            else:
                delay = 0
        if ret == 97:
            current_frame -= 1
        if ret == 100:
            current_frame += 1


def main(root, hz=10):
    global canvas, cam_dirs, cam_lists, delay, Hz
    Hz = hz
    cam_dirs = [os.path.join(root, cn) for cn in cam_names]
    assert (len(cam_dirs) == 4), "4-way fisheye only."
    cam_lists = [sorted(os.listdir(d)) for d in cam_dirs]
    print(list(zip(list(len(x) for x in cam_lists), cam_names)))
    cv2.namedWindow("Fisheye Player")

    delay = 0 if Hz is None else 1000 // Hz
    canvas = None
    cv2.createTrackbar(TRACKERBAR_NAME, APP_NAME, 0, min(list(len(x) for x in cam_lists)), on_trackbar)
    cv2.createButton("Exit", exit, None, cv2.QT_PUSH_BUTTON, 1)
    play_from_current(canvas, cam_dirs, cam_lists)
    print("All DONE")


if __name__ == '__main__':
    # root = sys.argv[1]
    root = "/home/jian/output/hibag_V71C005_default_000_20210709180551_20210709180909"

    main(root, hz=30)

import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from scipy.interpolate import CubicSpline
from torch.utils.data import Dataset

import datasets.transforms as tf
from utils.affinity_fields import generateAFs


def shrink(im, size):
    s_w, s_h = size
    h, w, _ = im.shape
    ret = np.zeros((s_h, s_w))
    stride_w = w // s_w
    stride_h = h // s_h
    for i in range(0, h, stride_h):
        for j in range(0, w, stride_w):
            patch = im[i:i + stride_h, j:j + stride_w, :]
            if i // stride_h >= s_h or j // stride_w >= s_w:
                continue
            ret[i // stride_h, j // stride_w] = patch.max()
    return ret

def coord_op_to_ip_hm(x, y, scale):
    # (208*scale, 72*scale) --> (208*scale, 72*scale+14=590) --> (1664, 590) --> (1640, 590)
    if x is not None:
        x = int(scale * x)
        # x = x * 1640. / 1664.
    if y is not None:
        y = int(scale * y)
    return x, y

def coord_op_to_ip(x, y, scale):
    # (208*scale, 72*scale) --> (208*scale, 72*scale+14=590) --> (1664, 590) --> (1640, 590)
    if x is not None:
        x = int(scale * x)
        x = x * 1640. / 1664.
    if y is not None:
        y = int(scale * y + 14)
    return x, y


def coord_ip_to_op(x, y, scale):
    # (1640, 590) --> (1664, 590) --> (1664, 590-14=576) --> (1664/scale, 576/scale)
    if x is not None:
        x = x * 1664. / 1640.
        x = int(x / scale)
    if y is not None:
        y = int((y - 14) / scale)
    return x, y


def get_lanes_culane(seg_out, samp_factor):
    # seg_out: input_size/samp_factor

    # fit cubic spline to each lane
    # h_samples = range(589, 240, -10)
    h_samples = range(1080, 600, -10)
    cs = []
    lane_ids = np.unique(seg_out[seg_out > 0])
    for idx, t_id in enumerate(lane_ids):
        xs, ys = [], []
        for y_op in range(seg_out.shape[0]):
            x_op = np.where(seg_out[y_op, :] == t_id)[0]
            if x_op.size > 0:
                x_op = np.mean(x_op)
                x_ip, y_ip = coord_op_to_ip_hm(x_op, y_op, samp_factor)
                x_ip *= 3.75
                y_ip *= 2.109375 # 3.75 #
                xs.append(x_ip)
                ys.append(y_ip)
        if len(xs) >= 2:
            cs.append(CubicSpline(ys, xs, extrapolate=False))
        else:
            cs.append(None)
    # get x-coordinates from fitted spline
    lanes = []
    for idx, t_id in enumerate(lane_ids):
        lane = []
        if cs[idx] is not None:
            y_out = np.array(h_samples)
            x_out = cs[idx](y_out)
            for _x, _y in zip(x_out, y_out):
                if np.isnan(_x):
                    continue
                else:
                    lane += [_x, _y]
        else:
            pass
            # print("Lane completely missed!")
        if len(lane) <= 16:
            continue
        lanes.append(lane)
    return lanes


class CULane(Dataset):
    def __init__(self, path, image_set='train', random_transforms=True, img_transforms=None):
        super(CULane, self).__init__()
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        assert image_set in ('train', 'val', 'test'), "image_set is not valid!"
        self.input_size = (832, 288)  # W, H # original image res: (590, 1640) -> (590-14, 1640+24)/2
        # if image_set in ["val", "test"]:
        #     self.input_size = (1664, 576)
        self.output_stride = 8
        self.output_size = list([i // self.output_stride for i in self.input_size])  # TODO valid dividing
        if image_set in ["val", "test"]:
            self.training_scales_range = (.5, .5)
        else:
            self.training_scales_range = (0.5, 0.6)
        # self.samp_factor = 32 / (1 / self.training_scales_range[0])
        self.random_transforms = random_transforms

        self.data_dir_path = path
        self.image_set = image_set
        # normalization transform for input images
        self.mean = [0.485, 0.456, 0.406]  # [103.939, 116.779, 123.68]
        self.std = [0.229, 0.224, 0.225]  # [1, 1, 1]
        self.ignore_label = 255

        self.img_ts = img_transforms

        if self.random_transforms:
            self.transforms = transforms.Compose([
                tf.GroupRandomScale(size=self.training_scales_range,
                                    interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
                tf.GroupRandomCropRatio(size=self.input_size),
                tf.GroupRandomHorizontalFlip(),
                tf.GroupRandomRotation(degree=(-5, 5), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST),
                                       padding=(self.mean, (self.ignore_label,))),
                tf.GroupNormalize(mean=(self.mean, (0,)), std=(self.std, (1,))),
            ])
        else:
            self.transforms = transforms.Compose([
                tf.GroupRandomScale(size=(self.training_scales_range[0], self.training_scales_range[0]),
                                    interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
                tf.GroupNormalize(mean=(self.mean, (0,)), std=(self.std, (1,))),
            ])
        # print("Creating Index...")
        self.img_list = []
        self.seg_list = []
        self.create_index()
        print("Creating Index DONE")

    def create_index(self):
        # if self.image_set == "test":
        #     listfile = os.path.join(self.data_dir_path, "list", "test_split", "test0_normal.txt")
        # else:
        listfile = os.path.join(self.data_dir_path, "list", "{}.txt".format(self.image_set))
        if not os.path.exists(listfile):
            raise FileNotFoundError("List file doesn't exist. Label has to be generated! ...")
        with open(listfile) as f:
            for lid, line in enumerate(f):
                # if lid > 2000:
                #     break
                l = line.strip()
                if self.image_set == 'test':
                    path = os.path.join(self.data_dir_path, l[1:])
                    self.img_list.append(path)  # l[1:]  get rid of the first '/' so as for os.path.join
                else:
                    self.img_list.append(os.path.join(self.data_dir_path,
                                                      l[0:]))
                if self.image_set == 'test':
                    # path = os.path.join(self.data_dir_path, 'laneseg_label_w16_test', l[0:-3] + 'png')
                    self.seg_list.append(os.path.join(self.data_dir_path, 'laneseg_label_w16_test', l[0:-3] + 'png'))
                else:
                    self.seg_list.append(os.path.join(self.data_dir_path, 'laneseg_label_w16', l[0:-3] + 'png'))

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx]).astype(np.float32) / 255.  # (H, W, 3)
        if self.image_set in ["test"]:
            seg = np.zeros(img.shape[:2])  # Empty Test Label
            # seg = cv2.resize(seg, self.input_size, interpolation=cv2.INTER_NEAREST)
            # img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
        else:
            seg = cv2.imread(self.seg_list[idx], cv2.IMREAD_UNCHANGED)  # (H, W)

        seg = np.tile(seg[..., np.newaxis], (1, 1, 3))  # (H, W, 3)

        img = cv2.resize(img[14:, :, :], (1664, 576), interpolation=cv2.INTER_LINEAR)
        seg = cv2.resize(seg[14:, :, :], (1664, 576), interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, seg = self.transforms((img, seg))

        # seg = shrink(seg, (math.ceil(img.shape[1] / self.output_stride), math.ceil(img.shape[0] / self.output_stride)))
        seg = cv2.resize(seg, None, fx=1 / self.output_stride, fy=1 / self.output_stride,
                         interpolation=cv2.INTER_NEAREST)
        mask = seg[:, :, 0].copy()
        mask[seg[:, :, 0] >= 1] = 1  # binary-mask
        mask[seg[:, :, 0] == self.ignore_label] = self.ignore_label  # ignored px

        # create AFs
        seg_wo_ignore = seg[:, :, 0].copy()
        seg_wo_ignore[seg_wo_ignore == self.ignore_label] = 0
        vaf, haf = generateAFs(seg_wo_ignore.astype(np.long), viz=False)
        af = np.concatenate((vaf, haf[:, :, 0:1]), axis=2)

        # convert all outputs to torch tensors
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
        if self.img_ts is not None:
            img = self.img_ts(img)
        mask = torch.from_numpy(mask).contiguous().float().unsqueeze(0)
        seg = torch.from_numpy(seg[:, :, 0]).contiguous().long().unsqueeze(0)
        af = torch.from_numpy(af).permute(2, 0, 1).contiguous().float()

        # print(img.shape, mask.shape, seg.shape, af.shape)
        # exit()
        return img, seg, mask, af

    def __len__(self):
        return len(self.img_list)

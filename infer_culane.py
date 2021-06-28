import os
import json
import time
from datetime import datetime
from statistics import mean
import argparse

from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
import cv2
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.utils.data import DataLoader

import global_config
from datasets.culane import CULane, get_lanes_culane
from datasets.hmlane import HMLane
from models.dla.pose_dla_dcn import get_pose_net
from utils.affinity_fields import decodeAFs
from utils.metrics import match_multi_class
from utils.visualize import tensor2image, create_viz


parser = argparse.ArgumentParser('Options for inference with LaneAF models in PyTorch...')
parser.add_argument('--dataset-dir', type=str, default=None, help='path to dataset')
parser.add_argument('--output-dir', type=str, default=None, help='output directory for model and logs')
parser.add_argument('--snapshot', type=str, default=None, help='path to pre-trained model snapshot')
parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--no-cuda', action='store_true', default=False, help='do not use cuda for training')
parser.add_argument('--save-viz', action='store_true', default=False, help='save visualization depicting intermediate and final results')
parser.add_argument('--scale', type=int, default=8)


args = parser.parse_args()
# check args
if args.dataset_dir is None:
    assert False, 'Path to dataset not provided!'
if args.snapshot is None:
    assert False, 'Model snapshot not provided!'

# set batch size to 1 for visualization purposes
args.batch_size = 1

# setup args
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.output_dir is None:
    args.output_dir = datetime.now().strftime("%Y-%m-%d-%H%M%S-infer")
    args.output_dir = os.path.join('.', 'experiments', 'culane', args.output_dir)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
else:
    pass  # assert False, 'Output directory already exists!'

# store config in output directory
with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
    json.dump(vars(args), f)

# set random seed
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 10}
print(args.dataset_dir)
test_loader = DataLoader(HMLane(args.dataset_dir, 'val', False), **kwargs)

# create file handles
f_log = open(os.path.join(args.output_dir, "logs.txt"), "w")

# get test set filenames
with open(os.path.join(args.dataset_dir, "val.list")) as f:
    filenames = f.readlines()
filenames = [x.strip() for x in filenames] 


# test function
def test(net):
    net.eval()
    if global_config.use_half_infer:
        net = net.half()
    out_vid = None
    with tqdm(total=len(test_loader)) as t:
        for b_idx, sample in enumerate(test_loader):
            input_img, input_seg, input_mask, input_af, _ = sample
            if args.cuda:
                input_img = input_img.cuda(non_blocking=True)
                input_seg = input_seg.cuda(non_blocking=True)
            if global_config.use_half_infer:
                input_img = input_img.half()
                input_seg = input_seg.half()
            torch.cuda.synchronize()
            s = time.time()
            # do the forward pass
            outputs = net(input_img)[-1]
            torch.cuda.synchronize()
            e = time.time()
            infer_time = e - s
            # print(e - s)
            t.set_postfix(Inf_T="{:.6f}".format(infer_time))
            # convert to arrays
            img = tensor2image(input_img.detach(), np.array(global_config.mean),
                np.array(global_config.std))
            mask_out = tensor2image(torch.sigmoid(outputs['hm']).repeat(1, 3, 1, 1).detach(),
                np.array([0.0 for _ in range(3)], dtype='float32'), np.array([1.0 for _ in range(3)], dtype='float32'))
            vaf_out = np.transpose(outputs['vaf'][0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0))
            haf_out = np.transpose(outputs['haf'][0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0))

            # decode AFs to get lane instances
            torch.cuda.synchronize()
            s = time.time()
            seg_out = decodeAFs(mask_out[:, :, 0], vaf_out, haf_out, fg_thresh=60, err_thresh=3)

            torch.cuda.synchronize()
            e = time.time()
            # print(e - s)
            # re-assign lane IDs to match with ground truth
            seg_out = match_multi_class(seg_out.astype(np.int64), input_seg[0, 0, :, :].detach().cpu().numpy().astype(np.int64))

            # get results in output structure
            xy_coords = get_lanes_culane(seg_out, args.scale) # TODO xxxx
            #
            # xcenter = 1920//2
            # left_id = -1
            # left_dis = -1920
            # right_id = -1
            # right_dis = 1920
            # for id, lane in enumerate(xy_coords):
            #     xmean = mean(lane[::2])
            #     dis = xmean - xcenter
            #     if dis < 0:
            #         if dis > left_dis:
            #             left_dis = dis
            #             left_id = id
            #     elif dis > 0:
            #         if dis < right_dis:
            #             right_dis = dis
            #             right_id = id
            # xy_coords = [xy_coords[left_id], xy_coords[right_id]]

            # write results to file
            if not os.path.exists(os.path.join(args.output_dir, 'outputs', os.path.dirname(filenames[b_idx][1:]))):
                os.makedirs(os.path.join(args.output_dir, 'outputs', os.path.dirname(filenames[b_idx][1:])))

            with open(os.path.join(args.output_dir, 'outputs', filenames[b_idx][:-3]+'lines.txt'), 'w') as f:
                f.write('\n'.join(' '.join(map(str, _lane)) for _lane in xy_coords))


            # if b_idx > 500:
            #     break
            # create video visualization
            if args.save_viz:
                # print("Save VIZ")
                img_out = create_viz(img, seg_out.astype(np.uint8), vaf_out, scale=args.scale, draw_arrow=False)
                img_out = cv2.resize(img_out, global_config.image_size)
                # print(len(xy_coords))
                for points in xy_coords:
                    for x, y in zip(points[::2], points[1::2]):
                        img_out = cv2.circle(img_out, (int(x * 1), int(y * 1)), radius=4, color=(0, 0, 255), thickness=-1)

                cv2.imwrite("{:04d}.jpg".format(b_idx), img_out)
                # save_image(outputs['host'], "{:04d}.host.jpg".format(b_idx))
                # cv2.imshow("O", img_out)
                # cv2.waitKey(1)
            # print('Done with image {} out of {}...'.format(min(args.batch_size*(b_idx+1), len(test_loader.dataset)), len(test_loader.dataset)))
            t.update()
    return


if __name__ == "__main__":
    model = global_config.model(global_config.heads, stride=global_config.stride)
    sd = torch.load(args.snapshot)
    # new_sd = {}
    # for k, v in sd.items():
    #     new_sd[k.replace("module.", '')] = v
    model.load_state_dict(sd['model'], strict=True)

    if args.cuda:
        model.cuda()
    test(model)

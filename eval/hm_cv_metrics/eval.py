import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def _read_result(path):
    lines = open(path, 'r').readlines()[1:]
    lines = [line.strip() for line in lines]
    lines = ' '.join(lines)
    values = lines.split(' ')[1::2]
    keys = lines.split(' ')[0::2]
    keys = [key[:-1] for key in keys]
    res = {k: float(v) for k, v in zip(keys, values)}
    return res


def _test_list(gt_dir, pred_dir, files_list, output_file="tmp.txt", w_lane=30, iou=0.5, im_w=1640, im_h=590,
               bin_cmd='evaluate'):
    frame = 1
    # print('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (bin_cmd, gt_dir, pred_dir, gt_dir, files_list, w_lane, iou, im_w, im_h, frame, output_file))
    with open(os.devnull, "w") as f:
        subprocess.call(('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (
        bin_cmd, gt_dir, pred_dir, gt_dir, files_list, w_lane, iou, im_w, im_h, frame, output_file)).split(' '), stdout=f, stderr=f)
    # os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s > /dev/null 2> /dev/null' % (
    #     bin_cmd, gt_dir, pred_dir, gt_dir, files_list, w_lane, iou, im_w, im_h, frame, output_file))
    res = _read_result(output_file)
    return res


def eval_culane(lb_dir, pred_dir, sub_set, w_lane=30, iou=0.5, im_w=1640, im_h=590,
                bin_cmd=HERE + '/evaluate'):
    eval_type = ["split", "all", "normal", "crowd", "hlight", "shadow", "noline", "arrow", "curve", "cross", "night"]
    if sub_set not in eval_type:
        raise RuntimeError("No such label: {} in culane".format(sub_set))
    if sub_set == "all":
        list_file = os.path.join(lb_dir, 'list/test.txt')
    elif sub_set == "split":
        ret = {}
        for sub_type in eval_type[2:]:
            ret[sub_type] = eval_culane(lb_dir, pred_dir, sub_type, w_lane=w_lane, iou=iou, im_w=im_w, im_h=im_h)
        return ret
    else:
        list_file = os.path.join(lb_dir, 'list/test_split/test{}_{}.txt'.format(eval_type.index(sub_set) - 2, sub_set))
    ret = {}
    ret[sub_set] = _test_list(lb_dir, pred_dir, list_file, w_lane=w_lane, iou=iou, im_w=im_w, im_h=im_h,
                              bin_cmd=bin_cmd)
    return ret

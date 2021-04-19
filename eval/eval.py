import os


def read_result(path):
    lines = open(path, 'r').readlines()[1:]
    lines = ' '.join(lines)
    values = lines.split(' ')[1::2]
    keys = lines.split(' ')[0::2]
    keys = [key[:-1] for key in keys]
    res = {k: v for k, v in zip(keys, values)}
    return res


def test_list(gt_dir, pred_dir, files_list, output_file="tmp.txt", w_lane=30, iou=0.5, im_w=1640, im_h=590,
              bin_cmd='./evaluation/culane/evaluate'):
    frame = 1
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (
        bin_cmd, gt_dir, pred_dir, gt_dir, files_list, w_lane, iou, im_w, im_h, frame, output_file))
    res = read_result(output_file)
    return res


def eval_culane(lb_dir, pred_dir, type, w_lane=30, iou=0.5, im_w=1640, im_h=590,
                bin_cmd='./evaluation/culane/evaluate'):
    eval_type = ["split", "all", "norml", "crowd", "hlight", "shadow", "noline", "arrow", "curve", "cross", "night"]
    if type not in eval_type:
        raise RuntimeError("No such label in culane")
    if type == "all":
        list_file = os.path.join(lb_dir, 'list/test.txt')
    elif type == "split":
        ret = {}
        for sub_type in eval_type[2:]:
            ret[sub_type] = eval_culane(lb_dir, pred_dir, sub_type, w_lane=w_lane, iou=iou, im_w=im_w, im_h=im_h)
        return ret
    else:
        list_file = os.path.join(lb_dir, 'list/test_split/test0_{}.txt'.format(type))
    ret = test_list(lb_dir, pred_dir, list_file, w_lane=w_lane, iou=iou, im_w=im_w, im_h=im_h, bin_cmd=bin_cmd)
    return ret


def call_culane_eval(data_dir, exp_name, output_path):
    if data_dir[-1] != '/':
        data_dir = data_dir + '/'
    detect_dir = os.path.join(output_path, exp_name) + '/'

    w_lane = 30
    iou = 0.5;  # Set iou to 0.3 or 0.5
    im_w = 1640
    im_h = 590
    frame = 1
    list0 = os.path.join(data_dir, 'list/test_split/test0_normal.txt')
    list1 = os.path.join(data_dir, 'list/test_split/test1_crowd.txt')
    list2 = os.path.join(data_dir, 'list/test_split/test2_hlight.txt')
    list3 = os.path.join(data_dir, 'list/test_split/test3_shadow.txt')
    list4 = os.path.join(data_dir, 'list/test_split/test4_noline.txt')
    list5 = os.path.join(data_dir, 'list/test_split/test5_arrow.txt')
    list6 = os.path.join(data_dir, 'list/test_split/test6_curve.txt')
    list7 = os.path.join(data_dir, 'list/test_split/test7_cross.txt')
    list8 = os.path.join(data_dir, 'list/test_split/test8_night.txt')
    if not os.path.exists(os.path.join(output_path, 'txt')):
        os.mkdir(os.path.join(output_path, 'txt'))
    out0 = os.path.join(output_path, 'txt', 'out0_normal.txt')
    out1 = os.path.join(output_path, 'txt', 'out1_crowd.txt')
    out2 = os.path.join(output_path, 'txt', 'out2_hlight.txt')
    out3 = os.path.join(output_path, 'txt', 'out3_shadow.txt')
    out4 = os.path.join(output_path, 'txt', 'out4_noline.txt')
    out5 = os.path.join(output_path, 'txt', 'out5_arrow.txt')
    out6 = os.path.join(output_path, 'txt', 'out6_curve.txt')
    out7 = os.path.join(output_path, 'txt', 'out7_cross.txt')
    out8 = os.path.join(output_path, 'txt', 'out8_night.txt')

    eval_cmd = './evaluation/culane/evaluate'
    if platform.system() == 'Windows':
        eval_cmd = eval_cmd.replace('/', os.sep)

    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list0,w_lane,iou,im_w,im_h,frame,out0))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (
        eval_cmd, data_dir, detect_dir, data_dir, list0, w_lane, iou, im_w, im_h, frame, out0))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list1,w_lane,iou,im_w,im_h,frame,out1))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (
        eval_cmd, data_dir, detect_dir, data_dir, list1, w_lane, iou, im_w, im_h, frame, out1))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list2,w_lane,iou,im_w,im_h,frame,out2))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (
        eval_cmd, data_dir, detect_dir, data_dir, list2, w_lane, iou, im_w, im_h, frame, out2))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list3,w_lane,iou,im_w,im_h,frame,out3))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (
        eval_cmd, data_dir, detect_dir, data_dir, list3, w_lane, iou, im_w, im_h, frame, out3))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list4,w_lane,iou,im_w,im_h,frame,out4))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (
        eval_cmd, data_dir, detect_dir, data_dir, list4, w_lane, iou, im_w, im_h, frame, out4))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list5,w_lane,iou,im_w,im_h,frame,out5))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (
        eval_cmd, data_dir, detect_dir, data_dir, list5, w_lane, iou, im_w, im_h, frame, out5))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list6,w_lane,iou,im_w,im_h,frame,out6))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (
        eval_cmd, data_dir, detect_dir, data_dir, list6, w_lane, iou, im_w, im_h, frame, out6))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list7,w_lane,iou,im_w,im_h,frame,out7))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (
        eval_cmd, data_dir, detect_dir, data_dir, list7, w_lane, iou, im_w, im_h, frame, out7))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list8,w_lane,iou,im_w,im_h,frame,out8))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (
        eval_cmd, data_dir, detect_dir, data_dir, list8, w_lane, iou, im_w, im_h, frame, out8))
    res_all = {}
    res_all['res_normal'] = read_result(out0)
    res_all['res_crowd'] = read_result(out1)
    res_all['res_night'] = read_result(out8)
    res_all['res_noline'] = read_result(out4)
    res_all['res_shadow'] = read_result(out3)
    res_all['res_arrow'] = read_result(out5)
    res_all['res_hlight'] = read_result(out2)
    res_all['res_curve'] = read_result(out6)
    res_all['res_cross'] = read_result(out7)
    return res_all


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Options for eval culane.')
    parser.add_argument('--dataset-dir', type=str, default=None, help='path to dataset')
    parser.add_argument('--output-dir', type=str, default=None, help='path to prediction results.')
    parser.add_argument('--type', type=str, default=None, help='Subset of test')
    parser.add_argument('--im_w', type=int, default=1640, metavar='N',
                        help='Image width(px)')
    parser.add_argument('--im_h', type=int, default=590, metavar='N',
                        help='Image height(px)')
    parser.add_argument('--w_lane', type=int, default=30,
                        help='Width of lane.(px)')
    parser.add_argument('--iou', default=0.5, type=float,
                        help='IoU threshold for eval.')
    parser.add_argument('--output_file', default=None, type=str,
                        help='Output_file')
    args = parser.parse_args()
    lb_dir = args.dataset_dir
    pred_dir = args.output_dir
    im_h = args.im_h
    im_w = args.im_w
    w_lane = args.w_lane
    iou = args.iou
    output_file = args.output_file
    eval_culane(lb_dir, pred_dir, type, w_lane=w_lane, im_h=im_h, im_w=im_w)

import json
import os
import sys

import cv2
import numpy as np

lane_json_priority = [
    "lanelines_6.json",
    "lanelines.json",
    "json"
]

def filter_nothing(x):
    return True


def filter_lanes(x):
    return x['type'] != "Road_Edge"


def filter_road_edge(x):
    return x['type'] == "Road_Edge"


def filter_host_lanes(x):
    return x['type'] != "Road_Edge" and (x['index'] == 0 or x['index'] == 1 or x['index'] == -1)


def load_lines(dir, file, filter_func=filter_nothing):
    for line in open(file):
        fn = line.strip()
        path = os.path.join(dir, fn)
        # can = cv2.imread(os.path.join(dir, fn.replace("json", img_suff)))
        # can = cv2.imread(os.path.join(dir, "2021.05.22", "60a836f882e8f91b2afaaf3e", fn.replace("json", "png")))
        # print(can.shape)
        with open(path) as fp:
            jojo = json.load(fp)
            if "height" in jojo and "width" in jojo:
                canvas_size = (jojo["height"], jojo["width"])
            else:
                canvas_size = (1080, 1920)
            can = np.zeros(canvas_size, np.uint8)
            host_seg = np.zeros(canvas_size, np.uint8)

            re = filter(filter_func, jojo['lines'])
            re = list(re)
            # print(re)
            with open(os.path.join(dir, fn.replace('json', 'edge.lines.txt')), 'w') as f:
                for idx, edge in enumerate(re):
                    # print(edge['points'])
                    points = np.array(edge['points']).astype(np.int)
                    print(points.shape)
                    f.write(' '.join(' '.join(map(str, xy)) for xy in points))
                    f.write('\n')
                    cv2.polylines(can, [points], False, (idx + 1), thickness=20)

        # can = cv2.resize(can, (960, 540))
        # cv2.imshow("Ps", can)
        # cv2.waitKey(0)
        p = os.path.join(dir, "label_edges", fn.replace("json", "png"))
        print(p)
        cv2.imwrite(p, can)
        # cv2.imwrite()


def load_from_list(dir, list_file, filter_func=filter_lanes):
    if filter_func == filter_lanes:
        txt_suff = "lines.txt"
        lb_suff = "lines.png"
        host_suff = "host.png"
    elif filter_func == filter_road_edge:
        txt_suff = "edge.lines.txt"
        lb_suff = "edge.lines.png"
    elif filter_func == filter_host_lanes:
        txt_suff = "host.lines.txt"
        lb_suff = "host.lines.png"
    else:
        raise NotImplementedError()

    for png_idx, png_file in enumerate(open(list_file)):
        png_file = png_file.strip()
        lb_file = None
        png_file = png_file[:-3]
        for suff in lane_json_priority:
            lb_file_prob = os.path.join(dir, png_file+suff)
            if os.path.exists(lb_file_prob):
                lb_file = lb_file_prob
                break
        if lb_file is None:
            print("Json lb not found: ", lb_file)
            continue
        # print(lb_file)
        with open(lb_file) as fp:
            print(lb_file)
            jojo = json.load(fp)
            if "height" in jojo and "width" in jojo:
                canvas_size = (jojo["height"], jojo["width"])
            else:
                canvas_size = (1080, 1920)
            can = np.zeros(canvas_size, np.uint8)
            # host_seg = np.zeros(canvas_size, np.uint8)
            re = filter(filter_func, jojo['lines'])
            re = list(re)
            # print(len(re))
            txt_file = os.path.join(dir, png_file + txt_suff)
            with open(txt_file, 'w') as f:
                for idx, edge in enumerate(re):
                    # print(edge['points'])
                    points = np.array(edge['points']).astype(np.int)
                    # print(points.shape)
                    f.write(' '.join(' '.join(map(str, xy)) for xy in points))
                    f.write('\n')
                    # TODO maybe a bug in lanelines_6.json
                    if edge['index'] in [-1, 0, 1]:
                        cv2.polylines(can, [points], False, edge['index'] + 102, thickness=35)
                    else:
                        cv2.polylines(can, [points], False, (idx + 1), thickness=35)
            p = os.path.join(dir, png_file + lb_suff)
            # host_p = os.path.join(dir, png_file + host_suff)

            # print(p)
            cv2.imwrite(p, can)
            # cv2.imwrite(host_p, host_seg)
            if png_idx % 1000 == 0:
                print(can.max())
                print(png_file, "\t json = ", lb_file, "Output:", "\n\t", txt_file, "\n\t", p)


if __name__ == '__main__':
    dir = sys.argv[1]
    list_file = sys.argv[2]
    load_from_list(dir, list_file)
    # load_lines("/home/jian/data/lines/2021.05.06/60936861b2292955cdf7cabe",
    #            "/home/jian/data/lines/2021.05.06/60936861b2292955cdf7cabe/list.list")
    # load_lines("/home/jian/data/lines/2021.05.22/60a836f882e8f91b2afaaf3e/",
    #            "/home/jian/data/lines/2021.05.22/60a836f882e8f91b2afaaf3e/list.list")

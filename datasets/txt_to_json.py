import os.path
import sys
import json


def main(list_file, root):
    for line in open(list_file):
        fp = line.strip()
        jojo = {"lines": []}
        print(fp)
        with open(fp) as txt_file:
            for lid, tline in enumerate(txt_file):
                print('  Lane {}'.format(lid))
                points = list(map(float, tline.strip().split()))
                ys = points[::2]
                xs = points[1::2]
                assert(len(ys) == len(xs)), "Total {}, {} vs {}".format(len(points), len(ys), len(xs))
                jojo["lines"].append({"index": lid,
                                      "color": "White",
                                      "type": "Single_Solid",
                                      "function": "Normal",
                                      "points": [[y, x] for x, y in zip(xs, ys)]})
        with open(os.path.join(root, fp[:-9] + "lanelines_6.json"), 'w') as of:
            json.dump(jojo, of)


if __name__ == '__main__':
    list_file = sys.argv[1]
    root = sys.argv[2]
    main(list_file, root)

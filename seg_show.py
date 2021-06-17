import json
import sys

import cv2
import numpy as np


tc = {
    "Building.common": [203, 213, 104],
    "Building.tunnel": [2, 2, 169],
    "Marks.lane": [247, 129, 7],
    "MO.living": [236, 184, 69],
    "MO.no_id": [239, 86, 208],
    "MO.pedestrian": [31, 170, 7],
    "MO.temp_static": [24, 166, 169],
    "MO.vehicle.ground": [25, 39, 42],
    "Nature.vegetation": [252, 73, 124],
    "Separator.road_boundary": [52, 31, 161],
    "Separator.std": [156, 24, 38],
    "Separator.wall": [17, 213, 171],
    "Space.vehicle": [85, 219, 203],
    "sse-eraser": [75, 195, 52],
    "Traffic.facility.affiliated": [65, 100, 8],
    "Traffic.light": [237, 40, 140],
    "Traffic.sign": [169, 83, 76],
    "Void": [6, 235, 68],
}

# 1 = 27.83201655663012
# 2 = 2.0726020479939065
# 3 = 9.547816414458818
# 4 = 19.857747056502127
# 5 = 6.950361021109947
# 6 = 31.12081445092349
# 7 = 1.1341359434548972
# 8 = 0.8247122620187566
# 9 = 0.017324691277856723
# 10 = 0.471035614559273
# 11 = 0.035762300685562166
# 12 = 0.02925148123251013
# 13 = 0.10630538983163276
# 14 = 3.211784907068853e-06
# 15 = 0.00010513396638246966
# 16 = 6.423569814137706e-06

tid = {
    "Building.common": 1,
    "Building.tunnel": 2,
    "Marks.lane": 3,
    "MO.living": 4,
    "MO.no_id": 5,
    "MO.pedestrian": 6,
    "MO.temp_static": 7,
    "MO.vehicle.ground": 8,
    "Nature.vegetation": 9,
    "Separator.road_boundary": 10,
    "Separator.std": 11,
    "Separator.wall": 12,
    "Space.vehicle": 13,
    "sse-eraser": 255,
    "Traffic.facility.affiliated": 14,
    "Traffic.light": 15,
    "Traffic.sign": 16,
    "Void": 255,
}

stats = {}
# nums = [1640009193,
#         1204517243,
#         89698312,
#         413211507,
#         859405882,
#         300798531,
#         1346850220,
#         49083267,
#         35691993,
#         749780,
#         20385534,
#         1547725,
#         1265949,
#         4600697,
#         139,
#         4550,
#         278]
# print(sum(nums), 2878 * 1920 * 1080 - nums[0])
# for i in range(1, len(nums)):
#     print("#", i, "=", nums[i] / (2878 * 1920 * 1080 -nums[0])* 100)
# exit()

fl = [line.strip() for line in open(sys.argv[1])]

for jsf in fl:
    # jsf = sys.argv[1]
    print(jsf)
    with open(jsf) as f:
        js = json.load(f)

    canvas = cv2.imread(jsf.replace(".json", ".png"))
    # canvas =cv2.resize(canvas, (960, 540))
    gt_lb = np.zeros(canvas.shape[:2], np.uint8)
    gt_lb += 255

    # types = set([line['type'] for line in js['segmentation']])
    # tc = {}
    # for s in types:
    #     color = [random.randint(0, 255) for i in range(3)]
    #     print("\"" + s + "\"", ":", color, ",")
    #     tc[s] = color
    # print(types)
    for line in js['segmentation']:
        points = line['polygon']
        points = np.array(points, np.int32)
        # if line['type'] == 'Traffic.sign':
        cv2.fillPoly(canvas, pts=[points], color=tc[line['type']])
        cv2.fillPoly(gt_lb, pts=[points], color=tid[line['type']])

    for c in tid.values():
        if c in stats:
            stats[c] += (gt_lb == c).sum()
        else:
            stats[c] = (gt_lb == c).sum()
    # print(stats)
    for line in js['segmentation']:
        points = line['polygon']
        points = np.array(points, np.int32)
        if line['type'] is None:
            cv2.putText(canvas, "None", (points[0][0], points[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
            cv2.fillPoly(canvas, pts=[points], color=(255, 255, 255))
        else:
            cv2.putText(canvas, line['type'], (points[0][0], points[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128))

    # canvas = cv2.resize(canvas, (960, 540))
    png = cv2.imread(jsf.replace(".json", ".png"))
    png = cv2.resize(png, (960, 540))
    canvas = cv2.resize(canvas, (960, 540))
    m = np.zeros((540, 1920, 3), np.uint8)
    m[:, :960] = png
    m[:, 960:] = canvas
    # cv2.imshow("FF", m)
    # cv2.imshow("GT", gt_lb)
    # cv2.waitKey(1)
    # cv2.imwrite(jsf.replace(".json", "_output.png"), m)
    cv2.imwrite(jsf.replace(".json", "_label.png"), gt_lb)
ps = sum(stats.values())
print(ps)
for c, n in stats.items():
    print(c, n / ps * 100)
import cv2
import json
import sys
import numpy as np


jsf = sys.argv[1]
print(jsf)
with open(jsf) as f:
    js = json.load(f)

import random
canvas = cv2.imread(jsf.replace(".json", ".png"))
types = set([line['type'] for line in js['segmentation']])
tc = {}
for s in types:
    color = [random.randint(0, 255) for i in range(3)]
    tc[s] = color
print(types)

for line in js['segmentation']:
    points = line['polygon']
    points = np.array(points, np.int32)
    cv2.fillPoly(canvas, pts=[points], color=tc[line['type']])

canvas = cv2.resize(canvas, (960, 540))

png = cv2.imread(jsf.replace(".json", ".png"))
png = cv2.resize(png, (960, 540))
m = np.zeros((540, 1920, 3), np.uint8)
m[:, :960] = png
m[:, 960:] = canvas
# cv2.imshow("FF", m)

cv2.imwrite(jsf.replace(".json", "_output.png"), m)


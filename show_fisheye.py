import cv2
import json
import numpy as np
s = "6045a8a9e759b2271d504757"
p = cv2.imread("/home/jian/output/{}.jpg".format(s))
jojo = json.load(open("/home/jian/output/{}.json".format(s)))
points = np.array(jojo["objects"][0]["coordinates"]).astype(np.int)
cv2.polylines(p, [points], False, (255, 0, 0), thickness=20)
cv2.imshow("F", p)
cv2.waitKey()
import torch.nn as nn

nn.Transformer()


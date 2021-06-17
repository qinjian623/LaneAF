import torch
import sys
from models.erf.erfnet import EAFNet
from models.raw_resnet import DLAFPNAF
model = DLAFPNAF({"hm": 1, "haf": 1, "vaf": 2})
print(model)
# model = D4UNet()
# sd = torch.load(sys.argv[1])
# sd = sd['model']
# new_sd = {}
# for k, v in sd.items():
#     new_sd[k.replace("module.", '')] = v
# sd = {}
# for k, v in new_sd.items():
#     sd[k.replace("encoder.features.", '')] = v
# model.load_state_dict(sd, strict=True)
# model.cuda()

dummy_input = torch.randn(1, 3, 576, 1024)# .cuda()
# model = torchvision.models.alexnet(pretrained=True).cuda()
torch.onnx.export(model, dummy_input, "dla34_fpn.onnx", verbose=True)

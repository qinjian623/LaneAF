from torch.optim.lr_scheduler import CosineAnnealingLR

from models.raw_resnet import DLAFPNAF
from torch.optim import Adam,SGD
import torchvision.transforms as transforms

# Input
image_size = (1920, 1080)
resize_size = (1024, 1024)  # X, Y or X, Y
input_size = (512, 512)  # W, H
mean = [0.485, 0.456, 0.406]  # [103.939, 116.779, 123.68]
std = [0.229, 0.224, 0.225]  # [1, 1, 1]

stride = 8

heads = {'hm': 1, 'vaf': 2, 'haf': 1}
stride = 8
model = DLAFPNAF
optimizer = Adam
optimizer_kwargs = {

}

warmup_optimizer = Adam
warmup_optimizer_kwargs = {
    "lr": 1e-4,
    # "momentum": 0.9
}
warmup_epoch = 5
warmup_step = (1e-3 - 1e-4)/warmup_epoch

use_warmup = True
use_sync_bn = True
use_half_infer = True
scheduler = CosineAnnealingLR

augs = transforms.Compose([
    # transforms.RandomGrayscale(),
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
    transforms.RandomErasing(p=0.5, value="random", scale=(0.01, 0.08)),
])
augs = None

basic_loss = "wbce"
assert basic_loss in ["focal", "bce", "wbce"]


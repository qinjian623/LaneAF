import argparse
import json
import os
from datetime import datetime
from statistics import mean

import matplotlib
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import global_config
from datasets.hmlane import HMLane
from models.erf.encoder import ERFNet as Encoder

matplotlib.use('Agg')
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.utils.data import DataLoader

from models.loss import FocalLoss, IoULoss, RegL1Loss

import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

parser = argparse.ArgumentParser('Options for training LaneAF models in PyTorch...')
parser.add_argument('--dataset-dir', type=str, default=None, help='path to dataset')
parser.add_argument('--output-dir', type=str, default=None, help='output directory for model and logs')
parser.add_argument('--snapshot', type=str, default=None, help='path to pre-trained model snapshot')
parser.add_argument('--batch-size', type=int, default=32 * 4, metavar='N', help='batch size for training')
parser.add_argument('--epochs', type=int, default=90, metavar='N', help='number of epochs to train for')
parser.add_argument('--weight-decay', type=float, default=1e-3, metavar='WD', help='weight decay')
parser.add_argument('--loss-type', type=str, default='wbce', help='Type of classification loss to use (focal/bce/wbce)')
parser.add_argument('--log-schedule', type=int, default=10, metavar='N',
                    help='number of iterations to print/save log after')
parser.add_argument('--seed', type=int, default=None, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--no-cuda', action='store_true', default=False, help='do not use cuda for training')
parser.add_argument('--random-transforms', action='store_true', default=True,
                    help='apply random transforms to input during training')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 10)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:8848', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--pretrained', type=str, default=None, help='path to dataset')
args = parser.parse_args()


def save_model(net, optimizer, save_path, name):
    if isinstance(net, torch.nn.parallel.DistributedDataParallel):
        net = net.module
    if dist.get_rank() == 0:
        model_state_dict = net.state_dict()
        state = {'model': model_state_dict, 'optimizer': optimizer.state_dict()}
        # state = {'model': model_state_dict}
        assert os.path.exists(save_path)
        model_path = os.path.join(save_path, name)
        torch.save(state, model_path)


def save_epoch(net, optimizer, epoch, save_path):
    save_model(net, optimizer, save_path, 'ep%03d.pth' % epoch)


def save_best(net, optimizer, save_path):
    save_model(net, optimizer, save_path, 'best.pth')


# validation function
def val(net, val_loader, f_log, gpu):
    epoch_acc, epoch_f1 = list(), list()
    epoch_host_acc, epoch_host_f1 = list(), list()
    net.eval()
    with tqdm(total=len(val_loader)) as t:
        for b_idx, sample in enumerate(val_loader):
            input_img, input_seg, input_mask, input_af, host_mask = sample
            input_img = input_img.cuda(gpu, non_blocking=True)
            input_mask = input_mask.cuda(gpu, non_blocking=True)
            host_mask = host_mask.cuda(gpu, non_blocking=True)

            # do the forward pass
            outputs = net(input_img)[-1]

            # calculate losses and metrics
            _mask = (input_mask != val_loader.dataset.ignore_label).float()

            pred = torch.sigmoid(outputs['hm']).detach().cpu().numpy().ravel()
            target = input_mask.detach().cpu().numpy().ravel()
            pred[target == val_loader.dataset.ignore_label] = 0
            target[target == val_loader.dataset.ignore_label] = 0
            val_acc = accuracy_score((pred > 0.5).astype(np.int64), (target > 0.5).astype(np.int64))
            val_f1 = f1_score((target > 0.5).astype(np.int64), (pred > 0.5).astype(np.int64), zero_division=1)
            epoch_acc.append(val_acc)
            epoch_f1.append(val_f1)

            # pred = torch.sigmoid(outputs['host']).detach().cpu().numpy().ravel()
            # target = host_mask.detach().cpu().numpy().ravel()
            # pred[target == val_loader.dataset.ignore_label] = 0
            # target[target == val_loader.dataset.ignore_label] = 0
            # host_val_acc = accuracy_score((pred > 0.5).astype(np.int64), (target > 0.5).astype(np.int64))
            # host_val_f1 = f1_score((target > 0.5).astype(np.int64), (pred > 0.5).astype(np.int64), zero_division=1)
            # epoch_host_acc.append(host_val_acc)
            # epoch_host_f1.append(host_val_f1)

            # t.set_postfix(F1="{:.6f}".format(val_f1), H1="{:.6f}".format(host_val_f1))
            t.set_postfix(F1="{:.6f}".format(val_f1))
            t.update()

    # now that the epoch is completed calculate statistics and store logs

    avg_acc = torch.tensor(mean(epoch_acc)).cuda(gpu)
    avg_f1 = torch.tensor(mean(epoch_f1)).cuda(gpu)

    if len(epoch_host_acc) > 0:
        host_avg_acc = torch.tensor(mean(epoch_host_acc)).cuda(gpu)
        host_avg_f1 = torch.tensor(mean(epoch_host_f1)).cuda(gpu)
    else:
        host_avg_acc = torch.tensor(0.0).cuda(gpu)
        host_avg_f1 = torch.tensor(0.0).cuda(gpu)

    # Sync whole dataset, no need this in training.
    dist.all_reduce(avg_f1)
    dist.all_reduce(avg_acc)
    dist.all_reduce(host_avg_f1)
    dist.all_reduce(host_avg_acc)

    avg_f1 /= dist.get_world_size()
    avg_acc /= dist.get_world_size()
    host_avg_f1 /= dist.get_world_size()
    host_avg_acc /= dist.get_world_size()

    if dist.get_rank() == 0:
        print("\n------------------------ Validation metrics ------------------------")
        f_log.write("\n------------------------ Validation metrics ------------------------\n")
        print("Average accuracy for epoch = {:.4f}".format(avg_acc))
        f_log.write("Average accuracy for epoch = {:.4f}\n".format(avg_acc))
        print("Average F1 score for epoch = {:.4f}".format(avg_f1))
        f_log.write("Average F1 score for epoch = {:.4f}\n".format(avg_f1))
        print("--------------------------------------------------------------------\n")
        f_log.write("--------------------------------------------------------------------\n\n")

        print("Average host accuracy for epoch = {:.4f}".format(host_avg_acc))
        f_log.write("Average host accuracy for epoch = {:.4f}\n".format(host_avg_acc))
        print("Average host F1 score for epoch = {:.4f}".format(host_avg_f1))
        f_log.write("Average host F1 score for epoch = {:.4f}\n".format(host_avg_f1))
        print("--------------------------------------------------------------------\n")
        f_log.write("--------------------------------------------------------------------\n\n")

    return avg_acc.cpu().item(), avg_f1.cpu().item()


# training function
def train(net, train_loader, criterions, optimizer, f_log, epoch, gpu):
    epoch_loss_seg, epoch_loss_vaf, epoch_loss_haf, epoch_loss, epoch_acc, epoch_f1 = list(), list(), list(), list(), list(), list()
    epoch_loss_host, epoch_host_acc, epoch_host_f1 = list(), list(), list()
    net.train()
    criterion_1, criterion_2, criterion_reg = criterions
    metric_sampler_inter = 20
    with tqdm(total=len(train_loader)) as t:
        for b_idx, sample in enumerate(train_loader):
            input_img, _, input_mask, input_af, host_mask = sample
            input_img = input_img.cuda(gpu, non_blocking=True)
            input_mask = input_mask.cuda(gpu, non_blocking=True)
            input_af = input_af.cuda(gpu, non_blocking=True)
            host_mask = host_mask.cuda(gpu, non_blocking=True)

            # zero gradients before forward pass
            optimizer.zero_grad()

            # do the forward pass
            outputs = net(input_img)[-1]

            # calculate losses and metrics
            _mask = (input_mask != train_loader.dataset.ignore_label).float()
            loss_seg = criterion_1(outputs['hm'] * _mask, input_mask * _mask) + criterion_2(
                torch.sigmoid(outputs['hm']),
                input_mask)
            # loss_host = criterion_1(outputs['host'] * _mask, host_mask * _mask) + criterion_2(torch.sigmoid(outputs['host']),
            #                                                                                 host_mask)
            loss_vaf = 0.5 * criterion_reg(outputs['vaf'], input_af[:, :2, :, :], input_mask)
            loss_haf = 0.5 * criterion_reg(outputs['haf'], input_af[:, 2:3, :, :], input_mask)

            epoch_loss_seg.append(loss_seg.item())
            epoch_loss_vaf.append(loss_vaf.item())
            epoch_loss_haf.append(loss_haf.item())
            # epoch_loss_host.append(loss_host.item())
            loss = loss_seg + loss_vaf + loss_haf  # + loss_host
            epoch_loss.append(loss.item())

            # Make training faster
            if b_idx % metric_sampler_inter == 0:
                pred = torch.sigmoid(outputs['hm']).detach().cpu().numpy().ravel()
                target = input_mask.detach().cpu().numpy().ravel()
                pred[target == train_loader.dataset.ignore_label] = 0
                target[target == train_loader.dataset.ignore_label] = 0
                train_acc = accuracy_score((pred > 0.5).astype(np.int64), (target > 0.5).astype(np.int64))
                train_f1 = f1_score((target > 0.5).astype(np.int64), (pred > 0.5).astype(np.int64), zero_division=1)
                epoch_acc.append(train_acc)
                epoch_f1.append(train_f1)

                # pred = torch.sigmoid(outputs['host']).detach().cpu().numpy().ravel()
                # target = host_mask.detach().cpu().numpy().ravel()
                # pred[target == train_loader.dataset.ignore_label] = 0
                # target[target == train_loader.dataset.ignore_label] = 0
                # host_val_acc = accuracy_score((pred > 0.5).astype(np.int64), (target > 0.5).astype(np.int64))
                # host_val_f1 = f1_score((target > 0.5).astype(np.int64), (pred > 0.5).astype(np.int64), zero_division=1)
                # epoch_host_acc.append(host_val_acc)
                # epoch_host_f1.append(host_val_f1)

                # t.set_postfix(F1="{:.6f}".format(train_f1), H1="{:.6f}".format(host_val_f1))
                t.set_postfix(F1="{:.6f}".format(train_f1))
            loss.backward()
            optimizer.step()

            t.update()
            if b_idx % args.log_schedule == 0 and dist.get_rank() == 0:
                # print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tF1-score: {:.4f} \tSegloss: {:.4f}'.format(
                #     epoch, (b_idx + 1) * args.batch_size, len(train_loader.dataset),
                #            100. * (b_idx + 1) * args.batch_size / len(train_loader.dataset), loss.item(), train_f1,
                #     loss_seg.item()))
                f_log.write('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tF1-score: {:.4f}\n'.format(
                    epoch, (b_idx + 1) * args.batch_size, len(train_loader.dataset),
                           100. * (b_idx + 1) * args.batch_size / len(train_loader.dataset), loss.item(), train_f1))

    # now that the epoch is completed calculate statistics and store logs
    avg_loss_seg = mean(epoch_loss_seg)
    avg_loss_vaf = mean(epoch_loss_vaf)
    avg_loss_haf = mean(epoch_loss_haf)
    avg_loss = mean(epoch_loss)
    avg_acc = mean(epoch_acc)
    avg_f1 = mean(epoch_f1)
    if dist.get_rank() == 0:
        print("\n------------------------ Training metrics ------------------------")
        f_log.write("\n------------------------ Training metrics ------------------------\n")
        print("Average segmentation loss for epoch = {:.2f}".format(avg_loss_seg))
        f_log.write("Average segmentation loss for epoch = {:.2f}\n".format(avg_loss_seg))
        print("Average VAF loss for epoch = {:.2f}".format(avg_loss_vaf))
        f_log.write("Average VAF loss for epoch = {:.2f}\n".format(avg_loss_vaf))
        print("Average HAF loss for epoch = {:.2f}".format(avg_loss_haf))
        f_log.write("Average HAF loss for epoch = {:.2f}\n".format(avg_loss_haf))
        print("Average loss for epoch = {:.2f}".format(avg_loss))
        f_log.write("Average loss for epoch = {:.2f}\n".format(avg_loss))
        print("Average accuracy for epoch = {:.4f}".format(avg_acc))
        f_log.write("Average accuracy for epoch = {:.4f}\n".format(avg_acc))
        print("Average F1 score for epoch = {:.4f}".format(avg_f1))
        f_log.write("Average F1 score for epoch = {:.4f}\n".format(avg_f1))
        print("------------------------------------------------------------------\n")
        f_log.write("------------------------------------------------------------------\n\n")

    return net, avg_loss_seg, avg_loss_vaf, avg_loss_haf, avg_loss, avg_acc, avg_f1


def dist_print(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)


def worker(gpu, gpu_num, args):
    args.gpu = gpu
    args.rank = gpu
    print("{} -> {}\t{}/{}".format(args.backend, args.dist_url, args.rank, gpu_num))
    dist.init_process_group(backend=args.backend, init_method=args.dist_url, world_size=gpu_num, rank=args.rank)
    dist_print(datetime.now().strftime('[%Y/%m/%d %H:%M:%S]') + ' start training...')
    f_log = open(os.path.join(args.output_dir, "logs.txt"), "w") if args.rank == 0 else None

    # TODO pretrained backbone
    # if args.pretrained is not None:
    #     print("Loading pretrained weights from file {} ...".format(args.pretrained))
    #     encoder = Encoder(1000)
    #     sd = torch.load(args.pretrained, map_location="cpu")
    #     sd = sd['state_dict']
    #     new_sd = {}
    #     for k, v in sd.items():
    #         new_sd[k.replace("module.", '')] = v
    #     encoder.load_state_dict(new_sd)
    # else:
    #     encoder = None

    torch.set_num_threads(1)
    model = create_model(args)

    # Loss && Optimizer
    # BCE(Focal) loss applied to each pixel individually
    if global_config.basic_loss == 'focal':
        criterion_1 = FocalLoss(gamma=2.0, alpha=0.25, size_average=True)
    elif global_config.basic_loss == 'bce':
        criterion_1 = torch.nn.BCEWithLogitsLoss()
    elif global_config.basic_loss == 'wbce':
        # TODO Maybe greater?
        criterion_1 = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9.6]).cuda())
    else:
        print("No such loss: {}".format(args.loss_type))
        exit()

    criterion_1.cuda(args.gpu)
    criterion_2 = IoULoss().cuda(args.gpu)
    criterion_reg = RegL1Loss().cuda(args.gpu)

    warmup_optimizer = global_config.warmup_optimizer(model.parameters(),
                                                      weight_decay=args.weight_decay,
                                                      **global_config.warmup_optimizer_kwargs)

    optimizer = global_config.optimizer(model.parameters(), lr=1e-3, weight_decay=args.weight_decay,
                                        **global_config.optimizer_kwargs)

    dist_print("Optimizer: {}".format(optimizer.__class__))
    dist_print("Warmup Optimizer: {}".format(warmup_optimizer.__class__))
    dist_print("Loss : {}, {}, {}".format(criterion_1.__class__, criterion_2.__class__, criterion_reg.__class__))
    # Loss && Optimizer ready

    scheduler = global_config.scheduler(optimizer, args.epochs)
    train_dataset = HMLane(args.dataset_dir, 'train', args.random_transforms, img_transforms=global_config.augs)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    kwargs = {'batch_size': args.batch_size // gpu_num, 'num_workers': args.workers,
              'sampler': train_sampler}
    train_loader = DataLoader(train_dataset, **kwargs, pin_memory=True)

    val_dataset = HMLane(args.dataset_dir, 'val', False, None)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    kwargs = {'batch_size': args.batch_size // gpu_num // 2, 'num_workers': args.workers, 'sampler': val_sampler}
    val_loader = DataLoader(val_dataset, **kwargs)

    # TODO resume && finetune

    init_epoch = 0
    if global_config.use_warmup:
        for e in range(global_config.warmup_epoch):
            dist_print("Warmup epoch {}".format(e))
            train_sampler.set_epoch(e)
            train(model, train_loader, [criterion_1, criterion_2, criterion_reg], warmup_optimizer, f_log, e,
                  args.gpu)

            for param_group in warmup_optimizer.param_groups:
                param_group['lr'] += global_config.warmup_step
            val(model, val_loader, f_log, args.gpu)
        init_epoch = global_config.warmup_epoch

    best_f1 = 0.0
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch + init_epoch)
        train(model, train_loader,
              [criterion_1, criterion_2, criterion_reg],
              optimizer, f_log, epoch, args.gpu)
        scheduler.step()
        acc, f1 = val(model, val_loader, f_log, args.gpu)
        # Save dist space by 30%
        if epoch > 30:
            save_epoch(model, optimizer, epoch, args.output_dir)
            if f1 > best_f1:
                best_f1 = f1
                save_best(model, optimizer, args.output_dir)


def create_model(args, config):
    model = config.model(config.heads, stride=config.stride)
    torch.cuda.set_device(args.gpu)
    if args.snapshot is not None:
        load_checkpoint(args.snapshot, model)
    model.cuda(args.gpu)
    if config.use_sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    dist_print(model)
    return model


def load_checkpoint(checkpoint_path, model):
    sd = torch.load(checkpoint_path)
    sd = sd['model']
    new_sd = {}
    # TODO maybe trash code
    for k, v in sd.items():
        new_sd[k.replace("module.", '')] = v
    model.load_state_dict(new_sd, strict=True)


def mp_train():
    # check args
    if args.dataset_dir is None:
        assert False, 'Path to dataset not provided!'

    # setup args
    args.cuda = torch.cuda.is_available()
    if args.output_dir is None:
        args.output_dir = datetime.now().strftime("FastTraining-%Y%m%d-%H%M%S")
        args.output_dir = os.path.join('.', 'experiments', 'culane', args.output_dir)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        assert False, 'Output directory already exists!'

    # store config in output directory
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f)

    # set random seed
    if args.seed:
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    assert args.world_size == 1, "Only world size == 1 mp training now"
    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


if __name__ == "__main__":
    # MP training for hm
    mp_train()

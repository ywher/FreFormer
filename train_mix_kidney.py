import argparse
import logging
import os
import time
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer_mixRf as ViT_seg_mixRf
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer_mix_kidney import trainer_kidney

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')
parser.add_argument('--max_epochs', type=int, default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.003, help='segmentation network learning rate')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
parser.add_argument('--train_img_dir', type=str, default='./data/xinhua_kidney/train/image', help='train image dir')
parser.add_argument('--train_mask_dir', type=str, default='./data/xinhua_kidney/train/mask', help='train mask dir')
parser.add_argument('--train_rf_dir', type=str, default=None, help='train rf dir')
parser.add_argument('--val_img_dir', type=str, default='./data/xinhua_kidney/val/image', help='validation image dir')
parser.add_argument('--val_mask_dir', type=str, default='./data/xinhua_kidney/val/mask', help='validation mask dir')
parser.add_argument('--val_rf_dir', type=str, default=None, help='validation rf dir')
parser.add_argument('--use_rf', action='store_true', help='whether to use rf data')


args = parser.parse_args()

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # 创建模型保存路径，在exp下方根据日期设置子文件夹
    work_dir = f"./exp/{time.strftime('%Y_%m_%d_%H_%M_%S')}"
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    
    # 初始化模型
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes  # 1
    config_vit.n_skip = args.n_skip  # 3
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), 
                                  int(args.img_size / args.vit_patches_size))  # grid:(14,14); size:(16,16)
    
    net = ViT_seg_mixRf(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    net.load_from(weights=np.load(config_vit.pretrained_path))

    trainer_kidney(args, net, work_dir)
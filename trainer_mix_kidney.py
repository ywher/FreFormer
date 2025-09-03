import logging
import os
import random
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from datasets.dataset_kidney import KidneySegmentationDataset
from datasets.dataset_nonrf_kidney import NonRFKidneySegmentationDataset

from utils import dice_loss, dice_score, miou_score, hd95_score, bce_loss, compute_focal_loss

from sklearn.metrics import accuracy_score
from scipy.spatial.distance import directed_hausdorff


def setup_logging(work_dir, args):
    """设置日志配置"""
    logging.basicConfig(filename=os.path.join(work_dir, "log.txt"), 
                       level=logging.INFO,
                       format='[%(asctime)s.%(msecs)03d] %(message)s', 
                       datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


def create_datasets(args):
    """创建训练和验证数据集"""
    if not args.use_rf:
        db_train = NonRFKidneySegmentationDataset(
            img_dir=args.train_img_dir,
            mask_dir=args.train_mask_dir,
        )
        db_val = NonRFKidneySegmentationDataset(
            img_dir=args.val_img_dir,
            mask_dir=args.val_mask_dir,
        )
    else:
        db_train = KidneySegmentationDataset(
            img_dir=args.train_img_dir,
            mask_dir=args.train_mask_dir,
            rf_dir=args.train_rf_dir,
        )
        db_val = KidneySegmentationDataset(
            img_dir=args.val_img_dir,
            mask_dir=args.val_mask_dir,
            rf_dir=args.val_rf_dir,
        )
    
    logging.info(f"Train set length: {len(db_train)}")
    logging.info(f"Validation set length: {len(db_val)}")
    
    return db_train, db_val


def create_dataloaders(db_train, db_val, batch_size, args):
    """创建数据加载器"""
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, 
                           num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)
    
    return trainloader, valloader


def train_one_epoch(model, trainloader, optimizer, device, args, base_lr, iter_num, max_iterations, writer):
    """训练一个epoch"""
    model.train()
    epoch_loss = 0
    epoch_focal_loss = 0
    epoch_dice_loss = 0
    train_preds = []
    train_labels = []
    
    for i_batch, date_batch in enumerate(trainloader):
        # 解包数据并移动到设备
        if args.use_rf:
            image_batch, label_batch, rf_batch = date_batch
            rf_batch = rf_batch.to(device)
        else:
            image_batch, label_batch = date_batch
            rf_batch = None
        
        image_batch = image_batch.to(device)
        label_batch = label_batch.to(device)
        outputs = model(image_batch, rf_batch)
        
        # (B,1,H,W) -> (B,H,W)
        label_batch = label_batch.squeeze(1)
        
        # 计算损失 (只使用focal_loss + dice_loss，与原版保持一致)
        focal_loss = compute_focal_loss(outputs, label_batch)
        loss_dice = dice_loss(outputs, label_batch.unsqueeze(1))
        
        focal_weight = 1
        dice_weight = 2
        loss = focal_weight * focal_loss + dice_weight * loss_dice

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 学习率调整
        lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_
        
        # 记录损失
        epoch_loss += loss.item()
        epoch_focal_loss += focal_loss.item()
        epoch_dice_loss += loss_dice.item()
        
        # 收集预测结果
        with torch.no_grad():
            preds = torch.sigmoid(outputs) > 0.5
            preds = preds.squeeze(1)
            train_preds.append(preds.cpu().numpy().flatten())
            train_labels.append(label_batch.cpu().numpy().flatten())
        
        iter_num += 1
        
        # 记录训练信息
        if iter_num % 20 == 0:
            writer.add_scalar('train/lr', lr_, iter_num)
            writer.add_scalar('train/total_loss', loss.item(), iter_num)
            writer.add_scalar('train/loss_dice', loss_dice.item(), iter_num)
            writer.add_scalar('train/loss_focal', focal_loss.item(), iter_num)

    # 计算训练集指标
    avg_epoch_loss = epoch_loss / len(trainloader)
    avg_focal_loss = epoch_focal_loss / len(trainloader)
    avg_dice_loss = epoch_dice_loss / len(trainloader)
    
    train_preds = np.concatenate(train_preds)
    train_labels = np.concatenate(train_labels)
    train_dice = dice_score(train_labels, train_preds)

    return avg_epoch_loss, avg_focal_loss, avg_dice_loss, train_dice, iter_num


def validate_one_epoch(model, valloader, device, args):
    """验证一个epoch"""
    model.eval()
    val_loss = 0
    val_ce_loss = 0
    val_dice_loss = 0
    val_dice = 0
    val_miou = 0
    val_hd95 = 0
    total_samples = 0

    with torch.no_grad():
        for val_batch in valloader:
            # 解包验证数据并移动到设备
            if args.use_rf:
                image_batch, label_batch, rf_batch = val_batch
                rf_batch = rf_batch.to(device)
            else:
                image_batch, label_batch = val_batch
                rf_batch = None
            
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)
            
            outputs = model(image_batch, rf_batch)
            label_batch = label_batch.squeeze(1)
            
            # 计算验证损失 (与原版完全一致：ce_loss + dice_loss，无权重)
            loss_ce = bce_loss(outputs.squeeze(1), label_batch)
            loss_dice = dice_loss(outputs, label_batch.unsqueeze(1))
            loss = loss_ce + loss_dice

            val_loss += loss.item()
            val_ce_loss += loss_ce.item()
            val_dice_loss += loss_dice.item()
            
            # 收集验证集的预测结果
            preds = torch.sigmoid(outputs) > 0.5
            preds = preds.squeeze(1)
            
            # 计算指标 - 对于Dice和MIoU按batch计算（效率高）
            batch_preds = preds.cpu().numpy().reshape(-1)
            batch_labels = label_batch.cpu().numpy().reshape(-1)
            
            val_dice += dice_score(batch_preds, batch_labels)
            val_miou += miou_score(batch_preds, batch_labels)
            
            # HD95需要逐样本计算（2D图像）
            batch_preds_2d = preds.cpu().numpy()
            batch_labels_2d = label_batch.cpu().numpy()
            
            for b in range(batch_preds_2d.shape[0]):
                try:
                    hd95_val = hd95_score(batch_preds_2d[b], batch_labels_2d[b])
                    val_hd95 += hd95_val
                except:
                    # 处理HD95计算异常（如空预测或标签）
                    val_hd95 += 0
                
                total_samples += 1

    # 计算平均指标
    avg_val_loss = val_loss / len(valloader)
    avg_val_ce_loss = val_ce_loss / len(valloader)
    avg_val_dice_loss = val_dice_loss / len(valloader)
    val_dice = val_dice / len(valloader)
    val_miou = val_miou / len(valloader)
    val_hd95 = val_hd95 / total_samples  # HD95按样本数量平均

    return avg_val_loss, avg_val_ce_loss, avg_val_dice_loss, val_dice, val_miou, val_hd95


def log_metrics(writer, epoch_num, train_metrics, val_metrics):
    """记录指标到TensorBoard"""
    avg_epoch_loss, avg_focal_loss, avg_dice_loss, train_dice = train_metrics
    avg_val_loss, avg_val_ce_loss, avg_val_dice_loss, val_dice, val_miou, val_hd95 = val_metrics

    writer.add_scalar('epoch/train_loss', avg_epoch_loss, epoch_num)
    writer.add_scalar('epoch/train_focal_loss', avg_focal_loss, epoch_num)
    writer.add_scalar('epoch/train_dice_loss', avg_dice_loss, epoch_num)
    writer.add_scalar('epoch/train_dice_score', train_dice, epoch_num)
    writer.add_scalar('epoch/val_loss', avg_val_loss, epoch_num)
    writer.add_scalar('epoch/val_ce_loss', avg_val_ce_loss, epoch_num)
    writer.add_scalar('epoch/val_dice_loss', avg_val_dice_loss, epoch_num)
    writer.add_scalar('epoch/val_dice_score', val_dice, epoch_num)
    writer.add_scalar('epoch/val_miou_score', val_miou, epoch_num)
    writer.add_scalar('epoch/val_hd95_score', val_hd95, epoch_num)


def save_checkpoint(model, work_dir, epoch_num, val_dice, best_dice, min_delta):
    """保存检查点"""
    best_model_path = os.path.join(work_dir, 'best_model.pth')
    
    if val_dice > (best_dice + min_delta):
        if os.path.exists(best_model_path):
            os.remove(best_model_path)
        torch.save(model.state_dict(), best_model_path)
        logging.info(f"New best model saved at epoch {epoch_num}, Dice: {val_dice:.4f}")
        return True, val_dice
    
    return False, best_dice


def trainer_kidney(args, model, work_dir):
    """主训练函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 初始化日志和TensorBoard
    setup_logging(work_dir, args)
    writer = SummaryWriter(log_dir=os.path.join(work_dir, 'tensorboard'))
    
    # 训练参数
    base_lr = args.base_lr
    batch_size = args.batch_size * args.n_gpu
    
    # 创建数据集和数据加载器
    db_train, db_val = create_datasets(args)
    trainloader, valloader = create_dataloaders(db_train, db_val, batch_size, args)
    
    # 模型并行化
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    
    # 训练参数
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    early_stop_patience = 50
    no_improve_epochs = 0
    best_dice = 0
    best_epoch = 0
    min_delta = 0.0005
    
    logging.info(f"{len(trainloader)} iterations per epoch. {max_iterations} max iterations")
    
    # 训练循环
    for epoch_num in tqdm(range(max_epoch), desc="Training", ncols=70):
        # 训练一个epoch
        train_metrics = train_one_epoch(
            model, trainloader, optimizer, device, args, 
            base_lr, iter_num, max_iterations, writer
        )
        avg_epoch_loss, avg_focal_loss, avg_dice_loss, train_dice, iter_num = train_metrics

        # 验证一个epoch
        val_metrics = validate_one_epoch(model, valloader, device, args)
        avg_val_loss, avg_val_ce_loss, avg_val_dice_loss, val_dice, val_miou, val_hd95 = val_metrics

        # 记录指标
        log_metrics(writer, epoch_num, train_metrics[:4], val_metrics)
        
        # 日志记录 (添加HD95指标)
        logging.info(
            f'Epoch {epoch_num}: '
            f'Train Loss: {avg_epoch_loss:.4f} (Focal: {avg_focal_loss:.4f}, Dice: {avg_dice_loss:.4f}) | '
            f'Train Dice: {train_dice:.4f} | '
            f'Val Loss: {avg_val_loss:.4f} (CE: {avg_val_ce_loss:.4f}, Dice: {avg_val_dice_loss:.4f}) | '
            f'Val Dice: {val_dice:.4f}, Val MIoU: {val_miou:.4f}, Val HD95: {val_hd95:.4f}'
        )
        
        # 保存最佳模型
        improved, best_dice = save_checkpoint(model, work_dir, epoch_num, val_dice, best_dice, min_delta)
        
        if improved:
            best_epoch = epoch_num
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            logging.info(f"No improvement for {no_improve_epochs}/{early_stop_patience} epochs")
            
            # Early Stopping
            if no_improve_epochs >= early_stop_patience:
                logging.info(f"Early stopping at epoch {epoch_num}!")
                logging.info(f"Best Dice {best_dice:.4f} achieved at epoch {best_epoch}")
                writer.close()
                return "Training Early Stopped!"
    
    # 训练结束
    final_model_path = os.path.join(work_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    
    logging.info(f"Training completed. Best Dice {best_dice:.4f} at epoch {best_epoch}")
    writer.close()
    return "Training Finished!"
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import torch.nn.functional as F

# import SimpleITK as sitk

epsilon=1e-7

def compute_focal_loss(outputs, label_batch, alpha=0.5, gamma=1.5):
    """计算Focal Loss"""
    ce_loss_value = F.binary_cross_entropy_with_logits(
        outputs.squeeze(1),
        label_batch,
        reduction='mean'
    )
    p_t = torch.exp(-ce_loss_value)
    focal_loss = alpha * (1 - p_t)**gamma * ce_loss_value
    return focal_loss.mean()

def dice_loss(input, target):
    """
    二分类Dice Loss函数实现
    Args:
        input: (N,1,H,W) tensor with logits (before sigmoid)
        target: (N,1,H,W) tensor with binary values 0 or 1
    Returns:
        dice_loss: scalar
    """
    
    # 应用sigmoid激活
    input = torch.sigmoid(input)
    
    # 展平预测和标签
    input = input.view(-1)
    target = target.view(-1)
    
    # 计算交集和并集
    intersection = (input * target).sum()
    union = input.sum() + target.sum()
    
    # 计算Dice系数
    dice = (2. * intersection + epsilon) / (union + epsilon)
    
    # 返回Dice loss (1 - Dice)
    return 1 - dice

def bce_loss(input, target):
    """
    二分类BCE Loss函数
    Args:
        input: (N,) tensor with logits (before sigmoid)
        target: (N,) tensor with binary values 0 or 1
    Returns:
        bce_loss: scalar
    """
    return F.binary_cross_entropy_with_logits(input, target)

def ce_loss(input, target):
    """
    计算交叉熵损失 (多分类)
    Args:
        input: (N, C, H, W) tensor with logits (before softmax)
        target: (N, H, W) tensor with class indices
    Returns:
        ce_loss: scalar
    """
    # 应用softmax激活
    input = torch.softmax(input, dim=1)
    target = target.unsqueeze(1)  # (N, 1, H, W)
    # 计算交叉熵损失
    loss = -torch.sum(input * target, dim=1)
    return loss.mean()

def dice_score(y_gt, y_pred):
    # 确保数据类型为float，避免整数除法问题
    y_gt = y_gt.astype(np.float32)
    y_pred = y_pred.astype(np.float32)
    
    intersection = np.sum(y_gt * y_pred)
    union = np.sum(y_gt) + np.sum(y_pred)
    return (2. * intersection) / (union + epsilon)  # 添加小常数避免除以0
    
def miou_score(y_gt, y_pred):
    # 确保数据类型为float，避免整数除法问题
    y_gt = y_gt.astype(np.float32)
    y_pred = y_pred.astype(np.float32)
    
    # 计算交集和并集
    intersection = np.sum(y_gt * y_pred)
    union = np.sum(y_gt) + np.sum(y_pred) - intersection
    
    # 计算IoU (添加小常数避免除以0)
    iou = (intersection + epsilon) / (union + epsilon)
    
    return iou

def hd95_score(y_gt, y_pred):
    """
    计算95th percentile Hausdorff Distance
    Args:
        y_gt: ground truth binary mask
        y_pred: predicted binary mask
    Returns:
        hd95: 95th percentile Hausdorff Distance
    """
    # 确保数据类型为float，避免整数除法问题
    y_gt = y_gt.astype(np.float32)
    y_pred = y_pred.astype(np.float32)
    
    # 将数据转换为二进制（确保只有0和1）
    y_pred[y_pred > 0] = 1
    y_gt[y_gt > 0] = 1
    
    # 处理边界情况
    if y_pred.sum() > 0 and y_gt.sum() > 0:
        # 两个掩码都有前景像素时，计算HD95
        hd95 = metric.binary.hd95(y_pred, y_gt)
        return hd95
    elif y_pred.sum() > 0 and y_gt.sum() == 0:
        # 预测有前景但真实标签没有，返回一个大的距离值
        return 100.0  # 或者其他合适的大值
    else:
        # 其他情况（包括两者都没有前景），返回0
        return 0.0


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes



# def calculate_metric_percase(pred, gt):
#     pred[pred > 0] = 1
#     gt[gt > 0] = 1
#     if pred.sum() > 0 and gt.sum()>0:
#         dice = metric.binary.dc(pred, gt)
#         hd95 = metric.binary.hd95(pred, gt)
#         return dice, hd95
#     elif pred.sum() > 0 and gt.sum()==0:
#         return 1, 0
#     else:
#         return 0, 0


# def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
#     image, label = image.squeeze(0).cpu().detach().np(), label.squeeze(0).cpu().detach().np()
#     if len(image.shape) == 3:
#         prediction = np.zeros_like(label)
#         for ind in range(image.shape[0]):
#             slice = image[ind, :, :]
#             x, y = slice.shape[0], slice.shape[1]
#             if x != patch_size[0] or y != patch_size[1]:
#                 slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
#             input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
#             net.eval()
#             with torch.no_grad():
#                 outputs = net(input)
#                 out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
#                 out = out.cpu().detach().np()
#                 if x != patch_size[0] or y != patch_size[1]:
#                     pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
#                 else:
#                     pred = out
#                 prediction[ind] = pred
#     else:
#         input = torch.from_numpy(image).unsqueeze(
#             0).unsqueeze(0).float().cuda()
#         net.eval()
#         with torch.no_grad():
#             out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
#             prediction = out.cpu().detach().np()
#     metric_list = []
#     for i in range(1, classes):
#         metric_list.append(calculate_metric_percase(prediction == i, label == i))

#     if test_save_path is not None:
#         img_itk = sitk.GetImageFromArray(image.astype(np.float32))
#         prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
#         lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
#         img_itk.SetSpacing((1, 1, z_spacing))
#         prd_itk.SetSpacing((1, 1, z_spacing))
#         lab_itk.SetSpacing((1, 1, z_spacing))
#         sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
#         sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
#         sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
#     return metric_list
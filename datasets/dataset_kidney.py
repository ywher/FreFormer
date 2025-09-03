import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import scipy.io as sio

class KidneySegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, rf_dir, transform=None):
        """
        Args:
            img_dir (string): Directory with all the JPG images
            mask_dir (string): Directory with all the mask images
            transform (callable, optional): Optional transform to be applied
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = 'jpg'
        self.mask_suffix = 'png'
        self.rf_dir = rf_dir
        self.rf_suffix = 'mat'
        
        # 定义图像和掩码的转换
        self.img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        self.rf_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.img_names = [os.path.splitext(f)[0] for f in os.listdir(img_dir) 
                         if f.lower().endswith(self.img_suffix)]
        
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
    
        # 加载图像
        img_path = os.path.join(self.img_dir, f"{img_name}.{self.img_suffix}")
        img = Image.open(img_path).convert('L')
        img = np.array(img)
    
        # 加载掩码
        mask_path = os.path.join(self.mask_dir, f"{img_name}.{self.mask_suffix}")
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)
    
        # 加载RF数据
        rf_path = os.path.join(self.rf_dir, f"{img_name}_RF.{self.rf_suffix}")
        rf = self.load_rf_image(rf_path)
    
        # 确保所有数据形状一致
        assert img.shape == mask.shape, f"Image and mask shapes mismatch: {img.shape} vs {mask.shape}"
    
        # 应用转换
        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        mask = (mask > 0).float()

        if rf.dim() == 2:
            rf = rf.unsqueeze(0)
    
        return img, mask, rf
        
    def load_rf_image(self, rfimg_path):
        """
        Load RF image from .mat file and properly format it for PyTorch
        """
        mat_data = sio.loadmat(rfimg_path)
        frameRF = mat_data['frameRF']
        rf_data = frameRF[0]['data'][0]
    
        # 预处理步骤
        rf_data = np.abs(rf_data)
        rf_data = np.log(rf_data + 1e-6)
    
        # 标准化到[0,1]范围
        rf_data = (rf_data - rf_data.min()) / (rf_data.max() - rf_data.min() + 1e-6)
    
        # 转换为PyTorch张量并调整形状
        rf_data = torch.from_numpy(rf_data).float()
        rf_data = rf_data.unsqueeze(0)
    
        # 应用RF特定的transform
        if self.rf_transform:
            rf_data = self.rf_transform(rf_data)
    
        return rf_data
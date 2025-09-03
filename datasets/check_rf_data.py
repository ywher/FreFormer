import os
import torch
import numpy as np
import scipy.io as sio
from datetime import datetime
import argparse

def load_rf_image(rfimg_path):
    """
    Load RF image from .mat file and properly format it for PyTorch
    Based on the implementation in dataset_kidney.py
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

    return rf_data

def check_rf_files(rf_dir, output_txt_path="failed_rf_files.txt"):
    """
    测试RF数据文件夹中所有.mat文件的载入情况
    
    Args:
        rf_dir (str): RF数据文件夹路径
        output_txt_path (str): 输出失败文件名的txt文件路径
    
    Returns:
        dict: 包含成功和失败统计信息的字典
    """
    if not os.path.exists(rf_dir):
        print(f"错误: RF数据文件夹不存在: {rf_dir}")
        return None
    
    # 获取所有.mat文件
    mat_files = [f for f in os.listdir(rf_dir) if f.lower().endswith('.mat')]
    
    if not mat_files:
        print(f"警告: 在文件夹 {rf_dir} 中没有找到.mat文件")
        return {"total": 0, "success": 0, "failed": 0, "failed_files": []}
    
    print(f"开始测试 {len(mat_files)} 个RF文件...")
    
    failed_files = []
    success_count = 0
    
    # 创建输出文件路径（在当前脚本路径下）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, output_txt_path)
    
    for i, mat_file in enumerate(mat_files):
        rf_path = os.path.join(rf_dir, mat_file)
        
        try:
            # 尝试载入RF数据
            rf_data = load_rf_image(rf_path)
            
            # 检查数据是否有效
            if rf_data is None or rf_data.numel() == 0:
                raise ValueError("载入的RF数据为空")
            
            # 检查数据是否包含NaN或Inf
            if torch.isnan(rf_data).any() or torch.isinf(rf_data).any():
                raise ValueError("RF数据包含NaN或Inf值")
            
            success_count += 1
            
        except Exception as e:
            print(f"载入失败: {mat_file} - 错误: {str(e)}")
            failed_files.append(mat_file)
        
        # 显示进度
        if (i + 1) % 100 == 0 or (i + 1) == len(mat_files):
            print(f"进度: {i + 1}/{len(mat_files)} ({(i + 1) / len(mat_files) * 100:.1f}%)")
    
    # 将失败的文件名写入txt文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"RF文件载入测试报告\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试目录: {rf_dir}\n")
        f.write(f"总文件数: {len(mat_files)}\n")
        f.write(f"成功载入: {success_count}\n")
        f.write(f"载入失败: {len(failed_files)}\n")
        f.write(f"成功率: {success_count / len(mat_files) * 100:.2f}%\n")
        f.write("\n" + "="*50 + "\n")
        f.write("载入失败的文件列表:\n")
        f.write("="*50 + "\n")
        
        if failed_files:
            for failed_file in failed_files:
                f.write(f"{failed_file}\n")
        else:
            f.write("无载入失败的文件\n")
    
    # 输出统计结果
    result = {
        "total": len(mat_files),
        "success": success_count,
        "failed": len(failed_files),
        "failed_files": failed_files,
        "success_rate": success_count / len(mat_files) * 100
    }
    
    print(f"\n测试完成!")
    print(f"总文件数: {result['total']}")
    print(f"成功载入: {result['success']}")
    print(f"载入失败: {result['failed']}")
    print(f"成功率: {result['success_rate']:.2f}%")
    print(f"失败文件列表已保存到: {output_path}")
    
    return result

def main():
    """
    主函数 - 使用argparse解析命令行参数
    """
    parser = argparse.ArgumentParser(description='测试RF数据文件的载入情况')
    parser.add_argument('--rf_dir', type=str, required=True, 
                       help='RF数据文件夹路径')
    parser.add_argument('--output_file', type=str, default='failed_rf_files.txt',
                       help='输出失败文件名的txt文件名 (默认: failed_rf_files.txt)')
    
    args = parser.parse_args()
    
    # 检查输入路径是否存在
    if not os.path.exists(args.rf_dir):
        print(f"错误: 指定的RF数据文件夹不存在: {args.rf_dir}")
        return
    
    print(f"RF数据文件夹: {args.rf_dir}")
    print(f"输出文件: {args.output_file}")
    print("-" * 50)
    
    # 运行测试
    result = check_rf_files(args.rf_dir, args.output_file)
    
    if result and result['failed'] > 0:
        print(f"\n发现 {result['failed']} 个文件载入失败，请检查失败文件列表。")
    elif result:
        print(f"\n所有文件载入成功！")

if __name__ == "__main__":
    main()

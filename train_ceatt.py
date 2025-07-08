#!/usr/bin/env python3
"""
CeATT-TCPFormer训练启动脚本
简化的训练启动，确保所有参数正确设置
"""

import os
import sys
import subprocess

def main():
    """启动CeATT-TCPFormer训练"""
    
    # 检查配置文件是否存在
    config_path = "configs/h36m/CeATT_TCPFormer_h36m_243.yaml"
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    print("🚀 启动CeATT-TCPFormer训练")
    print("=" * 60)
    print(f"配置文件: {config_path}")
    print(f"模型类型: CeATT-Enhanced TCPFormer")
    print(f"数据集: Human3.6M (243帧)")
    print(f"CeATT配置: temporal_ratio=0.33, spatial_ratio=0.5")
    print("=" * 60)
    
    # 构建训练命令
    cmd = [
        "python", "train.py",
        "--config", config_path,
        "--new_checkpoint", "checkpoint/ceatt_tcpformer/",
        "--use_wandb", "True",
        "--wandb_name", "CeATT-TCPFormer-H36M-243"
    ]
    
    print("训练命令:")
    print(" ".join(cmd))
    print("\n开始训练...")
    print("=" * 60)
    
    try:
        # 启动训练
        result = subprocess.run(cmd, check=True)
        print("\n✅ 训练完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练失败: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️ 训练被用户中断")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

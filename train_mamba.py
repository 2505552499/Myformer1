#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mamba-Enhanced TCPFormer训练脚本
基于原始train.py修改，支持MambaInducedTransformer
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import wandb

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common.opt import opts
from common.utils import *
from common.load_data_hm36 import Fusion
from common.h36m_dataset import H36mDataset
from model.Model import MambaInducedTransformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/h36m/MambaTCP_h36m_243.yaml", 
                       help="Path to the config file.")
    parser.add_argument("--checkpoint", type=str, default="", 
                       help="Path to checkpoint for resuming training.")
    parser.add_argument("--evaluate", action="store_true", 
                       help="Evaluate model on test set.")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug mode.")
    return parser.parse_args()


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config):
    """创建Mamba-Enhanced模型"""
    model = MambaInducedTransformer(
        n_layers=config['n_layers'],
        dim_in=config['dim_in'],
        dim_feat=config['dim_feat'],
        dim_rep=config['dim_rep'],
        dim_out=config['dim_out'],
        mlp_ratio=config['mlp_ratio'],
        act_layer=getattr(nn, config['act_layer'].upper())(),
        attn_drop=config['attn_drop'],
        drop=config['drop'],
        drop_path=config['drop_path'],
        use_layer_scale=config['use_layer_scale'],
        layer_scale_init_value=config['layer_scale_init_value'],
        use_adaptive_fusion=config['use_adaptive_fusion'],
        num_heads=config['num_heads'],
        qkv_bias=config['qkv_bias'],
        qkv_scale=config['qkv_scale'],
        hierarchical=config['hierarchical'],
        num_joints=config['num_joints'],
        use_temporal_similarity=config['use_temporal_similarity'],
        temporal_connection_len=config['temporal_connection_len'],
        use_tcn=config['use_tcn'],
        graph_only=config['graph_only'],
        neighbour_num=config['neighbour_num'],
        n_frames=config['n_frames'],
        # Mamba特定参数
        mamba_d_state=config.get('mamba_d_state', 16),
        mamba_d_conv=config.get('mamba_d_conv', 4),
        mamba_expand=config.get('mamba_expand', 2),
        use_geometric_reorder=config.get('use_geometric_reorder', True),
        use_bidirectional=config.get('use_bidirectional', True),
        use_local_mamba=config.get('use_local_mamba', True)
    )
    
    # 打印模型信息
    model_info = model.get_model_info()
    print(f"[INFO] Model created: {model_info}")
    
    return model


def create_dataloader(config, subset='train'):
    """创建数据加载器"""
    if subset == 'train':
        dataset = H36mDataset(
            data_root=config['data_root'],
            subset_list=config['subset_list'],
            dt_file=config['dt_file'],
            mode='train',
            n_frames=config['n_frames'],
            num_joints=config['num_joints'],
            root_rel=config['root_rel']
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 8),
            pin_memory=config.get('pin_memory', True),
            drop_last=True
        )
    else:
        # 测试数据加载器
        dataset = H36mDataset(
            data_root=config['data_root'],
            subset_list=config['subset_list'],
            dt_file=config['dt_file'],
            mode='test',
            n_frames=config['n_frames'],
            num_joints=config['num_joints'],
            root_rel=config['root_rel']
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # 测试时batch_size=1
            shuffle=False,
            num_workers=config.get('num_workers', 8),
            pin_memory=config.get('pin_memory', True)
        )
    
    return dataloader


def compute_losses(pred, target, config):
    """计算各种损失"""
    losses = {}
    
    # MPJPE损失
    mpjpe = torch.mean(torch.norm(pred - target, dim=-1))
    losses['mpjpe'] = mpjpe * config['lambda_3d_pos']
    
    # 速度损失
    if config.get('lambda_3d_velocity', 0) > 0:
        pred_vel = pred[:, 1:] - pred[:, :-1]
        target_vel = target[:, 1:] - target[:, :-1]
        vel_loss = torch.mean(torch.norm(pred_vel - target_vel, dim=-1))
        losses['velocity'] = vel_loss * config['lambda_3d_velocity']
    
    # 总损失
    total_loss = sum(losses.values())
    losses['total'] = total_loss
    
    return losses


def train_one_epoch(model, dataloader, optimizer, config, epoch, device):
    """训练一个epoch"""
    model.train()
    total_losses = {}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # 计算损失
        losses = compute_losses(outputs, targets, config)
        
        # 反向传播
        losses['total'].backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 累积损失
        for key, value in losses.items():
            if key not in total_losses:
                total_losses[key] = 0
            total_losses[key] += value.item()
        
        # 更新进度条
        pbar.set_postfix({
            'MPJPE': f"{losses['mpjpe'].item():.2f}",
            'Total': f"{losses['total'].item():.2f}"
        })
        
        # 记录到wandb
        if config.get('use_wandb', False) and batch_idx % 100 == 0:
            wandb.log({
                'train/mpjpe': losses['mpjpe'].item(),
                'train/total_loss': losses['total'].item(),
                'epoch': epoch,
                'batch': batch_idx
            })
    
    # 计算平均损失
    avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}
    return avg_losses


def evaluate(model, dataloader, config, device):
    """评估模型"""
    model.eval()
    total_mpjpe = 0
    num_samples = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            
            # 计算MPJPE
            mpjpe = torch.mean(torch.norm(outputs - targets, dim=-1))
            total_mpjpe += mpjpe.item() * inputs.size(0)
            num_samples += inputs.size(0)
    
    avg_mpjpe = total_mpjpe / num_samples
    return avg_mpjpe


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    # 初始化wandb
    if config.get('use_wandb', False):
        wandb.init(
            project=config.get('wandb_project', 'MambaTCP'),
            entity=config.get('wandb_entity', None),
            name=config.get('wandb_name', 'mamba-tcp-experiment'),
            config=config
        )
    
    # 创建模型
    model = create_model(config)
    model = model.to(device)
    
    # 创建数据加载器
    train_loader = create_dataloader(config, 'train')
    test_loader = create_dataloader(config, 'test')
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['lr_decay'])
    
    # 训练循环
    best_mpjpe = float('inf')
    start_epoch = 0
    
    for epoch in range(start_epoch, config['epochs']):
        print(f"\n[INFO] Epoch {epoch}/{config['epochs']}")
        
        # 训练
        train_losses = train_one_epoch(model, train_loader, optimizer, config, epoch, device)
        print(f"Train MPJPE: {train_losses['mpjpe']:.2f}mm")
        
        # 评估
        if epoch % config.get('eval_frequency', 1) == 0:
            test_mpjpe = evaluate(model, test_loader, config, device)
            print(f"Test MPJPE: {test_mpjpe:.2f}mm")
            
            # 保存最佳模型
            if test_mpjpe < best_mpjpe:
                best_mpjpe = test_mpjpe
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_mpjpe': best_mpjpe,
                    'config': config
                }, os.path.join(config['checkpoint_dir'], 'best_model.pth'))
                print(f"[INFO] New best model saved! MPJPE: {best_mpjpe:.2f}mm")
            
            # 记录到wandb
            if config.get('use_wandb', False):
                wandb.log({
                    'test/mpjpe': test_mpjpe,
                    'best_mpjpe': best_mpjpe,
                    'epoch': epoch
                })
        
        # 更新学习率
        scheduler.step()
        
        # 定期保存检查点
        if epoch % config.get('save_frequency', 10) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }, os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pth'))
    
    print(f"\n[INFO] Training completed! Best MPJPE: {best_mpjpe:.2f}mm")


if __name__ == "__main__":
    main()

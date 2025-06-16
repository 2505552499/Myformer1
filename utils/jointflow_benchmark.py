#!/usr/bin/env python3
"""
JointFlow Performance Benchmark Script

This script provides utilities to benchmark and compare the performance
of TCPFormer with and without JointFlow integration.
"""

import torch
import time
import numpy as np
from typing import Dict, List, Tuple
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.Model import MemoryInducedTransformer


class JointFlowBenchmark:
    """Benchmark utility for JointFlow performance evaluation"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Benchmark device: {self.device}")
    
    def create_model(self, use_joint_flow=True, **kwargs):
        """Create TCPFormer model with or without JointFlow"""
        default_config = {
            'n_layers': 16,
            'dim_in': 3,
            'dim_feat': 128,
            'dim_rep': 512,
            'dim_out': 3,
            'mlp_ratio': 4,
            'hierarchical': False,
            'use_tcn': False,
            'graph_only': False,
            'n_frames': 243,
            'use_joint_flow': use_joint_flow,
            'motion_dim': 32,
            'joint_flow_dropout': 0.1
        }
        default_config.update(kwargs)
        
        model = MemoryInducedTransformer(**default_config).to(self.device)
        return model
    
    def benchmark_inference_speed(self, model, input_shape=(2, 243, 17, 3), 
                                 warmup_runs=10, benchmark_runs=100):
        """Benchmark model inference speed"""
        model.eval()
        
        # Create test input
        x = torch.randn(*input_shape).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(x)
        
        # Synchronize GPU
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(benchmark_runs):
                _ = model(x)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        avg_time = (end_time - start_time) / benchmark_runs
        fps = 1.0 / avg_time
        
        return {
            'avg_inference_time': avg_time,
            'fps': fps,
            'total_time': end_time - start_time,
            'runs': benchmark_runs
        }
    
    def benchmark_memory_usage(self, model, input_shape=(2, 243, 17, 3)):
        """Benchmark model memory usage"""
        if self.device.type != 'cuda':
            return {'error': 'Memory benchmark only available on CUDA'}
        
        model.eval()
        x = torch.randn(*input_shape).to(self.device)
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Measure memory
        with torch.no_grad():
            output = model(x)
        
        memory_stats = {
            'peak_memory_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
            'current_memory_mb': torch.cuda.memory_allocated() / 1024 / 1024,
            'cached_memory_mb': torch.cuda.memory_reserved() / 1024 / 1024
        }
        
        return memory_stats
    
    def count_parameters(self, model):
        """Count model parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        jointflow_params = 0
        if hasattr(model, 'joint_flow') and model.joint_flow is not None:
            jointflow_params = sum(p.numel() for p in model.joint_flow.parameters())
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'jointflow_params': jointflow_params,
            'jointflow_ratio': jointflow_params / total_params if total_params > 0 else 0
        }
    
    def compare_models(self, input_shape=(2, 243, 17, 3), **model_kwargs):
        """Compare TCPFormer with and without JointFlow"""
        print("=" * 60)
        print("JointFlow Performance Comparison")
        print("=" * 60)
        
        results = {}
        
        for use_jf, name in [(False, "Without JointFlow"), (True, "With JointFlow")]:
            print(f"\n--- {name} ---")
            
            # Create model
            model = self.create_model(use_joint_flow=use_jf, **model_kwargs)
            
            # Parameter count
            param_stats = self.count_parameters(model)
            print(f"Parameters: {param_stats['total_params']:,}")
            if use_jf:
                print(f"JointFlow params: {param_stats['jointflow_params']:,} "
                      f"({param_stats['jointflow_ratio']:.3%})")
            
            # Speed benchmark
            speed_stats = self.benchmark_inference_speed(model, input_shape)
            print(f"Inference time: {speed_stats['avg_inference_time']:.4f}s")
            print(f"FPS: {speed_stats['fps']:.2f}")
            
            # Memory benchmark
            memory_stats = self.benchmark_memory_usage(model, input_shape)
            if 'error' not in memory_stats:
                print(f"Peak memory: {memory_stats['peak_memory_mb']:.1f} MB")
            
            # Test output difference
            if not use_jf:
                # Store baseline model and output for comparison
                baseline_model = model
                x = torch.randn(*input_shape).to(self.device)
                with torch.no_grad():
                    baseline_output = baseline_model(x)
                results['baseline'] = {
                    'model': baseline_model,
                    'output': baseline_output,
                    'params': param_stats,
                    'speed': speed_stats,
                    'memory': memory_stats
                }
            else:
                # Compare with baseline
                x = torch.randn(*input_shape).to(self.device)
                with torch.no_grad():
                    jf_output = model(x)
                
                # Calculate output difference
                output_diff = torch.mean(torch.abs(jf_output - results['baseline']['output']))
                print(f"Output difference: {output_diff:.6f}")
                
                results['jointflow'] = {
                    'model': model,
                    'output': jf_output,
                    'params': param_stats,
                    'speed': speed_stats,
                    'memory': memory_stats,
                    'output_diff': output_diff.item()
                }
        
        # Summary comparison
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        
        if 'baseline' in results and 'jointflow' in results:
            baseline = results['baseline']
            jointflow = results['jointflow']
            
            param_increase = jointflow['params']['total_params'] - baseline['params']['total_params']
            param_ratio = param_increase / baseline['params']['total_params']
            
            speed_ratio = jointflow['speed']['avg_inference_time'] / baseline['speed']['avg_inference_time']
            
            print(f"Parameter increase: +{param_increase:,} ({param_ratio:.3%})")
            print(f"Speed ratio: {speed_ratio:.3f}x")
            print(f"Motion enhancement effect: {jointflow['output_diff']:.6f}")
            
            if 'peak_memory_mb' in baseline['memory'] and 'peak_memory_mb' in jointflow['memory']:
                memory_increase = jointflow['memory']['peak_memory_mb'] - baseline['memory']['peak_memory_mb']
                print(f"Memory increase: +{memory_increase:.1f} MB")
        
        return results


def main():
    """Main benchmark function"""
    benchmark = JointFlowBenchmark()
    
    # Test different configurations
    configs = [
        {"n_frames": 81, "name": "81 frames"},
        {"n_frames": 243, "name": "243 frames"},
    ]
    
    for config in configs:
        name = config.pop("name")
        print(f"\n{'='*20} {name} {'='*20}")
        
        input_shape = (2, config["n_frames"], 17, 3)
        results = benchmark.compare_models(input_shape=input_shape, **config)


if __name__ == "__main__":
    main()

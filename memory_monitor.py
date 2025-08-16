#!/usr/bin/env python3
"""
GPU Memory Monitor for Talk2GPT-oss
This script helps monitor GPU memory usage and can be run alongside the main app.
"""

import torch
import time
import psutil
import os
from datetime import datetime

def get_gpu_memory_info():
    """Get detailed GPU memory information."""
    if not torch.cuda.is_available():
        return None
    
    gpu_info = {}
    try:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            total = props.total_memory / (1024**3)
            free = total - reserved
            
            gpu_info[f'GPU_{i}'] = {
                'name': props.name,
                'total_memory': f"{total:.2f} GB",
                'allocated': f"{allocated:.2f} GB",
                'reserved': f"{reserved:.2f} GB",
                'free': f"{free:.2f} GB",
                'utilization': f"{(reserved/total)*100:.1f}%"
            }
    except Exception as e:
        gpu_info['error'] = str(e)
    
    return gpu_info

def get_system_memory_info():
    """Get system RAM information."""
    memory = psutil.virtual_memory()
    return {
        'total': f"{memory.total / (1024**3):.2f} GB",
        'available': f"{memory.available / (1024**3):.2f} GB",
        'used': f"{memory.used / (1024**3):.2f} GB",
        'percentage': f"{memory.percent:.1f}%"
    }

def monitor_memory(interval=5, duration=300):
    """Monitor memory usage for a specified duration."""
    print(f"Starting memory monitoring for {duration} seconds...")
    print(f"Checking every {interval} seconds")
    print("=" * 60)
    
    start_time = time.time()
    
    while time.time() - start_time < duration:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}]")
        
        # System memory
        sys_mem = get_system_memory_info()
        print(f"System RAM: {sys_mem['used']}/{sys_mem['total']} ({sys_mem['percentage']})")
        
        # GPU memory
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            for gpu_id, info in gpu_info.items():
                if 'error' not in info:
                    print(f"{gpu_id} ({info['name']}): {info['reserved']}/{info['total_memory']} ({info['utilization']})")
                else:
                    print(f"GPU Error: {info['error']}")
        else:
            print("No CUDA GPU available")
        
        print("-" * 40)
        time.sleep(interval)

def emergency_cleanup():
    """Emergency GPU memory cleanup function."""
    print("Performing emergency GPU memory cleanup...")
    
    if torch.cuda.is_available():
        try:
            # Clear all cached tensors
            torch.cuda.empty_cache()
            
            # Synchronize CUDA operations
            torch.cuda.synchronize()
            
            # Clear all CUDA streams
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
            
            print("✅ GPU memory cleanup completed!")
            
            # Show memory status after cleanup
            gpu_info = get_gpu_memory_info()
            if gpu_info:
                for gpu_id, info in gpu_info.items():
                    if 'error' not in info:
                        print(f"{gpu_id}: {info['free']} free out of {info['total_memory']}")
        
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
    else:
        print("No CUDA GPU available for cleanup")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU Memory Monitor for Talk2GPT-oss")
    parser.add_argument("--monitor", action="store_true", help="Start memory monitoring")
    parser.add_argument("--cleanup", action="store_true", help="Perform emergency cleanup")
    parser.add_argument("--status", action="store_true", help="Show current memory status")
    parser.add_argument("--interval", type=int, default=5, help="Monitoring interval in seconds")
    parser.add_argument("--duration", type=int, default=300, help="Monitoring duration in seconds")
    
    args = parser.parse_args()
    
    if args.cleanup:
        emergency_cleanup()
    elif args.monitor:
        monitor_memory(args.interval, args.duration)
    elif args.status:
        print("Current Memory Status:")
        print("=" * 30)
        
        # System memory
        sys_mem = get_system_memory_info()
        print(f"System RAM: {sys_mem['used']}/{sys_mem['total']} ({sys_mem['percentage']})")
        
        # GPU memory
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            for gpu_id, info in gpu_info.items():
                if 'error' not in info:
                    print(f"{gpu_id} ({info['name']}): {info['reserved']}/{info['total_memory']} ({info['utilization']})")
        else:
            print("No CUDA GPU available")
    else:
        parser.print_help()

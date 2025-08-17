#!/usr/bin/env python3
"""
并行实验监控工具
实时监控6个并行进程的运行状态和进度

使用方法：
python monitor_parallel.py --results-dir parallel_experiment_results
"""

import os
import time
import json
import argparse
from datetime import datetime, timedelta
import psutil

def get_process_info():
    """获取Python进程信息"""
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'cmdline']):
        try:
            if proc.info['name'] == 'python.exe' or proc.info['name'] == 'python':
                cmdline = proc.info['cmdline']
                if any('main.py' in arg or 'run_' in arg for arg in cmdline):
                    python_processes.append({
                        'pid': proc.info['pid'],
                        'cpu': proc.info['cpu_percent'],
                        'memory': proc.info['memory_percent'],
                        'cmdline': ' '.join(cmdline[-3:])  # 显示最后几个参数
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return python_processes

def monitor_experiments(results_dir, refresh_interval=30):
    """监控实验进度"""
    print("🔍 并行实验监控器启动")
    print("=" * 80)
    
    start_time = datetime.now()
    
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')  # 清屏
            
            print("🔍 并行实验监控器")
            print("=" * 80)
            print(f"⏰ 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🕐 运行时长: {datetime.now() - start_time}")
            print()
            
            # 检查结果目录
            if os.path.exists(results_dir):
                print(f"📁 结果目录: {results_dir}")
                
                # 列出已完成的实验
                levels = ['basic', 'load_efficiency', 'role_specialization', 
                         'collaboration', 'behavior_shaping', 'full']
                
                print("\n📊 实验进度:")
                print("-" * 60)
                for level in levels:
                    level_dir = os.path.join(results_dir, level)
                    if os.path.exists(level_dir):
                        completed = len(os.listdir(level_dir))
                        print(f"   {level.upper():<20}: {completed} 个实验完成")
                    else:
                        print(f"   {level.upper():<20}: 等待开始...")
                
                # 检查汇总文件
                summary_file = os.path.join(results_dir, "parallel_experiment_summary.json")
                if os.path.exists(summary_file):
                    try:
                        with open(summary_file, 'r', encoding='utf-8') as f:
                            summary = json.load(f)
                        print(f"\n✅ 实验已完成！")
                        print(f"📊 总耗时: {summary.get('total_duration', 0):.0f} 秒")
                        break
                    except:
                        pass
            else:
                print(f"⏳ 等待结果目录创建: {results_dir}")
            
            # 显示运行中的Python进程
            processes = get_process_info()
            if processes:
                print(f"\n🖥️ 运行中的Python进程 ({len(processes)} 个):")
                print("-" * 60)
                for proc in processes:
                    print(f"   PID {proc['pid']:<8}: CPU {proc['cpu']:<6.1f}% | "
                          f"内存 {proc['memory']:<6.1f}% | {proc['cmdline']}")
            else:
                print("\n💤 没有检测到实验进程")
            
            # 系统资源使用情况
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            print(f"\n💻 系统资源:")
            print("-" * 60)
            print(f"   CPU使用率: {cpu_percent:.1f}%")
            print(f"   内存使用: {memory.percent:.1f}% (可用: {memory.available // (1024**3):.1f}GB)")
            
            # GPU信息（如果可用）
            try:
                import torch
                if torch.cuda.is_available():
                    print(f"   GPU数量: {torch.cuda.device_count()}")
                    for i in range(torch.cuda.device_count()):
                        try:
                            # GPU基本信息
                            props = torch.cuda.get_device_properties(i)
                            total_memory = props.total_memory // (1024**3)
                            
                            # 尝试获取当前进程的GPU内存使用（可能为0，因为多进程）
                            try:
                                allocated = torch.cuda.memory_allocated(i) // (1024**3)
                                cached = torch.cuda.memory_reserved(i) // (1024**3)
                                print(f"   GPU {i} ({props.name}): {total_memory}GB 总容量")
                                if allocated > 0 or cached > 0:
                                    print(f"         当前进程使用: {allocated}GB (缓存: {cached}GB)")
                                else:
                                    print(f"         当前进程未使用GPU内存 (多进程环境正常)")
                            except:
                                print(f"   GPU {i} ({props.name}): {total_memory}GB 总容量")
                        except Exception as e:
                            print(f"   GPU {i}: 检测失败 ({e})")
                else:
                    print("   GPU: CUDA不可用")
            except ImportError:
                print("   GPU: 未安装PyTorch")
            except Exception as e:
                print(f"   GPU: 检测异常 ({e})")
            
            print(f"\n🔄 下次刷新: {refresh_interval} 秒后")
            print("按 Ctrl+C 退出监控")
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\n👋 监控器已停止")

def main():
    parser = argparse.ArgumentParser(description="监控并行实验进度")
    parser.add_argument("--results-dir", type=str, default="parallel_experiment_results",
                       help="实验结果目录")
    parser.add_argument("--interval", type=int, default=30,
                       help="刷新间隔（秒）")
    
    args = parser.parse_args()
    monitor_experiments(args.results_dir, args.interval)

if __name__ == "__main__":
    main()

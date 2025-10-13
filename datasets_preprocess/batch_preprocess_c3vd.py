#!/usr/bin/env python3
"""
批量预处理C3VD数据集的所有序列

这个脚本会自动发现并处理C3VD目录下的所有序列，
使用修正后的位姿矩阵处理方法。
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """
    批量处理所有C3VD序列

    注意：运行此脚本前请先激活conda环境：
    conda activate cut3r
    """
    # 检查conda环境
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if conda_env != 'cut3r':
        print("⚠️  警告：当前不在cut3r环境中")
        print("   请先运行：conda activate cut3r")
        print("   当前环境：", conda_env if conda_env else "未知")
        response = input("   是否继续？(y/N): ")
        if response.lower() != 'y':
            print("   已取消")
            return
        print()

    # 配置路径
    input_base_dir = "/hy-tmp/hy-tmp/CUT3R/dataset/C3VD"
    output_base_dir = "/hy-tmp/hy-tmp/CUT3R/dataset/processed_C3VD"
    preprocess_script = "/hy-tmp/hy-tmp/CUT3R/datasets_preprocess/preprocess_c3vd.py"
    
    # 检查输入目录
    if not os.path.exists(input_base_dir):
        print(f"❌ 输入目录不存在: {input_base_dir}")
        return
    
    # 检查预处理脚本
    if not os.path.exists(preprocess_script):
        print(f"❌ 预处理脚本不存在: {preprocess_script}")
        return
    
    # 获取所有序列
    all_sequences = [d for d in os.listdir(input_base_dir) 
                    if os.path.isdir(os.path.join(input_base_dir, d))]
    all_sequences.sort()
    
    print(f"🔍 发现 {len(all_sequences)} 个序列:")
    for i, seq in enumerate(all_sequences, 1):
        print(f"  {i:2d}. {seq}")
    print()
    
    # 创建输出目录
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 处理每个序列
    successful_count = 0
    failed_sequences = []
    
    for i, sequence_name in enumerate(all_sequences, 1):
        print(f"📁 处理序列 {i}/{len(all_sequences)}: {sequence_name}")
        print("=" * 60)
        
        # 路径设置
        input_sequence_dir = os.path.join(input_base_dir, sequence_name)
        output_sequence_dir = os.path.join(output_base_dir, f"C3VD_{sequence_name}")
        
        print(f"   输入目录: {input_sequence_dir}")
        print(f"   输出目录: {output_sequence_dir}")
        
        # 构建命令
        cmd = [
            "python3", preprocess_script,
            "--input_dir", input_sequence_dir,
            "--output_dir", output_sequence_dir,
            "--sequence_name", sequence_name
        ]
        
        try:
            # 运行预处理脚本
            print(f"   🚀 运行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10分钟超时
            
            if result.returncode == 0:
                successful_count += 1
                print(f"✅ 序列 {sequence_name} 处理成功")
                # 打印输出的最后几行
                if result.stdout:
                    lines = result.stdout.strip().split('\n')
                    for line in lines[-3:]:  # 显示最后3行
                        if line.strip():
                            print(f"   📝 {line}")
            else:
                print(f"❌ 序列 {sequence_name} 处理失败")
                failed_sequences.append(sequence_name)
                if result.stderr:
                    print(f"   错误信息: {result.stderr}")
                    
        except subprocess.TimeoutExpired:
            print(f"⏰ 序列 {sequence_name} 处理超时")
            failed_sequences.append(sequence_name)
        except Exception as e:
            print(f"❌ 序列 {sequence_name} 处理异常: {str(e)}")
            failed_sequences.append(sequence_name)
        
        print()
    
    # 总结
    print("=" * 60)
    print("🎯 批量处理完成总结:")
    print(f"   总序列数: {len(all_sequences)}")
    print(f"   成功处理: {successful_count}")
    print(f"   失败序列: {len(failed_sequences)}")
    
    if failed_sequences:
        print("   失败的序列:")
        for seq in failed_sequences:
            print(f"     - {seq}")
    else:
        print("   🎉 所有序列都处理成功！")
    
    print(f"\n📂 处理后的数据保存在: {output_base_dir}")


if __name__ == "__main__":
    main()

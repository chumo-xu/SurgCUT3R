#!/usr/bin/env python3
"""
统一评估框架基本功能测试
用于验证框架安装和基本功能是否正常
"""

import os
import sys
import json
import numpy as np
import tempfile
from pathlib import Path

# 添加框架路径
framework_dir = Path(__file__).parent
sys.path.insert(0, str(framework_dir))

def create_test_data():
    """创建测试数据"""
    print("📊 创建测试数据...")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    print(f"临时目录: {temp_dir}")
    
    # 生成测试深度数据
    num_frames = 20
    height, width = 480, 640
    
    # GT深度 (随机生成，单位米)
    gt_depths = np.random.uniform(0.1, 10.0, (num_frames, height, width)).astype(np.float32)
    gt_depth_path = os.path.join(temp_dir, 'gt_depths.npz')
    np.savez_compressed(gt_depth_path, data=gt_depths)
    
    # 预测深度 (在GT基础上添加噪声)
    pred_depths = gt_depths + np.random.normal(0, 0.1, gt_depths.shape).astype(np.float32)
    pred_depths = np.clip(pred_depths, 0.01, 50.0)  # 限制范围
    
    # 保存为npy文件目录
    pred_depth_dir = os.path.join(temp_dir, 'pred_depths')
    os.makedirs(pred_depth_dir)
    for i in range(num_frames):
        pred_path = os.path.join(pred_depth_dir, f'depth_{i:06d}.npy')
        np.save(pred_path, pred_depths[i])
    
    # 生成测试位姿数据 (单位米)
    gt_poses = []
    pred_poses = []
    
    for i in range(num_frames):
        # GT位姿 (cam2world格式)
        gt_pose = np.eye(4)
        gt_pose[:3, 3] = [i * 0.1, 0, 0]  # 沿X轴移动
        gt_poses.append(gt_pose)
        
        # 预测位姿 (添加噪声)
        pred_pose = gt_pose.copy()
        pred_pose[:3, 3] += np.random.normal(0, 0.01, 3)  # 位移噪声
        pred_poses.append(pred_pose)
    
    gt_poses = np.array(gt_poses)
    pred_poses = np.array(pred_poses)
    
    # 保存GT位姿
    gt_pose_path = os.path.join(temp_dir, 'gt_poses.npz')
    np.savez_compressed(gt_pose_path, data=gt_poses)
    
    # 保存预测位姿为npz文件目录
    pred_pose_dir = os.path.join(temp_dir, 'pred_poses')
    os.makedirs(pred_pose_dir)
    for i in range(num_frames):
        pred_path = os.path.join(pred_pose_dir, f'pose_{i:06d}.npz')
        np.savez(pred_path, pose=pred_poses[i])
    
    test_data = {
        'temp_dir': temp_dir,
        'gt_depth_path': gt_depth_path,
        'pred_depth_dir': pred_depth_dir,
        'gt_pose_path': gt_pose_path,
        'pred_pose_dir': pred_pose_dir,
        'num_frames': num_frames
    }
    
    print(f"✅ 测试数据创建完成: {num_frames}帧")
    return test_data


def test_framework_import():
    """测试框架导入"""
    print("\n🔍 测试框架导入...")
    
    try:
        from evaluators.unified_evaluator import UnifiedEvaluator
        from core.depth_evaluator import DepthEvaluator
        from core.pose_evaluator import PoseEvaluator
        from adapters.depth_adapter import DepthAdapter
        from adapters.pose_adapter import PoseAdapter
        print("✅ 框架导入成功")
        return True
    except ImportError as e:
        print(f"❌ 框架导入失败: {e}")
        return False


def test_adapters(test_data):
    """测试数据适配器"""
    print("\n🔍 测试数据适配器...")
    
    try:
        from adapters.depth_adapter import DepthAdapter
        from adapters.pose_adapter import PoseAdapter
        
        # 测试深度适配器
        depth_adapter = DepthAdapter(verbose=False)
        
        # 测试NPZ文件加载
        gt_depths = depth_adapter.load(test_data['gt_depth_path'], 'npz_file', 'm')
        print(f"✅ GT深度加载成功: {gt_depths.shape}")
        
        # 测试NPY目录加载
        pred_depths = depth_adapter.load(test_data['pred_depth_dir'], 'npy_dir', 'm')
        print(f"✅ 预测深度加载成功: {pred_depths.shape}")
        
        # 测试位姿适配器
        pose_adapter = PoseAdapter(verbose=False)
        
        # 测试NPZ文件加载
        gt_poses = pose_adapter.load(test_data['gt_pose_path'], 'npz_file', 'm')
        print(f"✅ GT位姿加载成功: {gt_poses.shape}")
        
        # 测试NPZ目录加载
        pred_poses = pose_adapter.load(test_data['pred_pose_dir'], 'npz_dir', 'm')
        print(f"✅ 预测位姿加载成功: {pred_poses.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 适配器测试失败: {e}")
        return False


def test_depth_evaluator(test_data):
    """测试深度评估器"""
    print("\n🔍 测试深度评估器...")
    
    try:
        from evaluators.unified_evaluator import UnifiedEvaluator
        
        evaluator = UnifiedEvaluator(verbose=False)
        
        results = evaluator.evaluate_depth_only(
            gt_depth_path=test_data['gt_depth_path'],
            pred_depth_path=test_data['pred_depth_dir'],
            gt_format='npz_file',
            pred_format='npy_dir',
            gt_unit='m',
            pred_unit='m'
        )
        
        # 检查结果
        assert 'depth_metrics' in results
        metrics = results['depth_metrics']
        required_metrics = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
        
        for metric in required_metrics:
            assert metric in metrics
            assert 'mean' in metrics[metric]
            assert 'std' in metrics[metric]
        
        print(f"✅ 深度评估成功: a1={metrics['a1']['mean']:.3f}, rmse={metrics['rmse']['mean']:.3f}m")
        return True
        
    except Exception as e:
        print(f"❌ 深度评估失败: {e}")
        return False


def test_pose_evaluator(test_data):
    """测试位姿评估器"""
    print("\n🔍 测试位姿评估器...")
    
    try:
        from evaluators.unified_evaluator import UnifiedEvaluator
        
        evaluator = UnifiedEvaluator(pose_window_size=5, verbose=False)
        
        results = evaluator.evaluate_pose_only(
            gt_pose_path=test_data['gt_pose_path'],
            pred_pose_path=test_data['pred_pose_dir'],
            gt_format='npz_file',
            pred_format='npz_dir',
            gt_unit='m',
            pred_unit='m'
        )
        
        # 检查结果
        assert 'pose_metrics' in results
        pose_metrics = results['pose_metrics']
        
        # 检查G-ATE
        assert 'gate' in pose_metrics
        gate = pose_metrics['gate']
        assert 'gate_rmse' in gate
        
        # 检查L-ATE
        assert 'late' in pose_metrics
        late = pose_metrics['late']
        assert 'late_rmse_mean' in late
        assert 'late_rmse_std' in late
        
        print(f"✅ 位姿评估成功: G-ATE={gate['gate_rmse']:.4f}m, L-ATE={late['late_rmse_mean']:.4f}±{late['late_rmse_std']:.4f}m")
        return True
        
    except Exception as e:
        print(f"❌ 位姿评估失败: {e}")
        return False


def test_complete_evaluation(test_data):
    """测试完整评估"""
    print("\n🔍 测试完整评估...")
    
    try:
        from evaluators.unified_evaluator import UnifiedEvaluator
        
        evaluator = UnifiedEvaluator(verbose=False)
        
        output_dir = os.path.join(test_data['temp_dir'], 'results')
        
        results = evaluator.evaluate_complete(
            gt_depth_path=test_data['gt_depth_path'],
            pred_depth_path=test_data['pred_depth_dir'],
            gt_pose_path=test_data['gt_pose_path'],
            pred_pose_path=test_data['pred_pose_dir'],
            output_dir=output_dir,
            gt_depth_format='npz_file',
            pred_depth_format='npy_dir',
            gt_pose_format='npz_file',
            pred_pose_format='npz_dir',
            gt_depth_unit='m',
            pred_depth_unit='m',
            gt_pose_unit='m',
            pred_pose_unit='m'
        )
        
        # 检查结果结构
        assert 'depth_evaluation' in results
        assert 'pose_evaluation' in results
        assert 'evaluation_summary' in results
        
        # 检查输出文件
        json_file = os.path.join(output_dir, 'complete_evaluation_results.json')
        txt_file = os.path.join(output_dir, 'evaluation_summary.txt')
        assert os.path.exists(json_file)
        assert os.path.exists(txt_file)
        
        # 验证JSON文件可以读取
        with open(json_file, 'r') as f:
            saved_results = json.load(f)
        
        print("✅ 完整评估成功")
        print(f"✅ 结果已保存到: {output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ 完整评估失败: {e}")
        return False


def cleanup_test_data(test_data):
    """清理测试数据"""
    import shutil
    try:
        shutil.rmtree(test_data['temp_dir'])
        print(f"🗑️ 清理临时数据: {test_data['temp_dir']}")
    except:
        print(f"⚠️ 无法清理临时数据: {test_data['temp_dir']}")


def main():
    """主测试函数"""
    print("🚀 统一评估框架基本功能测试")
    print("=" * 60)
    
    # 测试计数
    total_tests = 0
    passed_tests = 0
    
    # 1. 测试框架导入
    total_tests += 1
    if test_framework_import():
        passed_tests += 1
    
    # 创建测试数据
    test_data = create_test_data()
    
    try:
        # 2. 测试适配器
        total_tests += 1
        if test_adapters(test_data):
            passed_tests += 1
        
        # 3. 测试深度评估器
        total_tests += 1
        if test_depth_evaluator(test_data):
            passed_tests += 1
        
        # 4. 测试位姿评估器
        total_tests += 1
        if test_pose_evaluator(test_data):
            passed_tests += 1
        
        # 5. 测试完整评估
        total_tests += 1
        if test_complete_evaluation(test_data):
            passed_tests += 1
    
    finally:
        # 清理测试数据
        cleanup_test_data(test_data)
    
    # 测试总结
    print("\n" + "=" * 60)
    print("📊 测试总结")
    print("=" * 60)
    print(f"总测试数: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {total_tests - passed_tests}")
    print(f"成功率: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 所有测试通过！统一评估框架工作正常。")
        return 0
    else:
        print(f"\n⚠️ {total_tests - passed_tests}个测试失败，请检查框架安装。")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

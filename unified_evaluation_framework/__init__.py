"""
统一评估框架 (Unified Evaluation Framework)
===========================================

一个用于深度估计和位姿估计的统一评估框架，确保不同模型使用相同的评估标准和流程。

主要特性:
- 标准化的深度评估流程 (1e-3到150m范围，中值缩放对齐)
- 统一的位姿评估标准 (G-ATE全局 + L-ATE局部不重叠窗口)
- 多种数据格式的自动适配 (npy, npz, tiff, txt, json)
- 完整的结果输出和可视化

快速开始:
```python
from unified_evaluation_framework import UnifiedEvaluator

evaluator = UnifiedEvaluator(depth_min=1e-3, depth_max=150.0, pose_window_size=5)
results = evaluator.evaluate_complete(
    gt_depth_path="gt_depths.npz",
    pred_depth_path="pred_depths/", 
    gt_pose_path="gt_poses.npz",
    pred_pose_path="pred_poses/",
    output_dir="results/"
)
```

版本: 1.0.0
作者: Unified Evaluation Framework Team
"""

from evaluators.unified_evaluator import UnifiedEvaluator
from core.depth_evaluator import DepthEvaluator
from core.pose_evaluator import PoseEvaluator
from adapters.depth_adapter import DepthAdapter
from adapters.pose_adapter import PoseAdapter

__version__ = "1.0.0"
__all__ = [
    'UnifiedEvaluator',
    'DepthEvaluator', 
    'PoseEvaluator',
    'DepthAdapter',
    'PoseAdapter'
]

# 框架信息
FRAMEWORK_INFO = {
    'name': 'Unified Evaluation Framework',
    'version': __version__,
    'description': '深度估计和位姿估计的统一评估框架',
    'features': [
        '标准化深度评估流程',
        '统一位姿评估标准 (G-ATE + L-ATE)',
        '多格式数据自动适配',
        '完整结果输出和可视化'
    ],
    'supported_formats': {
        'depth': ['npy_dir', 'npz_file', 'tiff_dir'],
        'pose': ['npz_dir', 'npz_file', 'txt_file', 'json_file']
    }
}

def print_framework_info():
    """打印框架信息"""
    info = FRAMEWORK_INFO
    print("=" * 60)
    print(f"🚀 {info['name']} v{info['version']}")
    print("=" * 60)
    print(f"📝 {info['description']}")
    print("\n✨ 主要特性:")
    for feature in info['features']:
        print(f"  • {feature}")
    print("\n📦 支持格式:")
    print(f"  深度: {', '.join(info['supported_formats']['depth'])}")
    print(f"  位姿: {', '.join(info['supported_formats']['pose'])}")
    print("=" * 60)

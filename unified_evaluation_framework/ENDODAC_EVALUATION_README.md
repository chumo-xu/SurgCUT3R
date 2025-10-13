# EndoDAC模型评估脚本使用说明

## 概述

本脚本基于CUT3R的统一评估框架，为EndoDAC模型创建了专门的评估工具。该脚本使用与CUT3R完全相同的评估方法和指标，确保结果的一致性和可比较性。

## 文件结构

```
unified_evaluation_framework/
├── evaluate_endodac_results.py    # EndoDAC评估主脚本
├── run_endodac_evaluation.py      # 简化的运行接口
└── ENDODAC_EVALUATION_README.md   # 本说明文档
```

## 数据路径配置

### EndoDAC结果路径结构

```
/hy-tmp/hy-tmp/CUT3R/eval/endodac_results/hy-tmp/hy-tmp/EndoDAC/
├── evaluation_results_dataset8_0/npy_depth_results/  # 序列1深度结果
├── evaluation_results_dataset8_1/npy_depth_results/  # 序列2深度结果
├── evaluation_results_dataset8_2/npy_depth_results/  # 序列3深度结果
├── evaluation_results_dataset8_3/npy_depth_results/  # 序列4深度结果
├── evaluation_results_dataset9_0/npy_depth_results/  # 序列5深度结果
├── evaluation_results_dataset9_1/npy_depth_results/  # 序列6深度结果
├── evaluation_results_dataset9_2/npy_depth_results/  # 序列7深度结果
├── evaluation_results_dataset9_3/npy_depth_results/  # 序列8深度结果
└── EndoDACScaredPose/
    ├── 80/absolute_poses.npz  # dataset8_0位姿
    ├── 81/absolute_poses.npz  # dataset8_1位姿
    ├── 82/absolute_poses.npz  # dataset8_2位姿
    ├── 83/absolute_poses.npz  # dataset8_3位姿
    ├── 90/absolute_poses.npz  # dataset9_0位姿
    ├── 91/absolute_poses.npz  # dataset9_1位姿
    ├── 92/absolute_poses.npz  # dataset9_2位姿
    └── 93/absolute_poses.npz  # dataset9_3位姿
```

### 真值数据路径（与CUT3R共享）

```
/hy-tmp/hy-tmp/CUT3R/eval/eval_data/test/
├── dataset8/
│   ├── keyframe0/
│   ├── keyframe1/
│   ├── keyframe2/
│   └── keyframe3/
└── dataset9/
    ├── keyframe0/
    ├── keyframe1/
    ├── keyframe2/
    └── keyframe3/
```

## 序列映射关系

| 序列名称 | 深度结果路径 | 位姿结果路径 | GT路径 |
|---------|-------------|-------------|--------|
| dataset8_0 | evaluation_results_dataset8_0/npy_depth_results/ | EndoDACScaredPose/80/absolute_poses.npz | dataset8/keyframe0/ |
| dataset8_1 | evaluation_results_dataset8_1/npy_depth_results/ | EndoDACScaredPose/81/absolute_poses.npz | dataset8/keyframe1/ |
| dataset8_2 | evaluation_results_dataset8_2/npy_depth_results/ | EndoDACScaredPose/82/absolute_poses.npz | dataset8/keyframe2/ |
| dataset8_3 | evaluation_results_dataset8_3/npy_depth_results/ | EndoDACScaredPose/83/absolute_poses.npz | dataset8/keyframe3/ |
| dataset9_0 | evaluation_results_dataset9_0/npy_depth_results/ | EndoDACScaredPose/90/absolute_poses.npz | dataset9/keyframe0/ |
| dataset9_1 | evaluation_results_dataset9_1/npy_depth_results/ | EndoDACScaredPose/91/absolute_poses.npz | dataset9/keyframe1/ |
| dataset9_2 | evaluation_results_dataset9_2/npy_depth_results/ | EndoDACScaredPose/92/absolute_poses.npz | dataset9/keyframe2/ |
| dataset9_3 | evaluation_results_dataset9_3/npy_depth_results/ | EndoDACScaredPose/93/absolute_poses.npz | dataset9/keyframe3/ |

## 使用方法

### 1. 环境准备

首先激活conda环境：
```bash
conda activate cut3r-slam
```

### 2. 数据可用性检查

在运行评估前，建议先检查数据是否完整：
```bash
cd /hy-tmp/hy-tmp/CUT3R/unified_evaluation_framework
python run_endodac_evaluation.py --check-only
```

### 3. 列出支持的序列

查看所有支持的评估序列：
```bash
python run_endodac_evaluation.py --list-sequences
```

### 4. 评估所有序列（推荐）

运行完整的EndoDAC模型评估：
```bash
python run_endodac_evaluation.py
```

或者使用详细版本：
```bash
python evaluate_endodac_results.py --verbose
```

### 5. 评估单个序列

如果只想评估特定序列：
```bash
python run_endodac_evaluation.py --sequence dataset8_0
```

或者：
```bash
python evaluate_endodac_results.py --sequence dataset8_0 --verbose
```

### 6. 自定义输出路径

指定评估结果的输出目录：
```bash
python evaluate_endodac_results.py --output /path/to/custom/output
```

## 评估指标

脚本使用与CUT3R完全相同的评估指标：

### 深度估计指标
- **abs_rel**: 平均相对误差
- **sq_rel**: 平方相对误差
- **rmse**: 均方根误差 (单位: 米)
- **rmse_log**: 对数均方根误差
- **a1**: δ < 1.25 的像素比例
- **a2**: δ < 1.25² 的像素比例
- **a3**: δ < 1.25³ 的像素比例

### 位姿估计指标
- **G-ATE**: 全局绝对轨迹误差 (单位: 米)
- **L-ATE**: 局部绝对轨迹误差 (单位: 米)

## 输出结果

评估完成后，会在以下位置生成结果：

### 单序列结果
```
/hy-tmp/hy-tmp/CUT3R/eval/endodac_results/hy-tmp/hy-tmp/EndoDAC/unified_evaluation_results/{sequence_name}/
├── depth_evaluation/          # 深度评估详细结果
├── pose_evaluation/           # 位姿评估详细结果
└── visualizations/            # 可视化结果
```

### 总结果报告
```
/hy-tmp/hy-tmp/CUT3R/eval/endodac_results/hy-tmp/hy-tmp/EndoDAC/unified_evaluation_summary/
├── endodac_evaluation_summary.json    # JSON格式总结果
└── endodac_evaluation_report.txt      # 文本格式详细报告
```

## 性能评级

脚本会自动对结果进行性能评级：

### 深度估计评级
- **S级（优秀）**: a1 > 0.8
- **A级（良好）**: a1 > 0.6
- **B级（有待改进）**: a1 ≤ 0.6

### 位姿估计评级
- **S级（优秀）**: G-ATE < 0.01m && L-ATE < 0.01m
- **A级（良好）**: G-ATE < 0.05m && L-ATE < 0.05m
- **B级（有待改进）**: 其他情况

## 故障排除

### 常见问题

1. **路径不存在错误**
   ```
   ❌ 序列 dataset8_0 缺少路径:
      pred_depth_dir: /path/to/depth/results
   ```
   **解决方案**: 检查EndoDAC结果是否正确放置在指定路径

2. **深度目录为空**
   ```
   ❌ 序列 dataset8_0 深度目录中没有npy文件
   ```
   **解决方案**: 确保npy_depth_results目录包含.npy文件

3. **conda环境问题**
   ```
   ModuleNotFoundError: No module named 'xxx'
   ```
   **解决方案**: 确保已激活cut3r-slam环境
   ```bash
   conda activate cut3r-slam
   ```

### 调试模式

使用详细输出模式进行调试：
```bash
python evaluate_endodac_results.py --verbose
```

## 与CUT3R结果比较

由于使用相同的评估框架和指标，EndoDAC的结果可以直接与CUT3R的结果进行对比分析。两个模型的评估报告具有相同的结构和格式。

## 技术细节

- **深度单位**: 米 (m)
- **位姿单位**: 米 (m) 
- **深度范围**: 1mm - 150m
- **L-ATE窗口大小**: 5帧
- **评估框架**: 统一评估框架 (UnifiedEvaluator)



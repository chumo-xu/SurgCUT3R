# 统一评估框架 (Unified Evaluation Framework)

一个用于深度估计和位姿估计的统一评估框架，确保不同模型使用相同的评估标准和流程。

## 🎯 特性

### 深度评估
- **标准化流程**: 严格按照统一标准（调整尺寸→有效掩码→中值缩放→范围截断→误差计算）
- **有效范围**: 1e-3到150米的深度范围过滤
- **中值缩放对齐**: 处理相对深度vs绝对深度的尺度问题
- **多种指标**: abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

### 位姿评估
- **G-ATE (全局ATE)**: 起始帧对齐 + 最小二乘法尺度估计 + 全局RMSE计算
- **L-ATE (局部ATE)**: 不重叠5帧窗口 + 每窗口独立对齐 + RMSE统计 (mean±std)
- **严格对齐**: 确保起始帧对齐 + 尺度缩放后仍保持起点对齐
- **统一单位**: 输入输出位移单位统一为米(m)

### 数据格式兼容
- **深度格式**: npy_dir, npz_file, tiff_dir
- **位姿格式**: npz_dir, npz_file, txt_file, json_file  
- **自动适配**: 支持mm/m单位转换，多种数据结构自动识别
- **数据验证**: 输入数据质量检查和警告

## 📦 框架结构

```
unified_evaluation_framework/
├── core/                     # 核心评估模块
│   ├── depth_evaluator.py   # 深度评估器
│   ├── pose_evaluator.py    # 位姿评估器
│   └── utils.py             # 通用工具函数
├── adapters/                 # 数据格式适配器
│   ├── depth_adapter.py     # 深度数据适配器
│   └── pose_adapter.py      # 位姿数据适配器
├── evaluators/               # 统一评估接口
│   └── unified_evaluator.py # 统一评估器
├── examples/                 # 使用示例
│   └── evaluation_examples.py
├── configs/                  # 配置文件
│   └── default_config.json
└── README.md                # 本文档
```

## 🚀 快速开始

### 基本使用

```python
from evaluators import UnifiedEvaluator

# 初始化评估器
evaluator = UnifiedEvaluator(
    depth_min=1e-3,      # 深度范围: 1mm ~ 150m
    depth_max=150.0,
    pose_window_size=5,  # L-ATE窗口: 5帧不重叠
    verbose=True
)

# 完整评估 (深度 + 位姿)
results = evaluator.evaluate_complete(
    gt_depth_path="/path/to/gt_depths.npz",
    pred_depth_path="/path/to/predicted_depths/",
    gt_pose_path="/path/to/gt_poses.npz",
    pred_pose_path="/path/to/predicted_poses/",
    output_dir="/path/to/results/",
    # 深度配置
    gt_depth_format='npz_file',
    pred_depth_format='npy_dir',
    gt_depth_unit='m',
    pred_depth_unit='m',
    # 位姿配置  
    gt_pose_format='npz_file',
    pred_pose_format='npz_dir',
    gt_pose_unit='m',
    pred_pose_unit='m'
)

# 打印结果
evaluator.print_complete_results(results)
```

### 仅深度评估

```python
depth_results = evaluator.evaluate_depth_only(
    gt_depth_path="/path/to/gt_depths.npz",
    pred_depth_path="/path/to/predicted_depths/",
    gt_format='npz_file',
    pred_format='npy_dir',
    gt_unit='m',
    pred_unit='m'
)
```

### 仅位姿评估

```python
pose_results = evaluator.evaluate_pose_only(
    gt_pose_path="/path/to/gt_poses.npz",
    pred_pose_path="/path/to/predicted_poses/",
    gt_format='npz_file', 
    pred_format='npz_dir',
    gt_unit='m',
    pred_unit='m'
)
```

## 📊 支持的数据格式

### 深度数据格式

| 格式 | 说明 | 示例 |
|------|------|------|
| `npy_dir` | NPY文件目录，每个文件一帧深度 | `depth_000001.npy`, `depth_000002.npy` |
| `npz_file` | 单个NPZ文件，包含所有帧 | `all_depths.npz` (key: 'data' 或 'depths') |
| `tiff_dir` | TIFF文件目录，适用于SCARED数据集 | `scene_points00001.tiff` |

### 位姿数据格式

| 格式 | 说明 | 示例 |
|------|------|------|
| `npz_dir` | NPZ文件目录，每个文件一个4x4位姿矩阵 | `pose_000001.npz` |
| `npz_file` | 单个NPZ文件，包含所有位姿 | `all_poses.npz` (key: 'data' 或 'poses') |
| `txt_file` | 文本文件，每行16个元素或12个元素(KITTI) | `poses.txt` |
| `json_file` | JSON格式文件 | `poses.json` |

## 🎯 评估标准

### 深度评估流程
1. **调整到GT尺寸**: 将预测深度resize到GT尺寸
2. **创建有效掩码**: 1e-3到150m范围，排除无效像素
3. **中值缩放对齐**: `ratio = GT中值 / 预测中值`
4. **深度范围截断**: 限制预测深度极值
5. **计算误差指标**: abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

### 位姿评估流程

#### G-ATE (全局ATE)
1. 起始帧对齐: `pred[0] = gt[0]`
2. 最小二乘法估计尺度因子
3. 应用缩放并保证起点仍对齐
4. 计算全局ATE的RMSE

#### L-ATE (局部ATE) 
1. 不重叠窗口分割: 0-4, 5-9, 10-14...
2. 每个窗口执行G-ATE相同流程
3. 计算每个窗口的ATE RMSE
4. 汇总所有窗口RMSE的mean和std

## 📈 输出结果

### 深度指标
- `abs_rel`: 平均相对误差
- `sq_rel`: 平方相对误差
- `rmse`: 均方根误差 (m)
- `rmse_log`: 对数RMSE
- `a1, a2, a3`: δ < 1.25^n 准确率

### 位姿指标
- `gate_rmse`: 全局ATE RMSE (m)
- `late_rmse_mean`: 局部ATE RMSE平均值 (m)
- `late_rmse_std`: 局部ATE RMSE标准差 (m)

### 结果文件
- `complete_evaluation_results.json`: 完整JSON结果
- `evaluation_summary.txt`: 简化文本报告
- 包含每帧详细数据和处理统计信息

## 🔧 高级配置

### 自定义评估参数

```python
evaluator = UnifiedEvaluator(
    depth_min=0.01,          # 自定义深度范围
    depth_max=100.0,
    pose_window_size=10,     # 自定义L-ATE窗口大小
    verbose=False            # 关闭详细输出
)
```

### 格式特定参数

```python
# TIFF深度数据裁剪
results = evaluator.evaluate_depth_only(
    gt_depth_path="tiff_dir/",
    pred_depth_path="npy_dir/",
    gt_format='tiff_dir',
    pred_format='npy_dir',
    format_kwargs={
        'gt_kwargs': {'crop_rows': 1024}  # 裁剪前1024行
    }
)

# 文本位姿格式指定
results = evaluator.evaluate_pose_only(
    gt_pose_path="poses.txt",
    pred_pose_path="pred_dir/",
    gt_format='txt_file',
    pred_format='npz_dir',
    format_kwargs={
        'gt_kwargs': {'format_type': 'kitti'}  # KITTI格式
    }
)
```

## 💡 最佳实践

### 数据准备
1. **单位统一**: 确保深度和位姿数据单位正确（推荐使用米）
2. **格式规范**: 位姿使用cam2world格式的4x4矩阵
3. **数据验证**: 使用框架的数据验证功能检查输入质量

### 评估配置
1. **深度范围**: 根据数据集特点调整有效深度范围
2. **窗口大小**: L-ATE窗口大小可根据序列长度调整
3. **详细输出**: 首次使用时开启verbose了解处理过程

### 结果解读
1. **深度性能**: a1 > 0.8为优秀，0.6-0.8为良好
2. **位姿性能**: RMSE < 0.01m为优秀，0.01-0.05m为良好
3. **数据质量**: 注意处理统计中的有效帧比例和缩放比例

## 🐛 故障排除

### 常见问题

**1. 数据加载失败**
```
错误: 文件不存在 或 格式不支持
解决: 检查路径正确性，使用get_format_info()检测格式
```

**2. 有效像素太少**
```
错误: 有效像素数太少
解决: 调整深度范围参数，检查数据质量
```

**3. 位姿对齐失败**
```
错误: 起点对齐误差过大
解决: 检查位姿数据格式，确认为cam2world格式
```

**4. 窗口评估失败**
```
错误: 没有成功评估任何窗口
解决: 检查序列长度，调整窗口大小参数
```

## 📚 示例数据集

框架已在以下数据集上验证：
- **SCARED**: 内窥镜手术数据集
- **EndoDAC**: 内窥镜深度数据集  
- **KITTI**: 自动驾驶数据集

## 🤝 贡献

欢迎提交问题报告和改进建议！

## 📄 许可证

本项目采用MIT许可证。

---

🎉 **统一评估，标准一致，结果可信！**




#!/bin/bash

# CUT3R批量推理脚本 - Scared + StereoMIS数据集
# 作者: AI Assistant
# 日期: $(date)

echo "🚀 开始CUT3R批量推理 - Scared & StereoMIS数据集"
echo "================================================="

# 检查是否在正确的目录
if [[ ! -f "demo_online.py" ]]; then
    echo "❌ 错误：请确保在CUT3R根目录下运行此脚本"
    echo "当前目录: $(pwd)"
    echo "请运行: cd /hy-tmp/hy-tmp/CUT3R"
    exit 1
fi

# 检查conda环境
if [[ "$CONDA_DEFAULT_ENV" != "cut3r-slam" ]]; then
    echo "⚠️  警告：当前conda环境不是cut3r-slam"
    echo "请先运行: conda activate cut3r-slam"
    exit 1
fi

# 配置参数
MODEL_PATH="/hy-tmp/hy-tmp/CUT3R/src/checkpoints/train_scared_stage1_medical_adaptationV2new18/checkpoint-5.pth"
SIZE=256
VIS_THRESHOLD=1.5

# 检查模型文件是否存在
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "❌ 错误：模型文件不存在: $MODEL_PATH"
    exit 1
fi

# 询问用户要推理哪个数据集
echo "📋 请选择要推理的数据集："
echo "   1) Scared数据集 (8个序列)"
echo "   2) StereoMIS数据集 (8个序列)"
echo "   3) 两个数据集都推理 (16个序列)"
echo ""
read -p "请输入选择 (1-3): " dataset_choice

case $dataset_choice in
    1)
        echo "✅ 选择推理 Scared 数据集"
        DATASETS=("scared")
        ;;
    2)
        echo "✅ 选择推理 StereoMIS 数据集"
        DATASETS=("stereomis")
        ;;
    3)
        echo "✅ 选择推理两个数据集"
        DATASETS=("scared" "stereomis")
        ;;
    *)
        echo "❌ 无效选择，默认推理 Scared 数据集"
        DATASETS=("scared")
        ;;
esac

# Scared数据集配置
declare -a SCARED_SEQUENCES=(
    "scared:dataset8:keyframe0:Scared8_0_Left_Images:80:/hy-tmp/hy-tmp/CUT3R/eval/eval_data/test:/hy-tmp/hy-tmp/CUT3R/eval/cut3r-complete-inference/scared"
    "scared:dataset8:keyframe1:Scared8_1_Left_Images:81:/hy-tmp/hy-tmp/CUT3R/eval/eval_data/test:/hy-tmp/hy-tmp/CUT3R/eval/cut3r-complete-inference/scared"
    "scared:dataset8:keyframe2:Scared8_2_Left_Images:82:/hy-tmp/hy-tmp/CUT3R/eval/eval_data/test:/hy-tmp/hy-tmp/CUT3R/eval/cut3r-complete-inference/scared"
    "scared:dataset8:keyframe3:Scared8_3_Left_Images:83:/hy-tmp/hy-tmp/CUT3R/eval/eval_data/test:/hy-tmp/hy-tmp/CUT3R/eval/cut3r-complete-inference/scared"
    "scared:dataset9:keyframe0:Scared9_0_Left_Images:90:/hy-tmp/hy-tmp/CUT3R/eval/eval_data/test:/hy-tmp/hy-tmp/CUT3R/eval/cut3r-complete-inference/scared"
    "scared:dataset9:keyframe1:Scared9_1_Left_Images:91:/hy-tmp/hy-tmp/CUT3R/eval/eval_data/test:/hy-tmp/hy-tmp/CUT3R/eval/cut3r-complete-inference/scared"
    "scared:dataset9:keyframe2:Scared9_2_Left_Images:92:/hy-tmp/hy-tmp/CUT3R/eval/eval_data/test:/hy-tmp/hy-tmp/CUT3R/eval/cut3r-complete-inference/scared"
    "scared:dataset9:keyframe3:Scared9_3_Left_Images:93:/hy-tmp/hy-tmp/CUT3R/eval/eval_data/test:/hy-tmp/hy-tmp/CUT3R/eval/cut3r-complete-inference/scared"
)

# StereoMIS数据集配置
declare -a STEREOMIS_SEQUENCES=(
    "stereomis:StereoMIS_P2_1_2:rgb:12:/hy-tmp/hy-tmp/CUT3R/dataset/stereomis_testset:/hy-tmp/hy-tmp/CUT3R/eval/cut3r-complete-inference/stereomis"
    "stereomis:StereoMIS_P2_4_1:rgb:41:/hy-tmp/hy-tmp/CUT3R/dataset/stereomis_testset:/hy-tmp/hy-tmp/CUT3R/eval/cut3r-complete-inference/stereomis"
    "stereomis:StereoMIS_P2_4_3:rgb:43:/hy-tmp/hy-tmp/CUT3R/dataset/stereomis_testset:/hy-tmp/hy-tmp/CUT3R/eval/cut3r-complete-inference/stereomis"
    "stereomis:StereoMIS_P2_5_1:rgb:51:/hy-tmp/hy-tmp/CUT3R/dataset/stereomis_testset:/hy-tmp/hy-tmp/CUT3R/eval/cut3r-complete-inference/stereomis"
    "stereomis:StereoMIS_P2_5_3:rgb:53:/hy-tmp/hy-tmp/CUT3R/dataset/stereomis_testset:/hy-tmp/hy-tmp/CUT3R/eval/cut3r-complete-inference/stereomis"
    "stereomis:StereoMIS_P2_7_2:rgb:72:/hy-tmp/hy-tmp/CUT3R/dataset/stereomis_testset:/hy-tmp/hy-tmp/CUT3R/eval/cut3r-complete-inference/stereomis"
    "stereomis:StereoMIS_P2_8_1:rgb:81:/hy-tmp/hy-tmp/CUT3R/dataset/stereomis_testset:/hy-tmp/hy-tmp/CUT3R/eval/cut3r-complete-inference/stereomis"
    "stereomis:StereoMIS_P2_8_2:rgb:82:/hy-tmp/hy-tmp/CUT3R/dataset/stereomis_testset:/hy-tmp/hy-tmp/CUT3R/eval/cut3r-complete-inference/stereomis"
)

# 根据用户选择构建序列列表
declare -a ALL_SEQUENCES=()
for dataset in "${DATASETS[@]}"; do
    if [[ "$dataset" == "scared" ]]; then
        ALL_SEQUENCES+=("${SCARED_SEQUENCES[@]}")
    elif [[ "$dataset" == "stereomis" ]]; then
        ALL_SEQUENCES+=("${STEREOMIS_SEQUENCES[@]}")
    fi
done

# 初始化计数器
TOTAL_SEQUENCES=${#ALL_SEQUENCES[@]}
SUCCESSFUL_COUNT=0
FAILED_COUNT=0
declare -a SUCCESSFUL_SEQUENCES=()
declare -a FAILED_SEQUENCES=()

echo "📊 推理配置："
echo "   模型路径: $MODEL_PATH"
echo "   图像尺寸: $SIZE"
echo "   可视化阈值: $VIS_THRESHOLD"
echo "   总序列数: $TOTAL_SEQUENCES"
echo ""

# 开始推理循环
for i in "${!ALL_SEQUENCES[@]}"; do
    # 解析序列信息 - 新格式: dataset_type:seq_name:sub_dir:output_suffix:base_input_path:base_output_path
    IFS=':' read -r dataset_type seq_name sub_dir output_suffix base_input_path base_output_path <<< "${ALL_SEQUENCES[$i]}"
    
    if [[ "$dataset_type" == "scared" ]]; then
        SEQ_PATH="$base_input_path/$seq_name/$sub_dir"
        SEQUENCE_NAME="${seq_name}_${sub_dir}"
    elif [[ "$dataset_type" == "stereomis" ]]; then
        SEQ_PATH="$base_input_path/$seq_name/$sub_dir"
        SEQUENCE_NAME="$seq_name"
    fi
    
    OUTPUT_DIR="$base_output_path/$output_suffix"
    
    echo "🔍 推理序列 $((i+1))/$TOTAL_SEQUENCES: $SEQUENCE_NAME"
    echo "   输入路径: $SEQ_PATH"
    echo "   输出路径: $OUTPUT_DIR"
    
    # 检查输入路径是否存在
    if [[ ! -d "$SEQ_PATH" ]]; then
        echo "   ❌ 输入路径不存在，跳过此序列"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_SEQUENCES+=("$SEQUENCE_NAME (路径不存在)")
        echo ""
        continue
    fi
    
    # 检查输出目录是否已存在结果
    if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ]]; then
        echo "   ⚠️  输出目录已存在且非空，是否跳过？(y/n)"
        read -t 10 -n 1 skip_existing
        echo ""
        if [[ "$skip_existing" == "y" || "$skip_existing" == "Y" ]]; then
            echo "   ⏭️  跳过已存在的结果"
            SUCCESSFUL_COUNT=$((SUCCESSFUL_COUNT + 1))
            SUCCESSFUL_SEQUENCES+=("$SEQUENCE_NAME (已存在)")
            echo ""
            continue
        fi
    fi
    
    # 创建输出目录
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$(dirname "$base_output_path")"
    
    # 构建推理命令
    INFERENCE_CMD="python demo_online.py \
        --model_path '$MODEL_PATH' \
        --size $SIZE \
        --seq_path '$SEQ_PATH' \
        --vis_threshold $VIS_THRESHOLD \
        --output_dir '$OUTPUT_DIR'"
    
    echo "   🚀 开始推理..."
    echo "   命令: $INFERENCE_CMD"
    
    # 记录开始时间
    START_TIME=$(date +%s)
    
    # 执行推理命令
    if eval "$INFERENCE_CMD"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "   ✅ 推理成功 (耗时: ${DURATION}s)"
        SUCCESSFUL_COUNT=$((SUCCESSFUL_COUNT + 1))
        SUCCESSFUL_SEQUENCES+=("$SEQUENCE_NAME")
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "   ❌ 推理失败 (耗时: ${DURATION}s)"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_SEQUENCES+=("$SEQUENCE_NAME")
    fi
    
    echo ""
done

# 生成推理报告
echo "=============================================="
echo "🎉 CUT3R批量推理完成！"
echo "=============================================="
echo "📊 推理统计："
echo "   总序列数: $TOTAL_SEQUENCES"
echo "   成功序列: $SUCCESSFUL_COUNT"
echo "   失败序列: $FAILED_COUNT"
echo "   成功率: $(( SUCCESSFUL_COUNT * 100 / TOTAL_SEQUENCES ))%"

if [[ $SUCCESSFUL_COUNT -gt 0 ]]; then
    echo ""
    echo "✅ 成功序列列表："
    for seq in "${SUCCESSFUL_SEQUENCES[@]}"; do
        echo "   - $seq"
    done
fi

if [[ $FAILED_COUNT -gt 0 ]]; then
    echo ""
    echo "❌ 失败序列列表："
    for seq in "${FAILED_SEQUENCES[@]}"; do
        echo "   - $seq"
    done
fi

echo ""
echo "📁 推理结果保存在:"
for dataset in "${DATASETS[@]}"; do
    if [[ "$dataset" == "scared" ]]; then
        echo "   Scared: /hy-tmp/hy-tmp/CUT3R/eval/cut3r-complete-inference/scared"
    elif [[ "$dataset" == "stereomis" ]]; then
        echo "   StereoMIS: /hy-tmp/hy-tmp/CUT3R/eval/cut3r-complete-inference/stereomis"
    fi
done
echo "=============================================="

# 生成推理报告文件
REPORT_DIR="/hy-tmp/hy-tmp/CUT3R/eval/cut3r-complete-inference"
mkdir -p "$REPORT_DIR"
REPORT_FILE="$REPORT_DIR/batch_inference_report_$(date +%Y%m%d_%H%M%S).txt"
cat > "$REPORT_FILE" << EOF
CUT3R批量推理报告
生成时间: $(date)
==================================================

推理配置:
- 模型路径: $MODEL_PATH
- 图像尺寸: $SIZE
- 可视化阈值: $VIS_THRESHOLD
- 总序列数: $TOTAL_SEQUENCES

推理统计:
- 成功序列: $SUCCESSFUL_COUNT
- 失败序列: $FAILED_COUNT
- 成功率: $(( SUCCESSFUL_COUNT * 100 / TOTAL_SEQUENCES ))%

成功序列:
$(printf '%s\n' "${SUCCESSFUL_SEQUENCES[@]}")

失败序列:
$(printf '%s\n' "${FAILED_SEQUENCES[@]}")

==================================================
EOF

echo "📄 详细报告已保存: $REPORT_FILE"

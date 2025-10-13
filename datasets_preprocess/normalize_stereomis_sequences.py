#!/usr/bin/env python3
"""
Normalize StereoMIS sequences so that each sequence's rgb/depth/cam files are
compactly reindexed to start from 000000, preserving one-to-one correspondence.

- Scans all sequences under a root directory (each sequence contains rgb/, depth/, cam/)
- Verifies that within a sequence, the three modalities share the exact same frame index set
- Dry-run mode prints a concise report and sample mappings (no changes made)
- Apply mode performs a safe in-place two-phase rename per modality to avoid conflicts

Usage examples:
  Dry-run (recommended first):
    python normalize_stereomis_sequences.py \
      --root_dir /path/to/CUT3R/dataset/processed_stereomis

  Apply (in-place rename):
    python normalize_stereomis_sequences.py \
      --root_dir /path/to/CUT3R/dataset/processed_stereomis --apply

Notes:
- Assumes fixed extensions: rgb: .jpg, depth: .npy, cam: .npz
- Two-phase rename uses a temporary prefix that should not collide: __tmpalign__
"""

import argparse
import os
import re
import sys
from typing import Dict, List, Set, Tuple

RGB_DIR = "rgb"
DEPTH_DIR = "depth"
CAM_DIR = "cam"

EXTENSIONS = {
    RGB_DIR: ".jpg",
    DEPTH_DIR: ".npy",
    CAM_DIR: ".npz",
}

FRAME_RE = re.compile(r"^(\d{6})(\..+)$")
TMP_PREFIX = "__tmpalign__"


def list_numeric_frames(dir_path: str, expected_ext: str) -> Tuple[List[int], Dict[int, str]]:
    """List 6-digit numeric frames in a directory with a specific extension.

    Returns:
        sorted_indices: sorted list of integer frame indices
        index_to_name: mapping from index to filename
    """
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Missing directory: {dir_path}")

    indices: List[int] = []
    index_to_name: Dict[int, str] = {}

    for name in os.listdir(dir_path):
        m = FRAME_RE.match(name)
        if not m:
            continue
        index_str, ext = m.groups()
        if ext != expected_ext:
            continue
        try:
            idx = int(index_str)
        except ValueError:
            continue
        # Ignore temporary files from a previous interrupted run
        if name.startswith(TMP_PREFIX):
            continue
        indices.append(idx)
        index_to_name[idx] = name

    indices.sort()
    return indices, index_to_name


def is_already_compact(indices: List[int]) -> bool:
    if not indices:
        return True
    return indices[0] == 0 and indices == list(range(len(indices)))


def print_sequence_summary(seq_name: str, frames_by_modality: Dict[str, List[int]]):
    def mod_stats(mod: str) -> Tuple[int, int, int]:
        fr = frames_by_modality.get(mod, [])
        if not fr:
            return 0, -1, -1
        return len(fr), fr[0], fr[-1]

    rgb_n, rgb_min, rgb_max = mod_stats(RGB_DIR)
    d_n, d_min, d_max = mod_stats(DEPTH_DIR)
    c_n, c_min, c_max = mod_stats(CAM_DIR)

    same_sets = (
        set(frames_by_modality.get(RGB_DIR, []))
        == set(frames_by_modality.get(DEPTH_DIR, []))
        == set(frames_by_modality.get(CAM_DIR, []))
    )

    print("=" * 80)
    print(f"📁 序列: {seq_name}")
    print(f"  rgb:   count={rgb_n}, min={rgb_min:06d} max={rgb_max:06d}" if rgb_n else "  rgb:   count=0")
    print(f"  depth: count={d_n}, min={d_min:06d} max={d_max:06d}" if d_n else "  depth: count=0")
    print(f"  cam:   count={c_n}, min={c_min:06d} max={c_max:06d}" if c_n else "  cam:   count=0")
    print(f"  三类帧集合一致: {'✅' if same_sets else '❌'}")
    if same_sets and rgb_n:
        print(f"  归零后: 000000 ~ {rgb_n-1:06d} (共 {rgb_n} 帧)")


def print_sample_mapping(seq_name: str, frames_sorted: List[int]):
    if not frames_sorted:
        return
    n = len(frames_sorted)
    head = min(5, n)
    tail = min(5, n - head)
    samples: List[Tuple[int, int]] = []
    # First head
    for i in range(head):
        samples.append((frames_sorted[i], i))
    # Last tail
    for i in range(n - tail, n):
        samples.append((frames_sorted[i], i))

    print(f"  映射示例 (old -> new):")
    for old_idx, new_idx in samples:
        print(f"    {old_idx:06d} -> {new_idx:06d}")


def two_phase_rename(dir_path: str, frames_sorted: List[int], index_to_name: Dict[int, str], ext: str):
    """Perform in-place two-phase rename to compact indices to 000000..N-1.

    Phase A: rename real files to temporary names using TMP_PREFIX + new_index
    Phase B: rename temporary files to final names new_index.ext
    """
    if not frames_sorted:
        return

    # Phase A: to temporary names
    for new_idx, old_idx in enumerate(frames_sorted):
        old_name = index_to_name[old_idx]
        old_path = os.path.join(dir_path, old_name)
        tmp_name = f"{TMP_PREFIX}{new_idx:06d}{ext}"
        tmp_path = os.path.join(dir_path, tmp_name)
        if os.path.exists(tmp_path):
            raise RuntimeError(f"临时目标已存在(可能之前中断): {tmp_path}")
        os.rename(old_path, tmp_path)

    # Phase B: to final names
    for new_idx, _old_idx in enumerate(frames_sorted):
        tmp_name = f"{TMP_PREFIX}{new_idx:06d}{ext}"
        tmp_path = os.path.join(dir_path, tmp_name)
        final_name = f"{new_idx:06d}{ext}"
        final_path = os.path.join(dir_path, final_name)
        # If final exists but not from our temp phase, it's unexpected
        if os.path.exists(final_path):
            # This could only reasonably happen if the directory was already compact
            # and someone ran apply again without Phase A. To be safe, error out.
            raise RuntimeError(f"最终目标已存在，放弃以避免覆盖: {final_path}")
        os.rename(tmp_path, final_path)


def process_sequence(seq_dir: str, dry_run: bool) -> Tuple[str, bool, str]:
    """Process a single sequence directory.

    Returns: (seq_name, success, message)
    """
    seq_name = os.path.basename(seq_dir.rstrip(os.sep))
    rgb_dir = os.path.join(seq_dir, RGB_DIR)
    depth_dir = os.path.join(seq_dir, DEPTH_DIR)
    cam_dir = os.path.join(seq_dir, CAM_DIR)

    for d in (rgb_dir, depth_dir, cam_dir):
        if not os.path.isdir(d):
            return seq_name, False, f"缺少必要子目录: {d}"

    rgb_frames, rgb_map = list_numeric_frames(rgb_dir, EXTENSIONS[RGB_DIR])
    d_frames, d_map = list_numeric_frames(depth_dir, EXTENSIONS[DEPTH_DIR])
    c_frames, c_map = list_numeric_frames(cam_dir, EXTENSIONS[CAM_DIR])

    frames_equal = set(rgb_frames) == set(d_frames) == set(c_frames)
    frames_sorted = rgb_frames  # if equal, any of them is fine; use rgb order

    print_sequence_summary(seq_name, {RGB_DIR: rgb_frames, DEPTH_DIR: d_frames, CAM_DIR: c_frames})

    if not frames_equal:
        return seq_name, False, "三类帧集合不一致，已跳过（请检查数据）"

    if not frames_sorted:
        return seq_name, True, "空序列，跳过"

    print_sample_mapping(seq_name, frames_sorted)

    if dry_run:
        already = is_already_compact(frames_sorted)
        return seq_name, True, ("已是从000000开始且连续，Dry-Run 仅报告" if already else "Dry-Run 预演完成，无改动")

    # Apply mode: in-place two-phase per modality
    try:
        for sub, mapping, frames, ext in [
            (rgb_dir, rgb_map, rgb_frames, EXTENSIONS[RGB_DIR]),
            (depth_dir, d_map, d_frames, EXTENSIONS[DEPTH_DIR]),
            (cam_dir, c_map, c_frames, EXTENSIONS[CAM_DIR]),
        ]:
            if is_already_compact(frames):
                # To keep idempotence, skip this modality if already compact
                continue
            two_phase_rename(sub, frames_sorted, mapping, ext)
        return seq_name, True, "改名完成"
    except Exception as e:
        return seq_name, False, f"改名失败: {e}"


def find_sequences(root_dir: str) -> List[str]:
    seqs: List[str] = []
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"输入根目录不存在: {root_dir}")
    for name in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, name)
        if os.path.isdir(path):
            seqs.append(path)
    return seqs


def main():
    parser = argparse.ArgumentParser(description="将每个StereoMIS序列紧凑重排并起点归零(000000)")
    parser.add_argument("--root_dir", type=str, required=True, help="包含多个序列的根目录(每个序列含 rgb/depth/cam)")
    parser.add_argument("--apply", action="store_true", help="执行原地改名(两阶段)，默认只Dry-Run")
    args = parser.parse_args()

    root = args.root_dir
    dry_run = not args.apply

    print("=" * 80)
    print("🔍 开始扫描序列" + (" (Dry-Run)" if dry_run else " (Apply)"))
    print(f"根目录: {root}")

    sequences = find_sequences(root)
    if not sequences:
        print("未发现任何序列目录。")
        sys.exit(1)

    ok_cnt = 0
    fail_cnt = 0

    for seq_dir in sequences:
        seq_name, ok, msg = process_sequence(seq_dir, dry_run=dry_run)
        status = "✅" if ok else "❌"
        print(f"  结果: {status} {seq_name}: {msg}")
        if ok:
            ok_cnt += 1
        else:
            fail_cnt += 1

    print("=" * 80)
    print(f"完成: 成功 {ok_cnt} 个序列, 失败 {fail_cnt} 个序列")
    if dry_run:
        print("提示: 使用 --apply 执行原地改名 (两阶段)，建议先备份或确认Dry-Run结果。")


if __name__ == "__main__":
    main()


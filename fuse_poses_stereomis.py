import os
import numpy as np
import glob
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from tqdm import tqdm
import argparse

def get_paths_stereomis(base_output_dir, cut3r2_base_dir, stereomis_dataset_base, sequence_name):
    """Generates all necessary paths for a StereoMIS sequence."""
    paths = {
        "cut3r1_poses_dir": os.path.join(base_output_dir, sequence_name, "anchor_poses_cut3r1", "camera"),
        "cut3r2_chunks_dir": os.path.join(cut3r2_base_dir, sequence_name, "chunk_outputs"),
        "final_output_dir": os.path.join(base_output_dir, sequence_name, "final_corrected_poses", "camera"),
        "image_dir": os.path.join(stereomis_dataset_base, sequence_name, "rgb"),
    }
    return paths

def load_pose(file_path):
    """Loads a pose matrix from an .npz file."""
    try:
        with np.load(file_path) as data:
            if 'pose' not in data:
                raise KeyError(f"Key 'pose' not found in {file_path}")
            if 'intrinsics' not in data:
                 raise KeyError(f"Key 'intrinsics' not found in {file_path}")
            return data['pose'], data['intrinsics']
    except Exception as e:
        print(f"Error loading pose from {file_path}: {e}")
        return None, None

def save_pose(file_path, pose, intrinsics):
    """Saves a pose matrix and intrinsics to an .npz file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.savez(file_path, pose=pose, intrinsics=intrinsics)

def main():
    parser = argparse.ArgumentParser(description="Fuse poses for StereoMIS sequences.")
    parser.add_argument('--sequence_name', type=str, required=True, help='Name of the sequence to process (e.g., StereoMIS_P2_1_2)')
    parser.add_argument('--base_output_dir', type=str, default="/hy-tmp/hy-tmp/CUT3R/eval/cut3r1+cut3r2_stereomis", help='Base directory for fused results.')
    parser.add_argument('--cut3r2_base_dir', type=str, default="/hy-tmp/hy-tmp/CUT3R/eval/cut3r+loss/stereomis_inference_results", help='Base directory for cut3r2 chunks.')
    parser.add_argument('--stereomis_dataset_base', type=str, default="/hy-tmp/hy-tmp/CUT3R/dataset/stereomis_testset", help='Base directory of the StereoMIS dataset.')
    args = parser.parse_args()

    paths = get_paths_stereomis(args.base_output_dir, args.cut3r2_base_dir, args.stereomis_dataset_base, args.sequence_name)
    cut3r1_poses_dir = paths["cut3r1_poses_dir"]
    cut3r2_chunks_dir = paths["cut3r2_chunks_dir"]
    final_output_dir = paths["final_output_dir"]
    image_dir = paths["image_dir"]

    all_img_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not all_img_paths:
        print(f"FATAL: No images found in {image_dir} for sequence {args.sequence_name}")
        return

    num_frames = len(all_img_paths)
    anchor_step = 16
    anchor_mapping = list(range(0, num_frames, anchor_step))
    if num_frames - 1 not in anchor_mapping:
        anchor_mapping.append(num_frames - 1)
    anchor_mapping = sorted(list(set(anchor_mapping)))

    print(f"Processing StereoMIS sequence: {args.sequence_name}")
    print(f"Found {num_frames} total frames. Using {len(anchor_mapping)} anchors.")
    
    print("Loading cut3r1 anchor poses...")
    cut3r1_poses = {}
    cut3r1_intrinsics = {}
    cut3r1_files = sorted(glob.glob(os.path.join(cut3r1_poses_dir, "*.npz")))
    
    if len(cut3r1_files) != len(anchor_mapping):
        print(f"Error: Mismatch between number of anchor poses ({len(cut3r1_files)}) and anchor mapping ({len(anchor_mapping)}).")
        return
        
    for i, file_path in enumerate(cut3r1_files):
        original_frame_idx = anchor_mapping[i]
        pose, intrinsics = load_pose(file_path)
        if pose is not None:
            cut3r1_poses[original_frame_idx] = pose
            cut3r1_intrinsics[original_frame_idx] = intrinsics
            
    print(f"Loaded {len(cut3r1_poses)} anchor poses.")

    chunk_dirs = sorted(glob.glob(os.path.join(cut3r2_chunks_dir, "chunk_*")))
    if not chunk_dirs:
        print(f"FATAL: No chunk directories found in {cut3r2_chunks_dir}")
        return

    for chunk_dir in tqdm(chunk_dirs, desc=f"Fusing {args.sequence_name}"):
        chunk_name = os.path.basename(chunk_dir)
        parts = chunk_name.split('_')
        start_frame, end_frame = map(int, parts[-1].split('-'))

        anchor_start = start_frame
        anchor_end = end_frame

        if anchor_start not in cut3r1_poses or anchor_end not in cut3r1_poses:
            print(f"Warning: Missing anchor poses for chunk {chunk_name}. Skipping.")
            continue
            
        P1_start_w = cut3r1_poses[anchor_start]
        P1_end_w = cut3r1_poses[anchor_end]

        P2_start_c, intrinsics_start = load_pose(os.path.join(chunk_dir, "camera", f"{(start_frame - start_frame):06d}.npz"))
        P2_end_c, _ = load_pose(os.path.join(chunk_dir, "camera", f"{(end_frame - start_frame):06d}.npz"))
        
        if P2_start_c is None or P2_end_c is None:
            print(f"Warning: Could not load start/end poses for chunk {chunk_name}. Skipping.")
            continue

        T_align_start = P1_start_w @ np.linalg.inv(P2_start_c)
        P2_end_c_aligned = T_align_start @ P2_end_c
        T_error_correction = P1_end_w @ np.linalg.inv(P2_end_c_aligned)
        
        R_error = T_error_correction[:3, :3]
        t_error = T_error_correction[:3, 3]
        
        key_rots = R.from_matrix([np.eye(3), R_error])
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rots)

        num_frames_in_chunk = end_frame - start_frame
        if num_frames_in_chunk == 0:
            save_pose(os.path.join(final_output_dir, f"{start_frame:06d}.npz"), P1_start_w, intrinsics_start)
            continue

        for i in range(start_frame, end_frame + 1):
            P2_i_c, intrinsics_i = load_pose(os.path.join(chunk_dir, "camera", f"{(i - start_frame):06d}.npz"))
            if P2_i_c is None:
                continue

            P2_i_aligned = T_align_start @ P2_i_c
            alpha = (i - start_frame) / num_frames_in_chunk
            interp_rot_matrix = slerp([alpha]).as_matrix()[0]
            interp_trans_vector = alpha * t_error
            
            T_corr_i = np.eye(4)
            T_corr_i[:3, :3] = interp_rot_matrix
            T_corr_i[:3, 3] = interp_trans_vector
            
            P_final_i = T_corr_i @ P2_i_aligned
            save_pose(os.path.join(final_output_dir, f"{i:06d}.npz"), P_final_i, intrinsics_i)

    print("Fusion complete. Corrected poses saved to:", final_output_dir)

if __name__ == "__main__":
    main()

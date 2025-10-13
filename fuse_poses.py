import os
import numpy as np
import glob
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from tqdm import tqdm
import argparse

def get_paths(base_output_dir, cut3r2_base_dir, sequence_name):
    """Generates all necessary paths based on the sequence name."""
    
    # e.g., dual_evaluation_dataset9_keyframe3 -> dataset9, keyframe3
    parts = sequence_name.split('_')
    dataset_name = parts[2] # dataset9
    keyframe_name = parts[3] # keyframe3
    
    paths = {
        "cut3r1_poses_dir": os.path.join(base_output_dir, sequence_name, "anchor_poses_cut3r1", "camera"),
        "cut3r2_chunks_dir": os.path.join(cut3r2_base_dir, sequence_name, "chunk_outputs"),
        "final_output_dir": os.path.join(base_output_dir, sequence_name, "final_corrected_poses", "camera"),
        "image_dir": f"/hy-tmp/hy-tmp/CUT3R/eval/eval_data/test/{dataset_name}/{keyframe_name}/Scared{dataset_name.replace('dataset','')}_{keyframe_name.replace('keyframe','')}_Left_Images",
    }
    return paths

def load_pose(file_path):
    """Loads a pose matrix from an .npz file."""
    try:
        with np.load(file_path) as data:
            # Ensure keys 'pose' and 'intrinsics' exist, matching demo_online.py output
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
    parser = argparse.ArgumentParser(description="Fuse poses from cut3r1 (anchors) and cut3r2 (chunks).")
    parser.add_argument('--sequence_name', type=str, required=True, help='Name of the sequence to process (e.g., dual_evaluation_dataset9_keyframe3)')
    parser.add_argument('--base_output_dir', type=str, default="/hy-tmp/hy-tmp/CUT3R/eval/cut3r1+cut3r2", help='Base directory where all batch processing results are stored.')
    parser.add_argument('--cut3r2_base_dir', type=str, default="/hy-tmp/hy-tmp/CUT3R/eval/cut3r+loss", help='Base directory for the cut3r2 chunk results.')
    args = parser.parse_args()

    # --- Configuration ---
    paths = get_paths(args.base_output_dir, args.cut3r2_base_dir, args.sequence_name)
    cut3r1_poses_dir = paths["cut3r1_poses_dir"]
    cut3r2_chunks_dir = paths["cut3r2_chunks_dir"]
    final_output_dir = paths["final_output_dir"]
    image_dir = paths["image_dir"]

    # --- Step 1: Create Anchor Mapping and Load cut3r1 Poses ---
    all_img_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    if not all_img_paths:
        print(f"FATAL: No images found in {image_dir} for sequence {args.sequence_name}")
        return

    num_frames = len(all_img_paths)
    anchor_step = 16
    anchor_mapping = list(range(0, num_frames, anchor_step))
    if num_frames - 1 not in anchor_mapping:
        anchor_mapping.append(num_frames - 1)
    anchor_mapping = sorted(list(set(anchor_mapping)))

    print(f"Processing sequence: {args.sequence_name}")
    print(f"Found {num_frames} total frames. Using {len(anchor_mapping)} anchors.")
    
    print("Loading cut3r1 anchor poses...")
    cut3r1_poses = {}
    cut3r1_intrinsics = {}
    # The output files are named 000000.npz, 000001.npz, etc.
    cut3r1_files = sorted(glob.glob(os.path.join(cut3r1_poses_dir, "*.npz")))
    
    if len(cut3r1_files) != len(anchor_mapping):
        print(f"Error: Mismatch between number of anchor poses ({len(cut3r1_files)}) and anchor mapping ({len(anchor_mapping)}).")
        print("Anchor mapping:", anchor_mapping)
        return
        
    for i, file_path in enumerate(cut3r1_files):
        original_frame_idx = anchor_mapping[i]
        pose, intrinsics = load_pose(file_path)
        if pose is not None:
            cut3r1_poses[original_frame_idx] = pose
            cut3r1_intrinsics[original_frame_idx] = intrinsics
            
    print(f"Loaded {len(cut3r1_poses)} anchor poses.")

    # --- Step 2: Iterate Through Chunks and Fuse Poses ---
    chunk_dirs = sorted(glob.glob(os.path.join(cut3r2_chunks_dir, "chunk_*")))

    if not chunk_dirs:
        print(f"FATAL: No chunk directories found in {cut3r2_chunks_dir}")
        return

    for chunk_dir in tqdm(chunk_dirs, desc=f"Fusing {args.sequence_name}"):
        chunk_name = os.path.basename(chunk_dir)
        parts = chunk_name.split('_')
        start_frame, end_frame = map(int, parts[-1].split('-'))

        # Get the anchor poses for the start and end of the chunk
        # Find the closest available anchor >= start and <= end
        anchor_start = start_frame
        anchor_end = end_frame

        if anchor_start not in cut3r1_poses or anchor_end not in cut3r1_poses:
            print(f"Warning: Missing anchor poses for chunk {chunk_name}. Skipping.")
            continue
            
        P1_start_w = cut3r1_poses[anchor_start]
        P1_end_w = cut3r1_poses[anchor_end]

        # Load cut3r2 poses for the current chunk using RELATIVE paths
        P2_start_c, intrinsics_start = load_pose(os.path.join(chunk_dir, "camera", f"{(start_frame - start_frame):06d}.npz"))
        P2_end_c, _ = load_pose(os.path.join(chunk_dir, "camera", f"{(end_frame - start_frame):06d}.npz"))
        
        if P2_start_c is None or P2_end_c is None:
            print(f"Warning: Could not load start/end poses for chunk {chunk_name}. Skipping.")
            continue

        # --- Step 3: Calculate Transformations ---
        # Transformation to align the start of the chunk
        # T_align_start @ P2_start_c = P1_start_w
        T_align_start = P1_start_w @ np.linalg.inv(P2_start_c)

        # Apply start alignment to the end pose of the chunk
        P2_end_c_aligned = T_align_start @ P2_end_c

        # Transformation for the remaining error at the end
        # T_error_correction @ P2_end_c_aligned = P1_end_w
        T_error_correction = P1_end_w @ np.linalg.inv(P2_end_c_aligned)
        
        # Decompose the error correction into rotation and translation
        R_error = T_error_correction[:3, :3]
        t_error = T_error_correction[:3, 3]
        
        # Use Slerp for smooth rotation interpolation
        key_rots = R.from_matrix([np.eye(3), R_error])
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rots)

        # --- Step 4: Apply Correction to each frame in the chunk ---
        num_frames_in_chunk = end_frame - start_frame
        if num_frames_in_chunk == 0: # handle single frame chunks
            save_pose(os.path.join(final_output_dir, f"{start_frame:06d}.npz"), P1_start_w, intrinsics_start)
            continue

        for i in range(start_frame, end_frame + 1):
            # Load pose using RELATIVE path
            P2_i_c, intrinsics_i = load_pose(os.path.join(chunk_dir, "camera", f"{(i - start_frame):06d}.npz"))
            if P2_i_c is None:
                continue

            # First, align the entire chunk to the start anchor
            P2_i_aligned = T_align_start @ P2_i_c
            
            # Calculate interpolation factor
            alpha = (i - start_frame) / num_frames_in_chunk
            
            # Interpolate rotation and translation
            interp_rot_matrix = slerp([alpha]).as_matrix()[0]
            interp_trans_vector = alpha * t_error
            
            # Compose the interpolated correction transform
            T_corr_i = np.eye(4)
            T_corr_i[:3, :3] = interp_rot_matrix
            T_corr_i[:3, 3] = interp_trans_vector
            
            # Apply the interpolated correction
            P_final_i = T_corr_i @ P2_i_aligned
            
            # Save the final corrected pose
            save_pose(os.path.join(final_output_dir, f"{i:06d}.npz"), P_final_i, intrinsics_i)

    print("Fusion complete. Corrected poses saved to:", final_output_dir)

if __name__ == "__main__":
    main()

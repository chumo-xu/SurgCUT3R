import os
import numpy as np
import glob
from tqdm import tqdm
import argparse

def get_paths(base_output_dir, cut3r2_base_dir, final_eval_dir, sequence_name):
    """Generates all necessary paths for formatting, based on the sequence name."""
    paths = {
        # Input for cut3r2 stitching
        "cut3r2_chunks_dir": os.path.join(cut3r2_base_dir, sequence_name, "chunk_outputs"),
        # Input for packaging fused poses
        "fused_poses_dir": os.path.join(base_output_dir, sequence_name, "final_corrected_poses", "camera"),
        # Final output files
        "cut3r2_output_file": os.path.join(final_eval_dir, sequence_name, "cut3r2.npz"),
        "fused_output_file": os.path.join(final_eval_dir, sequence_name, "cut3r1+cut3r2.npz"),
    }
    # Create the specific output directory for the sequence
    os.makedirs(os.path.join(final_eval_dir, sequence_name), exist_ok=True)
    return paths

def load_pose_from_npz(file_path):
    """Loads a pose matrix from an .npz file."""
    try:
        with np.load(file_path) as data:
            if 'pose' not in data:
                raise KeyError(f"Key 'pose' not found in {file_path}")
            return data['pose']
    except Exception as e:
        print(f"Error loading pose from {file_path}: {e}")
        return None

def process_cut3r2_chunks(chunks_dir, output_file):
    """
    Processes cut3r2 chunk outputs, stitches them together by aligning anchors,
    and saves the result as a single .npz file.
    """
    print("--- Processing cut3r2 chunks ---")
    chunk_dirs = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*")))
    
    global_poses = {}
    last_anchor_pose_global = np.eye(4) # Initialize with identity

    for i, chunk_dir in enumerate(tqdm(chunk_dirs, desc="Stitching cut3r2 chunks")):
        chunk_name = os.path.basename(chunk_dir)
        parts = chunk_name.split('_')
        start_frame, end_frame = map(int, parts[-1].split('-'))

        # Load all poses within the current chunk
        local_poses = {}
        for frame_idx in range(start_frame, end_frame + 1):
            relative_frame_idx = frame_idx - start_frame
            pose_file = os.path.join(chunk_dir, "camera", f"{relative_frame_idx:06d}.npz")
            pose = load_pose_from_npz(pose_file)
            if pose is not None:
                local_poses[frame_idx] = pose

        if not local_poses:
            print(f"Warning: No poses found in chunk {chunk_name}. Skipping.")
            continue

        # For the first chunk, the global poses are just the local poses.
        if i == 0:
            global_poses.update(local_poses)
            last_anchor_pose_global = global_poses[end_frame]
        else:
            # For subsequent chunks, calculate the transformation
            # T_align @ local_poses[start_frame] = last_anchor_pose_global
            current_anchor_pose_local = local_poses[start_frame]
            T_align = last_anchor_pose_global @ np.linalg.inv(current_anchor_pose_local)
            
            # Apply transformation to all poses in the current chunk
            for frame_idx, pose in local_poses.items():
                global_poses[frame_idx] = T_align @ pose
            
            # Update the last anchor pose for the next iteration
            last_anchor_pose_global = global_poses[end_frame]
    
    # Save the stitched trajectory
    # The evaluator expects a single array of shape (N, 4, 4)
    sorted_poses = [global_poses[i] for i in sorted(global_poses.keys())]
    stacked_poses = np.stack(sorted_poses, axis=0)
    
    np.savez(output_file, data=stacked_poses)
    print(f"Successfully saved stitched cut3r2 poses to {output_file}")
    print(f"Final array shape: {stacked_poses.shape}")


def package_fused_poses(fused_poses_dir, output_file):
    """
    Packages the final fused poses into a single .npz file for evaluation.
    """
    print("\n--- Packaging fused cut3r1+cut3r2 poses ---")
    pose_files = sorted(glob.glob(os.path.join(fused_poses_dir, "*.npz")))
    
    packaged_poses_dict = {}
    for pose_file in tqdm(pose_files, desc="Packaging fused poses"):
        frame_idx_str = os.path.basename(pose_file).split('.')[0]
        frame_idx = int(frame_idx_str)
        pose = load_pose_from_npz(pose_file)
        if pose is not None:
            packaged_poses_dict[frame_idx] = pose
            
    # Sort by frame index and stack into a single (N, 4, 4) array
    sorted_poses = [packaged_poses_dict[i] for i in sorted(packaged_poses_dict.keys())]
    stacked_poses = np.stack(sorted_poses, axis=0)
    
    np.savez(output_file, data=stacked_poses)
    print(f"Successfully packaged fused poses to {output_file}")
    print(f"Final array shape: {stacked_poses.shape}")


def main():
    parser = argparse.ArgumentParser(description="Format poses from different methods for evaluation.")
    parser.add_argument('--sequence_name', type=str, required=True, help='Name of the sequence to process (e.g., dual_evaluation_dataset9_keyframe3)')
    parser.add_argument('--base_output_dir', type=str, default="/hy-tmp/hy-tmp/CUT3R/eval/cut3r1+cut3r2", help='Base directory where fused results are stored.')
    parser.add_argument('--cut3r2_base_dir', type=str, default="/hy-tmp/hy-tmp/CUT3R/eval/cut3r+loss", help='Base directory for the raw cut3r2 chunk results.')
    parser.add_argument('--final_eval_dir', type=str, default="/hy-tmp/hy-tmp/CUT3R/eval/test_all_sequences", help='Final directory to save the formatted .npz files for evaluation.')
    args = parser.parse_args()
    
    # --- Configuration ---
    paths = get_paths(args.base_output_dir, args.cut3r2_base_dir, args.final_eval_dir, args.sequence_name)
    
    # --- Execute Tasks ---
    process_cut3r2_chunks(paths["cut3r2_chunks_dir"], paths["cut3r2_output_file"])
    package_fused_poses(paths["fused_poses_dir"], paths["fused_output_file"])
    
    print(f"\nAll tasks completed for sequence: {args.sequence_name}")

if __name__ == "__main__":
    main()

import os
import numpy as np
import glob
from tqdm import tqdm
import argparse

def get_paths_stereomis_format(base_output_dir, cut3r2_base_dir, final_eval_dir, sequence_name):
    """Generates all necessary paths for formatting a StereoMIS sequence."""
    paths = {
        "cut3r2_chunks_dir": os.path.join(cut3r2_base_dir, sequence_name, "chunk_outputs"),
        "fused_poses_dir": os.path.join(base_output_dir, sequence_name, "final_corrected_poses", "camera"),
        "cut3r2_output_file": os.path.join(final_eval_dir, sequence_name, "cut3r2.npz"),
        "fused_output_file": os.path.join(final_eval_dir, sequence_name, "cut3r1+cut3r2.npz"),
    }
    os.makedirs(os.path.join(final_eval_dir, sequence_name), exist_ok=True)
    return paths

def load_pose_from_npz(file_path):
    """Loads a pose matrix from an .npz file, compatible with both formats."""
    try:
        with np.load(file_path) as data:
            if 'pose' in data:
                return data['pose']
            # Fallback for the format where data is stored under dict keys
            if '0' in data:
                return data['0']
        raise KeyError(f"Key 'pose' or '0' not found in {file_path}")
    except Exception as e:
        print(f"Error loading pose from {file_path}: {e}")
        return None

def process_cut3r2_chunks(chunks_dir, output_file):
    """
    Processes cut3r2 chunk outputs for StereoMIS, stitches them, and saves as a single .npz file.
    """
    print("--- Processing StereoMIS cut3r2 chunks ---")
    chunk_dirs = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*")))
    
    global_poses = {}
    last_anchor_pose_global = np.eye(4)

    for i, chunk_dir in enumerate(tqdm(chunk_dirs, desc="Stitching StereoMIS chunks")):
        chunk_name = os.path.basename(chunk_dir)
        parts = chunk_name.split('_')
        start_frame, end_frame = map(int, parts[-1].split('-'))

        local_poses = {}
        for frame_idx in range(start_frame, end_frame + 1):
            relative_frame_idx = frame_idx - start_frame
            pose_file = os.path.join(chunk_dir, "camera", f"{relative_frame_idx:06d}.npz")
            pose = load_pose_from_npz(pose_file)
            if pose is not None:
                local_poses[frame_idx] = pose

        if not local_poses:
            continue

        if i == 0:
            global_poses.update(local_poses)
            if end_frame in global_poses:
                 last_anchor_pose_global = global_poses[end_frame]
        else:
            if start_frame in local_poses:
                current_anchor_pose_local = local_poses[start_frame]
                T_align = last_anchor_pose_global @ np.linalg.inv(current_anchor_pose_local)
                
                for frame_idx, pose in local_poses.items():
                    global_poses[frame_idx] = T_align @ pose
                
                if end_frame in global_poses:
                    last_anchor_pose_global = global_poses[end_frame]

    sorted_poses = [global_poses[i] for i in sorted(global_poses.keys())]
    stacked_poses = np.stack(sorted_poses, axis=0)
    
    np.savez(output_file, data=stacked_poses)
    print(f"Successfully saved stitched StereoMIS cut3r2 poses to {output_file}")
    print(f"Final array shape: {stacked_poses.shape}")

def package_fused_poses(fused_poses_dir, output_file):
    """
    Packages the final fused StereoMIS poses into a single .npz file for evaluation.
    """
    print("\n--- Packaging fused StereoMIS cut3r1+cut3r2 poses ---")
    pose_files = sorted(glob.glob(os.path.join(fused_poses_dir, "*.npz")))
    
    packaged_poses_dict = {}
    for pose_file in tqdm(pose_files, desc="Packaging StereoMIS fused poses"):
        frame_idx_str = os.path.basename(pose_file).split('.')[0]
        frame_idx = int(frame_idx_str)
        pose = load_pose_from_npz(pose_file)
        if pose is not None:
            packaged_poses_dict[frame_idx] = pose
            
    sorted_poses = [packaged_poses_dict[i] for i in sorted(packaged_poses_dict.keys())]
    stacked_poses = np.stack(sorted_poses, axis=0)
    
    np.savez(output_file, data=stacked_poses)
    print(f"Successfully packaged fused StereoMIS poses to {output_file}")
    print(f"Final array shape: {stacked_poses.shape}")

def main():
    parser = argparse.ArgumentParser(description="Format StereoMIS poses for evaluation.")
    parser.add_argument('--sequence_name', type=str, required=True, help='Name of the sequence to process (e.g., StereoMIS_P2_1_2)')
    parser.add_argument('--base_output_dir', type=str, default="/hy-tmp/hy-tmp/CUT3R/eval/cut3r1+cut3r2_stereomis", help='Base directory for fused results.')
    parser.add_argument('--cut3r2_base_dir', type=str, default="/hy-tmp/hy-tmp/CUT3R/eval/cut3r+loss/stereomis_inference_results", help='Base directory for raw cut3r2 chunks.')
    parser.add_argument('--final_eval_dir', type=str, default="/hy-tmp/hy-tmp/CUT3R/eval/test_stereomis_sequences", help='Final directory for formatted .npz files.')
    args = parser.parse_args()
    
    paths = get_paths_stereomis_format(args.base_output_dir, args.cut3r2_base_dir, args.final_eval_dir, args.sequence_name)
    
    process_cut3r2_chunks(paths["cut3r2_chunks_dir"], paths["cut3r2_output_file"])
    package_fused_poses(paths["fused_poses_dir"], paths["fused_output_file"])
    
    print(f"\nAll formatting tasks completed for StereoMIS sequence: {args.sequence_name}")

if __name__ == "__main__":
    main()

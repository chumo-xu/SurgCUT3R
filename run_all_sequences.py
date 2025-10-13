import os
import subprocess
import glob

def run_command(command, env=None):
    """Executes a command and prints its output in real-time."""
    print(f"\n{'='*30}\nExecuting: {' '.join(command)}\n{'='*30}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    process.stdout.close()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)

def get_image_dir(sequence_name):
    """Constructs the image directory path from the sequence name."""
    parts = sequence_name.split('_')
    dataset_name = parts[2]
    keyframe_name = parts[3]
    dataset_num = dataset_name.replace('dataset', '')
    keyframe_num = keyframe_name.replace('keyframe', '')
    return f"/hy-tmp/hy-tmp/CUT3R/eval/eval_data/test/{dataset_name}/{keyframe_name}/Scared{dataset_num}_{keyframe_num}_Left_Images"

def main():
    # --- Main Configuration ---
    sequences = [
        "dual_evaluation_dataset8_keyframe0",
        "dual_evaluation_dataset8_keyframe1",
        "dual_evaluation_dataset8_keyframe2",
        "dual_evaluation_dataset8_keyframe3",
        "dual_evaluation_dataset9_keyframe0",
        "dual_evaluation_dataset9_keyframe1",
        "dual_evaluation_dataset9_keyframe2",
        "dual_evaluation_dataset9_keyframe3"
    ]
    
    cut3r1_model_path = "/hy-tmp/hy-tmp/CUT3R/src/checkpoints/train_scared_stage1_medical_adaptationV2new17stage2-2/checkpoint-2.pth"
    base_output_dir = "/hy-tmp/hy-tmp/CUT3R/eval/cut3r1+cut3r2" # Fused results will go here
    cut3r2_base_dir = "/hy-tmp/hy-tmp/CUT3R/eval/cut3r+loss" # Raw cut3r2 results
    final_eval_dir = "/hy-tmp/hy-tmp/CUT3R/eval/test_all_sequences" # Final .npz files for evaluation

    # Get conda environment info
    conda_base = os.environ.get("CONDA_PREFIX")
    if not conda_base:
        raise RuntimeError("CONDA_PREFIX is not set. Cannot determine conda environment path.")
    
    # Path to python executable in the cut3r environment
    python_executable = os.path.join(conda_base, 'envs', 'cut3r', 'bin', 'python3')
    if not os.path.exists(python_executable):
         python_executable = os.path.join(conda_base, 'bin', 'python3') # Fallback for base env
         print("Warning: could not find python in cut3r env, trying base env.")
         if not os.path.exists(python_executable):
              raise FileNotFoundError("Could not find python executable in conda envs.")

    # --- Main Loop ---
    for seq in sequences:
        print(f"\n{'#'*80}\n# Starting processing for sequence: {seq}\n{'#'*80}")
        
        # === Step 1: Generate cut3r1 anchor poses ===
        print(f"\n--- Step 1: Generating cut3r1 anchor poses for {seq} ---")
        image_dir = get_image_dir(seq)
        anchor_frame_dir = os.path.join(os.path.dirname(image_dir), "anchor_frames_temp")
        
        try:
            # 1a. Create temp dir and copy anchor frames
            os.makedirs(anchor_frame_dir, exist_ok=True)
            all_img_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
            num_frames = len(all_img_paths)
            anchor_indices = list(range(0, num_frames, 16))
            if num_frames - 1 not in anchor_indices:
                anchor_indices.append(num_frames - 1)
            
            for idx in anchor_indices:
                subprocess.run(['cp', all_img_paths[idx], anchor_frame_dir], check=True)

            # 1b. Run inference using demo_online.py
            anchor_output_dir = os.path.join(base_output_dir, seq, "anchor_poses_cut3r1")
            
            inference_cmd = [
                python_executable, '/hy-tmp/hy-tmp/CUT3R/demo_online.py',
                '--model_path', cut3r1_model_path,
                '--seq_path', anchor_frame_dir,
                '--output_dir', anchor_output_dir,
                '--no-viz'
            ]
            run_command(inference_cmd)
            
        finally:
            # 1c. Clean up temp dir
            print(f"Cleaning up temporary directory: {anchor_frame_dir}")
            subprocess.run(['rm', '-rf', anchor_frame_dir], check=True)
            
        # === Step 2: Fuse poses ===
        print(f"\n--- Step 2: Fusing poses for {seq} ---")
        fuse_cmd = [
            python_executable, '/hy-tmp/hy-tmp/CUT3R/fuse_poses.py',
            '--sequence_name', seq,
            '--base_output_dir', base_output_dir,
            '--cut3r2_base_dir', cut3r2_base_dir
        ]
        run_command(fuse_cmd)

        # === Step 3: Format poses for evaluation ===
        print(f"\n--- Step 3: Formatting poses for {seq} ---")
        format_cmd = [
            python_executable, '/hy-tmp/hy-tmp/CUT3R/format_poses_for_eval.py',
            '--sequence_name', seq,
            '--base_output_dir', base_output_dir,
            '--cut3r2_base_dir', cut3r2_base_dir,
            '--final_eval_dir', final_eval_dir
        ]
        run_command(format_cmd)
        
        print(f"\n{'#'*80}\n# Finished processing for sequence: {seq}\n{'#'*80}")

    print("\n\n🎉🎉🎉 All sequences processed successfully! 🎉🎉🎉")

if __name__ == "__main__":
    main()

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

def main():
    # --- Main Configuration ---
    sequences = [
        "StereoMIS_P2_1_2",
        "StereoMIS_P2_4_1",
        "StereoMIS_P2_5_3",
        "StereoMIS_P2_8_1"
    ]
    
    cut3r1_model_path = "/hy-tmp/hy-tmp/CUT3R/src/checkpoints/train_scared_stage1_medical_adaptationV2new17stage2-2/checkpoint-2.pth"
    # Define new base directories for StereoMIS results
    base_output_dir = "/hy-tmp/hy-tmp/CUT3R/eval/cut3r1+cut3r2_stereomis"
    cut3r2_base_dir = "/hy-tmp/hy-tmp/CUT3R/eval/cut3r+loss/stereomis_inference_results"
    final_eval_dir = "/hy-tmp/hy-tmp/CUT3R/eval/test_stereomis_sequences"
    
    # Path to the original StereoMIS dataset
    stereomis_dataset_base = "/hy-tmp/hy-tmp/CUT3R/dataset/stereomis_testset"

    # Get conda environment info
    conda_base = os.environ.get("CONDA_PREFIX")
    if not conda_base:
        raise RuntimeError("CONDA_PREFIX is not set. Cannot determine conda environment path.")
    
    python_executable = os.path.join(conda_base, 'envs', 'cut3r', 'bin', 'python3')
    if not os.path.exists(python_executable):
         python_executable = os.path.join(conda_base, 'bin', 'python3')
         print("Warning: could not find python in cut3r env, trying base env.")
         if not os.path.exists(python_executable):
              raise FileNotFoundError("Could not find python executable in conda envs.")

    # --- Main Loop ---
    for seq in sequences:
        print(f"\n{'#'*80}\n# Starting processing for StereoMIS sequence: {seq}\n{'#'*80}")
        
        image_dir = os.path.join(stereomis_dataset_base, seq, "rgb")
        
        # === Step 1: Generate cut3r1 anchor poses ===
        print(f"\n--- Step 1: Generating cut3r1 anchor poses for {seq} ---")
        anchor_frame_dir = os.path.join(stereomis_dataset_base, seq, "anchor_frames_temp")
        
        try:
            os.makedirs(anchor_frame_dir, exist_ok=True)
            all_img_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
            if not all_img_paths:
                print(f"Warning: No images found in {image_dir}. Skipping sequence {seq}.")
                continue
            
            num_frames = len(all_img_paths)
            anchor_indices = list(range(0, num_frames, 16))
            if num_frames - 1 not in anchor_indices:
                anchor_indices.append(num_frames - 1)
            
            for idx in anchor_indices:
                subprocess.run(['cp', all_img_paths[idx], anchor_frame_dir], check=True)

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
            print(f"Cleaning up temporary directory: {anchor_frame_dir}")
            subprocess.run(['rm', '-rf', anchor_frame_dir], check=True)
            
        # === Step 2: Fuse poses using the dedicated StereoMIS script ===
        print(f"\n--- Step 2: Fusing poses for {seq} ---")
        fuse_cmd = [
            python_executable, '/hy-tmp/hy-tmp/CUT3R/fuse_poses_stereomis.py',
            '--sequence_name', seq,
            '--base_output_dir', base_output_dir,
            '--cut3r2_base_dir', cut3r2_base_dir,
            '--stereomis_dataset_base', stereomis_dataset_base
        ]
        run_command(fuse_cmd)

        # === Step 3: Format poses for evaluation using the dedicated StereoMIS script ===
        print(f"\n--- Step 3: Formatting poses for {seq} ---")
        format_cmd = [
            python_executable, '/hy-tmp/hy-tmp/CUT3R/format_poses_stereomis.py',
            '--sequence_name', seq,
            '--base_output_dir', base_output_dir,
            '--cut3r2_base_dir', cut3r2_base_dir,
            '--final_eval_dir', final_eval_dir
        ]
        run_command(format_cmd)
        
        print(f"\n{'#'*80}\n# Finished processing for StereoMIS sequence: {seq}\n{'#'*80}")

    print("\n\n🎉🎉🎉 All StereoMIS sequences processed successfully! 🎉🎉🎉")


if __name__ == "__main__":
    main()

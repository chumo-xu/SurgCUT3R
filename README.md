
# SurgCUT3R: Surgical Scene-Aware Continuous Understanding of Temporal 3D Representation


<hr>

<br>
Official implementation of <strong>Surgical Scene-Aware Continuous Understanding of Temporal 3D Representation</strong>





## Getting Started

### Installation

1. Clone MASt3R-SLAM and SurgCUT3R.
First follow the instruction of CUT3R to install it. Then follow the instruction of MASt3R-SLAM to install it in the folder of CUT3R to make sure that the environment works in the training process.
2. We also have the docker for this project.


### Download Checkpoints

We currently provide checkpoints on Google Drive:

| [`surgcut3r_local.pth`](https://drive.google.com/file/d/1gcmuXFp5aHoqoKySidQkAZ8EZwZAhKJg/view?usp=drive_link) 
| [`surgcut3r_global.pth`](https://drive.google.com/file/d/1e3sASko5xfvm4lWzFTzxf3suvWxfuusa/view?usp=drive_link) 


### Inference

To run the inference code, you can use the following command:
```bash
# the following script will run inference offline and visualize the output with viser on port 8080
python demo.py --model_path MODEL_PATH --seq_path SEQ_PATH --size SIZE --vis_threshold VIS_THRESHOLD --output_dir OUT_DIR  # input can be a folder or a video
# Example:
#     python demo.py --model_path src/cut3r_512_dpt_4_64.pth --size 512 \
#         --seq_path examples/001 --vis_threshold 1.5 --output_dir tmp
#
#     python demo.py --model_path src/cut3r_224_linear_4.pth --size 224 \
#         --seq_path examples/001 --vis_threshold 1.5 --output_dir tmp

# the following script will run inference with global alignment and visualize the output with viser on port 8080
python demo_ga.py --model_path MODEL_PATH --seq_path SEQ_PATH --size SIZE --vis_threshold VIS_THRESHOLD --output_dir OUT_DIR
```



## Acknowledgements
Our code is based on the following awesome repositories:

- [DUSt3R](https://github.com/naver/dust3r)
- [MonST3R](https://github.com/Junyi42/monst3r.git)
- [Spann3R](https://github.com/HengyiWang/spann3r.git)
- [Viser](https://github.com/nerfstudio-project/viser)
- [CUT3R](https://github.com/CUT3R/CUT3R.git)
- [Endo3R](https://github.com/wrld/Endo3R.git)

We thank the authors for releasing their code!





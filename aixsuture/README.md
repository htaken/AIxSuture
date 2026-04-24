# Aachen Suturing
## Description
Evaluation of Aachen Suturing Dataset. Predicts skill score based on RGB frames of recorded video data.

## Paper and Dataset
**Full code will become available upon publication of our paper.**

Our publication using this code can be found [here](https://doi.org/10.1007/s11548-024-03093-3). The dataset used can be found [here](https://zenodo.org/record/7940583).
### Citing
The paper was presented at the 15th International Conference on Information Processing in Computer-Assisted Interventions (IPCAI 2024) and published in the International Journal of Computer Assisted Radiology and Surgery.
```
@article{Hoffmann2024,
  author    = {Hanna Hoffmann and Isabel Funke and Philipp Peters and Danush Kumar Venkatesh and Jan Egger and Dominik Rivoir and Rainer Röhrig and Frank Hölzle and Sebastian Bodenstedt and Marie-Christin Willemer and Stefanie Speidel and Behrus Puladi},
  title     = {AIxSuture: vision-based assessment of open suturing skills},
  journal   = {International Journal of Computer Assisted Radiology and Surgery},
  volume    = {19},
  number    = {6},
  pages     = {1045--1052},
  year      = {2024},
  month     = {jun},
  abstract  = {Efficient and precise surgical skills are essential in ensuring positive patient outcomes. By continuously providing real-time, data-driven, and objective evaluation of surgical performance, automated skill assessment has the potential to greatly improve surgical skill training. Whereas machine learning-based surgical skill assessment is gaining traction for minimally invasive techniques, this cannot be said for open surgery skills. Open surgery generally has more degrees of freedom when compared to minimally invasive surgery, making it more difficult to interpret. In this paper, we present novel approaches for skill assessment for open surgery skills.},
  issn      = {1861-6429},
  doi       = {10.1007/s11548-024-03093-3},
  url       = {https://doi.org/10.1007/s11548-024-03093-3}
}
```

This work was carried out at the National Center for Tumor Diseases (NCT) Dresden, Department of Translational Surgical Oncology.

## Visuals
...pending

## Usage
How to start
Simply clone this repository:

```shell
cd <the directory where the repo shall live>
git clone https://gitlab.com/nct_tso_public/aixsuture.git
cd aixsuture
```

In the following, we use CODE_DIR to refer to the absolute path to the code.

### Environment Setup (uv)
Dependencies are managed with [uv](https://docs.astral.sh/uv/). Install uv first,
then run `uv sync` to provision Python 3.8.20 and all dependencies into a local
`.venv`:

```shell
uv sync
```

This installs:

    torch==2.4.1 torchvision==0.19.1 (CUDA 11.8 wheels)
    numpy<2 pillow pyyaml matplotlib seaborn pandas openpyxl
    tqdm torchmetrics tensorboard parse opencv-python-headless

Prefix subsequent commands with `uv run` (e.g. `uv run python3 train.py ...`) so
they execute inside the managed environment.

Experiments were run using Python 3.8 and PyTorch/Cuda 11.8.

### Data Preparation
`preprocessing.py` extracts frames at the given `frame_rate` (in fps) from the videos. `DATA_DIR` is the directory containing the downloaded videos. The videos should not be in further subdirectories. `FRAMES_DIR` is the location where the extracted frames should be saved. Frames are sorted under subdirectories according to video: `FRAMES_DIR/<video name>/img_xxxxx.jpg`
```shell
cd /<path>/<to>/<this>/<repo>
DATA_DIR="/<path>/<to>/<video data>"
FRAMES_DIR="/<path>/<to>/<video frame output location>"

uv run python3 preprocessing.py --data_root $DATA_DIR --out_dir $FRAMES_DIR --frame_rate 1
```

### Training
In `train.py` the extracted frames from `FRAMES_DIR` are distributed into train, validation, and test splits, and the chosen model architecture is trained. `FRAMES_DIR` is the same directory as previously.
```shell
PRETRAIN_PATH="/<path>/<to>/<pretrain weights>/rgb_imagenet.pt"
OUT_DIR="/<path>/<to>/<model output location>"
uv run python3 train.py --exp swin_tiny --data_path $FRAMES_DIR --out $OUT_DIR --split '70_15_15' --snippet_length 64 --num_segments 12 --arch 'SWINTransformer_T'
```

### Training Parameters
Here some useful parameters are listed with their respective descriptions and options, if applicable.

| Arg | Info. | Options |
| --- | ----- | - |
| exp | Name of the experiment to run | |
| split | Train, validation, and test split. Separate split percentages with underscores. (ex.: 70_15_15) | |
| do_test | Include a run with test set after completing training | |
| data_path | Path to data folder, which contains the extracted images for each video. This path should also contain the annotations file OSATS.xlsx.| |
| data_preloading | Whether all image data should be loaded to RAM before starting network training. | |
| arch | Model architecture to use. | Inception3D, SWINTransformer_T, SWINTransformer_S, SWINTransformer_B |
| snippet_length | Number of frames in one video snippet. | |
| num_segments | Number of snippets processed by the Temporal Segment Network. | |

## Outputs
For the current run all output (program log, models, and tensorboard logs) are saved to `<OUT_DIR>/<EXP>_date/<EVAL_SCHEME>/<SPLIT>/time/`.

## Contributing
The code is based on [this code](https://gitlab.com/nct_tso_public/surgical_skill_classification).

## Authors and acknowledgment
Thanks to I. Funke for the initial source code and help. Also thanks to B. Puladi and the faculty of Medicine RWTH Aachen University for their dataset availability and support.

import argparse

num_cls_Kinetics = 400 # Used for I3D model

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description="Test model for video-based surgical skill classification.")
parser.register('type', 'bool', str2bool)

# Experiment
parser.add_argument('--exp', type=str, default='test', help="Name (description) of the experiment to run.")
parser.add_argument('--split', type=str, required=True, help="Train, validation, and test split used in training.")
parser.add_argument('--modality', type=str, default='RGB', choices=['RGB', 'Flow'], help="Used input modality for I3D.")
parser.add_argument('--rand_seed_data_split', type=int, default=42, help='Random seed for data split generator.')

# Data
parser.add_argument('--data_path', type=str, default="?",
                    help="Path to data folder, which contains the extracted (flow) images for each video. "
                         "One subfolder per video. This path should also contain the annotations file OSATS.xlsx")
parser.add_argument('--do_horizontal_flip', type='bool', default=True,
                    help="Whether or not data augmentation should include a random horizontal flip.")
parser.add_argument('--data_preloading', type='bool', default=True,
                    help="Whether or not all image data should be loaded to RAM before starting network training.")
parser.add_argument('--image_tmpl', type=str, default='img_{:05d}.jpg', help='Image file type of data in DATA_PATH.')

# Model
parser.add_argument('--arch', type=str, default="Inception3D", choices=['Inception3D', 'Pretrained-Inception-v3', 'SWINTransformer_T', 'SWINTransformer_S', 'SWINTransformer_B'],
                    help="Model architecture to use.")
parser.add_argument('--snippet_length', type=int, default=64, help="Number of frames constituting one video snippet.")
parser.add_argument('--num_segments', type=int, default=10,
                    help="Number of snippets processed by the Temporal Segment Network.")
parser.add_argument('--pretrain_path', type=str, required=True, help="Path to pretrained model weights.")

# Testing
parser.add_argument('-j', '--workers', type=int, default=4, help="Number of threads used for data loading.")
parser.add_argument('--out', type=str, default="?",
                    help="Path to output folder, where all models and results will be stored.")



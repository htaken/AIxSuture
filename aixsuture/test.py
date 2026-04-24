import os.path
import datetime
import torch

import util

from torchmetrics import MetricCollection
from torchmetrics.wrappers import ClasswiseWrapper
from torchmetrics.classification import Accuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix

from models import TSN
from dataset import DatasetHandler
from test_args import parser
from transforms import GroupNormalize
from train import validate, save_conf_matrix
from util import log_metrics, log_predictions

device_gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")

def parse_params_from_log(logfile):
    file_name = 'best.pt'


def main():
    global args
    pretrain_path = args.pretrain_path
    if args.out == '?':
        output_folder = os.path.dirname(args.pretrain_path)
        print(f'Output path not specified. Setting to {output_folder}')
    else:
        output_folder = os.path.join(args.out, datetime.datetime.now().strftime("%Y%m%d") + "_" + args.exp,
                                     str(args.split), datetime.datetime.now().strftime("%H%M"))
        os.makedirs(output_folder)

    # ===== set up loggers =====
    f_log = open(os.path.join(output_folder, "test_log.txt"), "w")  # program log
    def log(msg): # simplifies log calls to normal program log
        util.log(f_log, msg)

    log("Used parameters...")
    for arg in sorted(vars(args)):
        log("\t" + str(arg) + " : " + str(getattr(args, arg)))

    # ===== start =====
    log('Doing test run.')
    # load model with best F1 score
    test_model = TSN(args.num_class, 1,  # num_segments = 1 to predict each snippet independently
                     args.modality, base_model=args.arch,
                     new_length=args.snippet_length,
                     consensus_type='avg', before_softmax=True, dropout=0.0, partial_bn=False,
                     use_three_input_channels=False)
    test_model.load_state_dict(torch.load(pretrain_path))
    test_model = test_model.to(device_gpu)

    # load data
    normalize = GroupNormalize(test_model.input_mean, test_model.input_std)
    train_augmentation = test_model.get_augmentation(args.do_horizontal_flip)
    filename = os.path.join(args.data_path, "OSATS.xlsx")

    dataset_handler = DatasetHandler(args.data_path, filename, num_segments=args.num_segments,
                                     new_length=args.snippet_length, modality=args.modality, image_tmpl=args.image_tmpl,
                                     transform=train_augmentation, normalize=normalize, random_shift=True,
                                     test_mode=False,
                                     video_suffix=args.video_suffix,
                                     return_3D_tensor=test_model.is_3D_architecture,
                                     return_three_channels=False,
                                     preload_to_RAM=args.data_preloading, data_seed=args.rand_seed_data_split,
                                     data_split=args.split)
    test_loader = torch.utils.data.DataLoader(dataset_handler.test, batch_size=1, shuffle=False,
                                              num_workers=args.workers)
    log("Loaded {} test examples".format(test_loader.dataset.__len__()))

    test_metrics = MetricCollection(
        {'multiclassaccuracy': Accuracy(task="multiclass", num_classes=3),
         'multiclassrecall': MulticlassRecall(num_classes=3, average='macro'),
         'multiclassprecision': MulticlassPrecision(num_classes=3, average='macro'),
         # macro average bc isabel does it, class imbalance, and bc micro averaged precision = accuracy
         'classwiserecall': ClasswiseWrapper(MulticlassRecall(num_classes=3, average=None),
                                             prefix='classwiserecall_'),
         # prefix specified bc metrics collection would call it muulticlassrecall and that gets confusing with the averaged recall metric
         'multiclassf1': MulticlassF1Score(num_classes=3, average='macro'),
         'classwisef1': ClasswiseWrapper(MulticlassF1Score(num_classes=3, average=None), prefix='classwisef1_'),
         'confusionmatrix': MulticlassConfusionMatrix(num_classes=3)}
    ).to(device_gpu)

    test_metrics, test_results = validate(test_loader, test_model, test_metrics)

    log_predictions(os.path.join(output_folder, "best_model_preds.csv"), test_results)
    log_metrics(output_folder, test_metrics)
    save_conf_matrix(output_folder, test_metrics["confusionmatrix"].to(device_cpu))


    log("Done.")
    f_log.close()


if __name__ == '__main__':
    args = parser.parse_args()

    args.num_class = 3
    args.video_suffix = ""
    if args.modality == 'Flow':
        args.image_tmpl = 'flow_{}_{:05d}.jpg'

    if args.data_path == '?':
        print("Please specify the path to your (flow) image data using the --data_path option or set an appropriate "
              "default in train_opts.py!")
    else:
        main()

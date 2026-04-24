# Copyright (C) 2023  National Center of Tumor Diseases (NCT) Dresden, Division of Translational Surgical Oncology

import datetime
import os.path
import string
import torch
import seaborn

import util

from tqdm import tqdm  # progress bars
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MetricCollection
from torchmetrics.wrappers import ClasswiseWrapper
from torchmetrics.classification import Accuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix

from dataset import AachenDataSet, DatasetHandler
from models import TSN
from train_opts import parser
from transforms import GroupNormalize
from util import AverageMeter, log_metrics, log_predictions, drawConfusionMatrix

device_gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")


def forward_pass(data, target, model):
    data = data.to(device_gpu)
    target = target.to(device_gpu)
    output = model(data)  # forward pass
    return output, target


def validate(data_loader, model, metrics_collection):
    metrics_collection.reset()
    model.eval()  # Optional when not using Model Specific layer
    with torch.inference_mode():
        results = []
        for _, batch in enumerate(data_loader):
            trial_id, data, target = batch  # dimensions: data: test_segments x C (x D) x W x H, target: dim(1)
            out, target = forward_pass(data, target, model)  # dimensions: test_segments x num_class dim(1x3)

            pred = torch.mean(out, 0).squeeze() # average over all segments
            _output = torch.nn.Softmax(dim=-1)(pred)  # dim -1 uses given tensor size
            prediction = torch.tensor([torch.argmax(_output)]).to(
                device_gpu)  # rewrap as tensor, so dimensions fit for metric calculation dim(1)

            results.append([trial_id, target.item(), prediction.item()])
            metrics_collection(prediction, target)
        metrics = metrics_collection.compute()
    return metrics, results


def save_conf_matrix(output_folder, confusion_matrix):
    cf_plot = drawConfusionMatrix(confusion_matrix)
    fig = cf_plot.get_figure()
    fig.savefig(os.path.join(output_folder, f"Best_model_confusion_matrix.svg"))


def train_final_swin(model):
    for param in model.base_model.features[6].parameters():  # last sequential layer with two swin blocks
        param.requires_grad = True


def get_list_of_videos_in_split(video_record_list):
    video_names_list = []
    for video in video_record_list:
        video_names_list += [video.trial]
    return video_names_list


def main():
    global args

    if not torch.cuda.is_available():
        print("GPU not found - exit")
        return

    if len([t for t in string.Formatter().parse(args.data_path)]) > 1:
        args.data_path = args.data_path.format(args.task)

    output_folder = os.path.join(args.out, datetime.datetime.now().strftime("%Y%m%d") + "_" + args.exp,
                                 str(args.split), datetime.datetime.now().strftime("%H%M"))
    os.makedirs(output_folder)

    # ===== set up loggers =====
    f_log = open(os.path.join(output_folder, "log.txt"), "w")  # program log
    def log(msg): # simplifies log calls to normal program log
        util.log(f_log, msg)

    log("Used parameters...")
    for arg in sorted(vars(args)):
        log("\t" + str(arg) + " : " + str(getattr(args, arg)))

    # ===== set up model =====

    consensus_type = 'avg'
    model = TSN(args.num_class, args.num_segments, args.modality, base_model=args.arch, new_length=args.snippet_length,
                consensus_type=consensus_type, before_softmax=True, dropout=args.dropout, partial_bn=False,
                use_three_input_channels=args.three_channel_flow, pretrained_model=args.pretrain_path)

    # freeze weights

    if 'SWINTransformer' in args.arch:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.base_model.head.parameters():
            param.requires_grad = True
        if args.pretrain_head == -1:
            train_final_swin(model)
    elif args.arch == 'Inception3D':
        if args.pretrain_path is None:
            log("Train model from scratch")
            for param in model.parameters():
                param.requires_grad = True
        else:
            for param in model.base_model.parameters(): #freeze all weights
                param.requires_grad = False
            for param in model.base_model.logits.parameters(): # unfreeze weights for top three layers
                param.requires_grad = True
            for param in model.base_model.Mixed_5c.parameters():
                param.requires_grad = True
            for param in model.base_model.Mixed_5b.parameters():
                param.requires_grad = True
    elif args.arch == 'Pretrained-Inception-v3':
        for param in model.base_model.parameters():
            param.requires_grad = False
        for param in model.base_model.fc_action.parameters():
            param.requires_grad = True
        for name, module in model.base_model.named_modules():
            if name.startswith("mixed_10"):
                for param in module.parameters():
                    param.requires_grad = True

    # ===== set up data loaders =====
    normalize = GroupNormalize(model.input_mean, model.input_std)
    train_augmentation = model.get_augmentation(args.do_horizontal_flip)

    filename = os.path.join(args.data_path, "OSATS.xlsx")

    dataset_handler = DatasetHandler(args.data_path, filename, num_segments=args.num_segments,
                                     new_length=args.snippet_length, modality=args.modality, image_tmpl=args.image_tmpl,
                                     transform=train_augmentation, normalize=normalize, random_shift=True,
                                     test_mode=False, video_suffix=args.video_suffix,
                                     return_3D_tensor=model.is_3D_architecture,
                                     return_three_channels=args.three_channel_flow,
                                     preload_to_RAM=args.data_preloading, data_seed=args.rand_seed_data_split,
                                     data_split=args.split)

    # to log videos in splits
    train_list = get_list_of_videos_in_split(dataset_handler.train.videorecord_dataset)
    valid_list = get_list_of_videos_in_split(dataset_handler.validation.videorecord_dataset)
    log(f'Train split: {train_list}')
    log(f'Validation split: {valid_list}')

    # load data in dataLoaders
    train_loader = torch.utils.data.DataLoader(dataset_handler.train, batch_size=args.batch_size, shuffle=True,
                                                     num_workers=args.workers, pin_memory=True)
    log("Loaded {} training videos".format(train_loader.dataset.__len__()))
    log("Total train batches {}".format(train_loader.dataset.__len__()/args.batch_size))


    valid_loader = torch.utils.data.DataLoader(dataset_handler.validation, batch_size=1, shuffle=False,
                                                     num_workers=args.workers)
    log("Loaded {} validation examples".format(valid_loader.dataset.__len__()))

    # ===== set up training =====
    tensorboard_folder = os.path.join(output_folder, "tensorboard")
    os.makedirs(tensorboard_folder)
    writer = SummaryWriter(log_dir=tensorboard_folder)  # tensorboard logging
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    log("param count: {}".format(sum(p.numel() for p in model.parameters())))
    log("trainable params: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # ===== set up metrics =====
    train_acc_metric = Accuracy(task="multiclass", num_classes=3).to(device_gpu)

    validation_metrics = MetricCollection(
        {'multiclassaccuracy': Accuracy(task="multiclass", num_classes=3).to(device_gpu),
         'multiclassrecall': MulticlassRecall(num_classes=3, average='macro').to(device_gpu),
         'multiclassprecision': MulticlassPrecision(num_classes=3, average='macro').to(device_gpu),
         # macro average bc isabel does it, class imbalance, and bc micro averaged precision = accuracy
         'classwiserecall': ClasswiseWrapper(MulticlassRecall(num_classes=3, average=None),
                                             prefix='classwiserecall_').to(device_gpu),
         # prefix specified bc metrics collection would call it muulticlassrecall and that gets confusing with the averaged recall metric
         'multiclassf1': MulticlassF1Score(num_classes=3, average='macro').to(device_gpu),
         'classwisef1': ClasswiseWrapper(MulticlassF1Score(num_classes=3, average=None), prefix='classwisef1_').to(
             device_gpu),
         'confusionmatrix': MulticlassConfusionMatrix(num_classes=3).to(device_gpu)}
    ).to(device_gpu)

    # ===== start! =====

    log("Start training...")

    model = model.to(device_gpu)
    torch.backends.cudnn.benchmark = True

    best_f1 = 0
    for epoch in tqdm(range(0, args.epochs), desc="Training epoch: "):
        train_loss = AverageMeter()
        train_acc_metric.reset()

        if epoch == args.pretrain_head:
            log("Training additional SWIN layers")
            train_final_swin(model)
            log("trainable params: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

        model.train()
        for _, batch in tqdm(enumerate(train_loader), desc="batch"):

            optimizer.zero_grad()

            data, target = batch
            batch_size = target.size(0)

            output, target = forward_pass(data, target, model)
            loss = criterion(output, target)
            loss.backward()  # backwards pass
            optimizer.step()  # update weights
            train_loss.update(loss.item(), batch_size)

            _output = torch.nn.Softmax(dim=1)(output)
            _, predicted = torch.max(_output.data, 1)  # max apparently also outputs index of max, has to be something with the 1
            
            train_acc_metric(predicted, target)
        train_acc = train_acc_metric.compute()

        # validation run - one validation run for each epoch
        metrics, results = validate(valid_loader, model, validation_metrics)

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:  # eval
            valid_acc = metrics['multiclassaccuracy']
            precision = metrics['multiclassprecision']
            recall = metrics['multiclassrecall']
            f1 = metrics['multiclassf1']
            log(f"Epoch {epoch}: Train loss: {train_loss.avg:.4f} Train acc: {train_acc:.3f} Valid acc: "
                f"{valid_acc:.2f}\n\t\t\tPrecision: {precision:.4f}    Recall: {recall:.4f}       F1: {f1:.4f}")
            writer.add_scalar("Loss/train_avg", train_loss.avg, epoch)
            writer.add_scalar("Acc/train_avg", train_acc, epoch)
            writer.add_scalar("Acc/valid_avg", valid_acc, epoch)
            writer.add_scalar('Precision', precision, epoch)
            writer.add_scalar('Recall/macro_avg', recall, epoch)
            writer.add_scalar('F1/macro_avg', f1, epoch)
            for i in range(3): # for each class
                writer.add_scalar(f'Recall/class_{i}', metrics[f'classwiserecall_{i}'], epoch)
                writer.add_scalar(f'F1/class_{i}', metrics[f'classwisef1_{i}'], epoch)
            writer.add_figure('Confusion_Matrix', drawConfusionMatrix(metrics['confusionmatrix'].to(device_cpu)), epoch)

            # save best model
            if f1 > best_f1 or epoch == 0:
                best_f1 = f1
                torch.save(model.state_dict(), os.path.join(output_folder, "best.pt"))
                log("Best model saved")

        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:  # save
            name = "model_" + str(epoch)
            model_file = os.path.join(output_folder, name + ".pth.tar")
            state = {'epoch': epoch + 1,
                     'arch': args.arch,
                     'state_dict': model.state_dict(),
                     }
            torch.save(state, model_file)
            log("Saved model to " + model_file)

            log_predictions(os.path.join(output_folder, "predictions_{}.csv".format(epoch)), results)

    writer.flush()
    log("Training complete.")

    # ======= test run =======
    if args.do_test:
        log('Doing test run.')
        # load model with best F1 score
        test_model = TSN(args.num_class, 1, # num_segments = 1 to predict each snippet independently
                         args.modality, base_model=args.arch,
                         new_length=args.snippet_length,
                         consensus_type=consensus_type, before_softmax=True, dropout=0.0, partial_bn=False,
                         use_three_input_channels=args.three_channel_flow)
        test_model.load_state_dict(torch.load(os.path.join(output_folder, "best.pt")))
        test_model = test_model.to(device_gpu)


        # load data
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
        if args.out == '?':
            print("Please specify the path to your output folder using the --out option or set an appropriate default "
                  "in train_opts.py!")
        else:
            main()

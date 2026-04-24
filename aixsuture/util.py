import os
import time
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from sys import stderr


def log(file, msg):
    """Log a message.

    :param file: File object to which the message will be written.
    :param msg:  Message to log (str).
    """
    print(time.strftime("[%d.%m.%Y %H:%M:%S]: "), msg, file=stderr)
    file.write(time.strftime("[%d.%m.%Y %H:%M:%S]: ") + msg + os.linesep)


def log_predictions(output_path, results):
    """

    :param output_path: full file path of log file
    :param results: results as list with Video, Target, Prediction
    """
    predictions_log = open(output_path, "w")
    predictions_log.write("Video, Target, Prediction" + os.linesep)
    for row in results:
        msg = "{},{:d},{:d}".format(row[0], row[1], row[2])
        print(msg)
        predictions_log.write(msg + os.linesep)
    predictions_log.close()


def log_metrics(output_folder, metrics):
    metrics_log = open(os.path.join(output_folder, "best_model_metrics.csv"), "w")
    metrics_log.write("Metric, Value" + os.linesep)
    for key, value in metrics.items():
        if "confusionmatrix" in key:
            continue  # confusion matrix saved/logged separately
        msg = f"{key}, {value.data:.4f}"
        metrics_log.write(msg + os.linesep)
    metrics_log.close()


def drawConfusionMatrix(confusion_matrix):
    '''

    :param confusion_matrix: confusion matrix as tensor, needs to be in cpu though (use tensor.to(...)
    :return: heatmap plot of confusion matrix
    '''
    # constant for classes
    classes = ('Novice', 'Intermediate', 'Expert')

    df_cm = pd.DataFrame(confusion_matrix, index=[i for i in classes],
                         columns=[i for i in classes])  # change to dataframe for class names in figure
    plt.figure(figsize=(12, 7))
    plt.tick_params(axis='both', which='major', labelsize=10, labelbottom=False, bottom=False, top=False, labeltop=True,
                    left=False)  # remove tick marks on axes and move x labels to top
    return sn.heatmap(df_cm, annot=True).get_figure()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


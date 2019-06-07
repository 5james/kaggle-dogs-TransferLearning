import logging
from torch.autograd import Variable
from argparse import ArgumentParser, RawTextHelpFormatter
import os
import time
import torch
import torch.nn as nn
import torch.nn.modules as modules
import torch.optim as optim
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler, random_split
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
from tensorboardX import SummaryWriter
import sys
import matplotlib
import matplotlib.pyplot as plt
import copy
import pickle
from textwrap import wrap
import re
import itertools
import random
import torchnet
import torch
import plotly
import plotly.graph_objs as go

matplotlib.use('agg')
sys.path.append("../4/")
from cnn_codes_extraction import *

np.set_printoptions(threshold=sys.maxsize)

DATA_DIR = '../data/'
LOG_DIR = 'logs/'

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

NUM_CLASSES = 120

IMAGE_INPUT_SIZE = 224

TRAIN_PART = 0.8
VALIDATION_PART = 0.1
TEST_PART = 0.1

parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
parser.add_argument("-e", "--experiment-dir", dest="EXPERIMENT_DIR", help="Pack whole experiment in one directory"
                                                                          "with predeclared filenames",
                    metavar="FILE", type=str, default=None)
parser.add_argument("--batch_size", dest="BATCH_SIZE", help="DataLoader batch size",
                    type=int, default=64)
parser.add_argument("--num_workers", dest="NUM_WORKERS", help="DataLoader number of workers",
                    type=int, default=8)
parser.add_argument("--epochs", dest="EPOCHS", help="number of epochs",
                    type=int, default=30)
parser.add_argument("--learning_rate", dest="LEARNING_RATE", help="learning rate",
                    type=float, default=0.04)
parser.add_argument("--weight_decay", dest="WEIGHT_DECAY", help="weight decay (L2)",
                    type=float, default=0)
parser.add_argument("--momentum", dest="MOMENTUM", help="momentum for optimizers",
                    type=float, default=0.9)
parser.add_argument("--step_size", dest="STEP_SIZE", help="step size (step LR)",
                    type=int, default=25)
parser.add_argument("--gamma", dest="GAMMA", help="gamma (step LR)",
                    type=float, default=0.1)
parser.add_argument("--use_scheduler", dest="USE_SCHEDULER", help="use scheduler for optimizer",
                    action='store_true')
parser.add_argument("--nogpu", dest="NOGPU", help="Specify if you don't want to use GPU (CUDA)",
                    action='store_true')
parser.add_argument("--roc_drawing", dest="ROC_DRAWING", help="ROC will be drawn once every N-th epochs. Specify N.",
                    type=int, default=5)


def plot_confusion_matrix(cm, classes, title='Confusion matrix'):
    # np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()

    fig = plt.figure(figsize=(20, 20), dpi=640, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=6)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=3, rotation=-90, ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=6)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=3, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i, j] != 0 else '.', horizontalalignment="center", fontsize=4,
                verticalalignment='center', color="black")
    fig.set_tight_layout(True)
    # fig.canvas.draw()
    #
    # # Now we can save it to a numpy array.
    # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return fig


def train_model_all(model, criterion, optimizer, scheduler, num_epochs, model_name):
    confusion_matrix = torchnet.meter.ConfusionMeter(NUM_CLASSES)
    accuracy_meter = torchnet.meter.ClassErrorMeter(topk=[1, 5], accuracy=True)
    total_loss_meter = 0
    auc_meter_list = [torchnet.meter.AUCMeter() for _ in range(NUM_CLASSES)]
    auc_avg = torchnet.meter.AUCMeter()
    average_precision = torchnet.meter.APMeter()
    mean_squared_error = torchnet.meter.MSEMeter()

    for epoch in range(num_epochs):
        logger.info('-' * 60)
        logger.info('Epoch {}/{}'.format(epoch + 1, num_epochs))
        logger.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            confusion_matrix.reset()
            accuracy_meter.reset()
            total_loss_meter = 0
            for auc_meter in auc_meter_list:
                auc_meter.reset()
            auc_avg.reset()
            average_precision.reset()
            mean_squared_error.reset()
            data_processed = 0

            if phase == 'train':
                if args.USE_SCHEDULER:
                    scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data.
            for data in data_loaders[phase]:
                # get the inputs
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # indicate progress
                data_processed += len(labels)
                print('{}/{}'.format(data_processed, datasets_len[phase]), end='\r')

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                # basic
                total_loss_meter += float(loss.item()) * float(inputs.size(0))
                confusion_matrix.add(outputs.data.squeeze(), labels.type(torch.LongTensor))
                accuracy_meter.add(outputs.data, labels.data)
                for ii in range(len(labels.data)):
                    label = int(labels.data[ii])
                    hot_one = [0] * NUM_CLASSES
                    hot_one[label] = 1
                    hot_one = np.array(hot_one)
                    average_precision.add(outputs.data[ii], hot_one)

                # auc meter
                for ii in range(NUM_CLASSES):
                    targets = []
                    for jj in labels.data:
                        targets.append(1 if int(jj) == ii else 0)
                    auc_meter_list[ii].add(outputs.data[:, ii], np.array(targets))
                    auc_avg.add(outputs.data[:, ii], np.array(targets))

            logger.info('{}'.format(phase))
            epoch_loss = total_loss_meter / datasets_len[phase]
            logger.info('Epoch Loss / Dataset Len = {:6.4f}'.format(epoch_loss))
            logger.info('Epoch Accuracy Top1 = {:6.4f}'.format(accuracy_meter.value(k=1)))
            logger.info('Epoch Accuracy Top5 = {:6.4f}'.format(accuracy_meter.value(k=5)))
            # logger.info('Confusion matrix:\n{}'.format(confusion_matrix.value()))

            figure = plot_confusion_matrix(confusion_matrix.value(), all_classes)
            # logger.info('Average Precision = \n{}'.format(average_precision.value()))
            logger.info('Mean Average Precision = {:6.4f}'.format(float(average_precision.value().mean())))

            if phase == 'train':
                writer.add_scalar(model_name + '/Train/Loss', epoch_loss, epoch)
                writer.add_scalar(model_name + '/Train/Accuracy-top1', accuracy_meter.value(k=1), epoch)
                writer.add_scalar(model_name + '/Train/Accuracy-top5', accuracy_meter.value(k=5), epoch)
                writer.add_scalar(model_name + '/Train/Mean-Avg-Precision', float(average_precision.value().mean()),
                                  epoch)
                # writer.add_image('Train/Confusion-Matrix', torch.from_numpy(figure), epoch)
                writer.add_figure(model_name + '/Train/Confusion-Matrix', figure, epoch)
            elif phase == 'val':
                writer.add_scalar(model_name + '/Val/Loss', epoch_loss, epoch)
                writer.add_scalar(model_name + '/Val/Accuracy-top1', accuracy_meter.value(k=1), epoch)
                writer.add_scalar(model_name + '/Val/Accuracy-top5', accuracy_meter.value(k=5), epoch)
                writer.add_scalar(model_name + '/Val/Mean-Avg-Precision', float(average_precision.value().mean()),
                                  epoch)
                writer.add_figure(model_name + '/Val/Confusion-Matrix', figure, epoch)

            # ROC curve
            if epoch % args.ROC_DRAWING == 0:
                mid_lane = go.Scatter(x=[0, 1], y=[0, 1],
                                      mode='lines',
                                      line=dict(color='navy', width=2, dash='dash'),
                                      showlegend=False)
                auc, tpr, fpr = auc_avg.value()
                avg_lane = go.Scatter(x=fpr, y=tpr,
                                      mode='lines',
                                      line=dict(color='deeppink', width=1, dash='dot'),
                                      name='average ROC curve (area = {:.2f})'.format(float(auc)))
                traces = [mid_lane, avg_lane]
                for ii in range(NUM_CLASSES):
                    auc, tpr, fpr = auc_meter_list[ii].value()
                    color = 'rgb({}, {}, {})'.format(random.randint(0, 255), random.randint(0, 255),
                                                     random.randint(0, 255))
                    trace = go.Scatter(x=fpr, y=tpr,
                                       mode='lines',
                                       line=dict(color=color, width=1),
                                       name='{} (area = {:.2f})'.format(idx_to_class[ii], float(auc))
                                       )
                    traces.append(trace)
                layout = go.Layout(title='Receiver operating characteristic',
                                   xaxis=dict(title='False Positive Rate'),
                                   yaxis=dict(title='True Positive Rate'))
                plotly.offline.plot({
                    "data": traces,
                    "layout": layout
                }, auto_open=False, filename=EXPERIMENT_DIR + model_name + '-{}-{}.html'.format(epoch, phase))

    # # load best model weights
    # model.load_state_dict(best_model_wts)
    return model


def test_model_all(model, model_name):
    confusion_matrix = torchnet.meter.ConfusionMeter(NUM_CLASSES)
    accuracy_meter = torchnet.meter.ClassErrorMeter(topk=[1, 5], accuracy=True)
    total_loss_meter = 0
    auc_meter_list = [torchnet.meter.AUCMeter() for _ in range(NUM_CLASSES)]
    auc_avg = torchnet.meter.AUCMeter()
    average_precision = torchnet.meter.APMeter()
    data_processed = 0

    logger.info('-' * 60)
    logger.info('Test Model')
    logger.info('-' * 60)

    model.eval()  # Set model to evaluate mode

    # Iterate over data.
    for data in data_loaders['test']:
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # indicate progress
        data_processed += len(labels)
        print('{}/{}'.format(data_processed, datasets_len['test']), end='\r')

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

        # statistics
        # basic
        total_loss_meter += float(loss.item()) * float(inputs.size(0))
        confusion_matrix.add(outputs.data.squeeze(), labels.type(torch.LongTensor))
        accuracy_meter.add(outputs.data, labels.data)
        for ii in range(len(labels.data)):
            label = int(labels.data[ii])
            hot_one = [0] * NUM_CLASSES
            hot_one[label] = 1
            hot_one = np.array(hot_one)
            average_precision.add(outputs.data[ii], hot_one)
        # auc meter
        for ii in range(NUM_CLASSES):
            targets = []
            for jj in labels.data:
                targets.append(1 if int(jj) == ii else 0)
            auc_meter_list[ii].add(outputs.data[:, ii], np.array(targets))
            auc_avg.add(outputs.data[:, ii], np.array(targets))

    epoch_loss = total_loss_meter / datasets_len['test']
    logger.info('Epoch Loss / Dataset Len = {:6.4f}'.format(epoch_loss))
    logger.info('Epoch Accuracy Top1 = {:6.4f}'.format(accuracy_meter.value(k=1)))
    logger.info('Epoch Accuracy Top5 = {:6.4f}'.format(accuracy_meter.value(k=5)))
    figure = plot_confusion_matrix(confusion_matrix.value(), all_classes)
    # logger.info('Confusion matrix:\n{}'.format(confusion_matrix.value()))
    writer.add_figure('Test/Confusion-Matrix', figure, 0)
    logger.info('Mean Average Precision = {:6.4f}'.format(float(average_precision.value().mean())))

    # ROC curve
    mid_lane = go.Scatter(x=[0, 1], y=[0, 1],
                          mode='lines',
                          line=dict(color='navy', width=2, dash='dash'),
                          showlegend=False)
    auc, tpr, fpr = auc_avg.value()
    avg_lane = go.Scatter(x=fpr, y=tpr,
                          mode='lines',
                          line=dict(color='deeppink', width=1, dash='dot'),
                          name='average ROC curve (area = {:.2f})'.format(float(auc)))
    traces = [mid_lane, avg_lane]
    for ii in range(NUM_CLASSES):
        auc, tpr, fpr = auc_meter_list[ii].value()
        color = 'rgb({}, {}, {})'.format(random.randint(0, 255), random.randint(0, 255),
                                         random.randint(0, 255))
        trace = go.Scatter(x=fpr, y=tpr,
                           mode='lines',
                           line=dict(color=color, width=1),
                           name='{} (area = {:.2f})'.format(idx_to_class[ii], float(auc))
                           )
        traces.append(trace)
    layout = go.Layout(title='Receiver operating characteristic',
                       xaxis=dict(title='False Positive Rate'),
                       yaxis=dict(title='True Positive Rate'))
    plotly.offline.plot({
        "data": traces,
        "layout": layout
    }, auto_open=False, filename=EXPERIMENT_DIR + 'test.html')

    # # load best model weights
    # model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    # parse args
    args = parser.parse_args()

    EXPERIMENT_DIR = args.EXPERIMENT_DIR
    if EXPERIMENT_DIR is not None:
        if EXPERIMENT_DIR[-1] is not '/':
            EXPERIMENT_DIR += '/'
        if not os.path.isdir(EXPERIMENT_DIR):
            os.mkdir(EXPERIMENT_DIR)
        else:
            raise Exception('Experiment with name {} already exists'.format(EXPERIMENT_DIR))
    else:
        raise Exception('No experiment directory given.')

    # create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # make sure logging directory 'logs' is available
    if not os.path.isdir(EXPERIMENT_DIR + LOG_DIR):
        os.mkdir(EXPERIMENT_DIR + LOG_DIR)

    # create file handler which logs messages
    fh = logging.FileHandler(EXPERIMENT_DIR + LOG_DIR + str(os.path.basename(__file__).split('.')[0]) + '.log')
    fh.setLevel(logging.DEBUG)

    # create console handler to print to screen
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)

    # tensorboardX logger
    tensorboard_dir = EXPERIMENT_DIR + LOG_DIR + 'tensorboard'
    if not os.path.isdir(tensorboard_dir):
        os.mkdir(tensorboard_dir)
    writer = SummaryWriter(tensorboard_dir)

    FREEZE_LAYERS_NUMBER = 100


    def freeze_params(parameters):
        params = []
        for para in parameters:
            para.requires_grad_(False)
            params.append(para)
        reversed_params = params[::-1]
        for param in range(min(FREEZE_LAYERS_NUMBER, len(reversed_params))):
            reversed_params[param].requires_grad_(True)


    # create model
    model_ft = models.vgg19_bn(pretrained=True)
    freeze_params(model_ft.parameters())
    # # newly created layers have requires_grad == True
    # model_ft.classifier[6] = nn.Sequential(
    #     nn.Dropout(p=0.5),
    #     nn.Linear(512 * 7 * 7, 4096, bias=True),
    #     nn.ReLU(inplace=True),
    #     nn.Dropout(p=0.5),
    #     nn.Linear(4096, 4096, bias=True),
    #     nn.ReLU(inplace=True),
    #     nn.Linear(4096, NUM_CLASSES)
    # )
    model_ft.classifier[6] = nn.Linear(4096, NUM_CLASSES)

    # normalization
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(224, scale=(0.90, 1.0), ratio=(0.95, 1.05), interpolation=2),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize((IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]),
        'val': transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.Resize((IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]),
    }

    # prepare datasets for training, validating and testing with DataLoaders
    dataset_all = datasets.ImageFolder(root=os.path.abspath('../../data').replace('\\', '/'))
    dataset_all_len = int(len(dataset_all))
    datasets_len = {'train': int(dataset_all_len * TRAIN_PART), 'val': int(dataset_all_len * VALIDATION_PART)}
    datasets_len['test'] = int(dataset_all_len - datasets_len['train'] - datasets_len['val'])
    dataset_train, dataset_val, dataset_test = random_split(dataset_all,
                                                            [datasets_len['train'],
                                                             datasets_len['val'],
                                                             datasets_len['test']])
    dataset_train.dataset.transform = data_transforms['train']
    dataset_val.dataset.transform = data_transforms['val']
    dataset_test.dataset.transform = data_transforms['val']

    train_dataloader = DataLoader(dataset_train, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=args.NUM_WORKERS)
    val_dataloader = DataLoader(dataset_val, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=args.NUM_WORKERS)
    test_dataloader = DataLoader(dataset_test, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=args.NUM_WORKERS)

    data_loaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}

    # Create map index -> class name
    idx_to_class = res = dict((v, k) for k, v in dataset_all.class_to_idx.items())
    # logger.info(idx_to_class)
    # Get rid of rubbish in class name
    for i in idx_to_class:
        search_re = re.search(r'(n\d+-)(\w+)', idx_to_class[i], re.I)
        if search_re is not None:
            idx_to_class[i] = search_re.group(2)
    all_classes = [idx_to_class[i] for i in range(NUM_CLASSES)]
    logger.info(idx_to_class)

    # check if can use GPU
    device = torch.device("cuda" if torch.cuda.is_available() and not args.NOGPU else "cpu")

    # ------------------------------------------------------------------------------------------------------------------
    # Retrain whole network

    logger.info('-' * 90)
    logger.info('PHASE ONE')
    logger.info('Retrain whole network')
    logger.info('-' * 90)

    criterion = nn.CrossEntropyLoss()

    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            logger.info("\t{}".format(name))

    logger.info('SGD: lr = {};  momentum = {}, weight decay = {}'.format(
        args.LEARNING_RATE, args.MOMENTUM, args.WEIGHT_DECAY))

    optimizer_ft = optim.SGD(params_to_update, lr=args.LEARNING_RATE, momentum=args.MOMENTUM,
                             weight_decay=args.WEIGHT_DECAY)

    # Decay LR by a factor of x every y epochs
    logger.info('StepLR: step_size = {};  gamma = {}'.format(args.STEP_SIZE, args.GAMMA))
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=args.STEP_SIZE, gamma=args.GAMMA)

    model_ft = model_ft.to(device)

    model_ft = train_model_all(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=args.EPOCHS,
                               model_name='first')
    test_model_all(model_ft, 'first')

    first_network = copy.deepcopy(model_ft)

    # ------------------------------------------------------------------------------------------------------------------
    # Reduce and retrain network

    logger.info('-' * 90)
    logger.info('PHASE Two')
    logger.info('Reduce and retrain network')
    logger.info('-' * 90)

    criterion = nn.CrossEntropyLoss()

    # reduce classifier to ONE Linear layer
    model_ft.classifier = nn.Sequential(*list(model_ft.classifier.children())[:-3])
    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier[-1] = nn.Linear(num_ftrs, NUM_CLASSES)
    conv_list = list(model_ft.features.children())
    # reduce all batchnorm2d layers
    batchnorm2d_type = type(conv_list[1])
    for ii in range(len(conv_list)):
        if not (0 <= ii < len(conv_list)):
            continue
        layer = conv_list[ii]
        if isinstance(layer, batchnorm2d_type):
            conv_list = conv_list[:ii] + conv_list[(ii + 1):]
            ii -= 1
    model_ft.features = nn.Sequential(*conv_list)

    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            logger.info("\t{}".format(name))

    logger.info('SGD: lr = {};  momentum = {}, weight decay = {}'.format(
        args.LEARNING_RATE, 0, args.WEIGHT_DECAY))

    optimizer_ft = optim.SGD(params_to_update, lr=args.LEARNING_RATE, momentum=0,
                             weight_decay=args.WEIGHT_DECAY)

    # Decay LR by a factor of x every y epochs
    logger.info('StepLR: step_size = {};  gamma = {}'.format(args.STEP_SIZE, args.GAMMA))
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=args.STEP_SIZE, gamma=args.GAMMA)

    model_ft = model_ft.to(device)

    model_ft = train_model_all(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=args.EPOCHS,
                               model_name='second')
    test_model_all(model_ft, 'second')

    second_network = copy.deepcopy(model_ft)

    # ------------------------------------------------------------------------------------------------------------------
    # Extract SVM codes from first and second network
    use_gpu = torch.cuda.is_available() and not args.NOGPU

    remaining_classifier_len = len(first_network.classifier)
    first_network.classifier = nn.Sequential(*list(first_network.classifier.children())[:-remaining_classifier_len])

    filename_x = EXPERIMENT_DIR + 'first_cnn_codes_train_X.h5'
    filename_y = EXPERIMENT_DIR + 'first_cnn_codes_train_y.h5'
    extract_cnn_codes(first_network, data_loaders['train'], filename_x, filename_y, logger, args.BATCH_SIZE,
                      dataset_train, use_gpu)
    filename_x = EXPERIMENT_DIR + 'first_cnn_codes_val_X.h5'
    filename_y = EXPERIMENT_DIR + 'first_cnn_codes_val_y.h5'
    extract_cnn_codes(first_network, data_loaders['val'], filename_x, filename_y, logger, args.BATCH_SIZE,
                      dataset_val, use_gpu)
    filename_x = EXPERIMENT_DIR + 'first_cnn_codes_test_X.h5'
    filename_y = EXPERIMENT_DIR + 'first_cnn_codes_test_y.h5'
    extract_cnn_codes(first_network, data_loaders['test'], filename_x, filename_y, logger, args.BATCH_SIZE,
                      dataset_test, use_gpu)

    remaining_classifier_len = len(second_network.classifier)
    second_network.classifier = nn.Sequential(*list(second_network.classifier.children())[:-remaining_classifier_len])

    filename_x = EXPERIMENT_DIR + 'second_cnn_codes_train_X.h5'
    filename_y = EXPERIMENT_DIR + 'second_cnn_codes_train_y.h5'
    extract_cnn_codes(second_network, data_loaders['train'], filename_x, filename_y, logger, args.BATCH_SIZE,
                      dataset_train, use_gpu)
    filename_x = EXPERIMENT_DIR + 'second_cnn_codes_val_X.h5'
    filename_y = EXPERIMENT_DIR + 'second_cnn_codes_val_y.h5'
    extract_cnn_codes(second_network, data_loaders['val'], filename_x, filename_y, logger, args.BATCH_SIZE,
                      dataset_val, use_gpu)
    filename_x = EXPERIMENT_DIR + 'second_cnn_codes_test_X.h5'
    filename_y = EXPERIMENT_DIR + 'second_cnn_codes_test_y.h5'
    extract_cnn_codes(second_network, data_loaders['test'], filename_x, filename_y, logger, args.BATCH_SIZE,
                      dataset_test, use_gpu)

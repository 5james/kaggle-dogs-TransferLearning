import logging
from torch.autograd import Variable
from argparse import ArgumentParser, RawTextHelpFormatter
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler, random_split
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
from tensorboardX import SummaryWriter
import sys
import matplotlib
import copy
import pickle
import random
import re
import torchnet
import plotly
import plotly.graph_objs as go

sys.path.append("..")
from eval import *

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


# parser.add_argument("-fln", "--freeze_layers_number", dest="FREEZE_LAYERS_NUMBER", help="how many layers should be"
#                                                                                         "NOT frozen "
#                                                                                         "(counting from the end)",
#                     type=int, default=6)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    confusion_matrix = torchnet.meter.ConfusionMeter(NUM_CLASSES)
    accuracy_meter = torchnet.meter.ClassErrorMeter(topk=[1, 5], accuracy=True)
    total_loss_meter = 0
    auc_meter_list = [torchnet.meter.AUCMeter() for _ in range(NUM_CLASSES)]
    auc_avg = torchnet.meter.AUCMeter()

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

            if phase == 'train':
                writer.add_scalar('Train/Loss', epoch_loss, epoch)
                writer.add_scalar('Train/Accuracy-top1', accuracy_meter.value(k=1), epoch)
                writer.add_scalar('Train/Accuracy-top5', accuracy_meter.value(k=5), epoch)
            elif phase == 'val':
                writer.add_scalar('Val/Loss', epoch_loss, epoch)
                writer.add_scalar('Val/Accuracy-top1', accuracy_meter.value(k=1), epoch)
                writer.add_scalar('Val/Accuracy-top5', accuracy_meter.value(k=5), epoch)

            # ROC curve
            if epoch % 5 == 0:
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
                    color = 'rgb({}, {}, {})'.format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
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
                }, auto_open=False, filename='{}-{}.html'.format(epoch, phase))

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

    FREEZE_LAYERS_NUMBER = 6


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
    # newly created layers have requires_grad == True
    model_ft.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(512 * 7 * 7, 4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(4096, NUM_CLASSES)
    )

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
    logger.info(idx_to_class)

    # check if can use GPU
    device = torch.device("cuda" if torch.cuda.is_available() and not args.NOGPU else "cpu")
    model_ft = model_ft.to(device)

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

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=args.EPOCHS)

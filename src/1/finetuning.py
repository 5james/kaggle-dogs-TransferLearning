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
import copy
import pickle
import re

DATA_DIR = '../data/'
LOG_DIR = 'logs/'

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

NUM_CLASSES = 120

IMAGE_INPUT_SIZE = 224

TRAIN_PART = 0.8
VALIDATION_PART = 0.1
# TEST_PART = 0.1

parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
parser.add_argument("-s", "--split", dest="SPLIT", help="split between amount of training and testing images",
                    type=float, default=0.85)
parser.add_argument("-mf", "--model_filename", dest="MODEL_FILENAME",
                    help="name (path) to model of CNN to be saved",
                    type=str, default="model")
parser.add_argument("--batch_size", dest="BATCH_SIZE", help="DataLoader batch size",
                    type=int, default=64)
parser.add_argument("--num_workers", dest="NUM_WORKERS", help="DataLoader number of workers",
                    type=int, default=0)
parser.add_argument("--epochs", dest="EPOCHS", help="number of epochs",
                    type=int, default=100)
parser.add_argument("--learning_rate", dest="LEARNING_RATE", help="learning rate",
                    type=float, default=0.001)
parser.add_argument("--weight_decay", dest="WEIGHT_DECAY", help="weight decay (L2)",
                    type=float, default=0)
parser.add_argument("--momentum", dest="MOMENTUM", help="momentum for optimizers",
                    type=float, default=0.9)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc_train = 0.0
    best_acc_val = 0.0

    for epoch in range(num_epochs):
        logger.info('-' * 60)
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            running_corrects_preterm = 0
            preterm_size = 0
            if torch.__version__ == '0.3.1b0+4cf3225':
                confusion_matrix = torch.zeros(2, 2)
            else:
                confusion_matrix = torch.zeros([2, 2], dtype=torch.int32)

            # Iterate over data.
            for data in data_loaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if torch.__version__ == '0.3.1b0+4cf3225':
                    if use_gpu:
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)
                else:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                # if use_gpu:
                #     inputs = Variable(inputs.cuda())
                #     labels = Variable(labels.cuda())
                # else:
                #     inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if phase == 'train':
                    # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                    outputs, aux_outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss1 = criterion(outputs, labels)
                    loss2 = criterion(aux_outputs, labels)
                    loss = loss1 + 0.4 * loss2
                else:
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                if torch.__version__ == '0.3.1b0+4cf3225':
                    running_loss += float(loss.data[0]) * float(inputs.size(0))
                    # print(float(loss.data[0]), float(inputs.size(0)),
                    #       float(float(loss.data[0]) * float(inputs.size(0))), running_loss)
                else:
                    running_loss += float(loss.item()) * float(inputs.size(0))
                    # print(loss, loss.item(), inputs.size(0))
                running_corrects += torch.sum(preds == labels.data)

                for b_idx in range(args.BATCH_SIZE):
                    if int(labels.data[b_idx]) == 1:
                        preterm_size += 1
                        if int(preds[b_idx]) == 1:
                            running_corrects_preterm += 1
                    if int(preds[b_idx]) == 0 or int(preds[b_idx]) == 1:
                        confusion_matrix[int(labels.data[b_idx])][int(preds[b_idx])] += 1
                    # if int(labels.data[b_idx]) == 1 and not is_inception:
                    #     logger.info('Prediction: {}, outputs: {:.2f}, {:.2f}'.format(int(preds[b_idx]),
                    #                                                                float(outputs[b_idx][0]),
                    #                                                                float(outputs[b_idx][1])))
                    # elif int(labels.data[b_idx]) == 1:
                    #     logger.info('Prediction: {}, outputs main: {:.2f}, {:.2f}\t'
                    #                 '   aux: {:.2f}, {:.2f}'.format(int(preds[b_idx]),
                    #                                             float(outputs[b_idx][0]),
                    #                                             float(outputs[b_idx][1]),
                    #                                             float(aux_outputs[b_idx][0]),
                    #                                             float(aux_outputs[b_idx][1])))

            epoch_loss = running_loss / datasets_len[phase]
            epoch_acc = int(running_corrects) / datasets_len[phase]
            epoch_acc_preterm = int(running_corrects_preterm) / preterm_size

            logger.info('{} Loss: {:.4f} Acc: {:.4f}, Preterm_Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc,
                                                                                  epoch_acc_preterm))
            logger.info('\n' + str(confusion_matrix))
            if phase == 'train':
                writer.add_scalar('Train/Loss', epoch_loss, epoch)
                writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
            elif phase == 'val':
                writer.add_scalar('Test/Loss', epoch_loss, epoch)
                writer.add_scalar('Test/Accuracy', epoch_acc, epoch)

            # # deep copy and save the model with best accuracy
            # if phase == 'val' and epoch_acc >= best_acc_val:
            #     best_acc_val = epoch_acc
            #     torch.save(model.state_dict(), MODEL_FILENAME + '_test')
            #     logger.info('[test] Model with best accuracy overall saved.')
            #
            # # deep copy and save the model with best preterm accuracy
            # if phase == 'val' and epoch_acc_preterm >= best_acc_preterm_val:
            #     best_acc_preterm_val = epoch_acc_preterm
            #     torch.save(model.state_dict(), MODEL_FILENAME + '_test_preterm')
            #     logger.info('[test] Model with best accuracy in preterm saved.')
            # # deep copy and save the model with best accuracy
            # if phase == 'train' and epoch_acc >= best_acc_train:
            #     best_acc_train = epoch_acc
            #     torch.save(model.state_dict(), MODEL_FILENAME + '_train')
            #     logger.info('[train] Model with best accuracy overall saved.')
            #
            # # deep copy and save the model with best preterm accuracy
            # if phase == 'train' and epoch_acc_preterm >= best_acc_preterm_train:
            #     best_acc_preterm_train = epoch_acc_preterm
            #     torch.save(model.state_dict(), MODEL_FILENAME + '_train_preterm')
            #     logger.info('[train] Model with best accuracy in preterm saved.')

        logger.info('\n')

    # time_elapsed = time.time() - since
    # logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Best val Acc val: {:4f}'.format(best_acc_val))
    logger.info('Best val Acc train: {:4f}'.format(best_acc_train))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    # parse args
    args = parser.parse_args()

    # create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # make sure logging directory 'logs' is available
    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)

    # create file handler which logs messages
    fh = logging.FileHandler(LOG_DIR + str(os.path.basename(__file__).split('.')[0]) + '.log')
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
    tensorboard_dir = LOG_DIR + 'tensorboard'
    if not os.path.isdir(tensorboard_dir):
        os.mkdir(tensorboard_dir)
    writer = SummaryWriter(tensorboard_dir)

    # FREEZE_LAYERS_NUMBER = 1
    #
    #
    # def freeze_params(parameters):
    #     freeze_idx = 0
    #     for para in parameters:
    #         if freeze_idx >= FREEZE_LAYERS_NUMBER:
    #             para.requires_grad = False
    #         freeze_idx += 1

    # create model
    model_ft = models.vgg16(pretrained=True)
    model_ft.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(512 * 7 * 7, 4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(4096, NUM_CLASSES)
    )

    # freeze all fully-connected layers
    for para in model_ft.classifier.parameters():
        para.requires_grad = False
    # freeze_params(model_ft.parameters())

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
    datasets_len = {'train': int(dataset_all_len * TRAIN_PART), 'val': int(dataset_all_len - VALIDATION_PART)}
    datasets_len['test'] = int(dataset_all_len - datasets_len['train'] - datasets_len['val'])
    dataset_train, dataset_val, dataset_test = random_split(dataset_all,
                                                            [datasets_len['train'],
                                                             datasets_len['val'],
                                                             datasets_len['test']])
    dataset_train.trasform = data_transforms['train']
    dataset_val.trasform = data_transforms['val']
    dataset_test.trasform = data_transforms['test']

    train_dataloader = DataLoader(dataset_train, batch_size=args.BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=args.BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(dataset_test, batch_size=args.BATCH_SIZE, shuffle=True)

    data_loaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}

    # check if can use GPU
    if torch.__version__ == '0.3.1b0+4cf3225':
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            model_ft = model_ft.cuda()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    logger.info(
        'SGD: lr = {};  momentum = {}, weight decay = {}'.format(args.LEARNING_RATE, args.MOMENTUM, args.WEIGHT_DECAY))
    optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad, model_ft.parameters()),
                             lr=args.LEARNING_RATE, momentum=args.MOMENTUM, weight_decay=args.WEIGHT_DECAY)

    # Decay LR by a factor of 0.1 every 7 epochs
    step_size = 7
    gamma = 0.1
    logger.info('StepLR: step_size = {};  gamma = {}'.format(step_size, gamma))
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=args.EPOCHS)

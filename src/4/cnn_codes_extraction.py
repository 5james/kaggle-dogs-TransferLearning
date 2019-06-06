import logging
from torch.autograd import Variable
import numpy as np
import tables


def extract_cnn_codes(pretrained_model, train_loader, filename_X, filename_y, logger, batch_size, image_dataset,
                      use_gpu):
    X_training = None
    y_training = None
    f_X = None
    f_Y = None
    logger.info('Start extracting features')
    for num_iteration, value_iteration in enumerate(train_loader):
        print('' + str(num_iteration + 1) + '/' + str(int(len(image_dataset) / batch_size)), end='\r')
        images_batch, target_batch = value_iteration
        if use_gpu:
            input_images_batch = Variable(images_batch.cuda())
            input_target_batch = Variable(target_batch.cuda())
        else:
            input_images_batch = Variable(images_batch)
            input_target_batch = Variable(target_batch)

        X_training_batch_tmp = pretrained_model(input_images_batch).data.cpu().numpy().squeeze()
        X_training_batch_tmp = np.float16(X_training_batch_tmp)
        logger.info('\tBatch shape X: {}'.format(X_training_batch_tmp.shape))
        if f_X is None:
            ROW_SIZE_X = int(X_training_batch_tmp.size / batch_size)
            logger.info('\tRow size X: {}'.format(ROW_SIZE_X))
            NUM_COLUMNS_X = len(image_dataset)
            logger.info('\tNumber of columns X: {}'.format(NUM_COLUMNS_X))
            f_X = tables.open_file(filename_X, mode='w')
            atom_X = tables.Float16Atom()
            array_c_X = f_X.create_earray(f_X.root, 'data', atom_X, (0, ROW_SIZE_X))

        # Reshape training batch and put into table
        # If batch size == 1, then reshape whole X_training_batch_tmp
        # If batch size >  1, then reshape each numpy.ndarray in X_training_batch_tmp
        if batch_size == 1:
            array_c_X.append(np.reshape(X_training_batch_tmp, (1, ROW_SIZE_X)))
        elif batch_size > 1:
            for idx in range(len(images_batch)):
                array_c_X.append(np.reshape(X_training_batch_tmp[idx], (1, ROW_SIZE_X)))
        else:
            raise Exception('Wrong batch size')

        y_training_batch_tmp = input_target_batch.data.cpu().numpy()
        y_training_batch_tmp = np.int8(y_training_batch_tmp)
        logger.info('\tBatch shape y: {}'.format(y_training_batch_tmp.shape))
        if f_Y is None:
            ROW_SIZE_Y = int(y_training_batch_tmp.size / batch_size)
            logger.info('\tRow size y: {}'.format(ROW_SIZE_Y))
            NUM_COLUMNS_Y = len(image_dataset)
            logger.info('\tNumber of columns y: {}'.format(NUM_COLUMNS_Y))
            f_Y = tables.open_file(filename_y, mode='w')
            atom_Y = tables.Int8Atom()
            array_c_Y = f_Y.create_earray(f_Y.root, 'data', atom_Y, (0, ROW_SIZE_Y))

        # Reshape training batch and put into table
        # If batch size == 1, then reshape whole X_training_batch_tmp
        # If batch size >  1, then reshape each numpy.ndarray in X_training_batch_tmp
        if batch_size == 1:
            array_c_Y.append(np.reshape(y_training_batch_tmp, (1, 1)))
        elif batch_size > 1:
            for idx in range(len(target_batch)):
                array_c_Y.append(np.reshape(y_training_batch_tmp[idx], (1, 1)))
        else:
            raise Exception('Wrong batch size')

        # writer.add_scalar('CNN-Codes-extraction/iterations', num_iteration)
        # writer.add_scalar('CNN-Codes-extraction/iterations', num_iteration,
        #                   int((datetime.datetime.now() - time_start_cnn_extracting).total_seconds()))

    if f_X is not None:
        f_X.close()
    if f_Y is not None:
        f_Y.close()

    logger.info('Ended extracting features')

    f_X = tables.open_file(filename_X, mode='r')
    X_training = f_X.root.data.read()
    f_X.close()

    f_Y = tables.open_file(filename_y, mode='r')
    y_training = f_Y.root.data.read().squeeze()
    f_Y.close()

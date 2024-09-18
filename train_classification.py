"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn


from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from sklearn.metrics import confusion_matrix


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=7, type=int, choices=[7, 10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--folder', default='bones_wo_normals', help='bones_wo_normals / bones_with_normals')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True
   

def train(model, trainDataLoader, criterion, optimizer, scheduler, args, device=0, num_classes=7,epoch=100):
    
    mean_correct = []
    all_preds = []
    all_labels = []
    total_celoss = 0.0

    global_step = 0
#     scheduler.step()

    for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
        optimizer.zero_grad()

        points = points.data.numpy()
        points = provider.random_point_dropout(points)
        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        points = torch.Tensor(points)
        points = points.transpose(2, 1)

        if not args.use_cpu:
            points, target = points.to(device), target.to(device)

        pred, trans_feat = model(points)
        loss = criterion(pred, target.long(), trans_feat)
        pred_choice = pred.data.max(1)[1]

        ce_loss = nn.CrossEntropyLoss()(pred, target.long())
        total_celoss += ce_loss.item()

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

        # Acumular predicciones y etiquetas verdaderas
        all_preds.extend(pred_choice.cpu().numpy())
        all_labels.extend(target.cpu().numpy())

        loss.backward()
        optimizer.step()
        global_step += 1

    train_instance_acc = np.mean(mean_correct)
    
    scheduler.step()
    
    return train_instance_acc, total_celoss, all_labels, all_preds


def test(model, loader, criterion, num_class=7, vote_num=1):

    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_celoss = 0.0

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        vote_pool = torch.zeros(target.size()[0], num_class).cuda()

        for _ in range(vote_num):
            pred, trans_feat = classifier(points)
            vote_pool += pred
        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1]

        all_preds.extend(pred_choice.cpu().numpy())
        all_labels.extend(target.cpu().numpy())

        loss = criterion(pred, target.long(), trans_feat)
        total_loss += loss.item()

        ce_loss = nn.CrossEntropyLoss()(pred, target.long())
        total_celoss += ce_loss.item()

        # points = points.transpose(2, 1)
        # pred, _ = classifier(points)
        # pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
 
    return instance_acc, class_acc, total_celoss, all_labels, all_preds


def plot_confusion_matrix(cm, classes, output_file):
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.savefig(output_file)
    # plt.show()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...') 
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = os.path.join('data', args.folder)
    
    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='validation', process_data=args.process_data)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir))

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(), 
            lr=args.learning_rate,          # default 1e-3
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate    # default 1e-4
        )
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            classifier.parameters(),
            lr=args.learning_rate,          # default 1e-3
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate,   # default 1e-4
            amsgrad=False,
            maximize=False,
            foreach=None,
            capturable=False,
            differentiable=False,
            fused=None,
        )
    elif args.optimizer == 'ASGD':
        optimizer = torch.optim.ASGD( 
            classifier.parameters(),   
            lr=args.learning_rate,          # default 1e-3
            lambd = 0.0001,
            alpha = 0.75,
            t0 = 1000000.0,
            weight_decay=args.decay_rate,   # default 1e-4
        )
        
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    hyperpar_file = os.path.join(log_dir, 'hyperpar.txt')
    
    with open(hyperpar_file, 'w') as f:
        f.write("Hyperparameters\n\n")
        f.write(f"Batch size: \t {args.batch_size} \n"
                f"Model: \t {args.model} \n"
                f"Epoch: \t {args.epoch} \n"
                f"Learning rate: \t {args.learning_rate} \n"
                f"Number of points: \t {args.num_point} \n"
                f"Optimizer: \t {args.optimizer} \n"
                f"Log directory: \t {args.log_dir} \n"
                f"Decay rate: \t {args.decay_rate} \n"
                f"Objects folder: \t {args.folder} \n"
                f"Normals: \t {args.use_normals} \n"
                f"Process data: \t {args.process_data} \n"
                f"Use uniform sample: \t {args.use_uniform_sample} \n"
                )
        
    metrics_file = os.path.join(log_dir, 'metrics.txt')
    
    if os.path.isfile(metrics_file):
        with open(metrics_file, 'a+') as f:
            f.write("")
    else:
        with open(metrics_file, 'a+') as f:
            f.write("Epoch\tTrain_Accuracy\tValidation_Accuracy\tCrossEntropy_Train_Loss\tCrossEntropy_Valid_Loss\n")
    
    global train_labels_file, train_preds_file, valid_labels_file, valid_preds_file
    
    train_labels_file = os.path.join(log_dir, 'train_labels.txt')
    train_preds_file = os.path.join(log_dir, 'train_preds.txt')
    valid_labels_file = os.path.join(log_dir, 'valid_labels.txt')
    valid_preds_file = os.path.join(log_dir, 'valid_preds.txt')
    
    if os.path.isfile(train_labels_file):
        with open(train_labels_file, 'r') as file:
            train_labels = file.read()

    if os.path.isfile(train_preds_file):
        with open(train_preds_file, 'r') as file:
            train_preds = file.read()
            
    if os.path.isfile(valid_labels_file):
        with open(valid_labels_file, 'r') as file:
            valid_labels = file.read()

    if os.path.isfile(valid_preds_file):
        with open(valid_preds_file, 'r') as file:
            valid_preds = file.read()
    
    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
                
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        classifier=classifier.train()
        train_instance_acc, train_celoss, train_labels, train_preds = train(classifier, trainDataLoader, criterion, optimizer, scheduler, args, 0, args.num_category, epoch)
        
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():

            valid_instance_acc, class_acc, valid_celoss, valid_labels, valid_preds = test(classifier.eval(), testDataLoader, criterion, num_class=num_class)

            if (valid_instance_acc >= best_instance_acc):
                best_instance_acc = valid_instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc

            log_string('Validation Instance Accuracy: %f, Class Accuracy: %f' % (valid_instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (valid_instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': valid_instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            
            global_epoch += 1

        with open(metrics_file, 'a+') as f:
            f.write(f"{epoch}\t{train_instance_acc}\t{valid_instance_acc}\t{train_celoss}\t{valid_celoss}\n")
    
    logger.info('End of training...')

    with open(train_labels_file, 'w') as f:
        f.write(f"{train_labels}")      
    with open(train_preds_file, 'w') as f:
        f.write(f"{train_preds}")
    with open(valid_labels_file, 'w') as f:
        f.write(f"{valid_labels}")
    with open(valid_preds_file, 'w') as f:
        f.write(f"{valid_preds}")        


if __name__ == '__main__':
    
    args = parse_args()
    main(args)
"""
Author: Benny
Date: Nov 2019
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, cohen_kappa_score, matthews_corrcoef, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=5, type=int, choices=[7, 10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    return parser.parse_args()


def test(model, loader, num_class=7, vote_num=1):
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))
    all_preds = []
    all_labels = []

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        vote_pool = torch.zeros(target.size()[0], num_class).cuda()

        for _ in range(vote_num):
            pred, _ = classifier(points)
            vote_pool += pred
        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1]

        all_preds.extend(pred_choice.cpu().numpy())
        all_labels.extend(target.cpu().numpy())

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc, all_labels, all_preds


def main(args):

    python_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(python_path)

    names_path = os.path.join(script_dir,os.path.join(r'data\bones_with_normals_tri_red',classes_txt))
    with open(names_path, 'r', encoding='utf-8') as text_file:
        classes = [line.strip() for line in text_file.readlines()]

    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir
    logs_dir = os.path.join (experiment_dir, 'logs')
    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % logs_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/bones_with_normals_tri_red/'

    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    files = os.listdir(experiment_dir + '/logs')
    model_file = [f for f in files if f.startswith('pointnet')]
    model_name = model_file[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    log_dir = os.path.join(experiment_dir, 'logs')

    test_labels_file = os.path.join(log_dir, 'test_labels.txt')
    test_preds_file = os.path.join(log_dir, 'test_preds.txt')

    with torch.no_grad():
        instance_acc, class_acc, test_labels, test_preds = test(classifier.eval(), testDataLoader, vote_num=args.num_votes, num_class=num_class)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))


        
    with open(test_labels_file, 'w') as f:
        f.write(f"{test_labels}")
    with open(test_preds_file, 'w') as f:
        f.write(f"{test_preds}") 


if __name__ == '__main__':

    classes_txt = 'shape_names.txt'
    colours = ['lightcoral', 'gold', 'greenyellow', 'green', 'indigo', 'lightblue', 'pink', 'teal', 'fuchsia', 'moccasin', 'slategrey', 'lightgreen', 'lavender', 'chocolate', 'orange']

    args = parse_args()
    main(args)

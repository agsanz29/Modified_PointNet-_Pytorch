from matplotlib import axis
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, cohen_kappa_score, matthews_corrcoef, precision_recall_curve, f1_score, auc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def parse_args():
    
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')

    parser.add_argument('--folder', default='', help='same as before')
    parser.add_argument('--log_dir', default='', help='same as before')
    return parser.parse_args()
    
    return parser.parse_args()


def counting_classes(path, classes_txt, file):
    
    names_path = os.path.join(path,classes_txt)
    with open(names_path, 'r', encoding='utf-8') as text_file:
        classes = [line.strip() for line in text_file.readlines()]
        
    counter = {clase: 0 for clase in classes}
    
    part_class = os.path.join(path, file)
    with open(part_class, 'r') as f:
        for line in f:
            archivo = line.strip()
            for clase in classes:
                if archivo.startswith(clase):
                    counter[clase] +=1
    
    vector = [counter[clase] for clase in classes]
    
    return vector


def read_file(path_folder, name):
    path_folder = os.path.join(path_folder, name)

    with open(path_folder, 'r') as file:
        vectorimport = file.read()
    # Convertir los datos en una lista de enteros
    data_list = vectorimport.strip().replace('[', '').replace(']', '').split(',')
    numbers = [int(num) for num in data_list]
    vector = np.array(numbers).reshape(-1, 1)
    
    return vector


def plot_confusion_matrix(cm, classes, part, output_file):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label', fontstyle ='italic')
    plt.xlabel('Predicted label', fontstyle ='italic')
    plt.title(f'Multiclass {part} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_file)


def fun_multiclass_metrics (label, pred):
    
    accuracy = accuracy_score(label, pred)
    balanced_accuracy = balanced_accuracy_score (label, pred)
    micro_avg_precision = precision_score (label, pred, average='weighted')
    micro_avg_recall = recall_score (label, pred, average='weighted')
    f1score = f1_score(label, pred, average='weighted')
    kappa_cohen = cohen_kappa_score (label, pred)
    MCC = matthews_corrcoef (label, pred)

    metrics = [round(accuracy, 4), round(balanced_accuracy, 4), round(float(micro_avg_precision), 4), round(float(micro_avg_recall), 4), round(float(f1score), 4), round(kappa_cohen, 4), round(MCC, 4)]

    return metrics


def conversion(label,pred,i):

    for j in range(len(label)):
        if label[j] == i:
            label[j] = 0
        else:
            label[j] = 1
    for k in range(len(pred)):
        if pred[k] == i:
            pred[k] = 0
        else:
            pred[k] = 1

    return label, pred


def multiclass_metrics(label, preds):

    metrics_name = ["Accuracy", "Balanced Accuracy", "Weighted Precision", "Weighted Recall", "Weighted F1 score", "Kappa Cohen", "MCC"]
    multiclass_metrics_test = fun_multiclass_metrics(label, preds)

    return multiclass_metrics_test, metrics_name


def multiclass_output (logs_dir, save_dir, object_folder_names, classes, colours):

    test_labels = read_file(logs_dir, 'test_labels.txt')        
    test_preds = read_file(logs_dir, 'test_preds.txt')  
    
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    colours_to_index = {colour: idx for idx, colour in enumerate(colours)}

    fig, axs = plt.subplots(2, 2, figsize=(30, 20))
    
    num_axis_fontsize = 20
    legend_fontsize = 20
    axis_fontsize = 25
    title_fontsize = 27

    x = [0, 1]
    y = [0, 1]

    fpr_total_test = []
    tpr_total_test = []

    for i, cls in enumerate(class_to_index.keys()): 

        label_test = read_file(logs_dir, 'test_labels.txt') 
        pred_test = read_file(logs_dir, 'test_preds.txt') 

        bin_label_test, bin_pred_test = conversion(label_test, pred_test, i)
        fpr_bin_test, tpr_bin_test, _ = roc_curve(bin_label_test, bin_pred_test)

        title = list(class_to_index.keys())[i]
        colour_sel = list(colours_to_index.keys())[i]

        axs[0, 0].plot(fpr_bin_test, tpr_bin_test, color=colour_sel, label=title)
        axs[0, 0].plot(x, y, 'k--', linewidth=0.5)
        axs[0, 0].set_ylabel('Precision', fontsize=axis_fontsize)
        axs[0, 0].set_xlabel('Recall', fontsize=axis_fontsize)
        axs[0, 0].set_title('Test ROC Curve', fontsize=title_fontsize)
        axs[0, 0].tick_params(axis='both', labelsize=num_axis_fontsize)

        fpr_total_test.append(fpr_bin_test)        
        tpr_total_test.append(tpr_bin_test)
        
    num_train_classes = counting_classes(object_folder_names, 'shape_names.txt', 'train.txt')
    fpr_mean_test = np.average(fpr_total_test, weights=num_train_classes, axis=0)
    tpr_mean_test = np.average(tpr_total_test, weights=num_train_classes, axis=0)
    test_auc = auc(fpr_mean_test, tpr_mean_test)

    j = i + 1
    
    axs[0, 0].plot(fpr_mean_test, tpr_mean_test, color=list(colours_to_index.keys())[j], linewidth=4.5, label='Mean AUC')

    axs[0, 0].legend(fontsize = legend_fontsize, loc = 'lower right')

    axs[0, 1].plot(fpr_mean_test, tpr_mean_test, color=list(colours_to_index.keys())[j], label='Test Mean AUC')
    axs[0, 1].plot(x, y, 'k--', linewidth=0.5)
    axs[0, 1].set_ylabel('Precision', fontsize=axis_fontsize)
    axs[0, 1].set_xlabel('Recall', fontsize=axis_fontsize)
    axs[0, 1].set_title('Weighted Arithmetic Mean ROC Curve', fontsize=title_fontsize)
    axs[0, 1].legend(fontsize = legend_fontsize, loc = 'lower right')
    axs[0, 1].tick_params(axis='both', labelsize=num_axis_fontsize)
    axs[0, 1].text(0.765, 0.05, f'Weighted AUC Test = {test_auc:.2f}', fontsize=18, bbox=dict(facecolor='white', alpha=0.5))
    
    multiclass_metrics_test, metrics_name = multiclass_metrics(test_labels, test_preds)

    multiclass_metrics_test = np.array(multiclass_metrics_test).reshape(-1, 1)

    table_metrics = multiclass_metrics_test
    
    formatted_data = [[f'{value:.4f}' for value in row] for row in table_metrics]
    table = axs[1, 1].table(cellText = formatted_data, 
                            rowLabels = metrics_name, 
                            colLabels=['Test'],
                            loc='center',
                            cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(20)
    table.scale(0.5, 3)

    axs[1, 0].axis('off')  
    axs[1, 1].axis('off')  
    
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir,'13_Multiclass_Test_ROC.png'))


def main(classes_txt, colours, args):

    python_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(python_path)
    
    folder = args.log_dir
    object_folder = args.folder
    
    dir = r'C:\Users\calculus\Desktop\Deep_Learning_TissueBank\Class_v2'
    dir_names = os.path.join(dir,os.path.join('data',object_folder))
    dir_files = os.path.join(dir,r'log\classification')

    names_path = os.path.join(dir_names,classes_txt)
    
    with open(names_path, 'r', encoding='utf-8') as text_file:
        classes = [line.strip() for line in text_file.readlines()]

    experiment_dir = os.path.join(dir_files, folder)
    logs_dir = os.path.join (experiment_dir, 'logs')

    test_labels = read_file(logs_dir, 'test_labels.txt')        
    test_preds = read_file(logs_dir, 'test_preds.txt')  
          
    save_dir = os.path.join(experiment_dir, 'output_images')
    
    # Calculate confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plot_confusion_matrix(cm, classes, 'Test', os.path.join(save_dir, '12_multiclass_test_cm.png'))

    multiclass_output (logs_dir, save_dir, dir_names, classes, colours)

if __name__ == '__main__':

    classes_txt = 'shape_names.txt'
    
    colours = ['lightcoral', 'gold', 'greenyellow', 'green', 'indigo', 'lightblue', 'pink', 'teal', 'fuchsia', 'moccasin', 'slategrey', 'lightgreen', 'lavender', 'chocolate', 'orange']
    
    args = parse_args()
        
    main(classes_txt, colours, args)

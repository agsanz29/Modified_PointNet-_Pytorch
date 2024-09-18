from telnetlib import TN3270E
from tkinter import font
from turtle import TPen
from cycler import V
from matplotlib import figure
from matplotlib.lines import lineStyles
import matplotlib.pyplot as plt
import os
import numpy as np
import math
import seaborn as sns

from sklearn.calibration import label_binarize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, cohen_kappa_score, matthews_corrcoef, precision_recall_curve, f1_score, auc
from scipy.interpolate import make_interp_spline

import argparse


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')

    parser.add_argument('--folder', default='', help='same as before')
    parser.add_argument('--log_dir', default='', help='same as before')
    return parser.parse_args()


def data_acc_loss(metrics_path):
    # Inicializar listas para cada columna
    epoch = []
    train_accuracy = []
    valid_accuracy = []
    ce_train_loss = []
    ce_validation_loss = []
    
    # Abrir el archivo y leer las líneas
    with open(metrics_path, 'r') as archivo:
        # Leer todas las líneas del archivo
        lineas = archivo.readlines()

        # Iterar sobre las líneas, empezando desde la segunda línea (índice 1)
        for linea in lineas[1:]:
            # Eliminar el carácter de nueva línea al final y luego dividir por '\t'
            valores = linea.strip().split('\t')
            
            # Asumiendo que tienes 5 columnas
            if len(valores) == 5:
                epoch.append(float(valores[0]))  # Convertir a float si son números
                train_accuracy.append(float(valores[1]))
                valid_accuracy.append(float(valores[2]))
                ce_train_loss.append(float(valores[3]))
                ce_validation_loss.append(float(valores[4]))

    return epoch, train_accuracy, valid_accuracy, ce_train_loss, ce_validation_loss


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


def fit_acc_curve(x,y):
    plot_points = int((x[-1] - x[0] + 1)*5)
    x_fit = np.linspace(x[0], x[-1], plot_points)
    
    n = 4
    coef = np.polyfit(x, y, n)
    poly = np.poly1d(coef)
    y_fit = poly (x_fit)    
    
    return x_fit, y_fit


def fit_loss_curve(x,y):
    plot_points = int((x[-1] - x[0] + 1)*5)
    x_fit = np.linspace(x[0], x[-1], plot_points)
    
    n = 9
    coef = np.polyfit(x, y, n)
    poly = np.poly1d(coef)
    y_fit = poly (x_fit)    
    
    return x_fit, y_fit


def plot_acc_loss(epoch_vec, train_acc, valid_acc, ce_train_loss, ce_validation_loss, output_file):

    epoch_vec_new, train_acc_fit = fit_acc_curve(epoch_vec, train_acc)
    epoch_vec_new, valid_acc_fit = fit_acc_curve(epoch_vec, valid_acc)
    epoch_vec_new, ce_train_loss_fit = fit_loss_curve(epoch_vec, ce_train_loss)
    epoch_vec_new, ce_valid_loss_fit = fit_loss_curve(epoch_vec, ce_validation_loss)
        
    plt.figure(figsize=(20,30))

    tick_fontsize = 20
    axis_fontsize = 25
    title_fontsize = 30
    ancho = 3
    
    plt.subplot(2,1,1)
    plt.plot(epoch_vec, train_acc, label='Train Accuracy', color='cornflowerblue')
    plt.plot(epoch_vec, valid_acc, label='Validation Accuracy', color='darksalmon')
    plt.ylabel('Accuracy', fontsize=axis_fontsize)
    plt.xlabel('Epoch', fontsize=axis_fontsize)
    plt.title('Epoch - Accuracy Curve', fontsize=title_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(fontsize=tick_fontsize)
    plt.plot(epoch_vec_new, train_acc_fit, linestyle=':', linewidth=ancho, color='navy')
    plt.plot(epoch_vec_new, valid_acc_fit, linestyle=':', linewidth=ancho, color='red')

    plt.subplot(2,1,2)
    plt.plot(epoch_vec, ce_train_loss, label='Train Loss', color='cornflowerblue')
    plt.plot(epoch_vec, ce_validation_loss, label='Valid Loss', color='darksalmon')
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.ylabel('Loss', fontsize=axis_fontsize)
    plt.xlabel('Epoch', fontsize=axis_fontsize)
    plt.title('Epoch - CrossEntropy Loss Curve', fontsize=title_fontsize)
    plt.legend(fontsize=tick_fontsize)
    plt.plot(epoch_vec_new, ce_train_loss_fit, linestyle=':', linewidth=ancho, color='navy')
    plt.plot(epoch_vec_new, ce_valid_loss_fit, linestyle=':', linewidth=ancho, color='red')

    plt.savefig(output_file, bbox_inches='tight')


def read_file(path_folder, name):
    path_folder = os.path.join(path_folder, name)

    with open(path_folder, 'r') as file:
        vectorimport = file.read()
    # Convertir los datos en una lista de enteros
    data_list = vectorimport.strip().replace('[', '').replace(']', '').split(',')
    numbers = [int(num) for num in data_list]
    vector = np.array(numbers).reshape(-1, 1)
    
    return vector


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


def binary_metrics(label, pred):

    accuracy = accuracy_score(label, pred)
    precision = precision_score (label, pred)
    recall = recall_score (label, pred)
    f1score = f1_score(label, pred)
    kappa_cohen = cohen_kappa_score (label, pred)
    MCC = matthews_corrcoef (label, pred)

    metrics = [round(accuracy, 4), round(float(precision), 4), round(float(recall), 4), round(float(f1score), 4), round(kappa_cohen, 4), round(MCC, 4)]

    return metrics


def training_read(path):

    label_train = read_file(path,'train_labels.txt')
    pred_train = read_file(path,'train_preds.txt')
    label_valid = read_file(path,'valid_labels.txt')
    pred_valid = read_file(path,'valid_preds.txt')

    return label_train, pred_train, label_valid, pred_valid    # label_train, pred_train, label_valid, pred_valid = training_read(path)


def image_binary_training (path, class_to_index, i, output_file):

    label_train, pred_train, label_valid, pred_valid = training_read(path)

    bin_label_train, bin_pred_train = conversion(label_train, pred_train, i)
    bin_label_valid, bin_pred_valid = conversion(label_valid, pred_valid, i)

    cm_bin_train = confusion_matrix(bin_label_train, bin_pred_train)
    cm_bin_valid = confusion_matrix(bin_label_valid, bin_pred_valid)

    fpr_bin_train, tpr_bin_train, _ = roc_curve(bin_label_train, bin_pred_train)
    fpr_bin_valid, tpr_bin_valid, _ = roc_curve(bin_label_valid, bin_pred_valid)

    auc_bin_train = roc_auc_score(bin_label_train, bin_pred_train)
    auc_bin_valid = roc_auc_score(bin_label_valid, bin_pred_valid)

    title = list(class_to_index.keys())[i]

    train_metrics = binary_metrics(bin_label_train, bin_pred_train)
    valid_metrics = binary_metrics(bin_label_valid, bin_pred_valid)

    train_metrics = np.array(train_metrics).reshape(-1, 1)
    valid_metrics = np.array(valid_metrics).reshape(-1, 1)

    table_metrics = np.hstack((train_metrics, valid_metrics))
    metrics_name = ["Accuracy", "Precision", "Recall", "F1 Score", "Kappa Cohen", "MCC"]

    fig, axs = plt.subplots(2, 2, figsize=(30, 20))
    
    subaxis_fontsize = 20
    axis_fontsize = 24
    title_fontsize = 28

    x = [0, 1]
    y = [0, 1]

    # Matriz de confusión - subplot 1
    sns.heatmap(cm_bin_train, annot=True, fmt="d", cmap="Blues", xticklabels=['Positive', 'Negative'], yticklabels=['Positive', 'Negative'], ax=axs[0, 0], annot_kws={"fontsize":25})
    axs[0, 0].set_ylabel('True label', fontsize=axis_fontsize, fontstyle ='italic')
    axs[0, 0].set_xlabel('Predicted label', fontsize=axis_fontsize, fontstyle ='italic')
    axs[0, 0].set_title(f'Train: {title} vs All', fontsize=title_fontsize)
    axs[0, 0].tick_params(axis='both', labelsize = subaxis_fontsize)

    # Matriz de confusión - subplot 2
    sns.heatmap(cm_bin_valid, annot=True, fmt="d", cmap="Blues", xticklabels=['Positive', 'Negative'], yticklabels=['Positive', 'Negative'], ax=axs[1,0], annot_kws={"fontsize":20})
    axs[1, 0].set_ylabel('True label', fontsize=axis_fontsize, fontstyle ='italic')
    axs[1, 0].set_xlabel('Predicted label', fontsize=axis_fontsize, fontstyle ='italic')
    axs[1, 0].set_title(f'Validation: {title} vs All', fontsize=title_fontsize)
    axs[1, 0].tick_params(axis='both', labelsize = subaxis_fontsize)

    # Curva ROC - subplot 3
    axs[0, 1].plot(fpr_bin_train, tpr_bin_train, color='darkorange', label='Training')
    axs[0, 1].plot(fpr_bin_valid, tpr_bin_valid, color='darkblue', label='Validation')
    axs[0, 1].plot(x, y, 'k--', linewidth=0.5)
    axs[0, 1].legend(loc='lower right')
    axs[0, 1].text(0.89, 0.075, f'AUC Train = {auc_bin_train:.2f}\nAUC Valid = {auc_bin_valid:.2f}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    axs[0, 1].set_ylabel('Precision', fontsize=axis_fontsize)
    axs[0, 1].set_xlabel('Recall', fontsize=axis_fontsize)
    axs[0, 1].set_title(f'ROC Curve: {title} vs All', fontsize=title_fontsize)
    axs[0, 1].tick_params(axis='both', labelsize=subaxis_fontsize)

    formatted_data = [[f'{value:.4f}' for value in row] for row in table_metrics]
    # Tabla de métricas - subplot 4
    table = axs[1, 1].table(cellText=formatted_data,
                            rowLabels=metrics_name,
                            colLabels=['Training', 'Validation'],
                            loc='center',
                            cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(20)
    table.scale(0.75, 3)

    axs[1, 1].axis('off')     

    plt.savefig(output_file, bbox_inches = 'tight')


def binary_training(path, pathsave, classes):

    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    nclasses = max(class_to_index.values()) + 1

    fig, axs = plt.subplots(nclasses, 4, figsize=(60, 60))
    
    for i, cls in enumerate(class_to_index.keys()):  
        index = str(i + 5).zfill(2)
        name = f'{index}_Binary_Training_{cls}.png'
        output_file = os.path.join(pathsave, name)

        image_binary_training (path, class_to_index, i, output_file)


def plot_multicm(cm, classes, part, output_file):

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label', fontstyle ='italic')
    plt.xlabel('Predicted label', fontstyle ='italic')
    plt.title(f'Multiclass {part} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_file)


def fun_multiclass_metrics (label, pred):
    
    accuracy = accuracy_score(label, pred)
    balance_accuracy = balanced_accuracy_score (label, pred)
    micro_avg_precision = precision_score (label, pred, average='weighted')
    micro_avg_recall = recall_score (label, pred, average='weighted')
    f1score = f1_score(label, pred, average='weighted')
    kappa_cohen = cohen_kappa_score (label, pred)
    MCC = matthews_corrcoef (label, pred)

    metrics = [round(accuracy, 4), round(balance_accuracy, 4), round(float(micro_avg_precision), 4), round(float(micro_avg_recall), 4), round(float(f1score), 4), round(kappa_cohen, 4), round(MCC, 4)]

    return metrics


def multiclass_metrics(path):

    label_train, pred_train, label_valid, pred_valid = training_read(path)

    metrics_name = ["Accuracy", "Balance_Accuracy", "Weighted Precision", "Weighted Recall", "Weighted F1 Score", "Kappa Cohen", "MCC"]

    multiclass_metrics_train = fun_multiclass_metrics(label_train, pred_train)
    multiclass_metrics_valid = fun_multiclass_metrics(label_valid, pred_valid)

    return multiclass_metrics_train, multiclass_metrics_valid, metrics_name


def dim_check(vectors):
    vector3d = []
    for vector in vectors:
        if len(vector) == 2:
            media = np.mean(vector)
            vector_change = np.insert(vector, 1, media)
            vector3d.append(vector_change)
        else:
            vector3d.append(vector)
    return vector3d


def multiclass_output(path, path_save, object_folder_names, classes, colours):

    label_train, pred_train, label_valid, pred_valid = training_read(path)

    cm_train = confusion_matrix(label_train, pred_train)
    multiclass_training_cm = os.path.join(path_save, ' 01_multiclass_training_cm.png')
    plot_multicm(cm_train, classes, 'Training', multiclass_training_cm)

    cm_valid = confusion_matrix(label_valid, pred_valid)
    multiclass_valid_cm = os.path.join(path_save, ' 02_multiclass_valid_cm.png')
    plot_multicm(cm_valid, classes, 'Validation', multiclass_valid_cm)

    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    colours_to_index = {colour: idx for idx, colour in enumerate(colours)}

    fig, axs = plt.subplots(2, 3, figsize=(30, 20))
      
    num_axis_fontsize = 15
    legend_fontsize = 15
    axis_fontsize = 25
    title_fontsize = 27

    x = [0, 1]
    y = [0, 1]

    fpr_total_train = []
    tpr_total_train = []
    fpr_total_valid = []
    tpr_total_valid = []

    for i, cls in enumerate(class_to_index.keys()): 

        label_train, pred_train, label_valid, pred_valid = training_read(path)

        bin_label_train, bin_pred_train = conversion(label_train, pred_train, i)
        bin_label_valid, bin_pred_valid = conversion(label_valid, pred_valid, i)

        fpr_bin_train, tpr_bin_train, _ = roc_curve(bin_label_train, bin_pred_train)
        fpr_bin_valid, tpr_bin_valid, _ = roc_curve(bin_label_valid, bin_pred_valid)
           
        title = list(class_to_index.keys())[i]
        colour_sel = list(colours_to_index.keys())[i]

        axs[0, 0].plot(fpr_bin_train, tpr_bin_train, color=colour_sel, label=title)
        axs[0, 0].plot(x, y, 'k--', linewidth=0.5)
        axs[0, 0].set_ylabel('Precision', fontsize=axis_fontsize)
        axs[0, 0].set_xlabel('Recall', fontsize=axis_fontsize)
        axs[0, 0].set_title('Training ROC Curve', fontsize=title_fontsize)
        
        axs[0, 0].tick_params(axis='both', labelsize=num_axis_fontsize)


        axs[0, 1].plot(fpr_bin_valid, tpr_bin_valid, color=colour_sel, label=title)
        axs[0, 1].plot(x, y, 'k--', linewidth=0.5)
        axs[0, 1].set_ylabel('Precision', fontsize=axis_fontsize)
        axs[0, 1].set_xlabel('Recall', fontsize=axis_fontsize)
        axs[0, 1].set_title('Validation ROC Curve', fontsize=title_fontsize)

        axs[0, 1].tick_params(axis='both', labelsize=num_axis_fontsize)

        fpr_total_train.append(fpr_bin_train)
        tpr_total_train.append(tpr_bin_train)            
        fpr_total_valid.append(fpr_bin_valid)              
        tpr_total_valid.append(tpr_bin_valid)  
    
    fpr_total_train = dim_check(fpr_total_train)
    tpr_total_train = dim_check(tpr_total_train)
    fpr_total_valid = dim_check(fpr_total_valid)
    tpr_total_valid = dim_check(tpr_total_valid)
    
    num_train_classes = counting_classes(object_folder_names, 'shape_names.txt', 'train.txt')
    fpr_mean_train = np.average(fpr_total_train, weights=num_train_classes, axis=0)
    tpr_mean_train = np.average(tpr_total_train, weights=num_train_classes, axis=0)
    train_auc = auc(fpr_mean_train, tpr_mean_train)
    
    num_valid_classes = counting_classes(object_folder_names, 'shape_names.txt', 'validation.txt')
    fpr_mean_valid = np.average(fpr_total_valid, weights=num_valid_classes, axis=0)   
    tpr_mean_valid = np.average(tpr_total_valid, weights=num_valid_classes, axis=0)
    valid_auc = auc(fpr_mean_valid, tpr_mean_valid)

    j = i + 1
    k = i + 2

    axs[0, 0].plot(fpr_mean_train, tpr_mean_train, color=list(colours_to_index.keys())[j], linewidth=4.5, label='Mean AUC')
    axs[0, 1].plot(fpr_mean_valid, tpr_mean_valid, color=list(colours_to_index.keys())[k], linewidth=4.5, label='Mean AUC')

    axs[0, 0].legend(fontsize = legend_fontsize)
    axs[0, 1].legend(fontsize = legend_fontsize)    

    axs[0, 2].plot(fpr_mean_train, tpr_mean_train, color=list(colours_to_index.keys())[j], label='Training Mean AUC')
    axs[0, 2].plot(fpr_mean_valid, tpr_mean_valid, color=list(colours_to_index.keys())[k], label='Validation Mean AUC')
    axs[0, 2].plot(x, y, 'k--', linewidth=0.5)
    axs[0, 2].set_ylabel('Precision', fontsize=axis_fontsize)
    axs[0, 2].set_xlabel('Recall', fontsize=axis_fontsize)
    axs[0, 2].set_title('Weighted Arithmetic Mean ROC Curve', fontsize=title_fontsize)
    axs[0, 2].legend(fontsize = legend_fontsize, loc = 'lower right')
    axs[0, 2].tick_params(axis='both', labelsize=num_axis_fontsize)
    axs[0, 2].text(0.66, 0.1, f'Weigthed AUC Train = {train_auc:.2f}\nWeighted AUC Valid = {valid_auc:.2f}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    multiclass_metrics_train, multiclass_metrics_valid, metrics_name = multiclass_metrics(path)

    multiclass_metrics_train = np.array(multiclass_metrics_train).reshape(-1, 1)
    multiclass_metrics_valid = np.array(multiclass_metrics_valid).reshape(-1, 1)

    table_metrics = np.hstack((multiclass_metrics_train, multiclass_metrics_valid))
    
    formatted_data = [[f'{value:.4f}' for value in row] for row in table_metrics]
    table = axs[1, 1].table(cellText = formatted_data, 
                            rowLabels = metrics_name, 
                            colLabels=['Training', 'Validation'],
                            loc='center',
                            cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(20)
    table.scale(2, 3)

    axs[1, 0].axis('off')  
    axs[1, 1].axis('off')  
    axs[1, 2].axis('off')    
    
    plt.savefig(os.path.join(path_save,'04_Multiclass_Training_ROC.png'), bbox_inches='tight')


def main(metrics_txt, classes_txt, colours, args):

    python_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(python_path)

    folder = args.log_dir
    object_folder = args.folder
    # add_dir = r'log\classification\pointnet2_cls_ssg_normal' # w_normals
    add_dir = os.path.join(r'log\classification', folder)   # wo_normals

    folder_save = os.path.join(script_dir,os.path.join(add_dir,'output_images'))
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)

    folder_names = os.path.join(script_dir,os.path.join(r'data',folder))
    object_folder_names = os.path.join(script_dir, os.path.join(r'data',object_folder))
    names_path = os.path.join(object_folder_names,classes_txt)
    with open(names_path, 'r', encoding='utf-8') as text_file:
        classes = [line.strip() for line in text_file.readlines()]

    python_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(python_path)
    logs_path = os.path.join(script_dir,os.path.join(add_dir,'logs'))
    metrics_path = os.path.join(logs_path,metrics_txt)

    epoch, train_accuracy, valid_accuracy, ce_train_loss, ce_validation_loss = data_acc_loss(metrics_path)
    plot_acc_loss(epoch, train_accuracy, valid_accuracy, ce_train_loss, ce_validation_loss, os.path.join(folder_save, '03_plot_acc_loss.png'))

    binary_training(logs_path, folder_save, classes)
    multiclass_output(logs_path, folder_save, object_folder_names, classes, colours)


if __name__ == '__main__':

    metrics_txt = 'metrics.txt'
    classes_txt = 'shape_names.txt'

    colours = ['lightcoral', 'gold', 'greenyellow', 'green', 'indigo', 'lightblue', 'pink', 'teal', 'fuchsia', 'moccasin', 'slategrey', 'lightgreen', 'lavender', 'chocolate', 'orange']

    args = parse_args()

    main(metrics_txt, classes_txt, colours, args)
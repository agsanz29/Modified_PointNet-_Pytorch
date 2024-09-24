#######################################
# Define los cojuntos de entrenamiento, test y validación. 
# Crea TXT de las clases para clasificar, listado de los modelos 3D, y particiones.
#######################################

import os
import re
import random
from sklearn.model_selection import train_test_split
from decimal import Decimal, ROUND_DOWN
import argparse


# Define el directorio raíz y los archivos de salida
BASE_DIR = os.path.dirname(__file__)

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default='', help='objects folder')
    return parser.parse_args()


# Función para obtener claves de ordenamiento natural
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

args = parse_args()
folder = args.folder
root_dir = os.path.join(BASE_DIR, os.path.join('data', folder))
output_file = os.path.join(root_dir, 'filelist.txt')
shape_names_file = os.path.join(root_dir, 'shape_names.txt')
train_file = os.path.join(root_dir, 'train.txt')
validation_file = os.path.join(root_dir,'validation.txt')
test_file = os.path.join(root_dir, 'test.txt')

# Listas para almacenar las rutas de los archivos y nombres de subcarpetas
file_paths = []
subfolders = []


# Recorre el directorio
for subdir, _, files in os.walk(root_dir):
    # Obtiene la ruta relativa del subdirectorio
    rel_dir = os.path.relpath(subdir, root_dir)
    if rel_dir != ".":
        subfolders.append(rel_dir)
    for file in files:
        # Verifica si el archivo es un .txt y no es el archivo de salida
        if file.endswith('.txt') and file not in ['filelist.txt', 'shape_names.txt', 'train.txt', 'validation.txt','test.txt']:
            # Agrega la ruta relativa del archivo a la lista
            file_paths.append(f"{rel_dir}/{file}")


# Ordena las rutas de los archivos y nombres de subcarpetas usando la función de ordenamiento natural
file_paths.sort(key=natural_sort_key)
subfolders.sort()


# Escribe las rutas de los archivos en el archivo de salida
with open(output_file, 'w') as f:
    for file_path in file_paths:
        f.write(f"{file_path}\n")


# Escribe los nombres de las subcarpetas en el archivo shape_names.txt
with open(shape_names_file, 'w') as f:
    for subfolder in subfolders:
        f.write(f"{subfolder}\n")


# Divide los archivos en train y test (80%-20%)
# random.shuffle(file_paths)
# split_index = int(len(file_paths) * 0.8)
# train_files = file_paths[:split_index]
# test_files = file_paths[split_index:]


def create_labels(file_paths):
    labels = [os.path.dirname(path) for path in file_paths]
    return labels

def split_data(X, y, train_size, val_size, test_size, random_state=None):
    """
    Divide los datos en conjuntos de entrenamiento, validación y prueba.

    Parameters:
    - X: Características (features)
    - y: Etiquetas (labels)
    - train_size: Proporción de datos para el conjunto de entrenamiento
    - val_size: Proporción de datos para el conjunto de validación
    - test_size: Proporción de datos para el conjunto de prueba
    - random_state: Semilla para la aleatoriedad

    Returns:
    - X_train, X_val, X_test, y_train, y_val, y_test
    """
    
    # assert train_size + val_size + test_size == 1.0, "Las proporciones deben sumar 1."

    # Primero, dividimos los datos en entrenamiento + validación y prueba
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Luego, dividimos el conjunto de entrenamiento + validación en entrenamiento y validación
    val_size_adjusted = val_size / (train_size + val_size)  # ajustar el tamaño de validación al nuevo conjunto
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size_adjusted, random_state=random_state, stratify=y_train_val)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

labels = create_labels(file_paths)

# Dividir los datos
train_per = 0.6  
val_per   = 0.2  
test_per  = 0.2   

train_files, val_files, test_files, y_train, y_val, y_test = split_data(file_paths, labels, train_size=train_per, val_size=val_per, test_size=test_per, random_state=95)


# Función para extraer el nombre del archivo sin la ruta y la extensión
def get_file_name(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]


# Extrae los nombres de los archivos sin la ruta y la extensión
train_files = [get_file_name(f) for f in train_files]
val_files = [get_file_name(f) for f in val_files]
test_files = [get_file_name(f) for f in test_files]


# Ordena los nombres de los archivos alfabéticamente
train_files.sort(key=natural_sort_key)
val_files.sort(key=natural_sort_key)
test_files.sort(key=natural_sort_key)


# Escribe los archivos de entrenamiento en train.txt
with open(train_file, 'w') as f:
    for train_file_name in train_files:
        f.write(f"{train_file_name}\n")


# Escribe los archivos de validacion en validation.txt
with open(validation_file, 'w') as f:
    for val_file_name in val_files:
        f.write(f"{val_file_name}\n")


# Escribe los archivos de prueba en test.txt
with open(test_file, 'w') as f:
    for test_file_name in test_files:
        f.write(f"{test_file_name}\n")
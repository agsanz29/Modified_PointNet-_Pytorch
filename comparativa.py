import matplotlib.pyplot as plt
import pandas as pd
import os

def carga_datos(excel_file):
    
    df = pd.read_excel(excel_file)
    
    colum_vectors = {col: df[col].values for col in df.columns}
    
    return colum_vectors    


def grafico(colum_vectors, variable, indices, fig_path):

    df = pd.DataFrame(colum_vectors)

    dicc_filtr = df.iloc(indices)

    if pd.api.types.is_numeric_dtype(dicc_filtr[variable]):
        dicc_filtr = dicc_filtr.sort_values(by=variable)
    
    selected_colums = [variable] + list(dicc_filtr.columns[-2:])
    
    variable_x = selected_colums[0]
    train_acc = selected_colums[1]
    valid_acc = selected_colums[2]
    
    axis_fontsize = 15
    title_fontsize = 25

    plt.plot(variable_x, train_acc, label='Train Accuracy', color='cornflowerblue')
    plt.plot(variable_x, valid_acc, label='Validation Accuracy', color='darksalmon')
    plt.ylabel('Accuracy', fontsize=axis_fontsize)
    # plt.xlabel(x_name, fontsize=axis_fontsize)
    plt.title('Model Accuracy', fontsize=title_fontsize)
    plt.legend()
    
    plt.savefig(fig_path)


def main():
    
    log_path = r'C:\Users\calculus\Desktop\Deep_Learning_TissueBank\Class_v2\log\classification'
    excel_path = os.path.join(os.path.join(log_path, 'comparativa'), 'resumen_logs.xlsx')
    
    colum_vectors = carga_datos(excel_path)
    name = '01_DecayRate'
    variable = 4
    indices = [29, 32, 34]
    fig_path = os.path.join(os.path.join(log_path, 'comparativa'), name)
    
    grafico(colum_vectors, variable, indices, fig_path)



if __name__ == '__main__':
    
    main()
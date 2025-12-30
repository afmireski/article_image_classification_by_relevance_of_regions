# Backup das funções originais
# Este arquivo contém versões de backup das funções que foram modificadas durante o desenvolvimento

import numpy as np
from sklearn.model_selection import StratifiedKFold
from utils import show_predict_infos

def predict_knn_original(knn, X, y, title="", k_folds=5):
    """
    Função original de predição KNN com validação cruzada.
    
    Esta é uma versão de backup da função predict_knn original, mantida para:
    - Compatibilidade com código legado
    - Referência de implementação
    - Comparação de resultados
    
    Args:
        knn: Modelo KNN para treinar
        X (array): Features do dataset
        y (array): Labels do dataset
        title (str): Título para identificação nos logs
        k_folds (int): Número de folds para validação cruzada
        
    Returns:
        None: Função original não retornava valores, apenas imprimia métricas
    """
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    metrics = ([], [], [], [])
    rodada = 1
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        knn.fit(X_train, y_train)

        predict = knn.predict(X_test)

        accuracy, f1, recall, precision = show_predict_infos(y_test, predict, title=f"{title} - Rodada {rodada}")

        metrics[0].append(accuracy)
        metrics[1].append(f1)
        metrics[2].append(recall)
        metrics[3].append(precision)
                          
        rodada += 1

    mean_accuracy, std_accuracy = np.mean(metrics[0]), np.std(metrics[0])
    mean_f1, std_f1 = np.mean(metrics[1]), np.std(metrics[1])
    mean_recall, std_recall = np.mean(metrics[2]), np.std(metrics[2])
    mean_precision, std_precision = np.mean(metrics[3]), np.std(metrics[3])
    
    print(f"Avaliação média do modelo {title}:")
    print("Acurácia: %.2f +- %.2f" % (mean_accuracy, std_accuracy))
    print("F1: %.2f +- %.2f" % (mean_f1, std_f1))
    print("Recall: %.2f +- %.2f" % (mean_recall, std_recall))
    print("Precisão: %.2f +- %.2f" % (mean_precision, std_precision))
    print("====================================\n\n")

    # Atualização da variável global (comportamento original)
    global best_knn
    if 'best_knn' not in globals():
        best_knn = None
    if best_knn is None or mean_f1 > best_knn[2][0]:
        best_knn = (title, (mean_accuracy, std_accuracy), (mean_f1, std_f1), (mean_recall, std_recall), (mean_precision, std_precision))


def predict_dt_original(dt, X, y, title="", k_folds=5):
    """
    Função original de predição Decision Tree com validação cruzada.
    
    Versão de backup da função predict_dt original.
    """
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    metrics = ([], [], [], [])
    rodada = 1
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        dt.fit(X_train, y_train)

        predict = dt.predict(X_test)

        accuracy, f1, recall, precision = show_predict_infos(y_test, predict, title=f"{title} - Rodada {rodada}", cmap="Greens")

        metrics[0].append(accuracy)
        metrics[1].append(f1)
        metrics[2].append(recall)
        metrics[3].append(precision)
                          
        rodada += 1
    
    mean_accuracy, std_accuracy = np.mean(metrics[0]), np.std(metrics[0])
    mean_f1, std_f1 = np.mean(metrics[1]), np.std(metrics[1])
    mean_recall, std_recall = np.mean(metrics[2]), np.std(metrics[2])
    mean_precision, std_precision = np.mean(metrics[3]), np.std(metrics[3])
    
    print(f"Avaliação média do modelo {title}:")
    print("Acurácia: %.2f +- %.2f" % (mean_accuracy, std_accuracy))
    print("F1: %.2f +- %.2f" % (mean_f1, std_f1))
    print("Recall: %.2f +- %.2f" % (mean_recall, std_recall))
    print("Precisão: %.2f +- %.2f" % (mean_precision, std_precision))
    print("====================================\n\n")

    global best_dt
    if 'best_dt' not in globals():
        best_dt = None
    if best_dt is None or mean_f1 > best_dt[2][0]:
        best_dt = (title, (mean_accuracy, std_accuracy), (mean_f1, std_f1), (mean_recall, std_recall), (mean_precision, std_precision))


def predict_svm_original(svm, X, y, title="", k_folds=5):
    """
    Função original de predição SVM com validação cruzada.
    
    Versão de backup da função predict_svm original.
    """
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    metrics = ([], [], [], [])
    rodada = 1
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        svm.fit(X_train, y_train)

        predict = svm.predict(X_test)

        accuracy, f1, recall, precision = show_predict_infos(y_test, predict, title=f"{title} - Rodada {rodada}", cmap="Reds")

        metrics[0].append(accuracy)
        metrics[1].append(f1)
        metrics[2].append(recall)
        metrics[3].append(precision)
                          
        rodada += 1
    
    mean_accuracy, std_accuracy = np.mean(metrics[0]), np.std(metrics[0])
    mean_f1, std_f1 = np.mean(metrics[1]), np.std(metrics[1])
    mean_recall, std_recall = np.mean(metrics[2]), np.std(metrics[2])
    mean_precision, std_precision = np.mean(metrics[3]), np.std(metrics[3])
    
    print(f"Avaliação média do modelo {title}:")
    print("Acurácia: %.2f +- %.2f" % (mean_accuracy, std_accuracy))
    print("F1: %.2f +- %.2f" % (mean_f1, std_f1))
    print("Recall: %.2f +- %.2f" % (mean_recall, std_recall))
    print("Precisão: %.2f +- %.2f" % (mean_precision, std_precision))
    print("====================================\n\n")

    global best_svm
    if 'best_svm' not in globals():
        best_svm = None
    if best_svm is None or mean_f1 > best_svm[2][0]:
        best_svm = (title, (mean_accuracy, std_accuracy), (mean_f1, std_f1), (mean_recall, std_recall), (mean_precision, std_precision))


# Exemplo de uso das funções de backup:
"""
Para usar as funções originais, importe este módulo:

import backup_functions as backup

# Use a função original
backup.predict_knn_original(model, X, y, title="Teste Original")

# Compare com a nova implementação
new_model, metrics = train_model_knn(model, X, y, title="Teste Novo")
"""

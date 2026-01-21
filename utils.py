# Importa libs úteis para avaliação dos modelos
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from mytypes import ExperimentMetrics, ModelMetrics, StandardExperimentMetrics

def show_predict_infos(y, predict, title="", cmap="Blues", show_plots=True):
    """
    Calcula e exibe métricas de avaliação de modelos de classificação.
    
    Args:
        y: Labels verdadeiros
        predict: Predições do modelo
        title: Título para os gráficos
        cmap: Mapa de cores para a matriz de confusão
        show_plots: Se True, exibe matriz de confusão. Se False, apenas calcula métricas
        
    Returns:
        tuple: (accuracy, f1, recall, precision)
    """
    # Mostra a matriz de confusão apenas se solicitado
    if show_plots:
        show_confusion_matrix(y, predict, title, cmap)

    # Calcula e mostra as métricas de avaliação
    # A acurácia é a proporção de predições corretas sobre o total de predições
    accuracy = accuracy_score(y, predict)
    accuracy_percent = accuracy * 100
    
    if show_plots:
        print(f"A acurácia no conjunto de testes: {accuracy_percent:.2f}%")

    # O recall é a proporção de predições corretas sobre o total de instâncias de uma classe
    recall = recall_score(y, predict, average="macro")
    recall_percent = recall * 100
    
    if show_plots:
        print(f"A recall no conjunto de testes: {recall_percent:.2f}%")

    # A precisão é a proporção de predições corretas sobre o total de predições de uma classe
    precision = precision_score(y, predict, average="macro")
    precision_percent = precision * 100
    
    if show_plots:
        print(f"A precision no conjunto de testes: {precision_percent:.2f}%")

    # A F1 é a média harmônica entre a precisão e o recall
    f1 = f1_score(y, predict, average="macro")
    f1_percent = f1 * 100
    
    if show_plots:
        print(f"A F1 no conjunto de testes: {f1_percent:.2f}%")

        # Mostra um relatório com as métricas de classificação por classe e as métricas calculadas sobre o conjunto todo.
        print("\nRelatório de Classificação")
        print(classification_report(y, predict))

    return accuracy, f1, recall, precision
    
def show_confusion_matrix(y, predict, title="", cmap="Blues", verbose=False, save_dir='results/confusion_matrixs'):
    import os
    
    # Cria a matriz de confusão
    disp = ConfusionMatrixDisplay.from_predictions(y, predict, colorbar=False, cmap=cmap)
    fig = getattr(disp, "figure_", None)
    
    # Adiciona título se fornecido
    if len(title) > 0:
        plt.title(f"Matriz de Confusão {title}")
    plt.xlabel("Rótulo Previsto")
    plt.ylabel("Rótulo Real")
    
    # Salva a matriz de confusão se um título foi fornecido
    if len(title) > 0:
        # Cria o diretório se não existir
        os.makedirs(save_dir, exist_ok=True)
        
        # Gera nome do arquivo baseado no título
        filename = title.lower().replace(" ", "_").replace("-", "_").replace("+", "_")
        filename = "".join(c for c in filename if c.isalnum() or c == "_")  # Remove caracteres especiais
        filepath = os.path.join(save_dir, f"{filename}_confusion_matrix.png")
        
        # Salva a figura
        plt.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0.1)
        if verbose:
            print(f"💾 Matriz de confusão salva em: {filepath}")
    
    # Mostra a matriz apenas se verbose for True
    if verbose:
        plt.show()

    # Fecha a figura para evitar acumular muitas figuras abertas na memória
    if fig is not None:
        plt.close(fig)
    else:
        plt.close()

def show_relevance_experiment_metrics(metrics: ExperimentMetrics, title=""):
    """
    Exibe métricas de avaliação de modelos de classificação.
    
    Args:
        metrics: Tupla com as métricas ((accuracy, f1, recall, precision), especialistas_train_metrics)
        title: Título para exibição
    """
    (accuracy, f1, recall, precision), specialists_train_metrics = metrics

    def print_folds(folds):
        return ", ".join([f"{fold*100:.4f}%" for fold in folds])

    print("#" * 40)    
    print(f"Métricas Finais Relevância {title}:")
    print(f"   📊 Acurácia: {accuracy*100:.4f}%")
    print(f"   📊 F1: {f1*100:.4f}%")
    print(f"   📊 Recall: {recall*100:.4f}%")
    print(f"   📊 Precision: {precision*100:.4f}%")
    print("-" * 40)
    print("Métricas de Treinamento dos Especialistas:")
    for idx, train_metrics in enumerate(specialists_train_metrics):
        sp_accuracy = train_metrics['accuracy']
        sp_f1 = train_metrics['f1']
        sp_recall = train_metrics['recall']
        sp_precision = train_metrics['precision']

        print(f"   Especialista classe {idx}:")
        print(f"      1️⃣ Acurácia Média: {sp_accuracy['mean']*100:.4f}% +- {sp_accuracy['std']*100:.4f}%")
        print(f'        | Folds: {print_folds(sp_accuracy["folds"])}')
        print(f"      2️⃣ F1 Média: {sp_f1['mean']*100:.4f}% +- {sp_f1['std']*100:.4f}%")
        print(f'        | Folds: {print_folds(sp_f1["folds"])}')
        print(f"      3️⃣ Recall Médio: {sp_recall['mean']*100:.4f}% +- {sp_recall['std']*100:.4f}%")
        print(f'        | Folds: {print_folds(sp_recall["folds"])}')
        print(f"      4️⃣ Precision Média: {sp_precision['mean']*100:.4f}% +- {sp_precision['std']*100:.4f}%")
        print(f'        | Folds: {print_folds(sp_precision["folds"])}')
    print("#" * 40)
    1
def show_sum_experiment_metrics(metrics: ModelMetrics, title=""):
    """
    Exibe métricas de avaliação de modelos de classificação.
    
    Args:
        metrics: Tupla com as métricas (accuracy, f1, recall, precision)
        title: Título para exibição
    """
    accuracy, f1, recall, precision = metrics

    print("#" * 40)    
    print(f"Métricas Finais Soma {title}:")
    print(f"   📊 Acurácia: {accuracy*100:.4f}%")
    print(f"   📊 F1: {f1*100:.4f}%")
    print(f"   📊 Recall: {recall*100:.4f}%")
    print(f"   📊 Precision: {precision*100:.4f}%")
    print("-" * 40)
    print("As métricas dos especialistas são as mesmas da técnica de relevância.")
    print("#" * 40)

def show_standard_experiment_metrics(metrics: StandardExperimentMetrics, title=""):
    """
    Exibe métricas de avaliação de modelos de classificação.
    
    Args:
        metrics: Tupla com as métricas ((accuracy, f1, recall, precision), especialistas_train_metrics)
        title: Título para exibição
    """
    (accuracy, f1, recall, precision), train_metrics = metrics

    def print_folds(folds):
        return ", ".join([f"{fold*100:.4f}%" for fold in folds])

    print("#" * 40)    
    print(f"Métricas Finais {title}:")
    print(f"   📊 Acurácia: {accuracy*100:.4f}%")
    print(f"   📊 F1: {f1*100:.4f}%")
    print(f"   📊 Recall: {recall*100:.4f}%")
    print(f"   📊 Precision: {precision*100:.4f}%")
    print("-" * 40)
    print("Métricas do Treinamento:")

    print(train_metrics)
    
    train_accuracy = train_metrics['accuracy']
    train_f1 = train_metrics['f1']
    train_recall = train_metrics['recall']
    train_precision = train_metrics['precision']

    print(f"      1️⃣ Acurácia Média: {train_accuracy['mean']*100:.4f}% +- {train_accuracy['std']*100:.4f}%")
    print(f'        | Folds: {print_folds(train_accuracy["folds"])}')
    print(f"      2️⃣ F1 Média: {train_f1['mean']*100:.4f}% +- {train_f1['std']*100:.4f}%")
    print(f'        | Folds: {print_folds(train_f1["folds"])}')
    print(f"      3️⃣ Recall Médio: {train_recall['mean']*100:.4f}% +- {train_recall['std']*100:.4f}%")
    print(f'        | Folds: {print_folds(train_recall["folds"])}')
    print(f"      4️⃣ Precision Média: {train_precision['mean']*100:.4f}% +- {train_precision['std']*100:.4f}%")
    print(f'        | Folds: {print_folds(train_precision["folds"])}')
        
    print("#" * 40)
    

def show_metrics(metrics: ModelMetrics, title=""):
    """
    Exibe métricas de avaliação de modelos de classificação.
    
    Args:
        metrics: Tupla com as métricas (accuracy, f1, recall, precision))
        title: Título para exibição
    """
    accuracy, f1, recall, precision = metrics

    print("-" * 40)    
    print(f"Métricas {title}:")
    print(f"   📊 Acurácia: {accuracy*100:.4f}%")
    print(f"   📊 F1: {f1*100:.4f}%")
    print(f"   📊 Recall: {recall*100:.4f}%")
    print(f"   📊 Precision: {precision*100:.4f}%")
    
    print("-" * 40)
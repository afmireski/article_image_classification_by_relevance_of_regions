import numpy as np
import os
import json
import csv
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.base import BaseEstimator, clone

from mytypes import (
    MulticlassClassificationDataset,
    PredictResults,
    StandardModelResults,
    StandardClassificationResults,
)

from tools.relevance import compute_metrics
from utils import show_confusion_matrix, show_experiment_metrics
import time


def timed_standard_technique(
    base_model: BaseEstimator,
    folded_dataset: MulticlassClassificationDataset,
    class_names: List[str],
    true_labels: PredictResults,
    model_name: str = "Standard",
    k_folds: int = 5,
    experiment_dir: str = None,
    cmap: str = "Blues"
) -> StandardClassificationResults:
    """
    Wrapper around standard_technique que mede e exibe o tempo de execução.

    Returns:
        O mesmo retorno de standard_technique.
    """
    start = time.perf_counter()
    results = standard_technique(
        base_model=base_model,
        folded_dataset=folded_dataset,
        class_names=class_names,
        true_labels=true_labels,
        model_name=model_name,
        k_folds=k_folds,
        experiment_dir=experiment_dir,
        cmap=cmap,
    )
    end = time.perf_counter()
    elapsed = end - start

    # Formata o tempo decorrido
    if elapsed < 60:
        elapsed_str = f"{elapsed:.2f} s"
    else:
        mins = int(elapsed // 60)
        secs = elapsed % 60
        elapsed_str = f"{mins}m {secs:.2f}s"

    print(f"⏱️  Tempo de execução ({model_name}): {elapsed_str}\n")

    return results


def extract_standard_probabilities(
    base_model: BaseEstimator,
    folded_dataset: MulticlassClassificationDataset,
    title: str = "",
) -> StandardModelResults:
    """
    Extrai probabilidades de todas as classes para classificação multiclasse padrão.

    Args:
        base_model: Modelo sklearn configurado (ex: GridSearchCV)
        folded_dataset: Lista de tuplas ((X_train, y_train, train_images), (X_test, y_test, test_images))
        title: Nome do modelo para logs

    Returns:
        Dicionário {img_id: [prob_class_0, prob_class_1, ..., prob_class_n]} com probabilidades de todas as classes
    """
    probabilities = {}

    print(f"=== Iniciando treinamento padrão: {title} ===")
    print(f"Validação cruzada: {len(folded_dataset)} folds")
    print("-" * 50)

    # Validação cruzada
    for fold_idx, (train_set, test_set) in enumerate(folded_dataset):
        X_train, y_train, train_images = train_set
        X_test, y_test, test_images = test_set

        print(f"Fold {fold_idx + 1}/{len(folded_dataset)}")

        # Clona o modelo base para cada fold
        fold_model = clone(base_model)
        fold_model.fit(X_train, y_train)

        print(f"  Melhores parâmetros: {fold_model.best_params_}")

        # Extrai probabilidades de todas as classes
        predict_proba = fold_model.predict_proba(X_test)

        # Armazena probabilidades para cada imagem
        # Cada imagem aparece em apenas 1 fold de teste, então armazena diretamente o array
        for i, probs in enumerate(predict_proba):
            img_id = test_images[i]
            # Armazena o array completo de probabilidades [prob_0, prob_1, ..., prob_n]
            probabilities[img_id] = probs

    print("-" * 50)
    print(f"✅ Probabilidades extraídas para {len(probabilities)} imagens")
    return probabilities


def predict_standard_labels(probabilities: StandardModelResults) -> PredictResults:
    """
    Prediz labels com base nas probabilidades usando argmax.

    Args:
        probabilities: Dicionário {img_id: [prob_0, prob_1, ..., prob_n]}

    Returns:
        Dicionário {img_id: predicted_label} onde predicted_label é o índice da maior probabilidade
    """
    predicted_labels = {}

    for img_id, probs in probabilities.items():
        predicted_labels[img_id] = int(np.argmax(probs))

    print(f"✅ Labels preditos para {len(predicted_labels)} imagens")
    return predicted_labels


def save_standard_confusion_matrix(
    true_labels: List[int],
    predicted_labels: List[int],
    model_name: str,
    experiment_dir: str,
    cmap: str
) -> str:
    """
    Salva a matriz de confusão em arquivo JPG.

    Args:
        true_labels: Lista de labels verdadeiros
        predicted_labels: Lista de labels preditos
        model_name: Nome do modelo (ex: "KNN_LBP")
        experiment_dir: Diretório do experimento

    Returns:
        Caminho do arquivo salvo
    """
    # Cria diretório para matrizes de confusão
    cm_dir = os.path.join(experiment_dir, "confusion_matrixs")
    os.makedirs(cm_dir, exist_ok=True)

    # Define caminho do arquivo
    filename = f"{model_name}_confusion_matrix.jpg"
    filepath = os.path.join(cm_dir, filename)

    try:
        # Cria a figura da matriz de confusão
        show_confusion_matrix(
            true_labels,
            predicted_labels,
            title=f"{model_name} - Confusion Matrix",
            cmap=cmap,
        )

        # Salva a figura atual
        plt.savefig(filepath, format="jpg", bbox_inches="tight", dpi=150)
        plt.close()

        print(f"  💾 Matriz de confusão salva: {filepath}")
        return filepath

    except Exception as e:
        print(f"  ❌ Erro ao salvar matriz de confusão: {str(e)}")
        return ""


def export_standard_results_to_csv(
    standard_results: StandardClassificationResults,
    true_labels: PredictResults,
    model_name: str,
    experiment_dir: str,
) -> str:
    """
    Exporta os resultados da classificação padrão para um arquivo CSV.

    Args:
        standard_results: Resultado da função standard_technique
        true_labels: Dicionário com labels reais {img_id: label}
        model_name: Nome do modelo (ex: "KNN_LBP")
        experiment_dir: Diretório do experimento

    Returns:
        str: Caminho completo do arquivo CSV gerado
    """
    # Desempacota os resultados
    probabilities, predicted_labels, labels_list, model_metrics = standard_results

    # Extrai as métricas globais
    accuracy_global, f1_global, recall_global, precision_global = model_metrics

    # Cria o diretório de saída
    csv_dir = os.path.join(experiment_dir, "csv_exports")
    os.makedirs(csv_dir, exist_ok=True)

    # Define o nome do arquivo
    filename = f"{model_name}_results.csv"
    filepath = os.path.join(csv_dir, filename)

    print(f"📊 Exportando resultados para CSV: {model_name}")
    print(f"   📁 Arquivo: {filepath}")
    print(f"   🎯 Imagens: {len(predicted_labels)} amostras")
    print("-" * 50)

    # Função auxiliar para serializar arrays em JSON com precisão controlada
    def serialize_array(arr):
        if arr is None:
            return "[]"
        # Converte para lista e aplica precisão de 4 casas decimais
        arr_list = np.asarray(arr).tolist()

        # Recursivamente aplica round para elementos aninhados
        def round_nested(obj):
            if isinstance(obj, list):
                return [round_nested(item) for item in obj]
            elif isinstance(obj, float):
                return round(obj, 4)
            else:
                return obj

        rounded_arr = round_nested(arr_list)
        return json.dumps(rounded_arr)

    # Escreve o arquivo CSV
    with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "nome_imagem",
            "label_predito",
            "label_real",
            "acuracia_global",
            "f1_global",
            "recall_global",
            "precision_global",
            "probabilidades",
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Escreve uma linha para cada imagem
        processed_count = 0
        for img_id in sorted(predicted_labels.keys()):
            writer.writerow(
                {
                    "nome_imagem": img_id,
                    "label_predito": predicted_labels[img_id],
                    "label_real": true_labels.get(img_id, -1),
                    "acuracia_global": round(accuracy_global, 4),
                    "f1_global": round(f1_global, 4),
                    "recall_global": round(recall_global, 4),
                    "precision_global": round(precision_global, 4),
                    "probabilidades": serialize_array(probabilities.get(img_id)),
                }
            )
            processed_count += 1

    print(f"   ✅ {processed_count} linhas escritas com sucesso")
    print(f"   💾 Arquivo salvo: {filepath}")
    print("=" * 50)

    return filepath


def standard_technique(
    base_model: BaseEstimator,
    folded_dataset: MulticlassClassificationDataset,
    class_names: List[str],
    true_labels: PredictResults,
    model_name: str = "Standard",
    k_folds: int = 5,
    experiment_dir: str = None,
    cmap: str = "Blues",
) -> StandardClassificationResults:
    """
    Aplica técnica de classificação multiclasse padrão para imagens completas.

    Args:
        base_model: Modelo base a ser utilizado para classificação (ex: GridSearchCV com KNN, SVM, RF)
        folded_dataset: Dataset dividido em K-folds para validação cruzada
        class_names: Nomes das classes para a tarefa de classificação (ex: ['dogs', 'cats', 'lions', 'horses'])
        true_labels: Dicionário com labels reais {img_id: label}
        model_name: Nome do modelo a ser utilizado (padrão: "Standard")
        k_folds: Número de dobras para validação cruzada (padrão: 5)
        experiment_dir: Diretório para salvar resultados. Se None, cria automaticamente com timestamp

    Returns:
        StandardClassificationResults: Tupla contendo:
            - probabilities: Dicionário {img_id: [prob_0, prob_1, ..., prob_n]}
            - predicted_labels: Dicionário {img_id: predicted_label}
            - labels_list: Tupla (true_labels, predicted_labels) como listas
            - model_metrics: Tupla (accuracy, f1, recall, precision)
    """
    print("\n" + "=" * 60)
    print(f"🚀 INICIANDO CLASSIFICAÇÃO PADRÃO: {model_name}")
    print("=" * 60)
    print(f"   📊 Classes: {class_names}")
    print(f"   🔄 Validação cruzada: {k_folds} folds")
    print(f"   🎯 Total de imagens: {len(true_labels)}")
    print("=" * 60 + "\n")

    # ETAPA 1: Extrair probabilidades de todas as classes
    print("📈 ETAPA 1: Extração de probabilidades")
    probabilities = extract_standard_probabilities(
        base_model=base_model,
        folded_dataset=folded_dataset,
        title=model_name,
    )

    # ETAPA 2: Predizer labels com base nas probabilidades
    print("\n🎯 ETAPA 2: Predição de labels")
    predicted_labels = predict_standard_labels(probabilities)

    # ETAPA 3: Computar métricas
    print("\n📊 ETAPA 3: Computação de métricas")
    labels_list, model_metrics = compute_metrics(true_labels, predicted_labels)

    # ETAPA 4: Mostrar métricas
    print("\n📋 ETAPA 4: Exibição de métricas")
    show_experiment_metrics(model_metrics, title=model_name)

    # ETAPA 5: Criar diretório do experimento se não fornecido
    if experiment_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join("results", "standard", f"experimento_{timestamp}")
    
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"\n📁 Diretório do experimento: {experiment_dir}")

    # ETAPA 6: Salvar matriz de confusão
    print("\n🔲 ETAPA 5: Salvamento da matriz de confusão")
    true_y, predicted_y = labels_list
    show_confusion_matrix(
            true_y,
            predicted_y,
            title=f"{model_name} - Confusion Matrix",
            cmap=cmap,
        )

    # ETAPA 7: Exportar resultados para CSV
    print("\n💾 ETAPA 6: Exportação de resultados para CSV")
    standard_results = (probabilities, predicted_labels, labels_list, model_metrics)
    export_standard_results_to_csv(
        standard_results=standard_results,
        true_labels=true_labels,
        model_name=model_name,
        experiment_dir=experiment_dir,
    )

    print("\n" + "=" * 60)
    print(f"✅ CLASSIFICAÇÃO PADRÃO CONCLUÍDA: {model_name}")
    print("=" * 60)
    print(f"   📁 Resultados salvos em: {experiment_dir}")
    print(f"   🎯 Acurácia: {model_metrics[0]:.4f}")
    print(f"   📊 F1-Score: {model_metrics[1]:.4f}")
    print(f"   🔄 Recall: {model_metrics[2]:.4f}")
    print(f"   🎯 Precision: {model_metrics[3]:.4f}")
    print("=" * 60 + "\n")

    return standard_results

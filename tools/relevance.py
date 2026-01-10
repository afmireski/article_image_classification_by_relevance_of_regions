import importlib
import numpy as np
import os
import sys
import shutil
import zipfile
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Tuple, Dict
from sklearn.base import BaseEstimator, clone

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
)

from utils import (show_metrics, show_confusion_matrix)

# Import image tools functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import tools.image_tools as tit
import mytypes as mtp

importlib.reload(tit)
importlib.reload(mtp)

from tools.image_tools import reconstruct_image_from_regions

from mytypes import (
    PreparedSetsForClassification,
    ClassificationDataset,
    RelevanceModelResults,
    PredictResults,
    ModelMetrics,
    ModelLabels,
    RelevanceResults,
    ResultsKeyDict,
    RelevanceTrainResults
)

def calculate_intermediary_metrics(
    y_test, predict
) -> ModelMetrics:
    accuracy = accuracy_score(y_test, predict)
    f1 = f1_score(y_test, predict)
    recall = recall_score(y_test, predict)
    precision = precision_score(y_test, predict)

    return accuracy, f1, recall, precision

def calculate_train_metrics(
    metrics
):
    return {
        'accuracy': {
            'folds': metrics['accuracies'],
            'mean': np.mean(metrics['accuracies']),
            'std': np.std(metrics['accuracies'])
        },
        'f1': {
            'folds': metrics['f1s'],
            'mean': np.mean(metrics['f1s']),
            'std': np.std(metrics['f1s'])
        },
        'recall': {
            'folds': metrics['recalls'],
            'mean': np.mean(metrics['recalls']),
            'std': np.std(metrics['recalls'])
        },
        'precision': {
            'folds': metrics['precisions'],
            'mean': np.mean(metrics['precisions']),
            'std': np.std(metrics['precisions'])
        }
    }

def extract_model_results(
    base_model: BaseEstimator, folded_dataset: ClassificationDataset, title: str = ""
) -> RelevanceModelResults:
    """
    Extrai probabilidades positivas de um modelo usando validação cruzada.

    Args:
        base_model: Modelo sklearn configurado (ex: GridSearchCV)
        folded_dataset: Lista de tuplas ((X_train, y_train), (X_test, y_test))
        title: Nome do modelo para logs

    Returns:
        Array 1D com probabilidades da classe positiva (0) para todas as amostras
    """

    probabilities = {}

    # Validação cruzada para avaliação
    print(f"=== Iniciando treinamento assistido: {title} ===")
    print(f"Validação cruzada: {len(folded_dataset)} folds")
    print("=" * 50)

    intermediary_metrics = {
        'accuracies': [],
        'f1s': [],
        'recalls': [],
        'precisions': [],
    }

    # Validação cruzada para avaliação
    for fold_index, (train_set, test_set) in enumerate(folded_dataset):
        X_train, y_train, train_pieces_map = train_set
        X_test, y_test, test_pieces_map = test_set

        # Clona o modelo base para cada fold
        fold_model = clone(base_model)
        fold_model.fit(X_train, y_train)

        print(f"\n~~ Fold {fold_index + 1} ~~")
        print("Melhores parâmetros:", fold_model.best_params_)

        predict_probabilities = fold_model.predict_proba(X_test)
        predict_results = fold_model.predict(X_test)

        for i, p in enumerate(predict_probabilities):
            img = test_pieces_map[i]

            if probabilities.get(img) is None:
                probabilities[img] = []

            probabilities[img].append(p[0])  # Probabilidade da classe positiva (0)
        
        # Calcula métricas intermediárias para o fold atual
        metrics = calculate_intermediary_metrics(y_test, predict_results)

        accuracy, f1, recall, precision = metrics

        intermediary_metrics['accuracies'].append(accuracy)
        intermediary_metrics['f1s'].append(f1)
        intermediary_metrics['recalls'].append(recall)
        intermediary_metrics['precisions'].append(precision)

        show_metrics(metrics, title=f"Fold {fold_index + 1} - {title}")
        show_confusion_matrix(y_test, predict_results, title=f"{title} - Fold {fold_index + 1}",save_dir='results/confusion_matrixs/folds')

        print("~" * 25)
    
    # Calcula métricas de treinamento agregadas
    train_metrics = calculate_train_metrics(intermediary_metrics)

    print("=" * 50)
    return (probabilities, train_metrics)


def consolidate_model_results(specialists_results: List[ResultsKeyDict]) -> ResultsKeyDict:
    """
    Consolida resultados de múltiplos folds em um único dicionário.

    Args:
        specialists_results: Lista de dicionários {img_id: [prob_segment_0, prob_segment_1, ...]}

    Returns:
        Dicionário consolidado {img_id: [[specialist0_prob_segment_0, specialist0_prob_segment_1, ...], [specialist1_prob_segment_0, ...], ...]}
    """
    all_images = set()
    for specialist_results in specialists_results:
        all_images.update(specialist_results.keys())

    consolidated = {}

    for img in all_images:
        img_probs = []

        for results in specialists_results:
            probs = results.get(img)
            img_probs.append(probs if probs is not None else [])

        consolidated[img] = np.array(
            img_probs
        ).T  # Transpõe para ter shape (n_probabilities, n_specialists)

    return consolidated


def extract_specialists_probabilities(
    base_model: BaseEstimator,
    extract_func: callable,
    specialist_sets: PreparedSetsForClassification,
    class_names: List[str],
    model_name: str = "Specialist",
    k_folds: int = 5,
) -> RelevanceTrainResults:
    """
    Treina múltiplos especialistas e extrai suas probabilidades normalizadas.

    Args:
        base_model: Modelo base para treinar especialistas
        extract_func: Função para extrair probabilidades (ex: extract_model_results)
        specialist_sets: Lista de datasets divididos em folds para cada especialista
        class_names: Nomes das classes ['dogs', 'cats', 'lions', 'horses']
        model_name: Nome base para logs
        k_folds: Número de folds

    Returns:
        Matriz (n_amostras, n_especialistas) com probabilidades normalizadas
    """
    from joblib import parallel_backend

    extracted_probabilities = []
    specialists_train_metrics = []

    print(f"🚀 Iniciando treinamento de especialistas {model_name}")
    print(f"   📋 {len(specialist_sets)} especialistas para treinar")
    print(f"   🔄 Validação cruzada: {k_folds} folds")
    print("-" * 60)

    # Use threading backend para evitar problemas de memory mapping
    with parallel_backend("threading"):
        for i, dataset in enumerate(specialist_sets):
            class_name = class_names[i]
            specialist_title = f"{model_name}-Specialist-{class_name}"

            print(
                f"\n🎯 Treinando especialista {i+1}/{len(specialist_sets)}: {class_name}"
            )

            # Usa a função de treinamento assistido fornecida
            try:
                (specialist_probabilities, specialist_train_metrics) = extract_func(
                    base_model=base_model,
                    folded_dataset=dataset,
                    title=specialist_title,
                )

                # Adiciona o modelo treinado ao array de especialistas
                extracted_probabilities.append(specialist_probabilities)
                specialists_train_metrics.append(specialist_train_metrics)

            except Exception as e:
                print(
                    f"   ❌ Erro ao extrair probabilidades do especialista {class_name}: {str(e)}"
                )
                raise e

    print(f"\n🎉 Treinamento de especialistas {model_name} concluído!")
    print(f"   ✅ {len(extracted_probabilities)} especialistas treinados com sucesso")
    print(
        "   📦 Array retornado: raw_probabilities[i] = probabilidades do especialista da classe i"
    )
    print("=" * 60)

    images_probabilities = consolidate_model_results(extracted_probabilities)

    return normalize_probabilities(images_probabilities), specialists_train_metrics


def normalize_probabilities(probabilities: ResultsKeyDict) -> ResultsKeyDict:
    """
    Normaliza probabilidades de especialistas para somarem 1.0 por amostra.
    """

    normalized_probs = {}

    for img, probs in probabilities.items():
        probs = np.asarray(probs, dtype=float)
        if probs.ndim != 2:
            raise ValueError("prob_matrix deve ser 2D: (n_amostras, n_especialistas).")
        # Normaliza para somar 1.0 por amostra (linha)

        norm_probs = probs / probs.sum(axis=1, keepdims=True)

        normalized_probs[img] = norm_probs

    return normalized_probs  # Shape: (n_probabilities, n_specialists)


def shannon_entropy(probabilities: ResultsKeyDict, use_clip=False, eps=1e-12) -> ResultsKeyDict:
    """
    Calcula H(x_j) por amostra (linha) para uma matriz de probabilidades no formato (n_amostras, n_especialistas).
    H(x_j) = - sum_i P_i(x_j) * log_base(P_i(x_j)), com base = n_especialistas por padrão.

    Args:
        probabilities: dicionário {img_id: [[specialist0_prob_segment_0, specialist0_prob_segment_1, ...], [specialist1_prob_segment_0, ...], ...]}
        use_clip: Se True, usa np.clip para evitar log(0); se False, ignora zeros
        eps: Valor pequeno para np.clip

    Returns:
        entropias: dicionário {img_id: array (n_amostras,) com H(x_j) para cada amostra (linha)}
    """

    entropies = {}

    for img, probs in probabilities.items():
        P = np.asarray(probs, dtype=float)
        _, base = P.shape

        if use_clip:
            # versão "robusta", aproximação com eps
            P_safe = np.clip(P, eps, 1.0)
            logP_base = np.log(P_safe) / np.log(base)
        else:
            # versão "matemática pura", ignora zeros
            logP_base = np.zeros_like(P)
            mask = P > 0
            logP_base[mask] = np.log(P[mask]) / np.log(base)

        H = -np.sum(P * logP_base, axis=1)
        H = np.where(
            np.isclose(H, 0), 0.0, H
        )  # Remove -0.0 que apareceram em alguns casos
        entropies[img] = H

    return entropies


def shannon_entropy_manual(probabilities: ResultsKeyDict) -> ResultsKeyDict:
    """
    Calcula H(x_j) por amostra (linha) para uma matriz de probabilidades no formato (n_amostras, n_especialistas).
    H(x_j) = - sum_i P_i(x_j) * log_base(P_i(x_j)), com base = n_especialistas por padrão.

    Args:
        probabilities: dicionário {img_id: [[specialist0_prob_segment_0, specialist0_prob_segment_1, ...], [specialist1_prob_segment_0, ...], ...]}

    Returns:
        entropias: array (n_amostras,) com H(x_j) para cada amostra (linha)
    """
    entropies = {}
    for img, probs in probabilities.items():
        P = np.asarray(probs, dtype=float)
        if P.ndim != 2:
            raise ValueError("prob_matrix deve ser 2D: (n_amostras, n_especialistas).")
        _, base = P.shape

        H = []

        for probs in P:
            s = 0  # sum
            for p_i in probs:
                log_p_i = np.log(p_i) / np.log(base) if p_i > 0 else 0.0
                x = p_i * log_p_i
                s += x
            H.append(-s)
        entropies[img] = np.array(H)

    return entropies


def calculate_relevance(entropies: ResultsKeyDict) -> ResultsKeyDict:
    """
    Calcula R(x_j) para cada segmento de uma imagem a partir de suas entropias H(x_j).
    R(x_j) = 1 - H(x_j).

    Args:
        entropies: dicionário {img_id: [H(x_0), H(x_1), ..., H(x_n)]} com H(x_j) para cada amostra

    Returns:
        relevancias: dicionário {img_id: [R(x_0), R(x_1), ..., R(x_n)]} com R(x_j) para cada amostra
    """
    return {img: 1.0 - H for img, H in entropies.items()}


def calculate_max_relevance(
    relevances: ResultsKeyDict, probabilities: ResultsKeyDict
) -> ResultsKeyDict:
    """
    Calcula R_max(x_j) para as relevâncias R(x_j) de cada imagem, ponderando pela maior probabilidade entre especialistas.
    R_max(x_j) = R(x_j) * max(P(x_j)).

    Args:
        relevances: dicionário {img_id: [R(x_0), R(x_1), ..., R(x_n)]} com R(x_j) para cada segmento
        probabilities: dicionário {img_id: [[P_0(x_0), P_0(x_1), ...], [P_1(x_0), P_1(x_1), ...], ...]} com P(x_j) para cada segmento

    Returns:
        relevances: dicionário {img_id: [R_max(x_0), R_max(x_1), ..., R_max(x_n)]} com R_max(x_j) para cada amostra
    """
    result = {}
    for img, R in relevances.items():
        P = probabilities.get(img)

        # Converte para arrays numpy com dtype float para garantir compatibilidade
        R_array = np.asarray(R, dtype=float)
        P_array = np.asarray(P, dtype=float)

        # Calcula R_max = R * max(P) ao longo do eixo dos especialistas
        max_probs = P_array.max(axis=1)

        R_max = R_array * max_probs

        result[img] = R_max

    return result


def calculate_ponderate_votes(
    probabilities: ResultsKeyDict, max_relevances: ResultsKeyDict
) -> ResultsKeyDict:
    """
    Calcula votos ponderados para cada segmento de uma imagem.
    Voto ponderado = P(x_j) * R_max(x_j).

    Args:
        probabilities: dicionário {img_id: [[P_0(x_0), P_0(x_1), ...], [P_1(x_0), P_1(x_1), ...], ...]} com P(x_j) para cada amostra
        max_relevances: dicionário {img_id: [R_max(x_0), R_max(x_1), ..., R_max(x_n)]} com R_max(x_j) para cada amostra

    Returns:
        votos_ponderados: dicionário {img_id: [[V_0(x_0), V_0(x_1), ...], [V_1(x_0), V_1(x_1), ...], ...]} com votos ponderados para cada especialista
    """
    weighted_votes = {}
    for img, P in probabilities.items():
        R_max = max_relevances.get(img)

        # Converte para arrays numpy com dtype float para garantir compatibilidade
        P_array = np.asarray(P, dtype=float)
        R_max_array = np.asarray(R_max, dtype=float)

        # Calcula votos ponderados
        votes = (
            P_array * R_max_array[:, np.newaxis]
        )  # Broadcasting para multiplicar cada linha por R_max correspondente

        weighted_votes[img] = votes

    return weighted_votes


def calculate_accumulated_votes(ponderated_votes: ResultsKeyDict) -> ResultsKeyDict:
    """
    Calcula votos acumulados somando o valor do voto de todos os segmentos de um especialista.
    Voto acumulado = sum(P(x_j) * R_max(x_j)) ao longo dos especialistas.

    Args:
        ponderated_votes: dicionário {img_id: [[V_0(x_0), V_0(x_1), ...], [V_1(x_0), V_1(x_1), ...], ...]} com votos ponderados para cada especialista

    Returns:
        votos_acumulados: dicionário {img_id: [S_0, S_1, ..., S_n]} com votos acumulados para especialista
    """
    accumulated_votes = {}
    for img, votes in ponderated_votes.items():
        votes_specialists = np.asarray(votes, dtype=float)
        votes_by_pieces = (
            votes_specialists.T
        )  # Transpõe para shape (n_amostras, n_especialistas)
        accumulated = votes_by_pieces.sum(axis=1)  # Soma ao longo dos especialistas
        accumulated_votes[img] = accumulated

    return accumulated_votes


def predict_labels(accumulated_votes: ResultsKeyDict) -> PredictResults:
    """
    Determina o rótulo final de cada imagem com base nos votos acumulados.
    Rótulo = índice do especialista com maior voto acumulado.

    Args:
        accumulated_votes: dicionário {img_id: [S_0, S_1, ..., S_n]} com votos acumulados para especialista

    Returns:
        image_labels: dicionário {img_id: label} com rótulo final da imagem
    """
    image_labels = {}
    for img, votes in accumulated_votes.items():
        votes_array = np.asarray(votes, dtype=float)
        label = int(
            np.argmax(votes_array)
        )  # Índice do especialista com maior voto acumulado
        image_labels[img] = label

    return image_labels


def compute_metrics(
    true_labels: PredictResults, predicted_labels: PredictResults
) -> Tuple[ModelLabels, ModelMetrics]:
    true_y = []
    predicted_y = []

    for img, true_label in true_labels.items():
        pred_label = predicted_labels.get(img)
        true_y.append(true_label)
        predicted_y.append(pred_label)

    accuracy = accuracy_score(true_y, predicted_y)
    f1 = f1_score(true_y, predicted_y, average="macro")
    recall = recall_score(true_y, predicted_y, average="macro")
    precision = precision_score(true_y, predicted_y, average="macro")

    return (true_y, predicted_y), (accuracy, f1, recall, precision)


def generate_relevance_heatmaps(
    max_relevances: ResultsKeyDict,
    all_images_segmented: Dict[str, np.ndarray],
    model_name: str,
    colormap: str = "viridis",
    overlay_alpha: float = 0.5,
    save_grid_lines: bool = True,
    results_dir: str = "results",
    verbose: bool = False
) -> None:
    """
    Gera mapas de calor (heatmaps) a partir das relevâncias máximas dos segmentos de imagem.
    
    Args:
        max_relevances: Dicionário {img_id: [R_max_0, R_max_1, ...]} com relevâncias [0,1] por segmento
        all_images_segmented: Dicionário {img_id: regions_matrix} com imagens segmentadas
        model_name: Nome do modelo para organizar os resultados (ex: "SVM", "RandomForest")
        colormap: Nome do colormap matplotlib para cores do heatmap (padrão: "viridis")
        overlay_alpha: Transparência da sobreposição do heatmap (0.0=transparente, 1.0=opaco)
        save_grid_lines: Se True, desenha linhas de grade mostrando os limites dos segmentos
        results_dir: Diretório base para salvar os resultados
        verbose: Se True, exibe mensagens de log; caso contrário, suprime mensagens não essenciais
    
    Returns:
        None: Salva as imagens em /results/<model_name>/overlays/
    """
    if verbose:
        print(f"🎨 Gerando heatmaps de relevância para modelo: {model_name}")
        print(f"   📊 Colormap: {colormap}")
        print(f"   🎭 Transparência overlay: {overlay_alpha}")
        print(f"   🔲 Linhas de grade: {save_grid_lines}")
        print("-" * 60)
    
    # Verifica se há correspondência entre relevâncias e imagens segmentadas
    images_with_relevance = set(max_relevances.keys())
    images_segmented = set(all_images_segmented.keys())
    
    common_images = images_with_relevance.intersection(images_segmented)
    
    if len(common_images) == 0:
        print("❌ Erro: Nenhuma imagem comum encontrada entre relevâncias e segmentações")
        return
        
    if verbose:
        print(f"   ✅ {len(common_images)} imagens serão processadas")
    
    # Processa cada imagem
    processed_count = 0
    for img_id in sorted(common_images):
        try:
            # Obtém os dados da imagem
            relevance_values = np.asarray(max_relevances[img_id], dtype=float)
            regions_matrix = all_images_segmented[img_id]
            
            # Converte relevâncias 1D para matriz 2D que corresponde à grade de segmentação
            grid_shape = regions_matrix.shape
            total_segments = grid_shape[0] * grid_shape[1]
            
            # Verifica se o número de relevâncias corresponde ao número de segmentos
            if len(relevance_values) != total_segments:
                if verbose:
                    print(f"   ⚠️  Aviso {img_id}: {len(relevance_values)} relevâncias vs {total_segments} segmentos - ajustando...")
                # Ajusta o array se necessário (preenche com zeros ou trunca)
                if len(relevance_values) < total_segments:
                    # Preenche com zeros se faltam valores
                    padded_values = np.zeros(total_segments)
                    padded_values[:len(relevance_values)] = relevance_values
                    relevance_values = padded_values
                else:
                    # Trunca se há valores demais
                    relevance_values = relevance_values[:total_segments]
            
            # Reshape das relevâncias para a forma da grade (rows, cols)
            relevance_matrix = relevance_values.reshape(grid_shape)
            
            if verbose:
                print(f"   🔄 Processando {img_id}: grade {grid_shape[0]}x{grid_shape[1]} = {total_segments} segmentos")
                print(f"       Relevâncias: min={relevance_values.min():.3f}, max={relevance_values.max():.3f}")
            
            # STEP 2: Gerar heatmap colorido usando o colormap
            heatmap_image = create_segment_heatmap(
                regions_matrix, relevance_matrix, colormap
            )
            
            # STEP 3: Adicionar linhas de grade se solicitado
            if save_grid_lines:
                heatmap_with_grid = add_grid_lines_to_heatmap(
                    heatmap_image, regions_matrix.shape
                )
                if verbose:
                    print("       🔲 Linhas de grade adicionadas")
            else:
                heatmap_with_grid = heatmap_image
            
            # STEP 4: Criar overlay com a imagem original
            original_image = reconstruct_original_image(regions_matrix)
            overlay_image = create_heatmap_overlay(
                original_image, heatmap_with_grid, overlay_alpha
            )
            
            # STEP 5: Salvar a imagem
            save_success = save_overlay_image(
                overlay_image, img_id, model_name, results_dir
            )
            
            if save_success:
                if verbose:
                    print("       💾 Salvo com sucesso")
            else:
                print("       ❌ Erro ao salvar")
                
            if verbose:
                print(f"       🎨 Heatmap final: {overlay_image.shape} (altura, largura, RGB)")
                print(f"       🎭 Overlay criado com transparência: {overlay_alpha}")
            processed_count += 1
            
        except Exception as e:
            print(f"   ❌ Erro ao processar {img_id}: {str(e)}")
            continue
    
    if verbose:
        print(f"\n🎉 CONCLUÍDO: {processed_count} heatmaps de relevância gerados!")
        print(f"   📁 Salvos em: {results_dir}/{model_name}/overlays/")
        print(f"   🎨 Colormap: {colormap}")
        print(f"   🎭 Transparência: {overlay_alpha}")
        print(f"   🔲 Linhas de grade: {'Ativadas' if save_grid_lines else 'Desativadas'}")
        print("=" * 60)


def save_overlay_image(
    overlay_image: np.ndarray, 
    img_id: str, 
    model_name: str, 
    results_dir: str = "results"
) -> bool:
    """
    Salva a imagem overlay como arquivo JPG no diretório organizado.
    
    Args:
        overlay_image: Imagem overlay RGB com valores [0,1]
        img_id: Identificador da imagem (ex: "cat1", "dog5")
        model_name: Nome do modelo (ex: "SVM", "RandomForest")
        results_dir: Diretório base para salvar
        
    Returns:
        bool: True se salvou com sucesso, False caso contrário
    """
    try:
        # Cria estrutura de diretórios
        output_dir = os.path.join(results_dir, 'heatmaps', model_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Define o nome do arquivo
        filename = f"{img_id}_relevance_overlay.jpg"
        filepath = os.path.join(output_dir, filename)
        
        # Converte para formato de imagem (0-255, uint8)
        image_uint8 = (overlay_image * 255).astype(np.uint8)
        
        # Salva usando matplotlib sem mostrar a imagem
        plt.figure(figsize=(10, 10), dpi=100)
        plt.imshow(image_uint8)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filepath, format='jpg', bbox_inches='tight', 
                   pad_inches=0, dpi=150)
        plt.close()  # Importante: fecha a figura para não mostrar
        
        return True
        
    except Exception as e:
        print(f"       ⚠️  Erro ao salvar {img_id}: {str(e)}")
        return False


def reconstruct_original_image(regions_matrix: np.ndarray) -> np.ndarray:
    """
    Reconstrói a imagem original em escala de cinza a partir da matriz de regiões.
    
    Args:
        regions_matrix: Matriz de regiões segmentadas (rows, cols) onde cada célula contém pixels de um segmento
        
    Returns:
        np.ndarray: Imagem original reconstruída em escala de cinza
    """
    return reconstruct_image_from_regions(regions_matrix)


def create_heatmap_overlay(
    original_image: np.ndarray, 
    heatmap_rgb: np.ndarray, 
    alpha: float = 0.5
) -> np.ndarray:
    """
    Cria um overlay combinando a imagem original em escala de cinza com o heatmap colorido.
    
    Args:
        original_image: Imagem original em escala de cinza (altura, largura)
        heatmap_rgb: Heatmap colorido RGB (altura, largura, 3) com valores [0,1]
        alpha: Transparência do heatmap (0.0=invisível, 1.0=opaco, 0.5=semi-transparente)
        
    Returns:
        np.ndarray: Imagem overlay RGB com shape (altura, largura, 3) com valores [0,1]
    """
    # Verifica se as dimensões são compatíveis
    if original_image.shape[:2] != heatmap_rgb.shape[:2]:
        raise ValueError(
            f"Dimensões incompatíveis: original {original_image.shape[:2]} vs heatmap {heatmap_rgb.shape[:2]}"
        )
    
    # Converte a imagem original para RGB (3 canais) normalizando para [0,1]
    if len(original_image.shape) == 2:
        # Escala de cinza para RGB
        original_rgb = np.stack([original_image, original_image, original_image], axis=2)
    else:
        original_rgb = original_image.copy()
    
    # Normaliza a imagem original para [0,1] se necessário
    if original_rgb.max() > 1.0:
        original_rgb = original_rgb.astype(np.float32) / 255.0
    
    # Garante que o heatmap está em [0,1]
    heatmap_normalized = np.clip(heatmap_rgb, 0.0, 1.0)
    
    # Cria o overlay usando blend linear:
    # overlay = (1 - alpha) * original + alpha * heatmap
    overlay = (1.0 - alpha) * original_rgb + alpha * heatmap_normalized
    
    # Garante que os valores finais estão em [0,1]
    overlay = np.clip(overlay, 0.0, 1.0)
    
    return overlay


def add_grid_lines_to_heatmap(
    heatmap_rgb: np.ndarray, 
    grid_shape: tuple, 
    line_color: tuple = (0.0, 0.0, 0.0),
    line_width: int = 2
) -> np.ndarray:
    """
    Adiciona linhas de grade ao heatmap para mostrar os limites dos segmentos.
    
    Args:
        heatmap_rgb: Imagem heatmap RGB com shape (altura, largura, 3)
        grid_shape: Forma da grade de segmentação (rows, cols)
        line_color: Cor das linhas de grade em RGB [0,1] (padrão: preto)
        line_width: Largura das linhas em pixels (padrão: 2)
        
    Returns:
        np.ndarray: Heatmap com linhas de grade adicionadas
    """
    # Cria uma cópia para não modificar o original
    heatmap_with_grid = heatmap_rgb.copy()
    
    height, width, _ = heatmap_rgb.shape
    rows, cols = grid_shape
    
    # Calcula as dimensões de cada segmento
    segment_height = height // rows
    segment_width = width // cols
    
    # Desenha linhas horizontais (separando as linhas da grade)
    for row in range(1, rows):  # Não desenha na primeira linha (topo)
        y_position = row * segment_height
        # Desenha linha horizontal com a largura especificada
        for offset in range(line_width):
            if y_position + offset < height:
                heatmap_with_grid[y_position + offset, :] = line_color
    
    # Desenha linhas verticais (separando as colunas da grade)
    for col in range(1, cols):  # Não desenha na primeira coluna (esquerda)
        x_position = col * segment_width
        # Desenha linha vertical com a largura especificada
        for offset in range(line_width):
            if x_position + offset < width:
                heatmap_with_grid[:, x_position + offset] = line_color
    
    return heatmap_with_grid


def create_segment_heatmap(
    regions_matrix: np.ndarray, 
    relevance_matrix: np.ndarray, 
    colormap_name: str = "viridis"
) -> np.ndarray:
    """
    Cria um heatmap colorido a partir das relevâncias dos segmentos.
    
    Args:
        regions_matrix: Matriz de regiões segmentadas (rows, cols) onde cada célula contém pixels de um segmento
        relevance_matrix: Matriz de relevâncias (rows, cols) com valores [0,1] para cada segmento
        colormap_name: Nome do colormap matplotlib (ex: "viridis", "hot", "plasma")
        
    Returns:
        np.ndarray: Imagem heatmap RGB com shape (altura, largura, 3)
    """
    # Obtém o colormap
    colormap = cm.get_cmap(colormap_name)
    
    # Determina as dimensões da imagem final usando reconstrução real
    # Isso garante que o heatmap tenha exatamente as mesmas dimensões da imagem original
    reconstructed_temp = reconstruct_image_from_regions(regions_matrix)
    total_height, total_width = reconstructed_temp.shape
    
    # Determina as dimensões da grade
    rows, cols = regions_matrix.shape
    
    # Cria a imagem heatmap RGB
    heatmap_rgb = np.zeros((total_height, total_width, 3), dtype=np.float32)
    
    # Preenche cada segmento com sua cor correspondente à relevância
    # Usa a mesma lógica de posicionamento da reconstrução
    current_row = 0
    for row_idx in range(rows):
        current_col = 0
        row_height = 0
        
        for col_idx in range(cols):
            region = regions_matrix[row_idx, col_idx]
            if region is not None and hasattr(region, 'shape'):
                region_height, region_width = region.shape
                
                # Obtém a relevância deste segmento (valor já está em [0,1])
                relevance_value = relevance_matrix[row_idx, col_idx]
                
                # Converte relevância para cor RGB usando o colormap
                color_rgba = colormap(relevance_value)  # Retorna (R, G, B, A)
                color_rgb = color_rgba[:3]  # Pega apenas RGB, ignora alpha
                
                # Aplica a cor uniformemente a todo o segmento
                heatmap_rgb[current_row:current_row + region_height, 
                           current_col:current_col + region_width] = color_rgb
                
                current_col += region_width
                row_height = max(row_height, region_height)
        
        current_row += row_height
    
    return heatmap_rgb


def export_relevance_results_to_csv(
    relevance_results: RelevanceResults,
    true_labels: PredictResults,
    model_name: str,
    output_dir: str = "results",
    filename: str = None
) -> str:
    """
    Exporta os resultados da técnica de relevância para um arquivo CSV.
    
    Args:
        relevance_results: Resultado da função relevance_technique
        true_labels: Dicionário com labels reais {img_id: label}
        model_name: Nome do modelo (ex: "KNN_LBP", "SVM_GLCM")
        output_dir: Diretório de saída (padrão: "results")
        filename: Nome do arquivo CSV (padrão: auto-gerado a partir do model_name)
        
    Returns:
        str: Caminho completo do arquivo CSV gerado
        
    Example:
        >>> # Após executar relevance_technique
        >>> csv_path = export_relevance_results_to_csv(
        ...     relevance_results=relevance_results_knn_lbp,
        ...     true_labels=true_images_labels,
        ...     model_name="KNN_LBP"
        ... )
        >>> print(f"CSV salvo em: {csv_path}")
    """
    import csv
    import json
    import os
    
    # Desempacota os resultados
    (probabilities, entropies, relevances, max_relevances, 
     ponderated_votes, accumulated_votes, predicted_labels, 
     labels_list, model_metrics) = relevance_results
    
    # Extrai as métricas globais
    (accuracy_global, f1_global, recall_global, precision_global), specialists_train_metrics = model_metrics
    
    # Cria o diretório de saída
    csv_dir = os.path.join(output_dir, "csv_exports")
    os.makedirs(csv_dir, exist_ok=True)
    
    # Define o nome do arquivo se não fornecido
    base_name = model_name.lower().replace('-', '_').replace(' ', '_')
    if filename is None:
        filename = f"{base_name}_results.csv"

    # Nome base (sem sufixo) para gerar também o CSV de métricas
    
    metrics_filename = f"{base_name}_metrics.csv"
    
    results_filepath = os.path.join(csv_dir, filename)
    metrics_filepath = os.path.join(csv_dir, metrics_filename)
    
    print(f"📊 Exportando resultados para CSV: {model_name}")
    print(f"   📁 Arquivo com os resultados: {results_filepath}")
    print (f"   📁 Arquivo com as métricas: {metrics_filepath}")
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
    with open(results_filepath, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'nome_imagem', 'label_predito', 'label_real',
            'acuracia_global', 'f1_global', 'recall_global', 'precision_global',
            'probabilidades', 'entropias', 'relevancias', 'max_relevancia', 'votos_ponderados'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Escreve uma linha para cada imagem
        processed_count = 0
        for img_id in sorted(predicted_labels.keys()):
            try:
                row = {
                    'nome_imagem': img_id,
                    'label_predito': predicted_labels[img_id],
                    'label_real': true_labels.get(img_id, -1),  # -1 se não encontrar
                    'acuracia_global': round(accuracy_global, 4),
                    'f1_global': round(f1_global, 4),
                    'recall_global': round(recall_global, 4),
                    'precision_global': round(precision_global, 4),
                    'probabilidades': serialize_array(probabilities.get(img_id)),
                    'entropias': serialize_array(entropies.get(img_id)),
                    'relevancias': serialize_array(relevances.get(img_id)),
                    'max_relevancia': serialize_array(max_relevances.get(img_id)),
                    'votos_ponderados': serialize_array(ponderated_votes.get(img_id))
                }
                
                writer.writerow(row)
                processed_count += 1
                
            except Exception as e:
                print(f"   ⚠️  Erro ao processar {img_id}: {str(e)}")
                continue
    
    print(f"   ✅ {processed_count} linhas escritas com sucesso")
    print(f"   💾 Arquivo salvo: {results_filepath}")

    # -----------------------------
    # CSV de métricas do experimento
    # -----------------------------
    # Formato solicitado:
    # model,accuracy,accuracy_std,f1_score,f1_score_std,recall,recall_std,precision,precision_std
    # - Métricas globais e por fold não têm std: preencher com '###'
    # - Métricas médias por especialista têm mean e std

    def fmt_num(x):
        return f"{x:.4f}"  # já em escala percentual

    def fmt_std(x):
        return f"{x:.4f}"  # já em escala percentual

    tag_prefix = f"{base_name}_relevance"

    def to_percent(x):
        # Métricas internas são [0,1]; exporta em % (0-100)
        return float(x) * 100.0

    metrics_rows = []

    # Global
    metrics_rows.append(
        {
            "model": f"{tag_prefix}_global",
            "accuracy (%)": fmt_num(to_percent(accuracy_global)),
            "accuracy_std (+- %)": "###",
            "f1_score (%)": fmt_num(to_percent(f1_global)),
            "f1_score_std (+- %)": "###",
            "recall (%)": fmt_num(to_percent(recall_global)),
            "recall_std (+- %)": "###",
            "precision (%)": fmt_num(to_percent(precision_global)),
            "precision_std (+- %)": "###",
        }
    )

    # Especialistas: mean/std + folds
    for sp_idx, train_metrics in enumerate(specialists_train_metrics):
        sp_accuracy = train_metrics["accuracy"]
        sp_f1 = train_metrics["f1"]
        sp_recall = train_metrics["recall"]
        sp_precision = train_metrics["precision"]

        # Mean (com std)
        metrics_rows.append(
            {
                "model": f"{tag_prefix}_specialist_{sp_idx}_mean",
                "accuracy (%)": fmt_num(to_percent(sp_accuracy["mean"])),
                "accuracy_std (+- %)": fmt_std(to_percent(sp_accuracy["std"])),
                "f1_score (%)": fmt_num(to_percent(sp_f1["mean"])),
                "f1_score_std (+- %)": fmt_std(to_percent(sp_f1["std"])),
                "recall (%)": fmt_num(to_percent(sp_recall["mean"])),
                "recall_std (+- %)": fmt_std(to_percent(sp_recall["std"])),
                "precision (%)": fmt_num(to_percent(sp_precision["mean"])),
                "precision_std (+- %)": fmt_std(to_percent(sp_precision["std"])),
            }
        )

        # Folds (sem std)
        folds_count = min(
            len(sp_accuracy.get("folds", [])),
            len(sp_f1.get("folds", [])),
            len(sp_recall.get("folds", [])),
            len(sp_precision.get("folds", [])),
        )

        for fold_idx in range(folds_count):
            metrics_rows.append(
                {
                    "model": f"{tag_prefix}_specialist_{sp_idx}_fold{fold_idx + 1}",
                    "accuracy (%)": fmt_num(to_percent(sp_accuracy["folds"][fold_idx])),
                    "accuracy_std (+- %)": "###",
                    "f1_score (%)": fmt_num(to_percent(sp_f1["folds"][fold_idx])),
                    "f1_score_std (+- %)": "###",
                    "recall (%)": fmt_num(to_percent(sp_recall["folds"][fold_idx])),
                    "recall_std (+- %)": "###",
                    "precision (%)": fmt_num(to_percent(sp_precision["folds"][fold_idx])),
                    "precision_std (+- %)": "###",
                }
            )
    with open(metrics_filepath, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "model",
            "accuracy (%)",
            "accuracy_std (+- %)",
            "f1_score (%)",
            "f1_score_std (+- %)",
            "recall (%)",
            "recall_std (+- %)",
            "precision (%)",
            "precision_std (+- %)",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row)

    print(f"   ✅ {len(metrics_rows)} linhas de métricas escritas com sucesso")
    print(f"   💾 Arquivo salvo: {metrics_filepath}")
    print("=" * 50)
    
    return results_filepath


def export_all_relevance_results_to_csv(
    results_dict: Dict[str, Tuple[RelevanceResults, PredictResults]],
    output_dir: str = "results"
) -> List[str]:
    """
    Exporta múltiplos resultados de relevância para arquivos CSV separados.
    
    Args:
        results_dict: Dicionário {model_name: (relevance_results, true_labels)}
        output_dir: Diretório de saída
        
    Returns:
        List[str]: Lista de caminhos dos arquivos CSV gerados
        
    Example:
        >>> results = {
        ...     "KNN_LBP": (relevance_results_knn_lbp, true_images_labels),
        ...     "KNN_GLCM": (relevance_results_knn_glcm, true_images_labels),
        ...     "SVM_LBP": (relevance_results_svm_lbp, true_images_labels),
        ... }
        >>> csv_files = export_all_relevance_results_to_csv(results)
        >>> print(f"Gerados {len(csv_files)} arquivos CSV")
    """
    generated_files = []
    
    print(f"🚀 Exportando {len(results_dict)} modelos para CSV")
    print("=" * 60)
    
    for model_name, (relevance_results, true_labels) in results_dict.items():
        try:
            csv_path = export_relevance_results_to_csv(
                relevance_results=relevance_results,
                true_labels=true_labels,
                model_name=model_name,
                output_dir=output_dir
            )
            generated_files.append(csv_path)
            print()
            
        except Exception as e:
            print(f"❌ Erro ao exportar {model_name}: {str(e)}")
            print()
            continue
    
    print("🎉 Exportação concluída!")
    print(f"   ✅ {len(generated_files)} arquivos CSV gerados")
    print(f"   📁 Diretório: {output_dir}/csv_exports/")
    print("=" * 60)
    
    return generated_files


def zip_and_cleanup_results(
    results_dir: str = "results",
    experiment_dir_name: str = "experiments",
    folders_to_zip: List[str] = None,
    timestamp_format: str = "%Y%m%d_%H%M%S",
) -> str:
    """
    Compacta (zip) pastas de resultados e limpa as pastas originais.

    Args:
        results_dir: Diretório base onde as pastas existem (default: 'results')
        experiment_dir_name: Subdiretório onde os zips serão guardados (default: 'experiments')
        folders_to_zip: Lista de subpastas dentro de results_dir a serem compactadas. Se None, usa
                        ['confusion_matrixs', 'csv_exports', 'heatmaps'].
        timestamp_format: Formato do timestamp usado no nome do arquivo zip.

    Returns:
        Caminho completo para o arquivo zip criado.
    """
    if folders_to_zip is None:
        folders_to_zip = ["confusion_matrixs", "csv_exports", "heatmaps"]

    # Garante que o diretório base existe
    base_dir = os.path.abspath(results_dir)
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Diretório de resultados não encontrado: {base_dir}")

    # Cria diretório de experiments
    experiments_dir = os.path.join(base_dir, experiment_dir_name)
    os.makedirs(experiments_dir, exist_ok=True)

    # Timestamp para o nome do arquivo
    ts = datetime.now().strftime(timestamp_format)
    zip_name = f"experimento_{ts}.zip"
    zip_path = os.path.join(experiments_dir, zip_name)

    # Cria o zip e adiciona as pastas se existirem
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for folder in folders_to_zip:
            folder_path = os.path.join(base_dir, folder)
            if not os.path.exists(folder_path):
                # pula se não existir
                continue

            # Percorre a pasta e adiciona arquivos mantendo a estrutura relativa
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    abs_file = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_file, base_dir)
                    zf.write(abs_file, arcname=rel_path)

    # Depois de criado o zip, remove as pastas compactadas
    for folder in folders_to_zip:
        folder_path = os.path.join(base_dir, folder)
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path)
            except Exception as e:
                print(f"⚠️  Falha ao remover {folder_path}: {e}")

    print(f"✅ Resultados compactados em: {zip_path}")
    return zip_path


def relevance_technique(
    base_model: BaseEstimator,
    specialist_sets: PreparedSetsForClassification,
    class_names: List[str],
    true_labels: PredictResults,
    model_name: str = "Specialist",
    k_folds: int = 5,
) -> RelevanceResults:
    """
    Aplica a técnica de relevância para classificar imagens usando um modelo base e conjuntos de especialistas.

    Args:
        base_model: modelo base a ser utilizado para classificação
        specialist_sets: conjuntos de especialistas preparados para classificação
        class_names: nomes das classes para a tarefa de classificação
        model_name: nome do modelo a ser utilizado (padrão: "Specialist")
        k_folds: número de dobras para validação cruzada (padrão: 5)

    Returns:
        resultados: resultados da classificação
    """

    probabilities, specialists_train_metrics = extract_specialists_probabilities(
        base_model=base_model,
        extract_func=extract_model_results,
        specialist_sets=specialist_sets,
        class_names=class_names,
        model_name=model_name,
        k_folds=k_folds,
    )

    entropies = shannon_entropy(probabilities)

    relevances = calculate_relevance(entropies)

    max_relevances = calculate_max_relevance(relevances, probabilities)

    ponderated_votes = calculate_ponderate_votes(probabilities, max_relevances)

    accumulated_votes = calculate_accumulated_votes(ponderated_votes)

    predicted_labels = predict_labels(accumulated_votes)

    labels_list, model_metrics = compute_metrics(
        true_labels, predicted_labels
    )

    return (
        probabilities,
        entropies,
        relevances,
        max_relevances,
        ponderated_votes,
        accumulated_votes,
        predicted_labels,
        labels_list,
        (model_metrics, specialists_train_metrics)
    )

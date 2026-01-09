#!/usr/bin/env python3
"""
Script para classificação de imagens usando a estratégia de Soma de Probabilidades.

Esta estratégia soma as probabilidades de todos os pedaços da imagem para cada especialista.
O especialista com a maior soma total determina o label predito para a imagem.

USO:
    # Executar pipeline completo
    python sum_of_probabilities.py <input_csv> <output_dir> [model_name]
    
    # Executar testes de validação
    python sum_of_probabilities.py

EXEMPLO:
    python sum_of_probabilities.py \\
        results/experiments/exp_20251030/csv_exports/svm_lbp_glcm_lpq_results.csv \\
        results/sum_probabilities \\
        "SVM_LBP_GLCM_LPQ_SumProb"

Autor: TCC Image Classification by Relevance of Regions
Data: 5 de novembro de 2025
"""

import csv
import json
import numpy as np
import os
import sys
from typing import Dict, List, Tuple

# Importa funções utilitárias do projeto
from utils import show_confusion_matrix, show_experiment_metrics

# Importa a função compute_metrics da técnica de relevância
from tools.relevance import compute_metrics


# ============================================================================
# ETAPA 2: Função de Leitura do CSV
# ============================================================================

def read_relevance_csv(filepath: str) -> Dict[str, Dict]:
    """
    Lê o CSV de resultados da técnica de relevância e extrai dados necessários.
    
    Args:
        filepath: Caminho para o arquivo CSV de entrada
        
    Returns:
        Dicionário {img_id: {'label_real': int, 'label_predito': int, 'probabilidades': np.ndarray,
                             'acuracia': float, 'f1': float, 'recall': float, 'precision': float}}
        
    Raises:
        FileNotFoundError: Se o arquivo não existir
        ValueError: Se o CSV estiver malformado
    """
    print("📖 ETAPA 2: Leitura do CSV de entrada")
    print(f"   📁 Arquivo: {filepath}")
    print("-" * 60)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Arquivo não encontrado: {filepath}")
    
    data = {}
    
    with open(filepath, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Valida que as colunas necessárias existem
        required_columns = ['nome_imagem', 'label_real', 'label_predito', 'probabilidades']
        for col in required_columns:
            if col not in reader.fieldnames:
                raise ValueError(f"❌ Coluna obrigatória ausente no CSV: {col}")
        
        processed_count = 0
        for row in reader:
            img_id = row['nome_imagem']
            label_real = int(row['label_real'])
            label_predito = int(row['label_predito'])
            
            # Desserializa as probabilidades de JSON para numpy array
            try:
                prob_list = json.loads(row['probabilidades'])
                probabilidades = np.array(prob_list, dtype=float)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"   ⚠️  Erro ao desserializar probabilidades de {img_id}: {e}")
                continue
            
            # Armazena os dados (incluindo métricas para comparação posterior)
            data[img_id] = {
                'label_real': label_real,
                'label_predito': label_predito,
                'probabilidades': probabilidades,
                'acuracia': float(row.get('acuracia_global', 0.0)),
                'f1': float(row.get('f1_global', 0.0)),
                'recall': float(row.get('recall_global', 0.0)),
                'precision': float(row.get('precision_global', 0.0))
            }
            
            processed_count += 1
    
    print(f"   ✅ {processed_count} imagens lidas com sucesso")
    print(f"   📊 Exemplo de dados (primeira imagem):")
    
    # Mostra um exemplo dos dados lidos
    first_img = list(data.keys())[0]
    first_data = data[first_img]
    print(f"      - Imagem: {first_img}")
    print(f"      - Label real: {first_data['label_real']}")
    print(f"      - Label predito (relevância): {first_data['label_predito']}")
    print(f"      - Shape probabilidades: {first_data['probabilidades'].shape}")
    print(f"      - Primeiras probabilidades:")
    print(f"        {first_data['probabilidades'][:2]}")  # Mostra primeiros 2 pedaços
    print("=" * 60)
    print()
    
    return data


# ============================================================================
# ETAPA 3: Cálculo da Soma de Probabilidades
# ============================================================================

def calculate_probability_sums(probabilities: np.ndarray) -> np.ndarray:
    """
    Calcula a soma de probabilidades para cada especialista.
    
    Args:
        probabilities: Matriz (n_pedaços, n_especialistas) com probabilidades
                      Exemplo: (9, 4) para 9 pedaços e 4 especialistas
    
    Returns:
        np.ndarray: Array 1D com a soma das probabilidades de cada especialista
                   Shape: (n_especialistas,)
    
    Example:
        >>> probs = np.array([[0.6879, 0.2081, 0.0135, 0.0905],
        ...                   [0.0262, 0.97, 0.0039, 0.0]])
        >>> calculate_probability_sums(probs)
        array([0.7141, 1.1781, 0.0174, 0.0905])
    """
    # Soma as probabilidades por coluna (cada coluna é um especialista)
    # axis=0 significa somar ao longo das linhas (pedaços)
    sums = np.sum(probabilities, axis=0)
    
    return sums


# ============================================================================
# ETAPA 4: Predição dos Labels
# ============================================================================

def predict_label_by_sum(probability_sums: np.ndarray) -> int:
    """
    Prediz o label com base na maior soma de probabilidades.
    
    O especialista com a maior soma de probabilidades determina o label predito.
    
    Args:
        probability_sums: Array 1D com somas de probabilidades por especialista
                         Shape: (n_especialistas,)
                         Exemplo: [2.7256, 6.0131, 0.0643, 0.1970]
    
    Returns:
        int: Índice do especialista com maior soma (label predito)
             Valores: 0=dog, 1=cat, 2=lion, 3=horse
    
    Example:
        >>> sums = np.array([2.7256, 6.0131, 0.0643, 0.1970])
        >>> predict_label_by_sum(sums)
        1  # cat (especialista 1 tem a maior soma)
    """
    # Retorna o índice do especialista com maior soma
    predicted_label = np.argmax(probability_sums)
    
    return int(predicted_label)


# ============================================================================
# ETAPA 5: Processamento de Todas as Imagens
# ============================================================================

def process_all_images(
    data: Dict[str, Dict]
) -> Tuple[Dict[str, int], Dict[str, np.ndarray], Dict[str, int]]:
    """
    Processa todas as imagens calculando somas e predizendo labels.
    
    Args:
        data: Dicionário {img_id: {'label_real': int, 'probabilidades': np.ndarray}}
              retornado pela função read_relevance_csv()
    
    Returns:
        Tupla contendo:
        - predicted_labels: {img_id: label_predito}
        - probability_sums: {img_id: [soma_esp_0, soma_esp_1, ...]}
        - true_labels: {img_id: label_real}
    """
    print("🔄 ETAPA 5: Processamento de todas as imagens")
    print(f"   📊 Total de imagens: {len(data)}")
    print("-" * 60)
    
    predicted_labels = {}
    probability_sums = {}
    true_labels = {}
    
    processed_count = 0
    
    for img_id, img_data in data.items():
        try:
            # Extrai dados da imagem
            probabilidades = img_data['probabilidades']
            label_real = img_data['label_real']
            
            # Calcula somas de probabilidades (ETAPA 3)
            sums = calculate_probability_sums(probabilidades)
            
            # Prediz o label (ETAPA 4)
            predicted_label = predict_label_by_sum(sums)
            
            # Armazena resultados
            predicted_labels[img_id] = predicted_label
            probability_sums[img_id] = sums
            true_labels[img_id] = label_real
            
            processed_count += 1            
                
        except Exception as e:
            print(f"   ⚠️  Erro ao processar {img_id}: {e}")
            continue
    
    print(f"   ✅ {processed_count} imagens processadas com sucesso")
    print(f"   📊 Labels preditos: {len(predicted_labels)}")
    print(f"   📊 Somas calculadas: {len(probability_sums)}")
    print(f"   📊 Labels reais: {len(true_labels)}")
    
    # Mostra estatísticas das predições
    unique_predicted = {}
    for label in predicted_labels.values():
        unique_predicted[label] = unique_predicted.get(label, 0) + 1
    
    print(f"\n   📈 Distribuição de predições:")
    label_names = {0: "dog", 1: "cat", 2: "lion", 3: "horse"}
    for label, count in sorted(unique_predicted.items()):
        label_name = label_names.get(label, f"label_{label}")
        percentage = (count / len(predicted_labels)) * 100
        print(f"      {label} ({label_name}): {count} imagens ({percentage:.2f}%)")
    
    print("=" * 60)
    print()
    
    return predicted_labels, probability_sums, true_labels


# ============================================================================
# ETAPA 6: Cálculo de Métricas Globais
# ============================================================================

def compute_global_metrics(
    true_labels: Dict[str, int],
    predicted_labels: Dict[str, int]
) -> Tuple[Tuple[List[int], List[int]], Tuple[float, float, float, float]]:
    """
    Calcula métricas globais de classificação usando a mesma função da técnica de relevância.
    
    Wrapper para tools.relevance.compute_metrics() para garantir consistência total
    entre a técnica de relevância e a técnica de soma de probabilidades.
    
    Args:
        true_labels: Dicionário {img_id: label_real}
        predicted_labels: Dicionário {img_id: label_predito}
    
    Returns:
        Tupla ((true_y, predicted_y), (accuracy, f1, recall, precision))
        - true_y, predicted_y: Listas com labels alinhados
        - accuracy, f1, recall, precision: Métricas no intervalo [0, 1]
    
    Example:
        >>> true = {"img1": 0, "img2": 1, "img3": 1}
        >>> pred = {"img1": 0, "img2": 1, "img3": 0}
        >>> (labels, metrics) = compute_global_metrics(true, pred)
        >>> accuracy, f1, recall, precision = metrics
    """
    print("📊 ETAPA 6: Cálculo de métricas globais")
    print(f"   📈 Total de amostras: {len(true_labels)}")
    print("-" * 60)
    
    # Usa a mesma função da técnica de relevância para garantir consistência
    labels_list, model_metrics = compute_metrics(true_labels, predicted_labels)
    
    true_y, predicted_y = labels_list
    
    print(f"   ✅ Amostras processadas: {len(true_y)}")
    print()
    
    # Exibe as métricas usando show_metrics() para consistência
    show_experiment_metrics(model_metrics, title="Soma de Probabilidades")
    
    print("=" * 60)
    print()
    
    return labels_list, model_metrics


# ============================================================================
# ETAPA 7: Geração da Matriz de Confusão
# ============================================================================

def generate_confusion_matrix(
    true_y: List[int],
    predicted_y: List[int],
    output_dir: str,
    model_name: str,
    cmap: str = "Blues",
    use_custom_dir: bool = True
) -> str:
    """
    Gera e salva a matriz de confusão para os resultados.
    
    Args:
        true_y: Lista com labels reais
        predicted_y: Lista com labels preditos
        output_dir: Diretório base para salvar a matriz
        model_name: Nome do modelo para o título e nome do arquivo
        cmap: Mapa de cores da matriz (padrão: "Blues")
        use_custom_dir: Se True, salva em <output_dir>/confusion_matrixs/; 
                       Se False, usa "results/confusion_matrixs/" (padrão: True)
        
    Returns:
        str: Caminho completo do arquivo da matriz de confusão salva
        
    Example:
        >>> true = [0, 1, 1, 0, 2]
        >>> pred = [0, 1, 0, 0, 2]
        >>> path = generate_confusion_matrix(true, pred, "results", "SumProb_SVM", cmap="Reds")
        >>> print(f"Matriz salva em: {path}")
    """
    print("📊 ETAPA 7: Geração da matriz de confusão")
    print(f"   📈 Modelo: {model_name}")
    print(f"   📁 Diretório base: {output_dir}")
    print(f"   🎨 Colormap: {cmap}")
    print("-" * 60)
        
    # Define o diretório de salvamento
    if use_custom_dir:
        save_dir = os.path.join(output_dir, "confusion_matrixs")
    else:
        save_dir = "results/confusion_matrixs"
    
    os.makedirs(save_dir, exist_ok=True)
    

    # Usa show_confusion_matrix() do utils.py para consistência
    filename = model_name.lower().replace(" ", "_").replace("-", "_").replace("+", "_")
    filename = "".join(c for c in filename if c.isalnum() or c == "_")    
    filepath = os.path.join(save_dir, f"{filename}_confusion_matrix.png")
    show_confusion_matrix(
        y=true_y,
        predict=predicted_y,
        title=model_name,
        cmap=cmap,
        verbose=False,
        save_dir=save_dir
    )
    
    print(f"   ✅ Matriz de confusão gerada")
    print(f"   💾 Arquivo salvo: {filepath}")
    print("=" * 60)
    print()
    
    return filepath


# ============================================================================
# ETAPA 8: Exportação para CSV
# ============================================================================

def export_to_csv(
    predicted_labels: Dict[str, int],
    true_labels: Dict[str, int],
    probability_sums: Dict[str, np.ndarray],
    probabilities: Dict[str, np.ndarray],
    model_metrics: Tuple[float, float, float, float],
    output_filepath: str
) -> str:
    """
    Exporta os resultados da soma de probabilidades para CSV.
    
    Args:
        predicted_labels: Dicionário {img_id: label_predito}
        true_labels: Dicionário {img_id: label_real}
        probability_sums: Dicionário {img_id: [soma_esp_0, soma_esp_1, ...]}
        probabilities: Dicionário {img_id: probabilidades_originais}
        model_metrics: Tupla (accuracy, f1, recall, precision)
        output_filepath: Caminho completo do arquivo CSV de saída
        
    Returns:
        str: Caminho do arquivo CSV gerado
        
    Example:
        >>> path = export_to_csv(pred_labels, true_labels, sums, probs, metrics, "results/output.csv")
        >>> print(f"CSV salvo em: {path}")
    """
    print("📊 ETAPA 8: Exportação para CSV")
    print(f"   📁 Arquivo de saída: {output_filepath}")
    print(f"   📈 Total de imagens: {len(predicted_labels)}")
    print("-" * 60)
    
    # Extrai métricas globais
    accuracy, f1, recall, precision = model_metrics
    
    # Cria o diretório se não existir
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    # Função auxiliar para serializar arrays (mesma do relevance.py)
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
    with open(output_filepath, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'nome_imagem', 'label_predito', 'label_real',
            'acuracia_global', 'f1_global', 'recall_global', 'precision_global',
            'probabilidades', 'somas_probabilidades'
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
                    'label_real': true_labels.get(img_id, -1),
                    'acuracia_global': round(accuracy, 4),
                    'f1_global': round(f1, 4),
                    'recall_global': round(recall, 4),
                    'precision_global': round(precision, 4),
                    'probabilidades': serialize_array(probabilities.get(img_id)),
                    'somas_probabilidades': serialize_array(probability_sums.get(img_id))
                }
                
                writer.writerow(row)
                processed_count += 1
                
            except Exception as e:
                print(f"   ⚠️  Erro ao processar {img_id}: {str(e)}")
                continue
    
    print(f"   ✅ {processed_count} linhas escritas com sucesso")
    print(f"   💾 Arquivo salvo: {output_filepath}")
    print("=" * 60)
    print()
    
    return output_filepath


# ============================================================================
# ETAPA 9: Geração de CSV Comparativo
# ============================================================================

# ============================================================================
# ETAPA 9: Geração de CSV Comparativo
# ============================================================================

def generate_comparison_csv(
    relevance_data: Dict[str, Dict],
    sum_predicted_labels: Dict[str, int],
    sum_true_labels: Dict[str, int],
    sum_metrics: Tuple[float, float, float, float],
    output_dir: str,
    model_name: str
) -> Tuple[str, str]:
    """
    Gera CSVs consolidados comparando resultados de relevância vs soma de probabilidades.
    
    Args:
        relevance_data: Dados do CSV de relevância {img_id: {'label_real', 'probabilidades', ...}}
        sum_predicted_labels: Labels preditos pela soma {img_id: label}
        sum_true_labels: Labels reais da soma {img_id: label}
        sum_metrics: Métricas da soma (accuracy, f1, recall, precision)
        output_dir: Diretório base do experimento
        model_name: Nome do modelo para identificação
        
    Returns:
        Tupla (comparison_csv_path, divergences_csv_path)
        
    Example:
        >>> paths = generate_comparison_csv(
        ...     relevance_data=data,
        ...     sum_predicted_labels=predicted_labels,
        ...     sum_true_labels=true_labels,
        ...     sum_metrics=model_metrics,
        ...     output_dir="results/exp",
        ...     model_name="SVM_LBP"
        ... )
    """
    print("=" * 80)
    print("📊 ETAPA 9: GERAÇÃO DE CSV COMPARATIVO")
    print("=" * 80)
    print(f"   📂 Diretório de saída: {output_dir}")
    print(f"   🏷️  Modelo: {model_name}")
    print("=" * 80)
    print()
    
    # ETAPA 9.1: Preparação dos dados
    print("📖 ETAPA 9.1: Preparação dos dados")
    print("-" * 60)
    
    # Extrai dados de relevância do formato interno
    relevance_labels = {}
    
    for img_id, img_data in relevance_data.items():
        relevance_labels[img_id] = {
            'label_predito': img_data.get('label_predito', -1),
            'label_real': img_data['label_real'],
            'acuracia': img_data.get('acuracia', 0.0),
            'f1': img_data.get('f1', 0.0),
            'recall': img_data.get('recall', 0.0),
            'precision': img_data.get('precision', 0.0)
        }
    
    print(f"   ✅ {len(relevance_labels)} imagens da técnica de relevância")
    
    # Prepara dados de soma
    sum_accuracy, sum_f1, sum_recall, sum_precision = sum_metrics
    sum_data = {}
    for img_id in sum_predicted_labels.keys():
        sum_data[img_id] = {
            'label_predito': sum_predicted_labels[img_id],
            'label_real': sum_true_labels[img_id],
            'acuracia': sum_accuracy,
            'f1': sum_f1,
            'recall': sum_recall,
            'precision': sum_precision
        }
    
    print(f"   ✅ {len(sum_data)} imagens da técnica de soma")
    print("=" * 60)
    print()
    
    # ETAPA 9.2: Validação de Consistência
    print("🔍 ETAPA 9.2: Validação de Consistência")
    print("-" * 60)
    
    relevance_imgs = set(relevance_labels.keys())
    sum_imgs = set(sum_data.keys())
    
    if relevance_imgs != sum_imgs:
        missing_in_sum = relevance_imgs - sum_imgs
        missing_in_relevance = sum_imgs - relevance_imgs
        
        if missing_in_sum:
            print(f"   ⚠️  Imagens em relevância mas não em soma: {len(missing_in_sum)}")
        
        if missing_in_relevance:
            print(f"   ⚠️  Imagens em soma mas não em relevância: {len(missing_in_relevance)}")
        
        raise ValueError("❌ Conjuntos de imagens não coincidem")
    
    print(f"   ✅ Conjuntos de imagens coincidem: {len(relevance_imgs)} imagens")
    
    # Verifica consistência de labels reais
    inconsistent_labels = []
    for img_id in relevance_imgs:
        if relevance_labels[img_id]['label_real'] != sum_data[img_id]['label_real']:
            inconsistent_labels.append(img_id)
    
    if inconsistent_labels:
        raise ValueError(f"❌ Labels reais não coincidem: {inconsistent_labels[:5]}")
    
    print("   ✅ Labels reais consistentes")
    print("=" * 60)
    print()
    
    # ETAPA 9.3: Processamento e Comparação
    print("🔄 ETAPA 9.3: Processamento e Comparação")
    print("-" * 60)
    
    comparison_rows = []
    divergence_rows = []
    
    acerto_stats = {'C': 0, 'R': 0, 'S': 0, 'N': 0}
    divergence_images = {'R': [], 'S': [], 'N': []}
    
    for img_id in sorted(relevance_imgs):
        rel_data = relevance_labels[img_id]
        sum_data_img = sum_data[img_id]
        
        label_real = rel_data['label_real']
        label_relevancia = rel_data['label_predito']
        label_soma = sum_data_img['label_predito']
        
        # Determina tipo de acerto
        rel_acertou = (label_relevancia == label_real)
        sum_acertou = (label_soma == label_real)
        
        if rel_acertou and sum_acertou:
            acerto = 'C'
        elif rel_acertou and not sum_acertou:
            acerto = 'R'
            divergence_images['R'].append(img_id)
        elif not rel_acertou and sum_acertou:
            acerto = 'S'
            divergence_images['S'].append(img_id)
        else:
            acerto = 'N'
            divergence_images['N'].append(img_id)
        
        acerto_stats[acerto] += 1
        
        # Calcula diferenças
        diff_acuracia = rel_data['acuracia'] - sum_data_img['acuracia']
        diff_f1 = rel_data['f1'] - sum_data_img['f1']
        diff_recall = rel_data['recall'] - sum_data_img['recall']
        diff_precision = rel_data['precision'] - sum_data_img['precision']
        
        # Monta linha para CSV de comparação
        comparison_row = {
            'imagem': img_id,
            'label_real': label_real,
            'label_relevancia': label_relevancia,
            'label_soma': label_soma,
            'acerto': acerto,
            'acuracia_relevancia': round(rel_data['acuracia'], 4),
            'acuracia_soma': round(sum_data_img['acuracia'], 4),
            'diferenca_acuracia': round(diff_acuracia, 4),
            'f1_relevancia': round(rel_data['f1'], 4),
            'f1_soma': round(sum_data_img['f1'], 4),
            'diferenca_f1': round(diff_f1, 4),
            'recall_relevancia': round(rel_data['recall'], 4),
            'recall_soma': round(sum_data_img['recall'], 4),
            'diferenca_recall': round(diff_recall, 4),
            'precision_relevancia': round(rel_data['precision'], 4),
            'precision_soma': round(sum_data_img['precision'], 4),
            'diferenca_precision': round(diff_precision, 4)
        }
        
        comparison_rows.append(comparison_row)
        
        # Adiciona ao CSV de divergências se não for acerto completo
        if acerto in ['R', 'S', 'N']:
            divergence_row = {
                'imagem': img_id,
                'label_real': label_real,
                'label_soma': label_soma,
                'label_relevancia': label_relevancia,
                'tipo_divergencia': acerto
            }
            divergence_rows.append(divergence_row)
    
    total = len(comparison_rows)
    print(f"   ✅ {total} linhas processadas")
    print("   📊 Estatísticas de acerto:")
    for tipo in ['C', 'R', 'S', 'N']:
        count = acerto_stats[tipo]
        percentage = (count / total) * 100
        tipo_nome = {'C': 'Ambos acertaram', 'R': 'Só Relevância', 'S': 'Só Soma', 'N': 'Nenhum acertou'}[tipo]
        print(f"      {tipo} ({tipo_nome}): {count} ({percentage:.2f}%)")
    print("=" * 60)
    print()
    
    # ETAPA 9.4: Exportação do CSV de Comparação
    print("💾 ETAPA 9.4: Exportação do CSV de Comparação")
    print("-" * 60)
    
    comp_dir = os.path.join(output_dir, "comp", "csv_exports")
    os.makedirs(comp_dir, exist_ok=True)
    
    comp_filename = f"{model_name.lower().replace('-', '_').replace(' ', '_')}_comparison.csv"
    comp_filepath = os.path.join(comp_dir, comp_filename)
    
    comparison_fieldnames = [
        'imagem', 'label_real', 'label_relevancia', 'label_soma', 'acerto',
        'acuracia_relevancia', 'acuracia_soma', 'diferenca_acuracia',
        'f1_relevancia', 'f1_soma', 'diferenca_f1',
        'recall_relevancia', 'recall_soma', 'diferenca_recall',
        'precision_relevancia', 'precision_soma', 'diferenca_precision'
    ]
    
    with open(comp_filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=comparison_fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)
    
    print(f"   ✅ CSV de comparação salvo: {comp_filepath}")
    print(f"   📊 {len(comparison_rows)} linhas escritas")
    print("=" * 60)
    print()
    
    # ETAPA 9.5: Exportação do CSV de Divergências
    print("💾 ETAPA 9.5: Exportação do CSV de Divergências")
    print("-" * 60)
    
    div_filename = f"{model_name.lower().replace('-', '_').replace(' ', '_')}_divergences.csv"
    div_filepath = os.path.join(comp_dir, div_filename)
    
    divergence_fieldnames = [
        'imagem', 'label_real', 'label_soma', 'label_relevancia', 'tipo_divergencia'
    ]
    
    with open(div_filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=divergence_fieldnames)
        writer.writeheader()
        writer.writerows(divergence_rows)
    
    print(f"   ✅ CSV de divergências salvo: {div_filepath}")
    print(f"   📊 {len(divergence_rows)} linhas escritas")
    print("=" * 60)
    print()
    
    # ETAPA 9.6: Resumo Final e Análise de Divergências
    print("=" * 80)
    print("📊 RESUMO DA COMPARAÇÃO")
    print("=" * 80)
    print()
    print(f"🎯 Modelo: {model_name}")
    print(f"📈 Total de imagens: {total}")
    print()
    print("📊 Distribuição de Acertos:")
    print("-" * 60)
    print(f"   ✅ Ambos acertaram (C): {acerto_stats['C']} ({acerto_stats['C']/total*100:.2f}%)")
    print(f"   🟡 Só Relevância (R): {acerto_stats['R']} ({acerto_stats['R']/total*100:.2f}%)")
    print(f"   🟠 Só Soma (S): {acerto_stats['S']} ({acerto_stats['S']/total*100:.2f}%)")
    print(f"   ❌ Nenhum acertou (N): {acerto_stats['N']} ({acerto_stats['N']/total*100:.2f}%)")
    print()
    
    # Calcula diferenças médias de métricas
    avg_diff_acuracia = sum(row['diferenca_acuracia'] for row in comparison_rows) / total
    avg_diff_f1 = sum(row['diferenca_f1'] for row in comparison_rows) / total
    avg_diff_recall = sum(row['diferenca_recall'] for row in comparison_rows) / total
    avg_diff_precision = sum(row['diferenca_precision'] for row in comparison_rows) / total
    
    print("📈 Diferenças Médias (Relevância - Soma):")
    print("-" * 60)
    print(f"   Acurácia: {avg_diff_acuracia:+.4f}")
    print(f"   F1-Score: {avg_diff_f1:+.4f}")
    print(f"   Recall: {avg_diff_recall:+.4f}")
    print(f"   Precision: {avg_diff_precision:+.4f}")
    print()
    
    # Lista imagens divergentes
    print("📋 Imagens com Divergências:")
    print("-" * 60)
    
    if divergence_images['R']:
        print(f"\n🟡 Só Relevância Acertou ({len(divergence_images['R'])} imagens):")
        for img in divergence_images['R']:
            rel_label = relevance_labels[img]['label_predito']
            sum_label = sum_data[img]['label_predito']
            real_label = relevance_labels[img]['label_real']
            print(f"   • {img}: Real={real_label}, Relevância={rel_label}, Soma={sum_label}")
    
    if divergence_images['S']:
        print(f"\n🟠 Só Soma Acertou ({len(divergence_images['S'])} imagens):")
        for img in divergence_images['S']:
            rel_label = relevance_labels[img]['label_predito']
            sum_label = sum_data[img]['label_predito']
            real_label = relevance_labels[img]['label_real']
            print(f"   • {img}: Real={real_label}, Relevância={rel_label}, Soma={sum_label}")
    
    if divergence_images['N']:
        print(f"\n❌ Nenhum Acertou ({len(divergence_images['N'])} imagens):")
        for img in divergence_images['N']:
            rel_label = relevance_labels[img]['label_predito']
            sum_label = sum_data[img]['label_predito']
            real_label = relevance_labels[img]['label_real']
            print(f"   • {img}: Real={real_label}, Relevância={rel_label}, Soma={sum_label}")
    
    print()
    print("📁 Arquivos Gerados:")
    print("-" * 60)
    print(f"   • Comparação: {comp_filepath}")
    print(f"   • Divergências: {div_filepath}")
    print("=" * 80)
    print()
    
    return comp_filepath, div_filepath
    """
    Gera CSVs consolidados comparando resultados de relevância vs soma de probabilidades.
    
    Args:
        relevance_csv_path: Caminho do CSV da técnica de relevância (para leitura se necessário)
        sum_csv_path: Caminho do CSV da técnica de soma (para referência)
        output_dir: Diretório base do experimento
        model_name: Nome do modelo para identificação
        relevance_data: (Opcional) Dados já lidos do CSV de relevância
        sum_predicted_labels: (Opcional) Labels preditos pela soma
        sum_true_labels: (Opcional) Labels reais da soma
        sum_metrics: (Opcional) Métricas da soma (accuracy, f1, recall, precision)
        
    Returns:
        Tupla (comparison_csv_path, divergences_csv_path)
        
    Example:
        >>> paths = generate_comparison_csv(
        ...     relevance_csv_path="results/exp/csv_exports/svm_lbp_results.csv",
        ...     sum_csv_path="results/exp/sum/csv_exports/svm_lbp_results.csv",
        ...     output_dir="results/exp",
        ...     model_name="SVM_LBP",
        ...     relevance_data=data,
        ...     sum_predicted_labels=predicted_labels,
        ...     sum_true_labels=true_labels,
        ...     sum_metrics=model_metrics
        ... )
    """
    print("=" * 80)
    print("📊 ETAPA 9: GERAÇÃO DE CSV COMPARATIVO")
    print("=" * 80)
    print(f"    Diretório de saída: {output_dir}")
    print(f"   🏷️  Modelo: {model_name}")
    print("=" * 80)
    print()
    
    # ETAPA 9.1: Preparação dos dados
    print("📖 ETAPA 9.1: Preparação dos dados")
    print("-" * 60)
    
    # Extrai dados de relevância do formato interno
    relevance_labels = {}
    
    for img_id, img_data in relevance_data.items():
        relevance_labels[img_id] = {
            'label_predito': img_data.get('label_predito', -1),
            'label_real': img_data['label_real'],
            'acuracia': img_data.get('acuracia', 0.0),
            'f1': img_data.get('f1', 0.0),
            'recall': img_data.get('recall', 0.0),
            'precision': img_data.get('precision', 0.0)
        }
    
    print(f"   ✅ {len(relevance_labels)} imagens da técnica de relevância")
    
    # Prepara dados de soma
    sum_accuracy, sum_f1, sum_recall, sum_precision = sum_metrics
    sum_data = {}
    for img_id in sum_predicted_labels.keys():
        sum_data[img_id] = {
            'label_predito': sum_predicted_labels[img_id],
            'label_real': sum_true_labels[img_id],
            'acuracia': sum_accuracy,
            'f1': sum_f1,
            'recall': sum_recall,
            'precision': sum_precision
        }
    
    print(f"   ✅ {len(sum_data)} imagens da técnica de soma")
    print("=" * 60)
    print()
    
    # ETAPA 9.2: Validação de Consistência
    print("🔍 ETAPA 9.2: Validação de Consistência")
    print("-" * 60)
    
    # Verifica se as imagens são as mesmas
    relevance_imgs = set(relevance_data.keys())
    sum_imgs = set(sum_data.keys())
    
    if relevance_imgs != sum_imgs:
        missing_in_sum = relevance_imgs - sum_imgs
        missing_in_relevance = sum_imgs - relevance_imgs
        
        if missing_in_sum:
            print(f"   ⚠️  Imagens em relevância mas não em soma: {len(missing_in_sum)}")
            print(f"      {list(missing_in_sum)[:5]}...")
        
        if missing_in_relevance:
            print(f"   ⚠️  Imagens em soma mas não em relevância: {len(missing_in_relevance)}")
            print(f"      {list(missing_in_relevance)[:5]}...")
        
        raise ValueError("❌ Conjuntos de imagens não coincidem entre os CSVs")
    
    print(f"   ✅ Conjuntos de imagens coincidem: {len(relevance_imgs)} imagens")
    
    # Verifica consistência de labels reais
    inconsistent_labels = []
    for img_id in relevance_imgs:
        if relevance_data[img_id]['label_real'] != sum_data[img_id]['label_real']:
            inconsistent_labels.append(img_id)
    
    if inconsistent_labels:
        print(f"   ⚠️  Labels reais inconsistentes: {len(inconsistent_labels)} imagens")
        raise ValueError(f"❌ Labels reais não coincidem: {inconsistent_labels[:5]}")
    
    print("   ✅ Labels reais consistentes entre os dados")
    print("=" * 60)
    print()
    
    # ETAPA 9.3: Processamento e Comparação
    print("🔄 ETAPA 9.3: Processamento e Comparação")
    print("-" * 60)
    
    comparison_rows = []
    divergence_rows = []
    
    # Contadores de acertos
    acerto_stats = {
        'C': 0,  # Ambos acertaram
        'R': 0,  # Só relevância
        'S': 0,  # Só soma
        'N': 0   # Nenhum acertou
    }
    
    # Listas de imagens por categoria
    divergence_images = {
        'R': [],  # Só relevância acertou
        'S': [],  # Só soma acertou
        'N': []   # Nenhum acertou
    }
    
    for img_id in sorted(relevance_imgs):
        rel_data = relevance_data[img_id]
        sum_data_img = sum_data[img_id]
        
        label_real = rel_data['label_real']
        label_relevancia = rel_data['label_predito']
        label_soma = sum_data_img['label_predito']
        
        # Determina tipo de acerto
        rel_acertou = (label_relevancia == label_real)
        sum_acertou = (label_soma == label_real)
        
        if rel_acertou and sum_acertou:
            acerto = 'C'
        elif rel_acertou and not sum_acertou:
            acerto = 'R'
            divergence_images['R'].append(img_id)
        elif not rel_acertou and sum_acertou:
            acerto = 'S'
            divergence_images['S'].append(img_id)
        else:
            acerto = 'N'
            divergence_images['N'].append(img_id)
        
        acerto_stats[acerto] += 1
        
        # Calcula diferenças
        diff_acuracia = rel_data['acuracia'] - sum_data_img['acuracia']
        diff_f1 = rel_data['f1'] - sum_data_img['f1']
        diff_recall = rel_data['recall'] - sum_data_img['recall']
        diff_precision = rel_data['precision'] - sum_data_img['precision']
        
        # Monta linha para CSV de comparação
        comparison_row = {
            'imagem': img_id,
            'label_real': label_real,
            'label_relevancia': label_relevancia,
            'label_soma': label_soma,
            'acerto': acerto,
            'acuracia_relevancia': round(rel_data['acuracia'], 4),
            'acuracia_soma': round(sum_data_img['acuracia'], 4),
            'diferenca_acuracia': round(diff_acuracia, 4),
            'f1_relevancia': round(rel_data['f1'], 4),
            'f1_soma': round(sum_data_img['f1'], 4),
            'diferenca_f1': round(diff_f1, 4),
            'recall_relevancia': round(rel_data['recall'], 4),
            'recall_soma': round(sum_data_img['recall'], 4),
            'diferenca_recall': round(diff_recall, 4),
            'precision_relevancia': round(rel_data['precision'], 4),
            'precision_soma': round(sum_data_img['precision'], 4),
            'diferenca_precision': round(diff_precision, 4)
        }
        
        comparison_rows.append(comparison_row)
        
        # Adiciona ao CSV de divergências se não for acerto completo
        if acerto in ['R', 'S', 'N']:
            divergence_row = {
                'imagem': img_id,
                'label_real': label_real,
                'label_soma': label_soma,
                'label_relevancia': label_relevancia,
                'tipo_divergencia': acerto
            }
            divergence_rows.append(divergence_row)
    
    print(f"   ✅ {len(comparison_rows)} linhas processadas")
    print(f"   📊 Estatísticas de acerto:")
    total = len(comparison_rows)
    for tipo, count in sorted(acerto_stats.items()):
        percentage = (count / total) * 100
        tipo_nome = {
            'C': 'Ambos acertaram',
            'R': 'Só Relevância',
            'S': 'Só Soma',
            'N': 'Nenhum acertou'
        }[tipo]
        print(f"      {tipo} ({tipo_nome}): {count} ({percentage:.2f}%)")
    print("=" * 60)
    print()
    
    # ETAPA 9.4: Exportação do CSV de Comparação
    print("💾 ETAPA 9.4: Exportação do CSV de Comparação")
    print("-" * 60)
    
    comp_dir = os.path.join(output_dir, "comp", "csv_exports")
    os.makedirs(comp_dir, exist_ok=True)
    
    comp_filename = f"{model_name.lower().replace('-', '_').replace(' ', '_')}_comparison.csv"
    comp_filepath = os.path.join(comp_dir, comp_filename)
    
    comparison_fieldnames = [
        'imagem', 'label_real', 'label_relevancia', 'label_soma', 'acerto',
        'acuracia_relevancia', 'acuracia_soma', 'diferenca_acuracia',
        'f1_relevancia', 'f1_soma', 'diferenca_f1',
        'recall_relevancia', 'recall_soma', 'diferenca_recall',
        'precision_relevancia', 'precision_soma', 'diferenca_precision'
    ]
    
    with open(comp_filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=comparison_fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)
    
    print(f"   ✅ CSV de comparação salvo: {comp_filepath}")
    print(f"   📊 {len(comparison_rows)} linhas escritas")
    print("=" * 60)
    print()
    
    # ETAPA 9.5: Exportação do CSV de Divergências
    print("💾 ETAPA 9.5: Exportação do CSV de Divergências")
    print("-" * 60)
    
    div_filename = f"{model_name.lower().replace('-', '_').replace(' ', '_')}_divergences.csv"
    div_filepath = os.path.join(comp_dir, div_filename)
    
    divergence_fieldnames = [
        'imagem', 'label_real', 'label_soma', 'label_relevancia', 'tipo_divergencia'
    ]
    
    with open(div_filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=divergence_fieldnames)
        writer.writeheader()
        writer.writerows(divergence_rows)
    
    print(f"   ✅ CSV de divergências salvo: {div_filepath}")
    print(f"   📊 {len(divergence_rows)} linhas escritas")
    print("=" * 60)
    print()
    
    # ETAPA 9.6: Resumo Final e Análise de Divergências
    print("=" * 80)
    print("📊 RESUMO DA COMPARAÇÃO")
    print("=" * 80)
    print()
    print(f"🎯 Modelo: {model_name}")
    print(f"📈 Total de imagens: {total}")
    print()
    print("📊 Distribuição de Acertos:")
    print("-" * 60)
    print(f"   ✅ Ambos acertaram (C): {acerto_stats['C']} ({acerto_stats['C']/total*100:.2f}%)")
    print(f"   🟡 Só Relevância (R): {acerto_stats['R']} ({acerto_stats['R']/total*100:.2f}%)")
    print(f"   🟠 Só Soma (S): {acerto_stats['S']} ({acerto_stats['S']/total*100:.2f}%)")
    print(f"   ❌ Nenhum acertou (N): {acerto_stats['N']} ({acerto_stats['N']/total*100:.2f}%)")
    print()
    
    # Calcula diferenças médias de métricas
    avg_diff_acuracia = sum(row['diferenca_acuracia'] for row in comparison_rows) / total
    avg_diff_f1 = sum(row['diferenca_f1'] for row in comparison_rows) / total
    avg_diff_recall = sum(row['diferenca_recall'] for row in comparison_rows) / total
    avg_diff_precision = sum(row['diferenca_precision'] for row in comparison_rows) / total
    
    print("📈 Diferenças Médias (Relevância - Soma):")
    print("-" * 60)
    print(f"   Acurácia: {avg_diff_acuracia:+.4f}")
    print(f"   F1-Score: {avg_diff_f1:+.4f}")
    print(f"   Recall: {avg_diff_recall:+.4f}")
    print(f"   Precision: {avg_diff_precision:+.4f}")
    print()
    
    # Lista imagens divergentes
    print("📋 Imagens com Divergências:")
    print("-" * 60)
    
    if divergence_images['R']:
        print(f"\n🟡 Só Relevância Acertou ({len(divergence_images['R'])} imagens):")
        for img in divergence_images['R']:
            rel_label = relevance_data[img]['label_predito']
            sum_label = sum_data[img]['label_predito']
            real_label = relevance_data[img]['label_real']
            print(f"   • {img}: Real={real_label}, Relevância={rel_label}, Soma={sum_label}")
    
    if divergence_images['S']:
        print(f"\n🟠 Só Soma Acertou ({len(divergence_images['S'])} imagens):")
        for img in divergence_images['S']:
            rel_label = relevance_data[img]['label_predito']
            sum_label = sum_data[img]['label_predito']
            real_label = relevance_data[img]['label_real']
            print(f"   • {img}: Real={real_label}, Relevância={rel_label}, Soma={sum_label}")
    
    if divergence_images['N']:
        print(f"\n❌ Nenhum Acertou ({len(divergence_images['N'])} imagens):")
        for img in divergence_images['N']:
            rel_label = relevance_data[img]['label_predito']
            sum_label = sum_data[img]['label_predito']
            real_label = relevance_data[img]['label_real']
            print(f"   • {img}: Real={real_label}, Relevância={rel_label}, Soma={sum_label}")
    
    print()
    print("📁 Arquivos Gerados:")
    print("-" * 60)
    print(f"   • Comparação: {comp_filepath}")
    print(f"   • Divergências: {div_filepath}")
    print("=" * 80)
    print()
    
    return comp_filepath, div_filepath


# ============================================================================
# ETAPA 10: Função Principal (main)
# ============================================================================

def main(input_csv: str, output_dir: str, model_name: str = "SumProbabilities", experiment_dir: str = None):
    """
    Executa o pipeline completo de classificação por soma de probabilidades.
    
    Args:
        input_csv: Caminho para o CSV de entrada (gerado pela técnica de relevância)
        output_dir: Diretório base para salvar os resultados
        model_name: Nome do modelo para identificação nos arquivos de saída
        experiment_dir: Diretório do experimento (para geração de comparação)
        
    Returns:
        Dict com os caminhos dos arquivos gerados
        
    Example:
        >>> results = main(
        ...     "results/experiments/exp_20251030/csv_exports/svm_lbp_results.csv",
        ...     "results/sum_probabilities",
        ...     "SVM_LBP_GLCM_SumProb"
        ... )
        >>> print(f"CSV: {results['csv']}")
        >>> print(f"Matriz: {results['confusion_matrix']}")
    """
    print("=" * 80)
    print("🚀 PIPELINE: CLASSIFICAÇÃO POR SOMA DE PROBABILIDADES")
    print("=" * 80)
    print(f"   📁 CSV de entrada: {input_csv}")
    print(f"   📂 Diretório de saída: {output_dir}")
    print(f"   🏷️  Modelo: {model_name}")
    print("=" * 80)
    print()
    
    try:
        # ETAPA 2: Leitura do CSV
        data = read_relevance_csv(input_csv)
        
        # ETAPA 5: Processamento de todas as imagens
        predicted_labels, probability_sums, true_labels = process_all_images(data)
        
        # ETAPA 6: Cálculo de métricas globais
        labels_list, model_metrics = compute_global_metrics(true_labels, predicted_labels)
        true_y, predicted_y = labels_list
        
        # ETAPA 7: Geração da matriz de confusão
        cmap = model_name.__contains__("SVM") and "Purples" or "Oranges"
        confusion_matrix_path = generate_confusion_matrix(
            true_y, predicted_y, output_dir, model_name, use_custom_dir=True, cmap=cmap
        )
        
        # ETAPA 8: Exportação para CSV
        # Extrai probabilidades originais
        original_probabilities = {img_id: img_data['probabilidades'] 
                                 for img_id, img_data in data.items()}
        
        # Define caminho do CSV de saída
        csv_dir = os.path.join(output_dir, "csv_exports")
        os.makedirs(csv_dir, exist_ok=True)
        
        csv_filename = f"{model_name.lower().replace('-', '_').replace(' ', '_')}_results.csv"
        csv_filepath = os.path.join(csv_dir, csv_filename)
        
        csv_path = export_to_csv(
            predicted_labels=predicted_labels,
            true_labels=true_labels,
            probability_sums=probability_sums,
            probabilities=original_probabilities,
            model_metrics=model_metrics,
            output_filepath=csv_filepath
        )
        
        # Resumo final
        print("=" * 80)
        print("🎉 PIPELINE CONCLUÍDO COM SUCESSO!")
        print("=" * 80)
        print("📊 Resumo dos Resultados:")
        print(f"   • Total de imagens: {len(predicted_labels)}")
        print(f"   • Acurácia: {model_metrics[0]*100:.2f}%")
        print(f"   • F1-Score: {model_metrics[1]*100:.2f}%")
        print()
        print("📁 Arquivos Gerados:")
        print(f"   • CSV: {csv_path}")
        print(f"   • Matriz de Confusão: {confusion_matrix_path}")
        print("=" * 80)
        print()
        
        # Resultado base
        result = {
            'csv': csv_path,
            'confusion_matrix': confusion_matrix_path,
            'metrics': model_metrics,
            'predicted_labels': predicted_labels,
            'true_labels': true_labels
        }
        
        # ETAPA 9: Geração de CSV Comparativo (se experiment_dir fornecido)
        if experiment_dir:
            print()
            print("🔄 Gerando CSV comparativo...")
            print()
            
            try:
                # Gera CSVs de comparação usando dados já processados
                comp_path, div_path = generate_comparison_csv(
                    relevance_data=data,
                    sum_predicted_labels=predicted_labels,
                    sum_true_labels=true_labels,
                    sum_metrics=model_metrics,
                    output_dir=experiment_dir,
                    model_name=model_name
                )
                
                # Adiciona aos resultados
                result['comparison_csv'] = comp_path
                result['divergences_csv'] = div_path
                
            except Exception as e:
                print(f"⚠️  Erro ao gerar CSV comparativo: {e}")
                import traceback
                traceback.print_exc()
        
        return result
        
    except FileNotFoundError as e:
        print(f"❌ ERRO: Arquivo não encontrado - {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ ERRO: Valor inválido - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERRO INESPERADO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def process_experiment_folder(experiment_path: str) -> Dict[str, Dict]:
    """
    Processa todos os CSVs de um experimento e gera resultados de soma de probabilidades.
    
    Args:
        experiment_path: Caminho para a pasta do experimento (ex: "results/experiments/experimento_20251030_195628")
        
    Returns:
        Dict com os resultados de cada modelo processado
        
    Example:
        >>> results = process_experiment_folder("results/experiments/experimento_20251030_195628")
        >>> print(f"Processados {len(results)} modelos")
    """
    print("=" * 80)
    print("🚀 PROCESSAMENTO EM LOTE: EXPERIMENTO COMPLETO")
    print("=" * 80)
    print(f"   📁 Experimento: {experiment_path}")
    print("=" * 80)
    print()
    
    # Valida que o diretório existe
    if not os.path.exists(experiment_path):
        raise FileNotFoundError(f"Diretório do experimento não encontrado: {experiment_path}")
    
    # Localiza o diretório csv_exports
    csv_dir = os.path.join(experiment_path, "csv_exports")
    if not os.path.exists(csv_dir):
        raise FileNotFoundError(f"Diretório csv_exports não encontrado: {csv_dir}")
    
    # Busca todos os arquivos CSV
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('_results.csv')]
    
    if not csv_files:
        raise ValueError(f"Nenhum arquivo CSV encontrado em: {csv_dir}")
    
    print(f"📊 Encontrados {len(csv_files)} arquivos CSV para processar:")
    for csv_file in sorted(csv_files):
        print(f"   • {csv_file}")
    print()
    
    # Define diretório de saída dentro do experimento
    output_base_dir = os.path.join(experiment_path, "sum")
    
    # Processa cada CSV
    all_results = {}
    processed_count = 0
    failed_count = 0
    
    for csv_file in sorted(csv_files):
        try:
            # Extrai o nome do modelo do arquivo CSV
            # Exemplo: "knn_glcm_lpq_results.csv" -> "KNN_GLCM_LPQ"
            model_name = csv_file.replace('_results.csv', '').upper()
            
            input_csv = os.path.join(csv_dir, csv_file)
            
            print("-" * 80)
            print(f"🔄 Processando: {csv_file}")
            print(f"   🏷️  Modelo: {model_name}")
            print("-" * 80)
            print()
            
            # Executa o pipeline para este CSV
            result = main(
                input_csv=input_csv, 
                output_dir=output_base_dir, 
                model_name=model_name,
                experiment_dir=experiment_path  # Passa o diretório do experimento
            )
            
            all_results[model_name] = result
            processed_count += 1
            
        except Exception as e:
            print(f"❌ ERRO ao processar {csv_file}: {e}")
            failed_count += 1
            continue
    
    # Resumo final
    print()
    print("=" * 80)
    print("🎉 PROCESSAMENTO EM LOTE CONCLUÍDO!")
    print("=" * 80)
    print(f"   ✅ Processados com sucesso: {processed_count}/{len(csv_files)}")
    if failed_count > 0:
        print(f"   ❌ Falhas: {failed_count}/{len(csv_files)}")
    print(f"   📂 Resultados salvos em: {output_base_dir}")
    print()
    print("📊 Resumo das Métricas:")
    print("-" * 80)
    for model_name, result in sorted(all_results.items()):
        f1 = result['metrics'][1] * 100
        acc = result['metrics'][0] * 100
        print(f"   • {model_name}:")
        print(f"      - Acurácia: {acc:.2f}%")
        print(f"      - F1-Score: {f1:.2f}%")
    
    # Resumo dos CSVs de comparação
    print()
    print("📋 CSVs de Comparação Gerados:")
    print("-" * 80)
    comp_count = sum(1 for r in all_results.values() if 'comparison_csv' in r)
    if comp_count > 0:
        print(f"   ✅ {comp_count} comparações geradas")
        print(f"   📂 Diretório: {experiment_path}/comp/csv_exports/")
    else:
        print("   ⚠️  Nenhuma comparação gerada")
    
    print("=" * 80)
    print()
    
    return all_results


# ============================================================================
# PONTO DE ENTRADA PARA TESTE E CLI
# ============================================================================

if __name__ == "__main__":
    # Verifica se foram passados argumentos de linha de comando
    if len(sys.argv) == 2 and sys.argv[1] not in ['-h', '--help', 'help']:
        # Modo LOTE: python sum_of_probabilities.py <experiment_folder>
        experiment_folder = sys.argv[1]
        
        print("📦 MODO LOTE: Processando experimento completo")
        print()
        
        results = process_experiment_folder(experiment_folder)
        
        sys.exit(0)
    
    elif len(sys.argv) >= 3:
        # Modo INDIVIDUAL: python sum_of_probabilities.py <input_csv> <output_dir> [model_name]
        input_csv_arg = sys.argv[1]
        output_dir_arg = sys.argv[2]
        model_name_arg = sys.argv[3] if len(sys.argv) > 3 else "SumProbabilities"
        
        print("🎯 MODO INDIVIDUAL: Executando pipeline com argumentos fornecidos")
        print()
        
        results = main(input_csv_arg, output_dir_arg, model_name_arg)
        
        sys.exit(0)
    
    elif len(sys.argv) == 2 and sys.argv[1] in ['-h', '--help', 'help']:
        # Modo AJUDA: Mostra instruções de uso
        print("=" * 80)
        print("📖 AJUDA: Classificação por Soma de Probabilidades")
        print("=" * 80)
        print()
        print("DESCRIÇÃO:")
        print("   Classifica imagens somando as probabilidades de todos os pedaços")
        print("   para cada especialista. O especialista com maior soma vence.")
        print()
        print("=" * 80)
        print("MODO 1: Processamento em Lote (Experimento Completo)")
        print("=" * 80)
        print()
        print("USO:")
        print("   python sum_of_probabilities.py <experiment_dir>")
        print()
        print("ARGUMENTOS:")
        print("   experiment_dir - Diretório do experimento contendo csv_exports/")
        print()
        print("EXEMPLO:")
        print("   python sum_of_probabilities.py results/experiments/experimento_20251030_195628")
        print()
        print("SAÍDA:")
        print("   Processa todos os CSVs em <experiment_dir>/csv_exports/ e salva:")
        print("   • CSVs: <experiment_dir>/sum/csv_exports/<model>_results.csv")
        print("   • Matrizes: <experiment_dir>/sum/confusion_matrixs/<model>_confusion_matrix.png")
        print()
        print("=" * 80)
        print("MODO 2: Processamento Individual (CSV Único)")
        print("=" * 80)
        print()
        print("USO:")
        print("   python sum_of_probabilities.py <input_csv> <output_dir> [model_name]")
        print()
        print("ARGUMENTOS:")
        print("   input_csv   - Caminho do CSV gerado pela técnica de relevância")
        print("   output_dir  - Diretório onde salvar os resultados")
        print("   model_name  - (Opcional) Nome do modelo para identificação")
        print()
        print("EXEMPLO:")
        print("   python sum_of_probabilities.py \\")
        print("       results/experiments/exp_20251030/csv_exports/svm_lbp_results.csv \\")
        print("       results/sum_probabilities \\")
        print("       \"SVM_LBP_GLCM_SumProb\"")
        print()
        print("SAÍDA:")
        print("   • CSV com resultados: <output_dir>/csv_exports/<model_name>_results.csv")
        print("   • Matriz de confusão: <output_dir>/confusion_matrixs/<model_name>_confusion_matrix.png")
        print()
        print("TESTES:")
        print("   python sum_of_probabilities.py")
        print("   (Executa testes de validação automáticos)")
        print("=" * 80)
        sys.exit(0)
    
    # Modo TESTE: Executa testes de validação
    print("🧪 MODO TESTE: Validando todas as etapas")
    print("=" * 60)
    print()
    
    # Arquivo de teste
    test_csv = "results/experiments/experimento_20251030_195628/csv_exports/svm_lbp_glcm_lpq_results.csv"
    
    if os.path.exists(test_csv):
        try:
            # ETAPA 2: Leitura do CSV
            data = read_relevance_csv(test_csv)
            print(f"✅ ETAPA 2 OK: {len(data)} imagens carregadas")
            print()
            
            # ETAPA 3 e 4: Teste rápido com primeira imagem
            print("🧮 ETAPA 3 e 4: Teste rápido de somas e predição")
            print("-" * 60)
            
            first_img = list(data.keys())[0]
            first_data = data[first_img]
            first_probs = first_data['probabilidades']
            first_label_real = first_data['label_real']
            
            sums = calculate_probability_sums(first_probs)
            predicted_label = predict_label_by_sum(sums)
            
            print(f"   🔍 Imagem teste: {first_img}")
            print(f"   📊 Somas: {sums}")
            print(f"   🎯 Predito={predicted_label}, Real={first_label_real}")
            
            if predicted_label == first_label_real:
                print("   ✅ Predição correta!")
            else:
                print("   ⚠️  Predição incorreta")
            
            print("=" * 60)
            print()
            
            # ETAPA 5: Processamento de todas as imagens
            predicted_labels, probability_sums, true_labels = process_all_images(data)
            
            # ETAPA 6: Cálculo de métricas globais
            labels_list, model_metrics = compute_global_metrics(
                true_labels, predicted_labels
            )
            
            true_y, predicted_y = labels_list
            accuracy, f1, recall, precision = model_metrics
            
            # ETAPA 7: Geração da matriz de confusão
            test_model_name = "TEST_SumProbabilities_SVM"
            confusion_matrix_path = generate_confusion_matrix(
                true_y, predicted_y, "results", test_model_name
            )
            
            # ETAPA 8: Exportação para CSV
            # Extrai as probabilidades originais do data
            original_probabilities = {img_id: img_data['probabilidades'] 
                                     for img_id, img_data in data.items()}
            
            test_output_csv = "results/test_sum_probabilities_results.csv"
            csv_path = export_to_csv(
                predicted_labels=predicted_labels,
                true_labels=true_labels,
                probability_sums=probability_sums,
                probabilities=original_probabilities,
                model_metrics=model_metrics,
                output_filepath=test_output_csv
            )
            
            # Validações das ETAPAS 6, 7 e 8
            print("🔍 VALIDAÇÕES DAS ETAPAS 6, 7 e 8")
            print("-" * 60)
            
            # Validações ETAPA 6
            assert 0.0 <= accuracy <= 1.0, f"❌ Acurácia fora do intervalo [0,1]: {accuracy}"
            assert 0.0 <= f1 <= 1.0, f"❌ F1 fora do intervalo [0,1]: {f1}"
            assert 0.0 <= recall <= 1.0, f"❌ Recall fora do intervalo [0,1]: {recall}"
            assert 0.0 <= precision <= 1.0, f"❌ Precision fora do intervalo [0,1]: {precision}"
            
            print(f"   ✅ Acurácia válida: {accuracy:.4f}")
            print(f"   ✅ F1 válido: {f1:.4f}")
            print(f"   ✅ Recall válido: {recall:.4f}")
            print(f"   ✅ Precision válida: {precision:.4f}")
            
            # Verifica consistência das listas retornadas
            assert len(true_y) == len(predicted_y), \
                f"❌ Listas desalinhadas: true_y={len(true_y)} vs predicted_y={len(predicted_y)}"
            assert len(true_y) == len(predicted_labels), \
                f"❌ Quantidade inconsistente: labels={len(true_y)} vs dicionário={len(predicted_labels)}"
            
            print(f"\n   ✅ Listas alinhadas: {len(true_y)} amostras")
            print("   ✅ Usando mesma função compute_metrics() da técnica de relevância")
            
            # Compara com acurácia preliminar calculada na ETAPA 5
            correct = sum(1 for img_id in predicted_labels if predicted_labels[img_id] == true_labels[img_id])
            preliminary_accuracy = correct / len(predicted_labels)
            
            assert abs(accuracy - preliminary_accuracy) < 0.0001, \
                f"❌ Acurácia inconsistente: compute_metrics={accuracy:.4f} vs preliminar={preliminary_accuracy:.4f}"
            
            print(f"   ✅ Consistência de acurácia verificada: {accuracy:.4f} == {preliminary_accuracy:.4f}")
            
            # Validações ETAPA 7
            assert os.path.exists(confusion_matrix_path), \
                f"❌ Arquivo de matriz de confusão não foi criado: {confusion_matrix_path}"
            
            print(f"\n   ✅ Matriz de confusão salva: {confusion_matrix_path}")
            print("   ✅ Usando mesma função show_confusion_matrix() do projeto")
            
            # Validações ETAPA 8
            assert os.path.exists(csv_path), \
                f"❌ Arquivo CSV não foi criado: {csv_path}"
            
            # Lê o CSV gerado para validar o conteúdo
            with open(csv_path, 'r', encoding='utf-8') as f:
                csv_reader = csv.DictReader(f)
                csv_rows = list(csv_reader)
            
            assert len(csv_rows) == len(predicted_labels), \
                f"❌ CSV tem {len(csv_rows)} linhas, esperado {len(predicted_labels)}"
            
            # Valida que as colunas esperadas existem
            expected_columns = [
                'nome_imagem', 'label_predito', 'label_real',
                'acuracia_global', 'f1_global', 'recall_global', 'precision_global',
                'probabilidades', 'somas_probabilidades'
            ]
            
            actual_columns = csv_rows[0].keys() if csv_rows else []
            for col in expected_columns:
                assert col in actual_columns, f"❌ Coluna '{col}' ausente no CSV"
            
            # Valida uma linha do CSV
            first_row = csv_rows[0]
            assert first_row['acuracia_global'] == str(round(accuracy, 4)), \
                f"❌ Acurácia no CSV inconsistente"
            
            print(f"\n   ✅ CSV exportado: {csv_path}")
            print(f"   ✅ {len(csv_rows)} linhas escritas")
            print(f"   ✅ Todas as colunas esperadas presentes")
            print(f"   ✅ Formato consistente com CSV de entrada")
            
            print("=" * 60)
            print()
            print("✅ TODAS AS ETAPAS (1, 2, 3, 4, 5, 6, 7, 8) CONCLUÍDAS COM SUCESSO!")
            
        except Exception as e:
            print(f"❌ ERRO NO TESTE: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print(f"⚠️  Arquivo de teste não encontrado: {test_csv}")
        print("   Execute o script com: python sum_of_probabilities.py <input_csv>")

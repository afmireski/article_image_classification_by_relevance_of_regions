import importlib
import numpy as np
import random

from typing import Dict, List, Tuple

import mytypes as mtp
importlib.reload(mtp)

from mytypes import (
    SpecialistSet,
    PreparedSetsForClassification,
    FoldData,
    FoldDataFull,
    ClassificationData,
    ClassificationDataFull,
    ClassificationDataset,
    MulticlassClassificationDataset,
    PreparedMulticlassSetsForClassification
)


def merge_categories_dicts(
    categories: List[str], textures_dict: Dict[str, Dict[str, np.ndarray]]
) -> Tuple[Dict[str, np.ndarray], List[str], Dict[str, int]]:
    merged_dict = {}
    labels = []
    labels_dict = {}
    for category_idx, category in enumerate(categories):
        category_dict = textures_dict[category]
        n_images = 0
        for img, features in category_dict.items():
            merged_dict[img] = features
            labels.append(category)
            labels_dict[img] = category_idx
            n_images += 1
        print(f"Categoria: {category}, Número de elementos: {n_images}")

    return (merged_dict, labels, labels_dict)


def combine_sets(sets: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
    """
    Combina conjuntos de features em todas as combinações possíveis.

    Args:
        sets: Lista de dicionários {image_name: features_array}
        labels: Lista de nomes dos conjuntos (ex: ["LBP", "GLCM", "LPQ"])

    Returns:
        Lista com todos os conjuntos originais + todas as combinações possíveis

    Example:
        Para 3 conjuntos [LBP, GLCM, LPQ], retorna 7 conjuntos:
        [LBP, GLCM, LPQ, LBP+GLCM, LBP+LPQ, GLCM+LPQ, LBP+GLCM+LPQ]
    """
    from itertools import combinations

    output_sets = []
    n_sets = len(sets)

    # Gera todas as combinações possíveis (de 1 até n elementos)
    for r in range(1, n_sets + 1):
        for combo_indices in combinations(range(n_sets), r):
            # Combina os conjuntos selecionados
            combined_dict = {}

            # Para cada imagem, combina as features dos conjuntos selecionados
            image_names = sets[
                0
            ].keys()  # Assume que todos os conjuntos têm as mesmas imagens

            for img_name in image_names:
                combined_features = []

                # Concatena features dos conjuntos selecionados
                for idx in combo_indices:
                    features = sets[idx][img_name]
                    combined_features.append(features)

                # Concatena todas as features
                if len(combined_features) == 1:
                    # Apenas um conjunto, não precisa concatenar
                    combined_dict[img_name] = combined_features[0]
                else:
                    # Múltiplos conjuntos - concatena ao longo do eixo das features (axis=1)
                    # Mantém a estrutura de segmentos: (n_segments, n_features_combined)
                    combined_dict[img_name] = np.concatenate(combined_features, axis=1)

            output_sets.append(combined_dict)

    return output_sets


def combine_sets_full(sets: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
    """
    Combina conjuntos de features de imagens completas (não-segmentadas) em todas as combinações possíveis.
    
    Diferente de combine_sets, esta função trabalha com features 1D (imagens completas),
    onde cada imagem tem um único vetor de features ao invés de múltiplos segmentos.

    Args:
        sets: Lista de dicionários {image_name: features_array_1D}

    Returns:
        Lista com todos os conjuntos originais + todas as combinações possíveis

    Example:
        Para 3 conjuntos [LBP, GLCM, LPQ], retorna 7 conjuntos:
        [LBP, GLCM, LPQ, LBP+GLCM, LBP+LPQ, GLCM+LPQ, LBP+GLCM+LPQ]
    """
    from itertools import combinations

    output_sets = []
    n_sets = len(sets)

    # Gera todas as combinações possíveis (de 1 até n elementos)
    for r in range(1, n_sets + 1):
        for combo_indices in combinations(range(n_sets), r):
            # Combina os conjuntos selecionados
            combined_dict = {}

            # Para cada imagem, combina as features dos conjuntos selecionados
            image_names = sets[0].keys()  # Assume que todos os conjuntos têm as mesmas imagens

            for img_name in image_names:
                combined_features = []

                # Concatena features dos conjuntos selecionados
                for idx in combo_indices:
                    features = sets[idx][img_name]
                    combined_features.append(features)

                # Concatena todas as features
                if len(combined_features) == 1:
                    # Apenas um conjunto, não precisa concatenar
                    combined_dict[img_name] = combined_features[0]
                else:
                    # Múltiplos conjuntos - concatena ao longo do eixo 0 (features são 1D)
                    # Para imagens completas: (n_features,) + (n_features,) = (n_features_combined,)
                    combined_dict[img_name] = np.concatenate(combined_features, axis=0)

            output_sets.append(combined_dict)

    return output_sets


def generate_texture_dicts(
    categories: List[str],
    lbp_dict: Dict[str, Dict[str, np.ndarray]],
    glcm_dict: Dict[str, Dict[str, np.ndarray]],
    lpq_dict: Dict[str, Dict[str, np.ndarray]],
) -> Tuple[List[Dict[str, np.ndarray]], List[str], Dict[str, int]]:
    """
    Gera dicionários de texturas mesclados e suas combinações para imagens segmentadas.

    Args:
        categories: Lista de categorias
        lbp_dict: Dicionário LBP por categoria
        glcm_dict: Dicionário GLCM por categoria
        lpq_dict: Dicionário LPQ por categoria

    Returns:
        Tupla com (lista_de_conjuntos_combinados, labels, true_images_labels)
    """
    (lbp_set, labels, true_images_labels) = merge_categories_dicts(categories, lbp_dict)
    (glcm_set, _, _) = merge_categories_dicts(categories, glcm_dict)
    (lpq_set, _, _) = merge_categories_dicts(categories, lpq_dict)

    # Combina todos os conjuntos
    sets = [lbp_set, glcm_set, lpq_set]

    combined_sets = combine_sets(sets)

    return (combined_sets, labels, true_images_labels)


def generate_texture_dicts_full(
    categories: List[str],
    lbp_dict: Dict[str, Dict[str, np.ndarray]],
    glcm_dict: Dict[str, Dict[str, np.ndarray]],
    lpq_dict: Dict[str, Dict[str, np.ndarray]],
) -> Tuple[List[Dict[str, np.ndarray]], List[str], Dict[str, int]]:
    """
    Gera dicionários de texturas mesclados e suas combinações para imagens completas (não-segmentadas).
    
    Diferente de generate_texture_dicts, esta função trabalha com features 1D
    (imagens completas sem segmentação).

    Args:
        categories: Lista de categorias
        lbp_dict: Dicionário LBP por categoria {category: {img: features_1D}}
        glcm_dict: Dicionário GLCM por categoria {category: {img: features_1D}}
        lpq_dict: Dicionário LPQ por categoria {category: {img: features_1D}}

    Returns:
        Tupla com (lista_de_conjuntos_combinados, labels, true_images_labels)
    """
    (lbp_set, labels, true_images_labels) = merge_categories_dicts(categories, lbp_dict)
    (glcm_set, _, _) = merge_categories_dicts(categories, glcm_dict)
    (lpq_set, _, _) = merge_categories_dicts(categories, lpq_dict)

    # Combina todos os conjuntos usando a versão específica para imagens completas
    sets = [lbp_set, glcm_set, lpq_set]

    combined_sets = combine_sets_full(sets)

    return (combined_sets, labels, true_images_labels)


def show_features_summary(
    combined_sets: List[Dict[str, np.ndarray]], labels: List[str]
):
    """
    Exibe um resumo das features dos conjuntos combinados.

    Args:
        combined_sets: Lista de dicionários com conjuntos combinados
        labels: Lista de labels das imagens
    """
    print("=" * 50)
    print("RESUMO DOS CONJUNTOS DE FEATURES")
    print("=" * 50)

    # Informações gerais
    total_images = len(labels)
    n_sets = len(combined_sets)

    print(f"📊 Total de imagens: {total_images}")
    print(f"🔧 Total de conjuntos gerados: {n_sets}")
    print(f"📋 Labels únicos: {set(labels)}")
    print()

    # Nomes dos conjuntos baseados na combinação
    set_names = [
        "LBP",
        "GLCM",
        "LPQ",
        "LBP+GLCM",
        "LBP+LPQ",
        "GLCM+LPQ",
        "LBP+GLCM+LPQ",
    ]

    # Para cada conjunto combinado
    for i, feature_set in enumerate(combined_sets):
        set_name = set_names[i] if i < len(set_names) else f"Conjunto_{i+1}"

        # Pega a primeira imagem para analisar a estrutura
        first_image_name = next(iter(feature_set.keys()))
        first_features = feature_set[first_image_name]

        n_images_in_set = len(feature_set)

        if len(first_features.shape) == 1:
            # Features simples (1D)
            n_features = first_features.shape[0]
            print(f"🔹 {set_name}:")
            print(f"   📁 Imagens: {n_images_in_set}")
            print(f"   🎯 Features por imagem: {n_features}")
        else:
            # Features segmentadas (2D)
            n_segments, n_features = first_features.shape
            total_features = n_segments * n_features
            total_segments = n_images_in_set * n_segments
            print(f"🔹 {set_name}:")
            print(f"   📁 Imagens: {n_images_in_set}")
            print(f"   🧩 Segmentos por imagem: {n_segments}")
            print(f"   🧱 Total de segmentos do conjunto: {total_segments}")
            print(f"   🎯 Features por segmento: {n_features}")
            print(f"   📈 Total features por imagem: {total_features}")

        print()

    print("=" * 50)


def split_data_in_folds(
    data: SpecialistSet,
    k_folds=5,
    random_state=42,
) -> List[FoldData]:
    """
    Constrói dados de treino com k-folds a partir do conjunto de especialistas.

    Args:
        data: Tupla com (class_features, no_class_features, true_map)
        k_folds: Número de folds para validação cruzada
        train_factor: Fator de treino (não usado nesta implementação, mantido para compatibilidade)
        random_state: Seed para reprodutibilidade

    Returns:
        Dados organizados em variáveis separadas para posterior organização
    """

    (
        (class_features, len_class_features),
        (no_class_features, len_no_class_features),
        true_map,
    ) = data

    # Configurar seed para reprodutibilidade
    random.seed(random_state)
    np.random.seed(random_state)

    # 1. Separar chaves por classe e embaralhar
    class_images = list(class_features.keys())
    no_class_images = list(no_class_features.keys())

    # Embaralhar cada classe separadamente
    random.shuffle(class_images)
    random.shuffle(no_class_images)

    print(f"📊 Total imagens classe: {len_class_features}")
    print(f"📊 Total imagens não-classe: {len_no_class_features}")
    print(f"🔄 Dividindo em {k_folds} folds...")

    # 2. Dividir cada classe em k_folds partes aproximadamente iguais
    def divide_in_k_parts(items, k):
        """Divide uma lista em k partes aproximadamente iguais"""
        n = len(items)
        base_size = n // k
        rest = n % k

        parts = []
        begin = 0

        for i in range(k):
            # Distribui o resto nas primeiras partições
            part_size = base_size + (1 if i < rest else 0)
            end = begin + part_size
            parts.append(items[begin:end])
            begin = end

        return parts

    class_folds = divide_in_k_parts(class_images, k_folds)
    no_class_folds = divide_in_k_parts(no_class_images, k_folds)

    # Verificar distribuição
    for i in range(k_folds):
        print(
            f"  Fold {i}: {len(class_folds[i])} classe, {len(no_class_folds[i])} não-classe"
        )

    # 3. Construir cada fold
    folds_data = []

    # print("=" * 20)
    # print(class_images)

    for k in range(k_folds):
        print(f"\n🔧 Construindo fold {k}...")

        # Teste = parte k de cada classe
        test_class_images = class_folds[k]
        test_no_class_images = no_class_folds[k]

        # Treino = todas as outras partes
        train_class_images = []
        train_no_class_images = []

        for i in range(k_folds):
            if i != k:  # Excluir a parte usada para teste
                train_class_images.extend(class_folds[i])
                train_no_class_images.extend(no_class_folds[i])

        # 4. Construir dicionários para este fold
        # Dados de treino
        train_class_dict = {img: class_features[img] for img in train_class_images}
        train_no_class_dict = {
            img: no_class_features[img] for img in train_no_class_images
        }
        train_true_map = {}

        # Adicionar ao mapeamento verdadeiro
        for img in train_class_images:
            train_true_map[img] = true_map[img]
        for img in train_no_class_images:
            train_true_map[img] = true_map[img]

        # Dados de teste
        test_class_dict = {img: class_features[img] for img in test_class_images}
        test_no_class_dict = {
            img: no_class_features[img] for img in test_no_class_images
        }
        test_true_map = {}

        # Adicionar ao mapeamento verdadeiro
        for img in test_class_images:
            test_true_map[img] = true_map[img]
        for img in test_no_class_images:
            test_true_map[img] = true_map[img]

        # print("+" * 20)
        # print(train_class_dict.keys())
        # print("-" * 20)
        # print(test_class_dict.keys())
        # print("+" * 20)

        # Armazenar dados do fold em variáveis separadas
        fold_data = {
            "fold_id": k,
            "train_class_features": train_class_dict,
            "train_no_class_features": train_no_class_dict,
            "train_true_map": train_true_map,
            "test_class_features": test_class_dict,
            "test_no_class_features": test_no_class_dict,
            "test_true_map": test_true_map,
            "train_class_count": len(train_class_images),
            "train_no_class_count": len(train_no_class_images),
            "test_class_count": len(test_class_images),
            "test_no_class_count": len(test_no_class_images),
            "train_total": len(train_class_images) + len(train_no_class_images),
            "test_total": len(test_class_images) + len(test_no_class_images),
        }

        folds_data.append(fold_data)

        print(
            f"  ✅ Treino: {len(train_class_images)} classe + {len(train_no_class_images)} não-classe = {fold_data['train_total']}"
        )
        print(
            f"  ✅ Teste: {len(test_class_images)} classe + {len(test_no_class_images)} não-classe = {fold_data['test_total']}"
        )

    print(f"\n🎉 {k_folds} folds construídos com sucesso!")
    # print("=" * 20)

    # Retornar dados em variáveis separadas para fase 2
    return folds_data


def _extract_features_and_labels(
    class_features: Dict[str, np.ndarray],
    no_class_features: Dict[str, np.ndarray],
    true_map: Dict[str, int],
) -> ClassificationData:
    """
    Extrai features e labels de dicionários de classe e não-classe.

    Args:
        class_features: Dicionário com features da classe
        no_class_features: Dicionário com features não-classe
        piece_map: Diz a qual imagem cada pedaço pertence

    Returns:
        ClassificationData com X, y e features_map
    """
    # Combinar todos os dicionários de features
    all_features = {**class_features, **no_class_features}

    # Listas para acumular features e labels
    X_list = []
    y_list = []
    features_map = {}

    current_position = 0

    # Processar cada imagem
    for img_name, features in all_features.items():
        # Obter o label verdadeiro da imagem
        label = true_map[img_name]

        # Features pode ser 1D (features simples) ou 2D (features segmentadas)
        if len(features.shape) == 1:
            # Features simples - uma linha por imagem
            n_segments = 1
            features_2d = features.reshape(1, -1)
        else:
            # Features segmentadas - múltiplas linhas por imagem
            n_segments = features.shape[0]
            features_2d = features

        # Adicionar features ao array X
        X_list.append(features_2d)

        # Adicionar labels para todos os segmentos desta imagem
        y_list.extend([label] * n_segments)

        # Registrar posições no mapa
        end_pos = current_position + n_segments

        for i in range(current_position, end_pos):
            # Registra para o pedaço, a qual imagem ele pertence.
            features_map[i] = img_name

        current_position = end_pos

    # Concatenar todas as features
    X = np.vstack(X_list)
    y = np.array(y_list)

    return (X, y, features_map)

def _extract_features_and_labels_full(
    feats: Dict[str, np.ndarray],
    true_map: Dict[str, int],
) -> ClassificationDataFull:
    """
    Extrai features e labels de dicionários para classificação multiclasse.
    
    Diferente de _extract_features_and_labels, esta função trabalha diretamente
    com todas as classes juntas (não há separação classe/não-classe como nos especialistas).

    Args:
        features: Dicionário {image_name: features_array} com features de todas as imagens
        true_map: Dicionário {image_name: label_index} com o índice da classe de cada imagem

    Returns:
        ClassificationDataFull: Tupla (X, y, features_map) onde:
            - X: np.ndarray com features empilhadas
            - y: np.ndarray com labels correspondentes
    """

    # Listas para acumular features e labels
    X_list = []
    y_list = []
    images = []

    # Processar cada imagem
    for img_name, feats in feats.items():
        # Obter o label verdadeiro da imagem
        label = true_map[img_name]

        # Features pode ser 1D (features simples) ou 2D (features segmentadas)
        

        # Adicionar features ao array X
        X_list.append(feats)

        # Adicionar labels para todos os segmentos desta imagem
        y_list.append(label)

        # Lista de imagens no conjunto
        images.append(img_name)

    # Concatenar todas as features
    X = np.vstack(X_list)
    y = np.array(y_list)

    return (X, y, images)


def build_classification_data(
    folded_data: List[FoldData],
) -> ClassificationDataset:
    """
    Converte dados de folds em formato adequado para classificação.

    Para cada fold:
    1. Extrai features de cada imagem para um array sequencial (X)
    2. Em paralelo, popula um array de rótulos (y) com a classe da imagem
    3. Salva em um dicionário o mapa de posições de cada imagem no array
    4. Faz isso para treino e teste

    Args:
        folded_data: Lista de folds com dados brutos

    Returns:
        Lista de folds processados com dados formatados para classificação
    """
    processed_folds = []

    print("🔄 Convertendo folds para formato de classificação...")

    for fold_data in folded_data:
        fold_id = fold_data["fold_id"]
        print(f"\n📂 Processando fold {fold_id}...")

        # Processar dados de treino
        print("  🏋️ Processando dados de treino...")
        train_data = _extract_features_and_labels(
            fold_data["train_class_features"],
            fold_data["train_no_class_features"],
            fold_data["train_true_map"],
        )

        # Processar dados de teste
        print("  🧪 Processando dados de teste...")
        test_data = _extract_features_and_labels(
            fold_data["test_class_features"],
            fold_data["test_no_class_features"],
            fold_data["test_true_map"],
        )

        # Criar fold processado
        processed_fold = (train_data, test_data)

        processed_folds.append(processed_fold)

        # Log informativo
        train_X_shape = train_data[0].shape
        test_X_shape = test_data[0].shape
        train_images = len(train_data[2])
        test_images = len(test_data[2])

        print(
            f"  ✅ Treino: {train_images} imagens → X{train_X_shape}, y{train_data[1].shape}"
        )
        print(
            f"  ✅ Teste: {test_images} imagens → X{test_X_shape}, y{test_data[1].shape}"
        )

    print(f"\n🎉 {len(processed_folds)} folds processados com sucesso!")

    return processed_folds


def build_classification_data_full(
    folded_data: List[FoldDataFull],
    verbose=True,
) -> MulticlassClassificationDataset:
    """
    Converte dados de folds de classificação multiclasse para formato adequado para treinamento.
    
    Diferente de build_classification_data, esta função trabalha com FoldDataFull 
    (classificação multiclasse normal, não-especialistas).
    
    Para cada fold:
    1. Extrai features de cada imagem para um array sequencial (X)
    2. Em paralelo, popula um array de rótulos (y) com a classe da imagem
    3. Salva em um dicionário o mapa de posições de cada imagem no array
    4. Faz isso para treino e teste

    Args:
        folded_data: Lista de folds com dados brutos (FoldDataFull)
        verbose: Se True, exibe mensagens de progresso

    Returns:
        Lista de folds processados com dados formatados para classificação
    """
    processed_folds = []

    if verbose:
        print("🔄 Convertendo folds para formato de classificação...")

    for fold_data in folded_data:
        fold_id = fold_data["fold_id"]
        
        if verbose:
            print(f"\n📂 Processando fold {fold_id}...")

        # Processar dados de treino
        if verbose:
            print("  🏋️ Processando dados de treino...")
        
        # Para classificação multiclasse, todas as features estão em train_features
        train_data = _extract_features_and_labels_full(
            fold_data["train_features"],  # Todas as classes juntas
            fold_data["train_true_map"],
        )

        # Processar dados de teste
        if verbose:
            print("  🧪 Processando dados de teste...")
        
        test_data = _extract_features_and_labels_full(
            fold_data["test_features"],  # Todas as classes juntas
            fold_data["test_true_map"],
        )

        # Criar fold processado
        processed_fold = (train_data, test_data)

        processed_folds.append(processed_fold)

        # Log informativo
        if verbose:
            train_X_shape = train_data[0].shape
            test_X_shape = test_data[0].shape
            train_images = len(train_data[0])
            test_images = len(test_data[0])

            print(
                f"  ✅ Treino: {train_images} imagens → X{train_X_shape}, y{train_data[1].shape}"
            )
            print(
                f"  ✅ Teste: {test_images} imagens → X{test_X_shape}, y{test_data[1].shape}"
            )

    if verbose:
        print(f"\n🎉 {len(processed_folds)} folds processados com sucesso!")

    return processed_folds


def prepare_sets_for_classification(
    sets: List[SpecialistSet],
    k_folds=5,
    random_state=42,
) -> PreparedSetsForClassification:
    """
    Prepara os conjuntos de dados para classificação, dividindo-os em folds e extraindo as características.

    Args:
        sets: Lista de conjuntos de dados a serem preparados.
        k_folds: Número de folds para a validação cruzada.
        random_state: Semente para reprodutibilidade.

    Returns:
        Lista de conjuntos de dados preparados para classificação.
    """

    data = []
    for dataset in sets:
        folds = split_data_in_folds(dataset, k_folds=k_folds, random_state=random_state)
        classification_data = build_classification_data(folds)
        data.append(classification_data)

    return data


def prepare_full_image_sets_for_classification(
    sets: List[Tuple[Dict[str, np.ndarray], Dict[str, int]]],
    k_folds=5,
    random_state=42,
    verbose=True,
) -> PreparedMulticlassSetsForClassification:
    """
    Prepara múltiplos conjuntos de dados de imagens completas para classificação multiclasse.
    
    Esta função garante que todos os conjuntos usem as mesmas divisões de folds,
    permitindo comparação justa entre diferentes combinações de features (LBP, GLCM, LPQ, etc.).
    
    Args:
        sets: Lista de tuplas (X, y) onde:
            - X: Dict[str, np.ndarray] - mapeamento imagem → features
            - y: Dict[str, int] - mapeamento imagem → label (índice da classe)
        k_folds: Número de folds para validação cruzada
        random_state: Seed para reprodutibilidade
        verbose: Se True, exibe mensagens de progresso
    
    Returns:
        Lista de ClassificationDataset, um para cada conjunto de features
        
    Example:
        >>> sets = [
        ...     (X_lbp_full, true_images_labels),
        ...     (X_glcm_full, true_images_labels),
        ...     (X_lpq_full, true_images_labels),
        ... ]
        >>> prepared = prepare_full_image_sets_for_classification(sets, k_folds=5)
        >>> lbp_folds, glcm_folds, lpq_folds = prepared
    """
    if verbose:
        print("=" * 60)
        print("PREPARANDO CONJUNTOS PARA CLASSIFICAÇÃO MULTICLASSE")
        print("=" * 60)
        print(f"📊 Total de conjuntos de features: {len(sets)}")
        print(f"🔄 K-folds: {k_folds}")
        print(f"🎲 Random state: {random_state}")
        print()
    
    if not sets:
        raise ValueError("A lista de conjuntos não pode estar vazia")
    
    # Usar o primeiro conjunto para criar a divisão base
    # Todos os outros conjuntos usarão as mesmas imagens nas mesmas folds
    first_X, first_y = sets[0]
    
    if verbose:
        print("🔧 Criando divisão base de folds (será reutilizada para todos os conjuntos)...")
    
    # Criar a divisão base em folds
    base_folds = split_full_image_data_in_folds(
        first_X, 
        first_y, 
        k_folds=k_folds, 
        random_state=random_state,
        verbose=verbose
    )
    
    # Extrair apenas os nomes das imagens de cada fold (ignorando features do primeiro conjunto)
    # Isso nos dá a estrutura de divisão que será aplicada a todos os conjuntos
    fold_structure = []
    for fold in base_folds:
        # Como é classificação multiclasse, todas as features estão em train_features
        train_images = list(fold["train_features"].keys())
        test_images = list(fold["test_features"].keys())
        
        fold_structure.append({
            "fold_id": fold["fold_id"],
            "train_images": train_images,
            "test_images": test_images,
            "train_true_map": fold["train_true_map"],
            "test_true_map": fold["test_true_map"],
        })
    
    if verbose:
        print(f"\n✅ Estrutura de folds criada com sucesso!")
        print(f"📋 Cada fold terá:")
        print(f"   - Treino: {len(fold_structure[0]['train_images'])} imagens")
        print(f"   - Teste: {len(fold_structure[0]['test_images'])} imagens")
        print()
    
    # Processar cada conjunto de features usando a mesma estrutura de folds
    prepared_datasets = []
    
    for set_idx, (X_features, y_labels) in enumerate(sets):
        if verbose:
            print(f"{'=' * 60}")
            print(f"📦 Processando conjunto {set_idx + 1}/{len(sets)}")
            print(f"{'=' * 60}")
        
        # Verificar se as imagens são as mesmas
        if set(X_features.keys()) != set(first_X.keys()):
            raise ValueError(
                f"Conjunto {set_idx} tem imagens diferentes do conjunto base. "
                "Todos os conjuntos devem ter exatamente as mesmas imagens."
            )
        
        # Verificar se os labels são os mesmos
        if y_labels != first_y:
            raise ValueError(
                f"Conjunto {set_idx} tem labels diferentes do conjunto base. "
                "Todos os conjuntos devem ter os mesmos labels."
            )
        
        # Reconstruir folds usando a estrutura base mas com as features deste conjunto
        folds_data = []
        
        for fold_info in fold_structure:
            fold_id = fold_info["fold_id"]
            
            if verbose:
                print(f"\n🔧 Construindo fold {fold_id}...")
            
            # Construir dicionários de features para este fold
            train_features = {
                img: X_features[img] 
                for img in fold_info["train_images"]
            }
            test_features = {
                img: X_features[img] 
                for img in fold_info["test_images"]
            }
            
            fold_data = {
                "fold_id": fold_id,
                "train_features": train_features,
                "train_true_map": fold_info["train_true_map"],
                "test_features": test_features,
                "test_true_map": fold_info["test_true_map"],
                "train_total": len(train_features),
                "test_total": len(test_features),
            }
            
            folds_data.append(fold_data)
            
            if verbose:
                print(f"  ✅ Treino: {len(train_features)} imagens")
                print(f"  ✅ Teste: {len(test_features)} imagens")
        
        if verbose:
            print(f"\n🎉 {k_folds} folds construídos para conjunto {set_idx + 1}!")
        
        # Converter folds para formato de classificação usando a função específica
        if verbose:
            print("\n🔄 Convertendo para formato de classificação...")
        
        classification_data = build_classification_data_full(folds_data, verbose=verbose)
        prepared_datasets.append(classification_data)
        
        if verbose:
            print(f"✅ Conjunto {set_idx + 1} preparado com sucesso!")
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"🎊 TODOS OS {len(sets)} CONJUNTOS PREPARADOS COM SUCESSO!")
        print(f"{'=' * 60}\n")
    
    return prepared_datasets


def split_full_image_data_in_folds(
    X: Dict[str, np.ndarray],
    y: Dict[str, int],
    k_folds=5,
    random_state=42,
    verbose=True,
) -> List[FoldDataFull]:
    """
    Divide dados de imagens completas em K folds balanceadas por classe.
    
    Diferente de split_data_in_folds, esta função trabalha com classificação multiclasse
    (não especialistas), mantendo todas as classes juntas em cada fold.
    
    Args:
        X: Dicionário {image_name: features_array} com features de cada imagem
        y: Dicionário {image_name: label_index} com o índice da classe de cada imagem
        k_folds: Número de folds para validação cruzada
        random_state: Seed para reprodutibilidade
        verbose: Se True, imprime informações sobre o processo
        
    Returns:
        Lista de FoldData com dados de treino e teste para cada fold
        
    Example:
        >>> X = {"img1": np.array([...]), "img2": np.array([...]), ...}
        >>> y = {"img1": 0, "img2": 1, "img3": 0, ...}
        >>> folds = split_full_image_data_in_folds(X, y, k_folds=5)
    """
    
    # Configurar seed para reprodutibilidade
    random.seed(random_state)
    np.random.seed(random_state)
    
    # Validar que X e y têm as mesmas chaves
    assert set(X.keys()) == set(y.keys()), "X e y devem ter as mesmas chaves (imagens)"
    
    # 1. Agrupar imagens por classe
    images_by_class = {}
    for img_name, label in y.items():
        if label not in images_by_class:
            images_by_class[label] = []
        images_by_class[label].append(img_name)
    
    # Embaralhar cada classe separadamente
    for label in images_by_class:
        random.shuffle(images_by_class[label])
    
    num_classes = len(images_by_class)
    total_images = len(X)
    
    if verbose:
        print(f"📊 Total de imagens: {total_images}")
        print(f"📊 Total de classes: {num_classes}")
        print(f"🔄 Dividindo em {k_folds} folds...")
        for label, images in images_by_class.items():
            print(f"  Classe {label}: {len(images)} imagens")
    
    # 2. Dividir cada classe em k_folds partes aproximadamente iguais
    def divide_in_k_parts(items, k):
        """Divide uma lista em k partes aproximadamente iguais"""
        n = len(items)
        base_size = n // k
        rest = n % k
        
        parts = []
        begin = 0
        
        for i in range(k):
            # Distribui o resto nas primeiras partições
            part_size = base_size + (1 if i < rest else 0)
            end = begin + part_size
            parts.append(items[begin:end])
            begin = end
        
        return parts
    
    # Dividir cada classe em folds
    class_folds = {}
    for label, images in images_by_class.items():
        class_folds[label] = divide_in_k_parts(images, k_folds)
    
    # Verificar distribuição
    if verbose:
        for i in range(k_folds):
            fold_distribution = [len(class_folds[label][i]) for label in sorted(images_by_class.keys())]
            print(f"  Fold {i}: {fold_distribution} (total: {sum(fold_distribution)})")
    
    # 3. Construir cada fold
    folds_data = []
    
    for k in range(k_folds):
        if verbose:
            print(f"\n🔧 Construindo fold {k}...")
        
        # Teste = parte k de cada classe
        test_images = []
        for label in images_by_class:
            test_images.extend(class_folds[label][k])
        
        # Treino = todas as outras partes
        train_images = []
        for i in range(k_folds):
            if i != k:  # Excluir a parte usada para teste
                for label in images_by_class:
                    train_images.extend(class_folds[label][i])
        
        # 4. Construir dicionários para este fold
        # Dados de treino
        train_features = {img: X[img] for img in train_images}
        train_true_map = {img: y[img] for img in train_images}
        
        # Dados de teste
        test_features = {img: X[img] for img in test_images}
        test_true_map = {img: y[img] for img in test_images}
        
        # Armazenar dados do fold
        fold_data = {
            "fold_id": k,
            "train_features": train_features,
            "train_true_map": train_true_map,
            "test_features": test_features,
            "test_true_map": test_true_map,
            "train_count": len(train_images),
            "test_count": len(test_images),
            "train_total": len(train_images),
            "test_total": len(test_images),
        }
        
        folds_data.append(fold_data)
        
        if verbose:
            print(f"  ✅ Treino: {len(train_images)} imagens")
            print(f"  ✅ Teste: {len(test_images)} imagens")
    
    if verbose:
        print(f"\n🎉 {k_folds} folds construídos com sucesso!")
    
    return folds_data

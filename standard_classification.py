import numpy as np
import os
from dotenv import load_dotenv

# Import the image segmentation tools
import warnings
import time

import sys
import io


from tools.image_tools import (
    load_train_images_dict,
)

from lbp import (
    compute_lbp_for_each_category,
)
from glcm import (
    parallel_calculate_glcm_for_each_category,
)
from lpq import (
    extract_lpq_features_for_each_category,
)

from tools.data import (
    generate_texture_dicts_full,
    prepare_full_image_sets_for_classification,
)

# Importa as bibliotecas dos Classificadores
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Bibliotecas úteis para treinamento dos modelos
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tools.relevance import (
    zip_and_cleanup_results,
)

from classical import timed_standard_technique

# ==============================================================================
# SOLUÇÃO PARA PYTHON 3.13 + JOBLIB RESOURCE_TRACKER WARNINGS
# ==============================================================================
# Python 3.13 tem incompatibilidade conhecida com joblib (usado pelo GridSearchCV)
# Esses warnings não afetam a execução, apenas poluem o terminal
# Referência: https://github.com/joblib/joblib/issues/1621

# Configuração 1: Variáveis de ambiente para o joblib
os.environ['PYTHONWARNINGS'] = 'ignore::ResourceWarning'
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Limita workers paralelos

# Configuração 2: Força método 'spawn' para multiprocessing
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Já foi configurado anteriormente

# Configuração 3: Suprime warnings do resource_tracker
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
warnings.filterwarnings("ignore", message=".*resource_tracker.*")
warnings.filterwarnings("ignore", message=".*Cannot register.*")

# Configuração 4: Redireciona stderr do resource_tracker (mais agressivo)

# Cria um filtro para stderr que remove linhas de resource_tracker
class ResourceTrackerFilter(io.TextIOBase):
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.buffer = ""
    
    def write(self, text):
        # Filtra mensagens do resource_tracker
        if 'resource_tracker' not in text.lower() and 'cannot register' not in text.lower() and 'semlock' not in text.lower() and 'joblib_memmapping' not in text.lower():
            self.original_stderr.write(text)
        return len(text)
    
    def flush(self):
        self.original_stderr.flush()

# Aplica o filtro ao stderr
sys.stderr = ResourceTrackerFilter(sys.stderr)

# ==============================================================================

# Load environment variables from .env file
load_dotenv(".env", override=True)

# Marca o tempo de início do script (para medir tempo total de execução)
start_time = time.perf_counter()

# Suprime warnings do JobLib relacionados ao ResourceTracker
# Evita poluição do output com mensagens irrelevantes
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
warnings.filterwarnings("ignore", message=".*resource_tracker.*")
warnings.filterwarnings("ignore", message=".*Cannot register.*")


# FUNÇÃO LEGACY - Não utilizada na classificação padrão
# def timed_relevance_technique(
#     base_model,
#     specialist_sets,
#     class_names,
#     model_name,
#     k_folds,
#     true_labels,
# ):
#     """Wrapper around relevance_technique that measures and prints execution time.
#
#     Returns the same value as relevance_technique.
#     """
#     start = time.perf_counter()
#     results = relevance_technique(
#         base_model=base_model,
#         specialist_sets=specialist_sets,
#         class_names=class_names,
#         model_name=model_name,
#         k_folds=k_folds,
#         true_labels=true_labels,
#     )
#     end = time.perf_counter()
#     elapsed = end - start
#     # Format elapsed nicely
#     if elapsed < 60:
#         elapsed_str = f"{elapsed:.2f} s"
#     else:
#         mins = int(elapsed // 60)
#         secs = elapsed % 60
#         elapsed_str = f"{mins}m {secs:.2f}s"
#
#     print(f"⏱️ Tempo de execução ({model_name}): {elapsed_str}")
#
#     return results


images_directory = "./images/pieces"
image_categories = ["dogs", "cats", "lions", "horses"]
n_examples = 4
NEEDS_RESIZE = int(os.getenv("NEEDS_RESIZE", True))
RESIZE_SIZE = int(os.getenv("RESIZE_SIZE", 512))
ACCEPTED_IMAGES_EXTENSIONS = (".jpg", ".jpeg", ".png")

train_images_by_categories = load_train_images_dict(
    images_directory,
    image_categories,
    ACCEPTED_IMAGES_EXTENSIONS,
    NEEDS_RESIZE,
)

# EXTRACAO DE FEATURES
features_full = []

## Extrair Features LBP para todas as imagens segmentadas


radius = 2
n_pixels = 8

lbps_by_categories_full = compute_lbp_for_each_category(
    image_categories, train_images_by_categories, radius, n_pixels
)
features_full.append(lbps_by_categories_full)

## Extrair Features GLCM para todas as imagens segmentadas
GLCM_DISTANCES = [1, 3, 5]
GLCM_ANGLES = np.deg2rad([0, 90, 180, 270])
GLCM_LEVELS = None
GLCM_FEATURES = [
    "contrast",
    "dissimilarity",
    "homogeneity",
    "ASM",
    "energy",
    "correlation",
]

glcms_by_category_full = parallel_calculate_glcm_for_each_category(
    image_categories,
    train_images_by_categories,
    GLCM_DISTANCES,
    GLCM_ANGLES,
    GLCM_FEATURES,
    GLCM_LEVELS,
)

features_full.append(glcms_by_category_full)

## Extrair Features LPQ para todas as imagens segmentadas
lpqs_dict_full = extract_lpq_features_for_each_category(image_categories, train_images_by_categories)

features_full.append(lpqs_dict_full)

# Matriz de características

(sets_full, labels, true_images_labels) = generate_texture_dicts_full(
    image_categories, features_full[0], features_full[1], features_full[2]
)

# Limpeza de variáveis desnecessárias após o cálculo das features
# Uma vez que as features estão calculadas e salvas, podemos liberar memória
# removendo variáveis intermediárias que não serão mais utilizadas

import gc  # Garbage collector

print("Limpando variáveis desnecessárias...")
print(f"Variáveis antes da limpeza: {len(locals())} variáveis locais")

# Variáveis de dados intermediários que podem ser removidas
variables_to_clean = [
    # Dados intermediários de features por categoria (já consolidados)
    "lbps_by_categories_full",
    "glcms_by_category_full",
    "lpq_features_dict_full",
    # Imagens de teste usadas para demonstração
    "test_images",
    # Variáveis temporárias do processo de merge (se existirem)
    "merge_lbp",
    "merge_glcm",
    "merge_lpq",
    "merge_labels",
    "data",
    # Arrays de features individuais antigos (estrutura mudou)
    "X_lbp",
    "X_glcm",
    "X_lpq",
    # Conjuntos antigos (estrutura mudou para combined_sets)
    "set_lbp",
    "set_glcm",
    "set_lpq",
    "set_lbp_glcm",
    "set_lbp_lpq",
    "set_glcm_lpq",
    "set_lbp_glcm_lpq",
    # Outras variáveis temporárias
    "auto_k",
    "auto_segmented",
    "actual_regions",
    "height",
    "width",
    "i",
    "image",
    # Variáveis de configuração temporárias
    "radius",
    "n_pixels",
    "n_examples",
]

# Conta quantas variáveis foram realmente removidas
removed_count = 0
for var_name in variables_to_clean:
    if var_name in locals():
        del locals()[var_name]
        removed_count += 1
        print(f"  ✓ Removida: {var_name}")

# Remove também variáveis globais se existirem
for var_name in variables_to_clean:
    if var_name in globals():
        del globals()[var_name]

# Força a coleta de lixo para liberar memória imediatamente
gc.collect()

print(f"\n✓ Limpeza concluída!")
print(f"  - {removed_count} variáveis removidas")
print(f"  - Memória liberada pela coleta de lixo")
print(f"  - Variáveis após limpeza: {len(locals())} variáveis locais")

# Mostra as principais variáveis que permanecem (estrutura atualizada)
essential_vars = [
    "image_categories",
    "train_images_by_categories",
    "combined_sets",
    "labels",
    "all_features_stored",
    "GLCM_DISTANCES",
    "GLCM_ANGLES",
    "GLCM_FEATURES",
]

print("\n📋 Principais variáveis mantidas:")
for var in essential_vars:
    if var in locals():
        if hasattr(locals()[var], "shape"):
            print(
                f"  ✓ {var}: {type(locals()[var]).__name__} {getattr(locals()[var], 'shape', '')}"
            )
        elif isinstance(locals()[var], (list, dict)):
            print(
                f"  ✓ {var}: {type(locals()[var]).__name__} (tamanho: {len(locals()[var])})"
            )
        else:
            print(f"  ✓ {var}: {type(locals()[var]).__name__}")

print(
    "\n💡 Variáveis limpas com sucesso! Agora você pode prosseguir com o treinamento dos modelos."
)
print(
    "🔄 Nova estrutura: combined_sets contém todos os conjuntos de features combinados."
)

# Divisisão do conjunto de dados

## Número de folds para a validação cruzada
K_FOLDS = 5
## Extraí os conjuntos e os labels da estrutura de dados:
[
    X_lbp_full,
    X_glcm_full,
    X_lpq_full,
    X_lbp_glcm_full,
    X_lbp_lpq_full,
    X_glcm_lpq_full,
    X_lbp_glcm_lpq_full,
] = sets_full

set_lbp_full = (X_lbp_full, true_images_labels)
set_glcm_full = (X_glcm_full, true_images_labels)
set_lpq_full = (X_lpq_full, true_images_labels)
set_lbp_glcm_full = (X_lbp_glcm_full, true_images_labels)
set_lbp_lpq_full = (X_lbp_lpq_full, true_images_labels)
set_glcm_lpq_full = (X_glcm_lpq_full, true_images_labels)
set_lbp_glcm_lpq_full = (X_lbp_glcm_lpq_full, true_images_labels)

print(f"Classes: {image_categories}")

# Preparar conjuntos de imagens completas para classificação multiclasse
print("\n" + "=" * 60)
print("PREPARANDO CONJUNTOS DE IMAGENS COMPLETAS")
print("=" * 60)

full_image_sets = [
    set_lbp_full,
    set_glcm_full,
    set_lpq_full,
    set_lbp_glcm_full,
    set_lbp_lpq_full,
    set_glcm_lpq_full,
    set_lbp_glcm_lpq_full,
]

# Preparar todos os conjuntos de imagens completas de uma vez
# Isso garante que todos usem as mesmas divisões de folds
prepared_full_image_sets = prepare_full_image_sets_for_classification(
    full_image_sets,
    k_folds=K_FOLDS,
    random_state=42,
    verbose=True,
)

# Desempacotar os conjuntos preparados
[
    final_lbp_full_folds,
    final_glcm_full_folds,
    final_lpq_full_folds,
    final_lbp_glcm_full_folds,
    final_lbp_lpq_full_folds,
    final_glcm_lpq_full_folds,
    final_lbp_glcm_lpq_full_folds,
] = prepared_full_image_sets

print("\n✅ Conjuntos de imagens completas preparados com sucesso!")
print(f"   - LBP: {len(final_lbp_full_folds)} folds")
print(f"   - GLCM: {len(final_glcm_full_folds)} folds")
print(f"   - LPQ: {len(final_lpq_full_folds)} folds")
print(f"   - LBP+GLCM: {len(final_lbp_glcm_full_folds)} folds")
print(f"   - LBP+LPQ: {len(final_lbp_lpq_full_folds)} folds")
print(f"   - GLCM+LPQ: {len(final_glcm_lpq_full_folds)} folds")
print(f"   - LBP+GLCM+LPQ: {len(final_lbp_glcm_lpq_full_folds)} folds")
print("=" * 60 + "\n")


# Definição dos modelos
## KNN
K_ARRAY = [1, 3, 5, 7, 9]
DISTANCE_METRICS = ["euclidean", "manhattan", "minkowski"]


def tune_knn(k_array, distance_metrics):

    knn = KNeighborsClassifier()

    # Lista com as combinações de hiperparâmetros
    parameters = {"knn__n_neighbors": k_array, "knn__metric": distance_metrics}

    pipe = Pipeline([("scaler", StandardScaler()), ("knn", knn)])

    model = GridSearchCV(pipe, parameters, n_jobs=-1)

    return model


print("Tunando parâmetros do KNN...")
base_knn = tune_knn(K_ARRAY, DISTANCE_METRICS)

## Random Forest

N_ESTIMATORS = [100, 200]  # [100, 150, 200, 400]
MIN_SAMPLES_LEAFS = [3, 10, 15, 30]
MIN_SAMPLES_SPLITS = [2, 4, 16, 40]
MAX_DEPTHS = [None, 3, 8]  # [None, 2, 3, 4, 5, 8, 12]
MAX_FEATURES = ["sqrt"]  # ["sqrt", "log2", 0, 2, 0.5, None]
MAX_SAMPLES = [None]  # [None, 0.6, 0.8]
CRITERIONS = ["gini", "entropy"]
CLASS_WEIGHT = [None, "balanced_subsample"]  # [None, "balanced_subsample"]


def tune_rf(
    n_estimators,
    min_samples_leaf,
    min_samples_split,
    max_depth,
    max_features,
    max_samples,
    criterions,
    class_weight,
):
    rf = RandomForestClassifier(n_jobs=-1, random_state=42)

    # Lista com as combinações de hiperparâmetros
    parameters = {
        "rf__n_estimators": n_estimators,
        "rf__min_samples_leaf": min_samples_leaf,
        "rf__min_samples_split": min_samples_split,
        "rf__max_depth": max_depth,
        "rf__max_features": max_features,
        "rf__max_samples": max_samples,
        "rf__criterion": criterions,
        "rf__class_weight": class_weight,
    }

    pipe = Pipeline([("scaler", StandardScaler()), ("rf", rf)])

    model = GridSearchCV(pipe, parameters, n_jobs=-1)

    return model


print("Tunando parâmetros do Random Forest...")
base_rf = tune_rf(
    N_ESTIMATORS,
    MIN_SAMPLES_LEAFS,
    MIN_SAMPLES_SPLITS,
    MAX_DEPTHS,
    MAX_FEATURES,
    MAX_SAMPLES,
    CRITERIONS,
    CLASS_WEIGHT,
)

## SVM
KERNELS = ["rbf"]
C = [0.1, 1, 10, 100, 1000]
GAMMA = [2e-5, 2e-3, 2e-1, "auto", "scale"]


def tune_svm(
    kernels,
    c_array,
    gamma_array,
    # degree_array,
    # coef0_array,
):
    svm = SVC(probability=True, random_state=42)

    # Lista com as combinações de hiperparâmetros
    parameters = {
        "svm__kernel": kernels,
        "svm__C": c_array,
        "svm__gamma": gamma_array,
        # "svm__degree": degree_array,
        # "svm__coef0": coef0_array,
    }

    pipe = Pipeline([("scaler", StandardScaler()), ("svm", svm)])

    model = GridSearchCV(pipe, parameters, n_jobs=-1)

    return model


print("Tunando SVM...")
base_svm = tune_svm(
    KERNELS,
    C,
    GAMMA,
    # DEGREE,
    # COEF0,
)

# Classificação Padrão
## KNN
print("🚀 === CLASSIFICAÇÃO PADRÃO - KNN ===")

class_names = image_categories
base_name = "KNN"

# Diretório do experimento
standard_experiment_dir = "results"

print("\n📊 Classificando com LBP...")
standard_results_knn_lbp = timed_standard_technique(
    base_model=base_knn,
    folded_dataset=final_lbp_full_folds,
    class_names=class_names,
    true_labels=true_images_labels,
    model_name=f"{base_name}_LBP",
    k_folds=K_FOLDS,
    experiment_dir=standard_experiment_dir,
    cmap='Blues'
)

# GLCM
print("\n📊 Classificando com GLCM...")
standard_results_knn_glcm = timed_standard_technique(
    base_model=base_knn,
    folded_dataset=final_glcm_full_folds,
    class_names=class_names,
    true_labels=true_images_labels,
    model_name=f"{base_name}_GLCM",
    k_folds=K_FOLDS,
    experiment_dir=standard_experiment_dir,
    cmap='Blues'
)

# LPQ
print("\n📊 Classificando com LPQ...")
standard_results_knn_lpq = timed_standard_technique(
    base_model=base_knn,
    folded_dataset=final_lpq_full_folds,
    class_names=class_names,
    true_labels=true_images_labels,
    model_name=f"{base_name}_LPQ",
    k_folds=K_FOLDS,
    experiment_dir=standard_experiment_dir,
    cmap='Blues'
)

# LBP+GLCM
print("\n📊 Classificando com LBP+GLCM...")
standard_results_knn_lbp_glcm = timed_standard_technique(
    base_model=base_knn,
    folded_dataset=final_lbp_glcm_full_folds,
    class_names=class_names,
    true_labels=true_images_labels,
    model_name=f"{base_name}_LBP_GLCM",
    k_folds=K_FOLDS,
    experiment_dir=standard_experiment_dir,
    cmap='Blues'
)

# LBP+LPQ
print("\n📊 Classificando com LBP+LPQ...")
standard_results_knn_lbp_lpq = timed_standard_technique(
    base_model=base_knn,
    folded_dataset=final_lbp_lpq_full_folds,
    class_names=class_names,
    true_labels=true_images_labels,
    model_name=f"{base_name}_LBP_LPQ",
    k_folds=K_FOLDS,
    experiment_dir=standard_experiment_dir,
    cmap='Blues'
)

# GLCM+LPQ
print("\n📊 Classificando com GLCM+LPQ...")
standard_results_knn_glcm_lpq = timed_standard_technique(
    base_model=base_knn,
    folded_dataset=final_glcm_lpq_full_folds,
    class_names=class_names,
    true_labels=true_images_labels,
    model_name=f"{base_name}_GLCM_LPQ",
    k_folds=K_FOLDS,
    experiment_dir=standard_experiment_dir,
    cmap='Blues'
)

# LBP+GLCM+LPQ
print("\n📊 Classificando com LBP+GLCM+LPQ...")
standard_results_knn_lbp_glcm_lpq = timed_standard_technique(
    base_model=base_knn,
    folded_dataset=final_lbp_glcm_lpq_full_folds,
    class_names=class_names,
    true_labels=true_images_labels,
    model_name=f"{base_name}_LBP_GLCM_LPQ",
    k_folds=K_FOLDS,
    experiment_dir=standard_experiment_dir,
    cmap='Blues'
)

print("\n✅ Classificação padrão KNN concluída para todos os 7 conjuntos de features")
print("=" * 80)

print("🚀 === CLASSIFICAÇÃO PADRÃO - SVM ===")

base_name = "SVM"

# LBP
print("\n📊 Classificando com LBP...")
standard_results_svm_lbp = timed_standard_technique(
    base_model=base_svm,
    folded_dataset=final_lbp_full_folds,
    class_names=class_names,
    true_labels=true_images_labels,
    model_name=f"{base_name}_LBP",
    k_folds=K_FOLDS,
    experiment_dir=standard_experiment_dir,
    cmap='Reds'
)

# GLCM
print("\n📊 Classificando com GLCM...")
standard_results_svm_glcm = timed_standard_technique(
    base_model=base_svm,
    folded_dataset=final_glcm_full_folds,
    class_names=class_names,
    true_labels=true_images_labels,
    model_name=f"{base_name}_GLCM",
    k_folds=K_FOLDS,
    experiment_dir=standard_experiment_dir,
    cmap='Reds'
)

# LPQ
print("\n📊 Classificando com LPQ...")
standard_results_svm_lpq = timed_standard_technique(
    base_model=base_svm,
    folded_dataset=final_lpq_full_folds,
    class_names=class_names,
    true_labels=true_images_labels,
    model_name=f"{base_name}_LPQ",
    k_folds=K_FOLDS,
    experiment_dir=standard_experiment_dir,
    cmap='Reds'
)

# LBP+GLCM
print("\n📊 Classificando com LBP+GLCM...")
standard_results_svm_lbp_glcm = timed_standard_technique(
    base_model=base_svm,
    folded_dataset=final_lbp_glcm_full_folds,
    class_names=class_names,
    true_labels=true_images_labels,
    model_name=f"{base_name}_LBP_GLCM",
    k_folds=K_FOLDS,
    experiment_dir=standard_experiment_dir,
    cmap='Reds'
)

# LBP+LPQ
print("\n📊 Classificando com LBP+LPQ...")
standard_results_svm_lbp_lpq = timed_standard_technique(
    base_model=base_svm,
    folded_dataset=final_lbp_lpq_full_folds,
    class_names=class_names,
    true_labels=true_images_labels,
    model_name=f"{base_name}_LBP_LPQ",
    k_folds=K_FOLDS,
    experiment_dir=standard_experiment_dir,
    cmap='Reds'
)

# GLCM+LPQ
print("\n📊 Classificando com GLCM+LPQ...")
standard_results_svm_glcm_lpq = timed_standard_technique(
    base_model=base_svm,
    folded_dataset=final_glcm_lpq_full_folds,
    class_names=class_names,
    true_labels=true_images_labels,
    model_name=f"{base_name}_GLCM_LPQ",
    k_folds=K_FOLDS,
    experiment_dir=standard_experiment_dir,
    cmap='Reds'
)

# LBP+GLCM+LPQ
print("\n📊 Classificando com LBP+GLCM+LPQ...")
standard_results_svm_lbp_glcm_lpq = timed_standard_technique(
    base_model=base_svm,
    folded_dataset=final_lbp_glcm_lpq_full_folds,
    class_names=class_names,
    true_labels=true_images_labels,
    model_name=f"{base_name}_LBP_GLCM_LPQ",
    k_folds=K_FOLDS,
    experiment_dir=standard_experiment_dir,
    cmap='Reds'
)

print("\n✅ Classificação padrão SVM concluída para todos os 7 conjuntos de features")
print("=" * 80)

zip_and_cleanup_results(
    results_dir=standard_experiment_dir,
    experiment_dir_name='standards',
    folders_to_zip=["confusion_matrixs", "heatmaps", "csv_exports"],
)

# Tempo total do script
end_time = time.perf_counter()
total_elapsed = end_time - start_time
if total_elapsed < 60:
    total_elapsed_str = f"{total_elapsed:.2f} s"
else:
    mins = int(total_elapsed // 60)
    secs = total_elapsed % 60
    total_elapsed_str = f"{mins}m {secs:.2f}s"

print(f"⏳ Tempo total de execução do script: {total_elapsed_str}")

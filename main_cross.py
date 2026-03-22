import numpy as np
import os
import gc  # Garbage collector
from typing import Callable, Optional

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg", force=True)
from dotenv import load_dotenv

# Import the image segmentation tools
import warnings
import time

from tools.image_tools import (
    segment_images_by_category_auto,
    merge_image_categories_dicts,
    load_train_images_dict,
)

from lbp import (
    compute_lbp_for_segments_by_categories,
)
from glcm import parallel_calculate_glcm_for_each_category_segmented
from lpq import extract_lpq_features_for_each_category_segmented

from tools.data import (
    generate_texture_dicts,
    show_features_summary,
    build_image_folds_structure,
    prepare_specialist_sets_from_image_folds,
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
    generate_relevance_heatmaps_by_fold,
    relevance_technique_cross,
    export_relevance_cross_results_to_csv,
    zip_and_cleanup_results,
)

from utils import show_confusion_matrix, show_relevance_cross_experiment_metrics

# Load environment variables from .env file
load_dotenv(".env", override=True)

# Marca o tempo de início do script (para medir tempo total de execução)
start_time = time.perf_counter()

# Suprime warnings do JobLib relacionados ao ResourceTracker
# Evita poluição do output com mensagens irrelevantes
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
warnings.filterwarnings("ignore", message=".*resource_tracker.*")
warnings.filterwarnings("ignore", message=".*Cannot register.*")


def timed_relevance_technique(
    base_model,
    specialist_sets,
    class_names,
    model_name,
    k_folds,
    true_labels,
    after: Optional[Callable[[object], None]] = None,
):
    """Wrapper around relevance_technique that measures and prints execution time.

    Returns the same value as relevance_technique.
    """
    start = time.perf_counter()
    results = relevance_technique_cross(
        base_model=base_model,
        specialist_sets=specialist_sets,
        class_names=class_names,
        model_name=model_name,
        k_folds=k_folds,
        true_labels=true_labels,
    )
    end = time.perf_counter()
    elapsed = end - start
    # Format elapsed nicely
    if elapsed < 60:
        elapsed_str = f"{elapsed:.2f} s"
    else:
        mins = int(elapsed // 60)
        secs = elapsed % 60
        elapsed_str = f"{mins}m {secs:.2f}s"

    print(f"⏱️ Tempo de execução ({model_name}): {elapsed_str}")

    # Executa pós-processamento (export/plots) fora do tempo medido.
    if after is not None:
        after(results)

    return results


def _postprocess_relevance_cross_results(
    cross_results,
    *,
    title: str,
    confusion_cmap: str,
    heatmap_model_name: str,
    export_model_name: str,
    heatmap_colormap: str = "spring",
):
    """Pós-processa resultados cross (sem impactar o tempo medido da técnica)."""

    global_results, cv_model_metrics, fold_results, fold_model_metrics = cross_results

    (
        _probabilities,
        _entropies,
        _relevances,
        _max_relevances,
        _ponderated_votes,
        _accumulated_votes,
        _predicted_labels,
        (true_y, predicted_y),
        relevance_metrics,
    ) = global_results

    show_relevance_cross_experiment_metrics(
        relevance_metrics,
        cv_model_metrics,
        fold_model_metrics,
        title=title,
    )
    show_confusion_matrix(
        true_y,
        predicted_y,
        title=title,
        cmap=confusion_cmap,
    )
    generate_relevance_heatmaps_by_fold(
        fold_results=fold_results,
        all_images_segmented=all_images_segmented,
        model_name=heatmap_model_name,
        overlay_alpha=0.5,
        save_grid_lines=True,
        colormap=heatmap_colormap,
    )

    export_relevance_cross_results_to_csv(
        cross_results=cross_results,
        true_labels=true_images_labels,
        model_name=export_model_name,
    )

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

MIN_K = int(os.getenv("IMAGE_MIN_K", 4))
MAX_K = int(os.getenv("IMAGE_MAX_K", 20))
BASE_SIZE = int(os.getenv("IMAGE_BASE_SIZE", 512))
BASE_K = int(os.getenv("IMAGE_BASE_K", 9))

print(f"MIN_K: {MIN_K}, MAX_K: {MAX_K}, BASE_SIZE: {BASE_SIZE}, BASE_K: {BASE_K}")

segmented_train_images = segment_images_by_category_auto(
    images_by_category=train_images_by_categories,
    min_k=MIN_K,
    max_k=MAX_K,
    base_k=BASE_K,
    base_size=BASE_SIZE,
)

all_images_segmented = merge_image_categories_dicts(segmented_train_images)

# EXTRACAO DE FEATURES
features = []

## Extrair Features LBP para todas as imagens segmentadas


radius = 2
n_pixels = 8

lbps_by_categories = compute_lbp_for_segments_by_categories(
    image_categories, segmented_train_images, radius, n_pixels
)

features.append(lbps_by_categories)

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

glcms_by_category = parallel_calculate_glcm_for_each_category_segmented(
    image_categories,
    segmented_train_images,
    GLCM_DISTANCES,
    GLCM_ANGLES,
    GLCM_FEATURES,
    GLCM_LEVELS,
)

features.append(glcms_by_category)

## Extrair Features LPQ para todas as imagens segmentadas
lpqs_dict = extract_lpq_features_for_each_category_segmented(
    image_categories, segmented_train_images
)

features.append(lpqs_dict)

# Matriz de características

(sets, labels, true_images_labels) = generate_texture_dicts(
    image_categories, features[0], features[1], features[2]
)
show_features_summary(sets, labels)

# Limpeza de variáveis desnecessárias após o cálculo das features
# Uma vez que as features estão calculadas e salvas, podemos liberar memória
# removendo variáveis intermediárias que não serão mais utilizadas

print("Limpando variáveis desnecessárias...")
print(f"Variáveis antes da limpeza: {len(locals())} variáveis locais")

# Variáveis de dados intermediários que podem ser removidas
variables_to_clean = [
    # Dados de segmentação (já processados)
    "segmented_train_images",
    # Dados intermediários de features por categoria (já consolidados)
    "lbps_by_categories",
    "glcms_by_category",
    "lpq_features_dict",
    # Variáveis antigas que não existem mais
    "lbps_dict_categories",
    "lbps_histograms_by_categories",
    "glcm_features_by_category",
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

print("\n✓ Limpeza concluída!")
print(f"  - {removed_count} variáveis removidas")
print("  - Memória liberada pela coleta de lixo")
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
[X_lbp, X_glcm, X_lpq, X_lbp_glcm, X_lbp_lpq, X_glcm_lpq, X_lbp_glcm_lpq] = sets

set_lbp = (X_lbp, labels)
set_glcm = (X_glcm, labels)
set_lpq = (X_lpq, labels)
set_lbp_glcm = (X_lbp_glcm, labels)
set_lbp_lpq = (X_lbp_lpq, labels)
set_glcm_lpq = (X_glcm_lpq, labels)
set_lbp_glcm_lpq = (X_lbp_glcm_lpq, labels)

print(f"Classes: {image_categories}")

# -----------------------------------------------------------------------------
# CROSS: folds primeiro (por imagem) → especialistas derivados por fold
# -----------------------------------------------------------------------------

folds_structure = build_image_folds_structure(
    X_ref=X_lbp,
    y=true_images_labels,
    k_folds=K_FOLDS,
    random_state=42,
    verbose=True,
)

final_sp_lbp_sets = prepare_specialist_sets_from_image_folds(
    X_features=X_lbp,
    y=true_images_labels,
    class_names=image_categories,
    folds_structure=folds_structure,
    verbose=False,
)
final_sp_glcm_sets = prepare_specialist_sets_from_image_folds(
    X_features=X_glcm,
    y=true_images_labels,
    class_names=image_categories,
    folds_structure=folds_structure,
    verbose=False,
)
final_sp_lpq_sets = prepare_specialist_sets_from_image_folds(
    X_features=X_lpq,
    y=true_images_labels,
    class_names=image_categories,
    folds_structure=folds_structure,
    verbose=False,
)
final_sp_lbp_glcm_sets = prepare_specialist_sets_from_image_folds(
    X_features=X_lbp_glcm,
    y=true_images_labels,
    class_names=image_categories,
    folds_structure=folds_structure,
    verbose=False,
)
final_sp_lbp_lpq_sets = prepare_specialist_sets_from_image_folds(
    X_features=X_lbp_lpq,
    y=true_images_labels,
    class_names=image_categories,
    folds_structure=folds_structure,
    verbose=False,
)
final_sp_glcm_lpq_sets = prepare_specialist_sets_from_image_folds(
    X_features=X_glcm_lpq,
    y=true_images_labels,
    class_names=image_categories,
    folds_structure=folds_structure,
    verbose=False,
)
final_sp_lbp_glcm_lpq_sets = prepare_specialist_sets_from_image_folds(
    X_features=X_lbp_glcm_lpq,
    y=true_images_labels,
    class_names=image_categories,
    folds_structure=folds_structure,
    verbose=False,
)

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

# Relevância
## KNN
print("🚀 === CALCULANDO RELEVÂNCIA DOS MODELOS KNN ===")

class_names = image_categories
base_name = "KNN"

print("\n📊 Calculando relevância LBP...")
relevance_results_knn_lbp = timed_relevance_technique(
    base_model=base_knn,
    specialist_sets=final_sp_lbp_sets,
    class_names=class_names,
    model_name=f"{base_name}-LBP",
    k_folds=K_FOLDS,
    true_labels=true_images_labels,
    after=lambda cross_results: _postprocess_relevance_cross_results(
        cross_results,
        title=f"{base_name} LBP",
        confusion_cmap="Blues",
        heatmap_model_name=f"{base_name.lower()}_lbp",
        export_model_name=f"{base_name}_LBP",
    ),
)

# Treina especialistas para GLCM
print("\n📊 Calculando relevância GLCM...")
relevance_results_knn_glcm = timed_relevance_technique(
    base_model=base_knn,
    specialist_sets=final_sp_glcm_sets,
    class_names=class_names,
    model_name=f"{base_name}-GLCM",
    k_folds=K_FOLDS,
    true_labels=true_images_labels,
    after=lambda cross_results: _postprocess_relevance_cross_results(
        cross_results,
        title="KNN GLCM",
        confusion_cmap="Blues",
        heatmap_model_name=f"{base_name.lower()}_glcm",
        export_model_name=f"{base_name}_GLCM",
    ),
)

# Calcula relevância para LPQ
print("\n📊 Calculando relevância LPQ...")
relevance_results_knn_lpq = timed_relevance_technique(
    base_model=base_knn,
    specialist_sets=final_sp_lpq_sets,
    class_names=class_names,
    model_name=f"{base_name}-LPQ",
    k_folds=K_FOLDS,
    true_labels=true_images_labels,
    after=lambda cross_results: _postprocess_relevance_cross_results(
        cross_results,
        title="KNN LPQ",
        confusion_cmap="Blues",
        heatmap_model_name=f"{base_name.lower()}_lpq",
        export_model_name=f"{base_name}_LPQ",
    ),
)

# Calcula relevância para LBP+GLCM
print("\n📊 Calculando relevância LBP+GLCM...")
relevance_results_knn_lbp_glcm = timed_relevance_technique(
    base_model=base_knn,
    specialist_sets=final_sp_lbp_glcm_sets,
    class_names=class_names,
    model_name=f"{base_name}-LBP+GLCM",
    k_folds=K_FOLDS,
    true_labels=true_images_labels,
    after=lambda cross_results: _postprocess_relevance_cross_results(
        cross_results,
        title="KNN LBP+GLCM",
        confusion_cmap="Blues",
        heatmap_model_name=f"{base_name.lower()}_lbp_glcm",
        export_model_name=f"{base_name}_LBP_GLCM",
    ),
)

# Calcula relevância para LBP+LPQ
print("\n📊 Calculando relevância LBP+LPQ...")
relevance_results_knn_lbp_lpq = timed_relevance_technique(
    base_model=base_knn,
    specialist_sets=final_sp_lbp_lpq_sets,
    class_names=class_names,
    model_name=f"{base_name}-LBP+LPQ",
    k_folds=K_FOLDS,
    true_labels=true_images_labels,
    after=lambda cross_results: _postprocess_relevance_cross_results(
        cross_results,
        title="KNN LBP+LPQ",
        confusion_cmap="Blues",
        heatmap_model_name=f"{base_name.lower()}_lbp_lpq",
        export_model_name=f"{base_name}_LBP_LPQ",
    ),
)

# Calcula relevância para GLCM+LPQ
print("\n📊 Calculando relevância GLCM+LPQ...")
relevance_results_knn_glcm_lpq = timed_relevance_technique(
    base_model=base_knn,
    specialist_sets=final_sp_glcm_lpq_sets,
    class_names=class_names,
    model_name=f"{base_name}-GLCM+LPQ",
    k_folds=K_FOLDS,
    true_labels=true_images_labels,
    after=lambda cross_results: _postprocess_relevance_cross_results(
        cross_results,
        title="KNN GLCM+LPQ",
        confusion_cmap="Blues",
        heatmap_model_name=f"{base_name.lower()}_glcm_lpq",
        export_model_name=f"{base_name}_GLCM_LPQ",
    ),
)

# Calcula relevância para LBP+GLCM+LPQ
print("\n📊 Calculando relevância LBP+GLCM+LPQ...")
relevance_results_knn_lbp_glcm_lpq = timed_relevance_technique(
    base_model=base_knn,
    specialist_sets=final_sp_lbp_glcm_lpq_sets,
    class_names=class_names,
    model_name=f"{base_name}-LBP+GLCM+LPQ",
    k_folds=K_FOLDS,
    true_labels=true_images_labels,
    after=lambda cross_results: _postprocess_relevance_cross_results(
        cross_results,
        title="KNN LBP+GLCM+LPQ",
        confusion_cmap="Blues",
        heatmap_model_name=f"{base_name.lower()}_lbp_glcm_lpq",
        export_model_name=f"{base_name}_LBP_GLCM_LPQ",
    ),
)

# print("✅ Relevância calculada para todos os 7 conjuntos de features")
# print("📊 Métricas e matrizes de confusão geradas para cada combinação")
# print("\n🎉 === CÁLCULO DE RELEVÂNCIA KNN CONCLUÍDO ===")

# print("🚀 === CALCULANDO RELEVÂNCIA DOS MODELOS SVM ===")

# base_name = "SVM"

# print("\n📊 Calculando relevância LBP...")
# relevance_results_svm_lbp = timed_relevance_technique(
#     base_model=base_svm,
#     specialist_sets=final_sp_lbp_sets,
#     class_names=class_names,
#     model_name=f"{base_name}-LBP",
#     k_folds=K_FOLDS,
#     true_labels=true_images_labels,
#     after=lambda cross_results: _postprocess_relevance_cross_results(
#         cross_results,
#         title="SVM LBP",
#         confusion_cmap="Reds",
#         heatmap_model_name=f"{base_name.lower()}_lbp",
#         export_model_name=f"{base_name}_LBP",
#     ),
# )

# print("\n📊 Calculando relevância GLCM...")
# relevance_results_svm_glcm = timed_relevance_technique(
#     base_model=base_svm,
#     specialist_sets=final_sp_glcm_sets,
#     class_names=class_names,
#     model_name=f"{base_name}-GLCM",
#     k_folds=K_FOLDS,
#     true_labels=true_images_labels,
#     after=lambda cross_results: _postprocess_relevance_cross_results(
#         cross_results,
#         title="SVM GLCM",
#         confusion_cmap="Reds",
#         heatmap_model_name=f"{base_name.lower()}_glcm",
#         export_model_name=f"{base_name}_GLCM",
#     ),
# )

# print("\n📊 Calculando relevância LPQ...")
# relevance_results_svm_lpq = timed_relevance_technique(
#     base_model=base_svm,
#     specialist_sets=final_sp_lpq_sets,
#     class_names=class_names,
#     model_name=f"{base_name}-LPQ",
#     k_folds=K_FOLDS,
#     true_labels=true_images_labels,
#     after=lambda cross_results: _postprocess_relevance_cross_results(
#         cross_results,
#         title="SVM LPQ",
#         confusion_cmap="Reds",
#         heatmap_model_name=f"{base_name.lower()}_lpq",
#         export_model_name=f"{base_name}_LPQ",
#     ),
# )

# print("\n📊 Calculando relevância LBP+GLCM...")
# relevance_results_svm_lbp_glcm = timed_relevance_technique(
#     base_model=base_svm,
#     specialist_sets=final_sp_lbp_glcm_sets,
#     class_names=class_names,
#     model_name=f"{base_name}-LBP+GLCM",
#     k_folds=K_FOLDS,
#     true_labels=true_images_labels,
#     after=lambda cross_results: _postprocess_relevance_cross_results(
#         cross_results,
#         title="SVM LBP+GLCM",
#         confusion_cmap="Reds",
#         heatmap_model_name=f"{base_name.lower()}_lbp_glcm",
#         export_model_name=f"{base_name}_LBP_GLCM",
#     ),
# )

# print("\n📊 Calculando relevância LBP+LPQ...")
# relevance_results_svm_lbp_lpq = timed_relevance_technique(
#     base_model=base_svm,
#     specialist_sets=final_sp_lbp_lpq_sets,
#     class_names=class_names,
#     model_name=f"{base_name}-LBP+LPQ",
#     k_folds=K_FOLDS,
#     true_labels=true_images_labels,
#     after=lambda cross_results: _postprocess_relevance_cross_results(
#         cross_results,
#         title="SVM LBP+LPQ",
#         confusion_cmap="Reds",
#         heatmap_model_name=f"{base_name.lower()}_lbp_lpq",
#         export_model_name=f"{base_name}_LBP_LPQ",
#     ),
# )

# print("\n📊 Calculando relevância GLCM+LPQ...")
# relevance_results_svm_glcm_lpq = timed_relevance_technique(
#     base_model=base_svm,
#     specialist_sets=final_sp_glcm_lpq_sets,
#     class_names=class_names,
#     model_name=f"{base_name}-GLCM+LPQ",
#     k_folds=K_FOLDS,
#     true_labels=true_images_labels,
#     after=lambda cross_results: _postprocess_relevance_cross_results(
#         cross_results,
#         title="SVM GLCM+LPQ",
#         confusion_cmap="Reds",
#         heatmap_model_name=f"{base_name.lower()}_glcm_lpq",
#         export_model_name=f"{base_name}_GLCM_LPQ",
#     ),
# )

# print("\n📊 Calculando relevância LBP+GLCM+LPQ...")
# relevance_results_svm_lbp_glcm_lpq = timed_relevance_technique(
#     base_model=base_svm,
#     specialist_sets=final_sp_lbp_glcm_lpq_sets,
#     class_names=class_names,
#     model_name=f"{base_name}-LBP+GLCM+LPQ",
#     k_folds=K_FOLDS,
#     true_labels=true_images_labels,
#     after=lambda cross_results: _postprocess_relevance_cross_results(
#         cross_results,
#         title="SVM LBP+GLCM+LPQ",
#         confusion_cmap="Reds",
#         heatmap_model_name=f"{base_name.lower()}_lbp_glcm_lpq",
#         export_model_name=f"{base_name}_LBP_GLCM_LPQ",
#     ),
# )

# print("✅ Relevância calculada para todos os 7 conjuntos de features")
# print("📊 Métricas e matrizes de confusão geradas para cada combinação")
# print("\n🎉 === CÁLCULO DE RELEVÂNCIA SVM CONCLUÍDO ===")

zip_and_cleanup_results(
    results_dir="results",
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

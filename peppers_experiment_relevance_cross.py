import os
import time
import warnings
from typing import Callable, Optional

import matplotlib
import numpy as np
from dotenv import load_dotenv

os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg", force=True)

from tools.image_tools import (  # noqa: E402
    load_train_images_dict_gray,
    merge_image_categories_dicts,
    segment_images_by_category_auto,
)

from glcm import parallel_calculate_glcm_for_each_category_segmented  # noqa: E402
from lbp import compute_lbp_for_segments_by_categories  # noqa: E402
from lpq import extract_lpq_features_for_each_category_segmented  # noqa: E402

from tools.data import (  # noqa: E402
    build_image_folds_structure,
    generate_texture_dicts,
    prepare_specialist_sets_from_image_folds,
    show_features_summary,
)

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import GridSearchCV  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from tools.relevance import (  # noqa: E402
    export_relevance_cross_results_to_csv,
    generate_relevance_heatmaps_by_fold,
    relevance_technique_cross,
    zip_and_cleanup_results,
)

from utils import show_confusion_matrix, show_relevance_cross_experiment_metrics  # noqa: E402


# Load environment variables from .env file
load_dotenv(".env", override=True)

# Marca o tempo de início do script (para medir tempo total de execução)
start_time = time.perf_counter()

# Suprime warnings do JobLib relacionados ao ResourceTracker
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
    """Wrapper around relevance_technique_cross that measures execution time.

    Post-processing (export/plots) is intentionally excluded from the measured time.
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
    if elapsed < 60:
        elapsed_str = f"{elapsed:.2f} s"
    else:
        mins = int(elapsed // 60)
        secs = elapsed % 60
        elapsed_str = f"{mins}m {secs:.2f}s"

    print(f"⏱️ Tempo de execução ({model_name}): {elapsed_str}")

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
    show_confusion_matrix(true_y, predicted_y, title=title, cmap=confusion_cmap)

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


images_directory = "./images/experiment/peppers_two_classes"
image_categories = ["peperomia", "piper"]
NEEDS_RESIZE = int(os.getenv("NEEDS_RESIZE", True))
ACCEPTED_IMAGES_EXTENSIONS = (".jpg", ".jpeg", ".png")

train_images_by_categories = load_train_images_dict_gray(
    images_directory,
    image_categories,
    ACCEPTED_IMAGES_EXTENSIONS,
    NEEDS_RESIZE,
)

MIN_K = int(os.getenv("IMAGE_MIN_K", 4))
MAX_K = int(os.getenv("IMAGE_MAX_K", 20))
BASE_SIZE = int(os.getenv("IMAGE_BASE_SIZE", 512))
BASE_K = int(os.getenv("IMAGE_BASE_K", 9))
MIN_REGION_SIZE = int(os.getenv("IMAGE_MIN_REGION_SIZE", 128))

print(f"MIN_K: {MIN_K}, MAX_K: {MAX_K}, BASE_SIZE: {BASE_SIZE}, BASE_K: {BASE_K}, MIN_REGION_SIZE: {MIN_REGION_SIZE}")

segmented_train_images = segment_images_by_category_auto(
    images_by_category=train_images_by_categories,
    min_k=MIN_K,
    max_k=MAX_K,
    base_k=BASE_K,
    base_size=BASE_SIZE,
    min_region_size=MIN_REGION_SIZE
)

all_images_segmented = merge_image_categories_dicts(segmented_train_images)

# -----------------------------------------------------------------------------
# Extração de features
# -----------------------------------------------------------------------------
features = []

radius = 2
n_pixels = 8

lbps_by_categories = compute_lbp_for_segments_by_categories(
    image_categories, segmented_train_images, radius, n_pixels
)
features.append(lbps_by_categories)

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

lpqs_dict = extract_lpq_features_for_each_category_segmented(
    image_categories, segmented_train_images
)
features.append(lpqs_dict)

(sets, labels, true_images_labels) = generate_texture_dicts(
    image_categories, features[0], features[1], features[2]
)
show_features_summary(sets, labels)

# -----------------------------------------------------------------------------
# CROSS: folds primeiro (por imagem) → especialistas derivados por fold
# -----------------------------------------------------------------------------
K_FOLDS = 5
[X_lbp, X_glcm, X_lpq, X_lbp_glcm, X_lbp_lpq, X_glcm_lpq, X_lbp_glcm_lpq] = sets

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


# -----------------------------------------------------------------------------
# Modelos e tuning
# -----------------------------------------------------------------------------
K_ARRAY = [1, 3, 5, 7, 9]
DISTANCE_METRICS = ["euclidean", "manhattan", "minkowski"]


def tune_knn(k_array, distance_metrics):
    knn = KNeighborsClassifier()
    parameters = {"knn__n_neighbors": k_array, "knn__metric": distance_metrics}
    pipe = Pipeline([("scaler", StandardScaler()), ("knn", knn)])
    model = GridSearchCV(pipe, parameters, n_jobs=-1)
    return model


print("Tunando parâmetros do KNN...")
base_knn = tune_knn(K_ARRAY, DISTANCE_METRICS)

N_ESTIMATORS = [100, 200]
MIN_SAMPLES_LEAFS = [3, 10, 15, 30]
MIN_SAMPLES_SPLITS = [2, 4, 16, 40]
MAX_DEPTHS = [None, 3, 8]
MAX_FEATURES = ["sqrt"]
MAX_SAMPLES = [None]
CRITERIONS = ["gini", "entropy"]
CLASS_WEIGHT = [None, "balanced_subsample"]


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

KERNELS = ["rbf"]
C = [0.1, 1, 10, 100, 1000]
GAMMA = [2e-5, 2e-3, 2e-1, "auto", "scale"]


def tune_svm(kernels, c_array, gamma_array):
    svm = SVC(probability=True, random_state=42)
    parameters = {"svm__kernel": kernels, "svm__C": c_array, "svm__gamma": gamma_array}
    pipe = Pipeline([("scaler", StandardScaler()), ("svm", svm)])
    model = GridSearchCV(pipe, parameters, n_jobs=-1)
    return model


print("Tunando SVM...")
base_svm = tune_svm(KERNELS, C, GAMMA)


# -----------------------------------------------------------------------------
# Relevância (CROSS)
# -----------------------------------------------------------------------------
print("🚀 === CALCULANDO RELEVÂNCIA (CROSS) ===")

class_names = image_categories

print("\n📊 KNN - LBP...")
_ = timed_relevance_technique(
    base_model=base_knn,
    specialist_sets=final_sp_lbp_sets,
    class_names=class_names,
    model_name="KNN-LBP",
    k_folds=K_FOLDS,
    true_labels=true_images_labels,
    after=lambda cross_results: _postprocess_relevance_cross_results(
        cross_results,
        title="KNN LBP",
        confusion_cmap="Blues",
        heatmap_model_name="knn_lbp",
        export_model_name="KNN_LBP",
    ),
)

print("\n📊 KNN - GLCM...")
_ = timed_relevance_technique(
    base_model=base_knn,
    specialist_sets=final_sp_glcm_sets,
    class_names=class_names,
    model_name="KNN-GLCM",
    k_folds=K_FOLDS,
    true_labels=true_images_labels,
    after=lambda cross_results: _postprocess_relevance_cross_results(
        cross_results,
        title="KNN GLCM",
        confusion_cmap="Blues",
        heatmap_model_name="knn_glcm",
        export_model_name="KNN_GLCM",
    ),
)

print("\n📊 KNN - LPQ...")
_ = timed_relevance_technique(
    base_model=base_knn,
    specialist_sets=final_sp_lpq_sets,
    class_names=class_names,
    model_name="KNN-LPQ",
    k_folds=K_FOLDS,
    true_labels=true_images_labels,
    after=lambda cross_results: _postprocess_relevance_cross_results(
        cross_results,
        title="KNN LPQ",
        confusion_cmap="Blues",
        heatmap_model_name="knn_lpq",
        export_model_name="KNN_LPQ",
    ),
)

print("\n📊 KNN - LBP+GLCM...")
_ = timed_relevance_technique(
    base_model=base_knn,
    specialist_sets=final_sp_lbp_glcm_sets,
    class_names=class_names,
    model_name="KNN-LBP+GLCM",
    k_folds=K_FOLDS,
    true_labels=true_images_labels,
    after=lambda cross_results: _postprocess_relevance_cross_results(
        cross_results,
        title="KNN LBP+GLCM",
        confusion_cmap="Blues",
        heatmap_model_name="knn_lbp_glcm",
        export_model_name="KNN_LBP_GLCM",
    ),
)

print("\n📊 KNN - LBP+LPQ...")
_ = timed_relevance_technique(
    base_model=base_knn,
    specialist_sets=final_sp_lbp_lpq_sets,
    class_names=class_names,
    model_name="KNN-LBP+LPQ",
    k_folds=K_FOLDS,
    true_labels=true_images_labels,
    after=lambda cross_results: _postprocess_relevance_cross_results(
        cross_results,
        title="KNN LBP+LPQ",
        confusion_cmap="Blues",
        heatmap_model_name="knn_lbp_lpq",
        export_model_name="KNN_LBP_LPQ",
    ),
)

print("\n📊 KNN - GLCM+LPQ...")
_ = timed_relevance_technique(
    base_model=base_knn,
    specialist_sets=final_sp_glcm_lpq_sets,
    class_names=class_names,
    model_name="KNN-GLCM+LPQ",
    k_folds=K_FOLDS,
    true_labels=true_images_labels,
    after=lambda cross_results: _postprocess_relevance_cross_results(
        cross_results,
        title="KNN GLCM+LPQ",
        confusion_cmap="Blues",
        heatmap_model_name="knn_glcm_lpq",
        export_model_name="KNN_GLCM_LPQ",
    ),
)

print("\n📊 KNN - LBP+GLCM+LPQ...")
_ = timed_relevance_technique(
    base_model=base_knn,
    specialist_sets=final_sp_lbp_glcm_lpq_sets,
    class_names=class_names,
    model_name="KNN-LBP+GLCM+LPQ",
    k_folds=K_FOLDS,
    true_labels=true_images_labels,
    after=lambda cross_results: _postprocess_relevance_cross_results(
        cross_results,
        title="KNN LBP+GLCM+LPQ",
        confusion_cmap="Blues",
        heatmap_model_name="knn_lbp_glcm_lpq",
        export_model_name="KNN_LBP_GLCM_LPQ",
    ),
)

print("\n📊 SVM - LBP...")
_ = timed_relevance_technique(
    base_model=base_svm,
    specialist_sets=final_sp_lbp_sets,
    class_names=class_names,
    model_name="SVM-LBP",
    k_folds=K_FOLDS,
    true_labels=true_images_labels,
    after=lambda cross_results: _postprocess_relevance_cross_results(
        cross_results,
        title="SVM LBP",
        confusion_cmap="Reds",
        heatmap_model_name="svm_lbp",
        export_model_name="SVM_LBP",
    ),
)

print("\n📊 SVM - GLCM...")
_ = timed_relevance_technique(
    base_model=base_svm,
    specialist_sets=final_sp_glcm_sets,
    class_names=class_names,
    model_name="SVM-GLCM",
    k_folds=K_FOLDS,
    true_labels=true_images_labels,
    after=lambda cross_results: _postprocess_relevance_cross_results(
        cross_results,
        title="SVM GLCM",
        confusion_cmap="Reds",
        heatmap_model_name="svm_glcm",
        export_model_name="SVM_GLCM",
    ),
)

print("\n📊 SVM - LPQ...")
_ = timed_relevance_technique(
    base_model=base_svm,
    specialist_sets=final_sp_lpq_sets,
    class_names=class_names,
    model_name="SVM-LPQ",
    k_folds=K_FOLDS,
    true_labels=true_images_labels,
    after=lambda cross_results: _postprocess_relevance_cross_results(
        cross_results,
        title="SVM LPQ",
        confusion_cmap="Reds",
        heatmap_model_name="svm_lpq",
        export_model_name="SVM_LPQ",
    ),
)

print("\n📊 SVM - LBP+GLCM...")
_ = timed_relevance_technique(
    base_model=base_svm,
    specialist_sets=final_sp_lbp_glcm_sets,
    class_names=class_names,
    model_name="SVM-LBP+GLCM",
    k_folds=K_FOLDS,
    true_labels=true_images_labels,
    after=lambda cross_results: _postprocess_relevance_cross_results(
        cross_results,
        title="SVM LBP+GLCM",
        confusion_cmap="Reds",
        heatmap_model_name="svm_lbp_glcm",
        export_model_name="SVM_LBP_GLCM",
    ),
)

print("\n📊 SVM - LBP+LPQ...")
_ = timed_relevance_technique(
    base_model=base_svm,
    specialist_sets=final_sp_lbp_lpq_sets,
    class_names=class_names,
    model_name="SVM-LBP+LPQ",
    k_folds=K_FOLDS,
    true_labels=true_images_labels,
    after=lambda cross_results: _postprocess_relevance_cross_results(
        cross_results,
        title="SVM LBP+LPQ",
        confusion_cmap="Reds",
        heatmap_model_name="svm_lbp_lpq",
        export_model_name="SVM_LBP_LPQ",
    ),
)

print("\n📊 SVM - GLCM+LPQ...")
_ = timed_relevance_technique(
    base_model=base_svm,
    specialist_sets=final_sp_glcm_lpq_sets,
    class_names=class_names,
    model_name="SVM-GLCM+LPQ",
    k_folds=K_FOLDS,
    true_labels=true_images_labels,
    after=lambda cross_results: _postprocess_relevance_cross_results(
        cross_results,
        title="SVM GLCM+LPQ",
        confusion_cmap="Reds",
        heatmap_model_name="svm_glcm_lpq",
        export_model_name="SVM_GLCM_LPQ",
    ),
)

print("\n📊 SVM - LBP+GLCM+LPQ...")
_ = timed_relevance_technique(
    base_model=base_svm,
    specialist_sets=final_sp_lbp_glcm_lpq_sets,
    class_names=class_names,
    model_name="SVM-LBP+GLCM+LPQ",
    k_folds=K_FOLDS,
    true_labels=true_images_labels,
    after=lambda cross_results: _postprocess_relevance_cross_results(
        cross_results,
        title="SVM LBP+GLCM+LPQ",
        confusion_cmap="Reds",
        heatmap_model_name="svm_lbp_glcm_lpq",
        export_model_name="SVM_LBP_GLCM_LPQ",
    ),
)

zip_and_cleanup_results(
    results_dir="results",
    folders_to_zip=["confusion_matrixs", "heatmaps", "csv_exports"],
)

end_time = time.perf_counter()
total_elapsed = end_time - start_time
if total_elapsed < 60:
    total_elapsed_str = f"{total_elapsed:.2f} s"
else:
    mins = int(total_elapsed // 60)
    secs = total_elapsed % 60
    total_elapsed_str = f"{mins}m {secs:.2f}s"

print(f"⏳ Tempo total de execução do script: {total_elapsed_str}")

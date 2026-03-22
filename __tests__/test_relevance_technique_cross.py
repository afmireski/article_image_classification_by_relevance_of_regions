"""Testes para relevance_technique_cross (execução por fold).

Observação: os testes aqui seguem o padrão dos demais arquivos em __tests__/:
funções com asserts, compatíveis com pytest (se usado) e com execução direta.
"""

from __future__ import annotations

import sys
import os
import tempfile
from typing import Dict, List

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Garante que a raiz do projeto está no sys.path (execução direta do arquivo).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.relevance import (
    relevance_technique_cross,
    export_relevance_cross_results_to_csv,
)


def _make_synthetic_specialists_dataset(
    n_classes: int = 3,
    images_per_class: int = 6,
    segments_per_image: int = 4,
    n_features: int = 8,
    k_folds: int = 3,
    seed: int = 42,
):
    """Cria specialist_sets no formato esperado por relevance_technique(_cross).

    - Folds são disjuntos por imagem (todas as peças de uma imagem no mesmo fold).
    - Para cada especialista i: y=0 para imagens da classe i, y=1 caso contrário.
    """

    rng = np.random.default_rng(seed)

    # IDs e labels por imagem (nível de imagem)
    image_ids: List[str] = []
    image_labels: Dict[str, int] = {}
    for c in range(n_classes):
        for j in range(images_per_class):
            img_id = f"img_c{c}_{j}"
            image_ids.append(img_id)
            image_labels[img_id] = c

    # Embaralha imagens para distribuir nos folds
    shuffled_ids = image_ids[:]
    rng.shuffle(shuffled_ids)

    # Particiona imagens por fold
    folds_images: List[List[str]] = [[] for _ in range(k_folds)]
    for idx, img_id in enumerate(shuffled_ids):
        folds_images[idx % k_folds].append(img_id)

    # Features sintéticas: cada classe com um centro distinto
    class_centers = rng.normal(0, 3.0, size=(n_classes, n_features))

    # Pré-gera segmentos por imagem
    segments_X: Dict[str, np.ndarray] = {}
    for img_id in image_ids:
        c = image_labels[img_id]
        X_img = class_centers[c] + rng.normal(0, 0.5, size=(segments_per_image, n_features))
        segments_X[img_id] = X_img.astype(np.float64)

    # Monta specialist_sets: List[ClassificationDataset]
    specialist_sets = []

    for specialist_class in range(n_classes):
        dataset_folds = []

        for fold_idx in range(k_folds):
            test_imgs = set(folds_images[fold_idx])
            train_imgs = [img for img in image_ids if img not in test_imgs]
            test_imgs_list = [img for img in image_ids if img in test_imgs]

            # Concatena segmentos
            X_train = np.vstack([segments_X[img] for img in train_imgs])
            X_test = np.vstack([segments_X[img] for img in test_imgs_list])

            # Labels binários por segmento
            y_train = np.hstack(
                [
                    np.zeros(segments_per_image, dtype=int)
                    if image_labels[img] == specialist_class
                    else np.ones(segments_per_image, dtype=int)
                    for img in train_imgs
                ]
            )
            y_test = np.hstack(
                [
                    np.zeros(segments_per_image, dtype=int)
                    if image_labels[img] == specialist_class
                    else np.ones(segments_per_image, dtype=int)
                    for img in test_imgs_list
                ]
            )

            # pieces_map: idx_amostra -> img_id
            train_map = {
                idx: train_imgs[idx // segments_per_image] for idx in range(len(train_imgs) * segments_per_image)
            }
            test_map = {
                idx: test_imgs_list[idx // segments_per_image]
                for idx in range(len(test_imgs_list) * segments_per_image)
            }

            train_set = (X_train, y_train, train_map)
            test_set = (X_test, y_test, test_map)
            dataset_folds.append((train_set, test_set))

        specialist_sets.append(dataset_folds)

    class_names = [f"class_{i}" for i in range(n_classes)]
    return specialist_sets, class_names, image_labels, k_folds


def _make_base_model():
    # Modelo simples (sem GridSearchCV) para validar best_params_ condicional.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=3)),
    ])


def test_relevance_technique_cross_shapes_and_metrics():
    specialist_sets, class_names, true_labels, k_folds = _make_synthetic_specialists_dataset()
    base_model = _make_base_model()

    cross_results = relevance_technique_cross(
        base_model=base_model,
        specialist_sets=specialist_sets,
        class_names=class_names,
        true_labels=true_labels,
        model_name="TEST",
        k_folds=k_folds,
    )

    global_results, cv_model_metrics, fold_results, fold_model_metrics = cross_results

    (
        _probabilities,
        _entropies,
        _relevances,
        _max_relevances,
        _ponderated_votes,
        _accumulated_votes,
        predicted_labels_global,
        _labels_list,
        _metrics,
    ) = global_results

    # 1) Predições cobrem exatamente as imagens
    assert set(predicted_labels_global.keys()) == set(true_labels.keys())
    assert all(predicted_labels_global[img] is not None for img in true_labels.keys())

    # 2) CV metrics têm folds = k_folds
    assert len(cv_model_metrics["accuracy"]["folds"]) == k_folds
    assert len(fold_results) == k_folds
    assert len(fold_model_metrics) == k_folds

    # 3) Mean bate com média dos folds
    accs = [m[0] for m in fold_model_metrics]
    mean_acc = float(np.mean(accs))
    assert np.isclose(cv_model_metrics["accuracy"]["mean"], mean_acc)


def test_export_relevance_cross_metrics_rows():
    specialist_sets, class_names, true_labels, k_folds = _make_synthetic_specialists_dataset()
    base_model = _make_base_model()

    cross_results = relevance_technique_cross(
        base_model=base_model,
        specialist_sets=specialist_sets,
        class_names=class_names,
        true_labels=true_labels,
        model_name="TEST_MODEL",
        k_folds=k_folds,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = export_relevance_cross_results_to_csv(
            cross_results=cross_results,
            true_labels=true_labels,
            model_name="TEST_MODEL",
            output_dir=tmpdir,
        )

        assert os.path.exists(paths["global"])
        for i in range(1, k_folds + 1):
            assert os.path.exists(paths[f"fold{i}"])

        # Checa o metrics CSV global
        metrics_path = os.path.join(tmpdir, "csv_exports", "test_model_metrics.csv")
        assert os.path.exists(metrics_path)

        with open(metrics_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Linhas esperadas: global + mean + fold1..K
        assert "test_model_relevance_global" in content
        assert "test_model_relevance_mean" in content
        for i in range(1, k_folds + 1):
            assert f"test_model_relevance_fold{i}" in content


def run_all_tests():
    try:
        print("\n" + "=" * 60)
        print("TESTES: relevance_technique_cross")
        print("=" * 60)
        test_relevance_technique_cross_shapes_and_metrics()
        print("✅ shapes/metrics ok")
        test_export_relevance_cross_metrics_rows()
        print("✅ export/metrics rows ok")
        print("\n🎉 Todos os testes de relevance_technique_cross passaram!")
    except AssertionError as e:
        print(f"\n❌ TESTE FALHOU: {str(e)}")
        raise
    except Exception as e:
        print(f"\n❌ ERRO INESPERADO: {str(e)}")
        raise


if __name__ == "__main__":
    run_all_tests()

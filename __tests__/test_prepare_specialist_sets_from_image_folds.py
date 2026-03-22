"""Testes para preparação de especialistas a partir de folds globais por imagem.

Garante que, para cada fold, TODOS os especialistas compartilham exatamente o mesmo
conjunto de imagens em teste (alinhamento), e que folds são disjuntos por imagem.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np

# Garante que a raiz do projeto está no sys.path (execução direta do arquivo).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.data import build_image_folds_structure, prepare_specialist_sets_from_image_folds


def _make_segmented_dataset(
    n_classes: int = 4,
    images_per_class: int = 5,
    segments_per_image: int = 4,
    n_features: int = 6,
    seed: int = 123,
) -> tuple[Dict[str, np.ndarray], Dict[str, int], List[str]]:
    rng = np.random.default_rng(seed)

    class_names = [f"class_{i}" for i in range(n_classes)]
    X: Dict[str, np.ndarray] = {}
    y: Dict[str, int] = {}

    for c in range(n_classes):
        center = rng.normal(0, 2.0, size=(n_features,))
        for j in range(images_per_class):
            img_id = f"img_c{c}_{j}"
            X[img_id] = (center + rng.normal(0, 0.5, size=(segments_per_image, n_features))).astype(
                np.float64
            )
            y[img_id] = c

    return X, y, class_names


def test_prepare_specialist_sets_are_aligned_by_fold():
    X, y, class_names = _make_segmented_dataset()
    k_folds = 5

    folds_structure = build_image_folds_structure(
        X_ref=X,
        y=y,
        k_folds=k_folds,
        random_state=42,
        verbose=False,
    )

    specialist_sets = prepare_specialist_sets_from_image_folds(
        X_features=X,
        y=y,
        class_names=class_names,
        folds_structure=folds_structure,
        verbose=False,
    )

    assert len(specialist_sets) == len(class_names)
    assert all(len(ds) == k_folds for ds in specialist_sets)

    # 1) Para cada fold: conjunto de imagens de teste é igual para todos especialistas
    for fold_idx in range(k_folds):
        expected_test_images = set(folds_structure[fold_idx]["test_images"])

        seen_test_images_sets = []
        for sp_idx in range(len(class_names)):
            (_train_set, test_set) = specialist_sets[sp_idx][fold_idx]
            X_test, y_test, test_map = test_set

            assert X_test.shape[0] == y_test.shape[0]
            assert set(test_map.keys()) == set(range(len(y_test)))

            test_images_sp = set(test_map.values())
            seen_test_images_sets.append(test_images_sp)

            assert test_images_sp == expected_test_images
            assert test_images_sp.issubset(set(X.keys()))

        # Confirma igualdade entre especialistas (redundante, mas deixa a intenção explícita)
        base = seen_test_images_sets[0]
        assert all(s == base for s in seen_test_images_sets[1:])

    # 2) Folds disjuntos por imagem (test sets não podem se repetir)
    all_test = set()
    for fold_idx in range(k_folds):
        fold_test = set(folds_structure[fold_idx]["test_images"])
        assert all_test.isdisjoint(fold_test)
        all_test |= fold_test

    assert all_test == set(X.keys())


def run_all_tests():
    print("\n" + "=" * 60)
    print("TESTES: prepare_specialist_sets_from_image_folds")
    print("=" * 60)
    test_prepare_specialist_sets_are_aligned_by_fold()
    print("✅ OK")


if __name__ == "__main__":
    run_all_tests()

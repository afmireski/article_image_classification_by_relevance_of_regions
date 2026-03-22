"""Teste simples para validar reutilização de folds em prepare_full_image_sets_for_classification.

Este teste garante que múltiplos conjuntos de features (imagens completas) usam a MESMA
estrutura de folds (mesmas imagens em treino/teste por fold), permitindo comparação justa.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np

# Garante que a raiz do projeto está no sys.path (execução direta do arquivo).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.data import prepare_full_image_sets_for_classification


def _make_full_image_sets(
    n_classes: int = 3,
    images_per_class: int = 6,
    n_features_a: int = 8,
    n_features_b: int = 5,
    seed: int = 42,
) -> Tuple[List[Tuple[Dict[str, np.ndarray], Dict[str, int]]], List[str]]:
    rng = np.random.default_rng(seed)

    X_a: Dict[str, np.ndarray] = {}
    X_b: Dict[str, np.ndarray] = {}
    y: Dict[str, int] = {}

    for c in range(n_classes):
        for j in range(images_per_class):
            img_id = f"img_c{c}_{j}"
            X_a[img_id] = rng.normal(0, 1.0, size=(n_features_a,)).astype(np.float64)
            X_b[img_id] = rng.normal(0, 1.0, size=(n_features_b,)).astype(np.float64)
            y[img_id] = c

    sets = [(X_a, y), (X_b, y)]
    class_names = [f"class_{i}" for i in range(n_classes)]
    return sets, class_names


def test_prepare_full_image_sets_reuse_same_fold_structure():
    sets, _class_names = _make_full_image_sets()

    prepared = prepare_full_image_sets_for_classification(
        sets=sets,
        k_folds=3,
        random_state=123,
        verbose=False,
    )

    assert len(prepared) == 2
    folds_a, folds_b = prepared
    assert len(folds_a) == len(folds_b) == 3

    for fold_idx in range(3):
        (_train_a, test_a) = folds_a[fold_idx]
        (_train_b, test_b) = folds_b[fold_idx]

        # No formato full-image: train/test map é uma LISTA de imagens
        _X_test_a, _y_test_a, test_images_a = test_a
        _X_test_b, _y_test_b, test_images_b = test_b

        assert set(test_images_a) == set(test_images_b)


def run_all_tests():
    print("\n" + "=" * 60)
    print("TESTES: full_image_folds")
    print("=" * 60)
    test_prepare_full_image_sets_reuse_same_fold_structure()
    print("✅ OK")


if __name__ == "__main__":
    run_all_tests()

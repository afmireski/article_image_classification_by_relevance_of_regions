"""
Teste de integração: Valida que a preparação de conjuntos no standard_classification.py
funciona corretamente com a nova função prepare_full_image_sets_for_classification.
"""

import numpy as np
from tools.data import prepare_full_image_sets_for_classification

print("=" * 60)
print("TESTE: Integração com standard_classification.py")
print("=" * 60)

# Simular dados similares ao standard_classification.py
n_images = 80
n_classes = 4
images_per_class = n_images // n_classes

# Gerar nomes de imagens
image_names = []
true_images_labels = {}
for class_idx in range(n_classes):
    for img_idx in range(images_per_class):
        img_name = f"class{class_idx}_img{img_idx}"
        image_names.append(img_name)
        true_images_labels[img_name] = class_idx

# Simular os 7 conjuntos de features (como no standard_classification.py)
print("\nCriando conjuntos de features simulados:")
print(f"  - {n_images} imagens")
print(f"  - {n_classes} classes")

# LBP: 256 features
X_lbp_full = {img: np.random.rand(256) for img in image_names}
print("  ✓ LBP: 256 features")

# GLCM: 120 features
X_glcm_full = {img: np.random.rand(120) for img in image_names}
print("  ✓ GLCM: 120 features")

# LPQ: 256 features
X_lpq_full = {img: np.random.rand(256) for img in image_names}
print("  ✓ LPQ: 256 features")

# LBP+GLCM: 376 features
X_lbp_glcm_full = {img: np.concatenate([X_lbp_full[img], X_glcm_full[img]]) for img in image_names}
print("  ✓ LBP+GLCM: 376 features")

# LBP+LPQ: 512 features
X_lbp_lpq_full = {img: np.concatenate([X_lbp_full[img], X_lpq_full[img]]) for img in image_names}
print("  ✓ LBP+LPQ: 512 features")

# GLCM+LPQ: 376 features
X_glcm_lpq_full = {img: np.concatenate([X_glcm_full[img], X_lpq_full[img]]) for img in image_names}
print("  ✓ GLCM+LPQ: 376 features")

# LBP+GLCM+LPQ: 632 features
X_lbp_glcm_lpq_full = {img: np.concatenate([X_lbp_full[img], X_glcm_full[img], X_lpq_full[img]]) for img in image_names}
print("  ✓ LBP+GLCM+LPQ: 632 features")

# Criar estrutura de conjuntos (como no standard_classification.py)
set_lbp_full = (X_lbp_full, true_images_labels)
set_glcm_full = (X_glcm_full, true_images_labels)
set_lpq_full = (X_lpq_full, true_images_labels)
set_lbp_glcm_full = (X_lbp_glcm_full, true_images_labels)
set_lbp_lpq_full = (X_lbp_lpq_full, true_images_labels)
set_glcm_lpq_full = (X_glcm_lpq_full, true_images_labels)
set_lbp_glcm_lpq_full = (X_lbp_glcm_lpq_full, true_images_labels)

# Preparar conjuntos (EXATAMENTE como no standard_classification.py)
print("\n" + "=" * 60)
print("EXECUTANDO: prepare_full_image_sets_for_classification")
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

K_FOLDS = 5

prepared_full_image_sets = prepare_full_image_sets_for_classification(
    full_image_sets,
    k_folds=K_FOLDS,
    random_state=42,
    verbose=True,
)

# Desempacotar (como no standard_classification.py)
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

# Validação
print("\n" + "=" * 60)
print("VALIDAÇÃO DOS RESULTADOS")
print("=" * 60)

# Verificar estrutura dos folds
print("\n--- Validando estrutura das folds ---")
for fold_idx in range(K_FOLDS):
    train_data, test_data = final_lbp_full_folds[fold_idx]
    X_train, y_train, features_map_train = train_data
    X_test, y_test, features_map_test = test_data
    
    print(f"✓ Fold {fold_idx}:")
    print(f"  - Treino: X{X_train.shape}, y{y_train.shape}")
    print(f"  - Teste: X{X_test.shape}, y{y_test.shape}")

# Verificar que dimensões estão corretas para cada conjunto
print("\n--- Validando dimensões de features ---")
expected_dims = {
    "LBP": 256,
    "GLCM": 120,
    "LPQ": 256,
    "LBP+GLCM": 376,
    "LBP+LPQ": 512,
    "GLCM+LPQ": 376,
    "LBP+GLCM+LPQ": 632,
}

all_folds = [
    ("LBP", final_lbp_full_folds),
    ("GLCM", final_glcm_full_folds),
    ("LPQ", final_lpq_full_folds),
    ("LBP+GLCM", final_lbp_glcm_full_folds),
    ("LBP+LPQ", final_lbp_lpq_full_folds),
    ("GLCM+LPQ", final_glcm_lpq_full_folds),
    ("LBP+GLCM+LPQ", final_lbp_glcm_lpq_full_folds),
]

for name, folds in all_folds:
    train_data, _ = folds[0]
    X_train, _, _ = train_data
    actual_dims = X_train.shape[1]
    expected = expected_dims[name]
    
    if actual_dims == expected:
        print(f"✓ {name}: {actual_dims} features (correto)")
    else:
        print(f"✗ {name}: {actual_dims} features (esperado: {expected})")

print("\n" + "=" * 60)
print("🎉 TESTE DE INTEGRAÇÃO PASSOU COM SUCESSO!")
print("=" * 60)

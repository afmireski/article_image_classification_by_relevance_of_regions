"""
Teste para validar a função prepare_full_image_sets_for_classification
"""

import numpy as np
from tools.data import prepare_full_image_sets_for_classification

# Criar dados de teste simulando diferentes conjuntos de features
np.random.seed(42)

# 80 imagens, 4 classes (20 imagens por classe)
n_images = 80
n_classes = 4
images_per_class = n_images // n_classes

# Criar nomes de imagens
image_names = [f"img_{i:03d}" for i in range(n_images)]

# Criar labels (0, 0, 0, ..., 1, 1, 1, ..., 2, 2, 2, ..., 3, 3, 3)
true_labels = {}
for i, img in enumerate(image_names):
    class_idx = i // images_per_class
    true_labels[img] = class_idx

# Simular 3 conjuntos de features diferentes (LBP, GLCM, LPQ)
# LBP: 256 features
X_lbp = {img: np.random.rand(256) for img in image_names}

# GLCM: 120 features
X_glcm = {img: np.random.rand(120) for img in image_names}

# LPQ: 256 features
X_lpq = {img: np.random.rand(256) for img in image_names}

# Criar lista de conjuntos
sets = [
    (X_lbp, true_labels),
    (X_glcm, true_labels),
    (X_lpq, true_labels),
]

print("=" * 60)
print("TESTE: prepare_full_image_sets_for_classification")
print("=" * 60)
print(f"Total de imagens: {n_images}")
print(f"Total de classes: {n_classes}")
print(f"Imagens por classe: {images_per_class}")
print(f"\nConjuntos de features:")
print(f"  - LBP: {list(X_lbp.values())[0].shape[0]} features")
print(f"  - GLCM: {list(X_glcm.values())[0].shape[0]} features")
print(f"  - LPQ: {list(X_lpq.values())[0].shape[0]} features")
print()

# Testar a função
prepared_sets = prepare_full_image_sets_for_classification(
    sets,
    k_folds=5,
    random_state=42,
    verbose=True
)

print("\n" + "=" * 60)
print("VALIDAÇÃO DOS RESULTADOS")
print("=" * 60)

# Validar estrutura retornada
assert len(prepared_sets) == 3, f"Esperado 3 conjuntos, recebido {len(prepared_sets)}"
print(f"✓ Número de conjuntos correto: {len(prepared_sets)}")

# Validar cada conjunto
for idx, dataset in enumerate(prepared_sets):
    feature_names = ["LBP", "GLCM", "LPQ"]
    print(f"\n--- Conjunto {idx + 1} ({feature_names[idx]}) ---")
    
    # Cada dataset deve ter 5 folds
    assert len(dataset) == 5, f"Esperado 5 folds, recebido {len(dataset)}"
    print(f"  ✓ Número de folds: {len(dataset)}")
    
    # Validar cada fold
    for fold_idx, fold in enumerate(dataset):
        train_data, test_data = fold
        train_X, train_y, train_images = train_data
        test_X, test_y, test_images = test_data

        if fold_idx == 0:  # Mostrar detalhes apenas do primeiro fold
            print(f"  ✓ Fold 0:")
            print(f"    - Treino: X{train_X.shape}, y{train_y.shape}, {len(train_images)} imagens")
            print(f"    - Teste: X{test_X.shape}, y{test_y.shape}, {len(test_images)} imagens")

            # Verificar número de features
            expected_features = [256, 120, 256][idx]
            assert train_X.shape[1] == expected_features, \
                f"Esperado {expected_features} features, recebido {train_X.shape[1]}"
            print(f"    - Features corretas: {train_X.shape[1]}")

# Validar que todos os conjuntos usam as mesmas imagens nas mesmas folds
print("\n--- Validação: Mesmas imagens em todas as folds ---")

for fold_idx in range(5):
    # Pegar imagens de teste de cada conjunto para esta fold
    lbp_test_images = set(prepared_sets[0][fold_idx][1][2])
    glcm_test_images = set(prepared_sets[1][fold_idx][1][2])
    lpq_test_images = set(prepared_sets[2][fold_idx][1][2])

    # Todas devem ser iguais
    assert lbp_test_images == glcm_test_images == lpq_test_images, \
        f"Fold {fold_idx}: Imagens de teste diferentes entre conjuntos!"
    
    print(f"  ✓ Fold {fold_idx}: Todas as 3 conjuntos têm as mesmas {len(lbp_test_images)} imagens de teste")

# Validar que não há overlap entre folds
print("\n--- Validação: Sem overlap entre folds ---")
all_test_images = set()
for fold_idx in range(5):
    fold_test_images = set(prepared_sets[0][fold_idx][1][2])
    
    # Verificar se há overlap com folds anteriores
    overlap = all_test_images & fold_test_images
    assert len(overlap) == 0, f"Fold {fold_idx} tem overlap com folds anteriores: {overlap}"
    
    all_test_images.update(fold_test_images)

print(f"  ✓ Todas as {len(all_test_images)} imagens únicas aparecem exatamente uma vez")

# Validar balanceamento de classes
print("\n--- Validação: Balanceamento de classes ---")
for fold_idx in range(5):
    test_y = prepared_sets[0][fold_idx][1][1]
    
    # Contar classes
    unique, counts = np.unique(test_y, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    
    # Todas as classes devem ter a mesma quantidade
    counts_values = list(class_distribution.values())
    assert len(set(counts_values)) == 1, \
        f"Fold {fold_idx}: Classes desbalanceadas: {class_distribution}"
    
    if fold_idx == 0:
        print(f"  ✓ Fold {fold_idx}: Balanceado - {class_distribution}")

print("\n✓ Todas as folds estão balanceadas!")

print("\n" + "=" * 60)
print("🎉 TODOS OS TESTES PASSARAM COM SUCESSO!")
print("=" * 60)

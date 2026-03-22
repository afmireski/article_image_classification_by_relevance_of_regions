"""Script de teste para validar split_full_image_data_in_folds"""

import os
import sys

import numpy as np

# Garante que a raiz do projeto está no sys.path (execução direta do arquivo).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.data import split_full_image_data_in_folds

# Criar dados de teste simulados
# 4 classes com 20 imagens cada = 80 imagens totais
X = {}
y = {}

classes = 4
images_per_class = 20

for class_idx in range(classes):
    for img_idx in range(images_per_class):
        img_name = f"img_class{class_idx}_{img_idx}"
        # Features aleatórias (simulando features de imagem)
        X[img_name] = np.random.rand(100)  # 100 features por imagem
        y[img_name] = class_idx

print("=" * 60)
print("TESTE: split_full_image_data_in_folds")
print("=" * 60)
print(f"Total de imagens: {len(X)}")
print(f"Total de classes: {classes}")
print(f"Imagens por classe: {images_per_class}")
print()

# Testar com 5 folds
k_folds = 5

folds = split_full_image_data_in_folds(X, y, k_folds=k_folds, random_state=42, verbose=True)

print("\n" + "=" * 60)
print("VALIDAÇÃO DOS RESULTADOS")
print("=" * 60)

# Validar que não há overlap entre treino e teste
for fold_idx, fold in enumerate(folds):
    train_images = set(fold["train_features"].keys())
    test_images = set(fold["test_features"].keys())
    
    overlap = train_images.intersection(test_images)
    assert len(overlap) == 0, f"Fold {fold_idx}: Overlap detectado! {overlap}"
    
    print(f"\n✓ Fold {fold_idx}: Sem overlap entre treino e teste")
    print(f"  Treino: {len(train_images)} imagens")
    print(f"  Teste: {len(test_images)} imagens")
    
    # Validar balanceamento por classe no teste
    test_class_distribution = {}
    for img_name in test_images:
        label = fold["test_true_map"][img_name]
        test_class_distribution[label] = test_class_distribution.get(label, 0) + 1
    
    print(f"  Distribuição teste: {test_class_distribution}")
    
    # Validar que todas as imagens estão presentes
    all_fold_images = train_images.union(test_images)
    assert all_fold_images == set(X.keys()), f"Fold {fold_idx}: Faltam imagens!"

print("\n" + "=" * 60)
print("VALIDAÇÃO: Mesmas imagens em diferentes folds")
print("=" * 60)

# Validar que as mesmas imagens aparecem nos testes dos diferentes folds
# (para garantir que múltiplos conjuntos de features terão as mesmas divisões)
all_test_images = []
for fold_idx, fold in enumerate(folds):
    test_images = set(fold["test_features"].keys())
    all_test_images.append(test_images)
    
# Todas as imagens devem aparecer exatamente uma vez no teste
all_unique_test_images = set()
for test_set in all_test_images:
    all_unique_test_images.update(test_set)

assert all_unique_test_images == set(X.keys()), "Nem todas as imagens foram testadas!"
print("✓ Todas as imagens aparecem exatamente uma vez nos testes")

# Validar que não há duplicatas
for i in range(len(all_test_images)):
    for j in range(i + 1, len(all_test_images)):
        overlap = all_test_images[i].intersection(all_test_images[j])
        assert len(overlap) == 0, f"Overlap entre fold {i} e fold {j}: {overlap}"

print("✓ Não há overlap entre folds diferentes")

print("\n" + "=" * 60)
print("🎉 TODOS OS TESTES PASSARAM COM SUCESSO!")
print("=" * 60)

"""
Teste para validar generate_texture_dicts_full com features de imagens completas (1D).
"""
import numpy as np
from tools.data import generate_texture_dicts_full

print("=" * 60)
print("TESTE: generate_texture_dicts_full")
print("=" * 60)

# Simular dados de features de imagens completas (1D - sem segmentação)
categories = ["cats", "dogs", "horses", "lions"]
n_images_per_category = 5

# Features LBP (256 features)
lbp_dict = {}
for cat in categories:
    lbp_dict[cat] = {}
    for i in range(n_images_per_category):
        img_name = f"{cat}_{i:02d}"
        # Features 1D (não-segmentadas)
        lbp_dict[cat][img_name] = np.random.rand(256)

# Features GLCM (120 features)
glcm_dict = {}
for cat in categories:
    glcm_dict[cat] = {}
    for i in range(n_images_per_category):
        img_name = f"{cat}_{i:02d}"
        # Features 1D (não-segmentadas)
        glcm_dict[cat][img_name] = np.random.rand(120)

# Features LPQ (256 features)
lpq_dict = {}
for cat in categories:
    lpq_dict[cat] = {}
    for i in range(n_images_per_category):
        img_name = f"{cat}_{i:02d}"
        # Features 1D (não-segmentadas)
        lpq_dict[cat][img_name] = np.random.rand(256)

print(f"Categorias: {categories}")
print(f"Imagens por categoria: {n_images_per_category}")
print(f"Total de imagens: {len(categories) * n_images_per_category}")
print()

# Testar generate_texture_dicts_full
print("Gerando combinações de features...")
(combined_sets, labels, true_images_labels) = generate_texture_dicts_full(
    categories, lbp_dict, glcm_dict, lpq_dict
)

print(f"\n✅ Geradas {len(combined_sets)} combinações de features")
print()

# Validar estrutura
set_names = [
    "LBP",
    "GLCM", 
    "LPQ",
    "LBP+GLCM",
    "LBP+LPQ",
    "GLCM+LPQ",
    "LBP+GLCM+LPQ",
]

expected_feature_sizes = [
    256,      # LBP
    120,      # GLCM
    256,      # LPQ
    256+120,  # LBP+GLCM
    256+256,  # LBP+LPQ
    120+256,  # GLCM+LPQ
    256+120+256,  # LBP+GLCM+LPQ
]

print("=" * 60)
print("VALIDAÇÃO DOS RESULTADOS")
print("=" * 60)

all_correct = True

for i, (combined_set, expected_size) in enumerate(zip(combined_sets, expected_feature_sizes)):
    set_name = set_names[i]
    
    # Pegar primeira imagem para validar estrutura
    first_img = next(iter(combined_set.keys()))
    features = combined_set[first_img]
    
    # Verificar se é 1D
    if len(features.shape) != 1:
        print(f"❌ {set_name}: Esperado features 1D, mas obteve shape {features.shape}")
        all_correct = False
        continue
    
    # Verificar tamanho
    actual_size = features.shape[0]
    if actual_size != expected_size:
        print(f"❌ {set_name}: Esperado {expected_size} features, mas obteve {actual_size}")
        all_correct = False
    else:
        print(f"✓ {set_name}: {actual_size} features (1D) - Correto!")

print()

# Validar labels
if len(labels) == len(categories) * n_images_per_category:
    print(f"✓ Número correto de labels: {len(labels)}")
else:
    print(f"❌ Labels incorretos: esperado {len(categories) * n_images_per_category}, obteve {len(labels)}")
    all_correct = False

# Validar true_images_labels
if len(true_images_labels) == len(categories) * n_images_per_category:
    print(f"✓ Número correto de true_images_labels: {len(true_images_labels)}")
else:
    print(f"❌ true_images_labels incorretos")
    all_correct = False

print()
print("=" * 60)
if all_correct:
    print("🎉 TODOS OS TESTES PASSARAM COM SUCESSO!")
else:
    print("❌ ALGUNS TESTES FALHARAM")
print("=" * 60)

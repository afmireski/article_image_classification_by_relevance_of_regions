"""
Teste simples para validar prepare_full_image_sets_from_segmented_folds
"""
import numpy as np
import sys
sys.path.append('tools')

from tools.data import prepare_full_image_sets_from_segmented_folds, _extract_features_and_labels

def create_test_data():
    """Cria dados de teste simulados"""
    
    # Simular features segmentadas (4 segmentos por imagem, 10 features por segmento)
    segmented_class = {
        'cat1': np.random.rand(4, 10),
        'cat2': np.random.rand(4, 10),
    }
    segmented_no_class = {
        'dog1': np.random.rand(4, 10),
        'dog2': np.random.rand(4, 10),
    }
    
    # Simular features completas (1 feature vector de 20 dimensões por imagem)
    full_class = {
        'cat1': np.random.rand(20),
        'cat2': np.random.rand(20),
    }
    full_no_class = {
        'dog1': np.random.rand(20),
        'dog2': np.random.rand(20),
    }
    
    # True map (0 = classe positiva, 1 = classe negativa)
    true_map = {
        'cat1': 0,
        'cat2': 0,
        'dog1': 1,
        'dog2': 1,
    }
    
    # Criar SpecialistSet segmentado
    segmented_specialist_set = (
        (segmented_class, 2),
        (segmented_no_class, 2),
        true_map
    )
    
    # Criar SpecialistSet completo
    full_specialist_set = (
        (full_class, 2),
        (full_no_class, 2),
        true_map
    )
    
    return segmented_specialist_set, full_specialist_set


def create_simple_segmented_folds():
    """
    Cria manualmente uma fold segmentada simulada para teste.
    Simula o output de prepare_sets_for_classification
    """
    
    # Simular dados de treino (cat1, dog1)
    train_class_dict = {'cat1': np.random.rand(4, 10)}
    train_no_class_dict = {'dog1': np.random.rand(4, 10)}
    train_true_map = {'cat1': 0, 'dog1': 1}
    
    train_data = _extract_features_and_labels(
        train_class_dict,
        train_no_class_dict,
        train_true_map
    )
    
    # Simular dados de teste (cat2, dog2)
    test_class_dict = {'cat2': np.random.rand(4, 10)}
    test_no_class_dict = {'dog2': np.random.rand(4, 10)}
    test_true_map = {'cat2': 0, 'dog2': 1}
    
    test_data = _extract_features_and_labels(
        test_class_dict,
        test_no_class_dict,
        test_true_map
    )
    
    # Uma fold = (train_data, test_data)
    fold = (train_data, test_data)
    
    # Um dataset = lista de folds
    classification_dataset = [fold]
    
    # PreparedSetsForClassification = lista de datasets
    segmented_prepared_sets = [classification_dataset]
    
    return segmented_prepared_sets


def test_prepare_full_image_folds():
    """Testa a função prepare_full_image_sets_from_segmented_folds"""
    
    print("=" * 60)
    print("TESTE: prepare_full_image_sets_from_segmented_folds")
    print("=" * 60)
    
    # Criar dados de teste
    _, full_specialist_set = create_test_data()
    segmented_prepared_sets = create_simple_segmented_folds()
    
    print("\n📊 Dados de entrada criados:")
    print(f"  - Segmented prepared sets: {len(segmented_prepared_sets)} conjuntos")
    print(f"  - Full image sets: 1 specialist set")
    
    # Chamar a função
    print("\n🔄 Executando prepare_full_image_sets_from_segmented_folds...\n")
    full_prepared_sets = prepare_full_image_sets_from_segmented_folds(
        segmented_prepared_sets=segmented_prepared_sets,
        full_image_sets=[full_specialist_set],
        verbose=True
    )
    
    # Validações
    print("\n" + "=" * 60)
    print("VALIDAÇÕES")
    print("=" * 60)
    
    # 1. Número de conjuntos deve ser igual
    assert len(full_prepared_sets) == len(segmented_prepared_sets), \
        f"❌ Número de conjuntos não coincide: {len(full_prepared_sets)} vs {len(segmented_prepared_sets)}"
    print("✅ Número de conjuntos correto")
    
    # 2. Número de folds deve ser igual
    assert len(full_prepared_sets[0]) == len(segmented_prepared_sets[0]), \
        f"❌ Número de folds não coincide: {len(full_prepared_sets[0])} vs {len(segmented_prepared_sets[0])}"
    print("✅ Número de folds correto")
    
    # 3. Verificar estrutura da fold
    train_data_full, test_data_full = full_prepared_sets[0][0]
    train_data_seg, test_data_seg = segmented_prepared_sets[0][0]
    
    # 4. Número de imagens deve ser igual (mas não número de samples!)
    train_images_full = set(train_data_full[2].values())
    train_images_seg = set(train_data_seg[2].values())
    
    assert train_images_full == train_images_seg, \
        f"❌ Imagens de treino não coincidem: {train_images_full} vs {train_images_seg}"
    print(f"✅ Mesmas imagens de treino: {train_images_full}")
    
    test_images_full = set(test_data_full[2].values())
    test_images_seg = set(test_data_seg[2].values())
    
    assert test_images_full == test_images_seg, \
        f"❌ Imagens de teste não coincidem: {test_images_full} vs {test_images_seg}"
    print(f"✅ Mesmas imagens de teste: {test_images_full}")
    
    # 5. Verificar que imagens completas têm 1 linha por imagem (não segmentadas)
    n_train_images = len(train_images_full)
    n_test_images = len(test_images_full)
    
    assert train_data_full[0].shape[0] == n_train_images, \
        f"❌ Treino deveria ter {n_train_images} linhas, tem {train_data_full[0].shape[0]}"
    print(f"✅ Treino tem 1 linha por imagem: {train_data_full[0].shape}")
    
    assert test_data_full[0].shape[0] == n_test_images, \
        f"❌ Teste deveria ter {n_test_images} linhas, tem {test_data_full[0].shape[0]}"
    print(f"✅ Teste tem 1 linha por imagem: {test_data_full[0].shape}")
    
    # 6. Verificar que features têm dimensão correta (20 no nosso teste)
    assert train_data_full[0].shape[1] == 20, \
        f"❌ Features deveriam ter 20 dimensões, têm {train_data_full[0].shape[1]}"
    print(f"✅ Dimensão de features correta: {train_data_full[0].shape[1]}")
    
    # 7. Verificar labels
    assert len(train_data_full[1]) == n_train_images, \
        f"❌ Número de labels de treino incorreto"
    print(f"✅ Labels de treino corretos: {len(train_data_full[1])} labels")
    
    print("\n" + "=" * 60)
    print("🎉 TODOS OS TESTES PASSARAM!")
    print("=" * 60)


if __name__ == "__main__":
    test_prepare_full_image_folds()

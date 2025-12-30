"""
Script de teste para validar as funções de classificação padrão em classical.py
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classical import (
    extract_standard_probabilities,
    predict_standard_labels,
    save_standard_confusion_matrix,
    export_standard_results_to_csv,
    standard_technique,
    timed_standard_technique,
)

from mytypes import (
    MulticlassClassificationDataset,
    PredictResults,
    StandardModelResults,
)


def create_mock_dataset():
    """Cria um dataset mock para testes."""
    np.random.seed(42)
    
    # 4 classes, 20 imagens por classe (80 total)
    n_samples = 80
    n_features = 256  # Simulando LBP
    n_classes = 4
    
    # Gera features sintéticas
    X = np.random.rand(n_samples, n_features)
    y = np.repeat(np.arange(n_classes), 20)  # [0,0,...,1,1,...,2,2,...,3,3,...]
    
    # IDs das imagens
    image_ids = [f"img_{i}" for i in range(n_samples)]
    
    # Cria 5 folds manualmente
    fold_size = n_samples // 5
    folds = []
    
    for fold_idx in range(5):
        # Índices de teste para este fold
        test_start = fold_idx * fold_size
        test_end = (fold_idx + 1) * fold_size
        test_indices = list(range(test_start, test_end))
        
        # Índices de treino (todo o resto)
        train_indices = [i for i in range(n_samples) if i not in test_indices]
        
        # Cria train e test sets
        X_train = X[train_indices]
        y_train = y[train_indices]
        train_images = [image_ids[i] for i in train_indices]
        
        X_test = X[test_indices]
        y_test = y[test_indices]
        test_images = [image_ids[i] for i in test_indices]
        
        # Adiciona ao dataset no formato correto
        train_set = (X_train, y_train, train_images)
        test_set = (X_test, y_test, test_images)
        folds.append((train_set, test_set))
    
    # Cria true_labels (mapeamento img_id -> label)
    true_labels = {img_id: label for img_id, label in zip(image_ids, y)}
    
    return folds, true_labels


def create_mock_model():
    """Cria um modelo mock simples para testes."""
    knn = KNeighborsClassifier()
    parameters = {"knn__n_neighbors": [3, 5]}
    pipe = Pipeline([("scaler", StandardScaler()), ("knn", knn)])
    model = GridSearchCV(pipe, parameters, cv=2, n_jobs=-1)
    return model


def test_extract_standard_probabilities():
    """Testa a extração de probabilidades."""
    print("\n" + "=" * 60)
    print("TESTE 1: extract_standard_probabilities()")
    print("=" * 60)
    
    folds, _ = create_mock_dataset()
    model = create_mock_model()
    
    probabilities = extract_standard_probabilities(
        base_model=model,
        folded_dataset=folds,
        title="Test Model",
    )
    
    # Validações
    assert isinstance(probabilities, dict), "❌ Deve retornar dicionário"
    assert len(probabilities) == 80, f"❌ Esperado 80 imagens, obteve {len(probabilities)}"
    
    # Verifica estrutura das probabilidades
    for img_id, probs in probabilities.items():
        assert isinstance(probs, np.ndarray), f"❌ Probabilidades de {img_id} devem ser numpy array"
        assert len(probs) == 4, f"❌ Esperado 4 probabilidades (classes), obteve {len(probs)}"
        assert np.isclose(probs.sum(), 1.0, atol=1e-5), f"❌ Probabilidades devem somar 1.0, soma={probs.sum()}"
    
    print(f"✅ Teste passou!")
    print(f"   - {len(probabilities)} imagens processadas")
    print(f"   - Exemplo de probabilidades: {list(probabilities.values())[0]}")
    print(f"   - Soma das probabilidades: {list(probabilities.values())[0].sum():.6f}")
    
    return probabilities


def test_predict_standard_labels(probabilities):
    """Testa a predição de labels."""
    print("\n" + "=" * 60)
    print("TESTE 2: predict_standard_labels()")
    print("=" * 60)
    
    predicted_labels = predict_standard_labels(probabilities)
    
    # Validações
    assert isinstance(predicted_labels, dict), "❌ Deve retornar dicionário"
    assert len(predicted_labels) == 80, f"❌ Esperado 80 labels, obteve {len(predicted_labels)}"
    
    # Verifica que todos os labels estão no intervalo [0, 3]
    for img_id, label in predicted_labels.items():
        assert isinstance(label, int), f"❌ Label de {img_id} deve ser int"
        assert 0 <= label <= 3, f"❌ Label deve estar entre 0 e 3, obteve {label}"
    
    print(f"✅ Teste passou!")
    print(f"   - {len(predicted_labels)} labels preditos")
    print(f"   - Distribuição de classes: {np.bincount(list(predicted_labels.values()))}")
    
    return predicted_labels


def test_save_standard_confusion_matrix(predicted_labels, true_labels):
    """Testa o salvamento da matriz de confusão."""
    print("\n" + "=" * 60)
    print("TESTE 3: save_standard_confusion_matrix()")
    print("=" * 60)
    
    # Converte para listas
    true_y = [true_labels[img_id] for img_id in sorted(true_labels.keys())]
    predicted_y = [predicted_labels[img_id] for img_id in sorted(predicted_labels.keys())]
    
    experiment_dir = "results/standard/test_experiment"
    
    filepath = save_standard_confusion_matrix(
        true_labels=true_y,
        predicted_labels=predicted_y,
        model_name="TEST_MODEL",
        experiment_dir=experiment_dir,
        cmap='Blues'
    )
    
    # Validações
    import os
    assert filepath != "", "❌ Filepath não deve ser vazio"
    assert os.path.exists(filepath), f"❌ Arquivo não foi criado: {filepath}"
    assert filepath.endswith(".jpg"), "❌ Arquivo deve ser .jpg"
    
    print(f"✅ Teste passou!")
    print(f"   - Arquivo salvo: {filepath}")
    
    return filepath


def test_export_standard_results_to_csv(probabilities, predicted_labels, true_labels):
    """Testa a exportação para CSV."""
    print("\n" + "=" * 60)
    print("TESTE 4: export_standard_results_to_csv()")
    print("=" * 60)
    
    # Cria métricas mock
    from tools.relevance import compute_metrics
    labels_list, model_metrics = compute_metrics(true_labels, predicted_labels)
    
    # Cria standard_results
    standard_results = (probabilities, predicted_labels, labels_list, model_metrics)
    
    experiment_dir = "results/standard/test_experiment"
    
    filepath = export_standard_results_to_csv(
        standard_results=standard_results,
        true_labels=true_labels,
        model_name="TEST_MODEL",
        experiment_dir=experiment_dir,
    )
    
    # Validações
    import os
    assert os.path.exists(filepath), f"❌ Arquivo CSV não foi criado: {filepath}"
    assert filepath.endswith(".csv"), "❌ Arquivo deve ser .csv"
    
    # Verifica conteúdo do CSV
    import csv
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 80, f"❌ Esperado 80 linhas, obteve {len(rows)}"
        
        # Verifica colunas
        expected_columns = [
            "nome_imagem", "label_predito", "label_real",
            "acuracia_global", "f1_global", "recall_global", "precision_global",
            "probabilidades"
        ]
        assert rows[0].keys() == set(expected_columns), "❌ Colunas incorretas"
    
    print(f"✅ Teste passou!")
    print(f"   - Arquivo salvo: {filepath}")
    print(f"   - {len(rows)} linhas escritas")
    print(f"   - Colunas: {list(rows[0].keys())}")
    
    return filepath


def test_standard_technique():
    """Testa a função principal standard_technique."""
    print("\n" + "=" * 60)
    print("TESTE 5: standard_technique() - FUNÇÃO PRINCIPAL")
    print("=" * 60)
    
    folds, true_labels = create_mock_dataset()
    model = create_mock_model()
    class_names = ["class_0", "class_1", "class_2", "class_3"]
    
    results = standard_technique(
        base_model=model,
        folded_dataset=folds,
        class_names=class_names,
        true_labels=true_labels,
        model_name="FULL_TEST",
        k_folds=5,
        experiment_dir="results/standard/test_full_technique",
    )
    
    # Desempacota resultados
    probabilities, predicted_labels, labels_list, model_metrics = results
    
    # Validações
    assert isinstance(probabilities, dict), "❌ Probabilidades deve ser dict"
    assert isinstance(predicted_labels, dict), "❌ Predicted labels deve ser dict"
    assert isinstance(labels_list, tuple), "❌ Labels list deve ser tupla"
    assert isinstance(model_metrics, tuple), "❌ Model metrics deve ser tupla"
    assert len(model_metrics) == 4, "❌ Deve ter 4 métricas"
    
    accuracy, f1, recall, precision = model_metrics
    assert 0 <= accuracy <= 1, f"❌ Acurácia inválida: {accuracy}"
    assert 0 <= f1 <= 1, f"❌ F1 inválido: {f1}"
    assert 0 <= recall <= 1, f"❌ Recall inválido: {recall}"
    assert 0 <= precision <= 1, f"❌ Precision inválido: {precision}"
    
    print(f"✅ Teste passou!")
    print(f"   - Acurácia: {accuracy:.4f}")
    print(f"   - F1-Score: {f1:.4f}")
    print(f"   - Recall: {recall:.4f}")
    print(f"   - Precision: {precision:.4f}")
    
    return results


def test_timed_standard_technique():
    """Testa o wrapper com timer."""
    print("\n" + "=" * 60)
    print("TESTE 6: timed_standard_technique() - WRAPPER COM TIMER")
    print("=" * 60)
    
    folds, true_labels = create_mock_dataset()
    model = create_mock_model()
    class_names = ["class_0", "class_1", "class_2", "class_3"]
    
    results = timed_standard_technique(
        base_model=model,
        folded_dataset=folds,
        class_names=class_names,
        true_labels=true_labels,
        model_name="TIMED_TEST",
        k_folds=5,
        experiment_dir="results/standard/test_timed_technique",
    )
    
    # Validações (mesmo que standard_technique)
    assert results is not None, "❌ Resultados não devem ser None"
    assert len(results) == 4, "❌ Deve retornar tupla com 4 elementos"
    
    print(f"✅ Teste passou! (com medição de tempo)")
    
    return results


def run_all_tests():
    """Executa todos os testes."""
    print("\n" + "🧪" * 30)
    print("EXECUTANDO SUITE COMPLETA DE TESTES - classical.py")
    print("🧪" * 30)
    
    try:
        # Teste 1
        probabilities = test_extract_standard_probabilities()
        
        # Teste 2
        predicted_labels = test_predict_standard_labels(probabilities)
        
        # Teste 3
        _, true_labels = create_mock_dataset()
        test_save_standard_confusion_matrix(predicted_labels, true_labels)
        
        # Teste 4
        test_export_standard_results_to_csv(probabilities, predicted_labels, true_labels)
        
        # Teste 5
        test_standard_technique()
        
        # Teste 6
        test_timed_standard_technique()
        
        print("\n" + "🎉" * 30)
        print("TODOS OS TESTES PASSARAM COM SUCESSO!")
        print("🎉" * 30)
        print("\n✅ Funções validadas:")
        print("   1. extract_standard_probabilities()")
        print("   2. predict_standard_labels()")
        print("   3. save_standard_confusion_matrix()")
        print("   4. export_standard_results_to_csv()")
        print("   5. standard_technique()")
        print("   6. timed_standard_technique()")
        print("\n✅ Arquivos de teste criados em:")
        print("   - results/standard/test_experiment/")
        print("   - results/standard/test_full_technique/")
        print("   - results/standard/test_timed_technique/")
        
    except AssertionError as e:
        print(f"\n❌ TESTE FALHOU: {str(e)}")
        raise
    except Exception as e:
        print(f"\n❌ ERRO INESPERADO: {str(e)}")
        raise


if __name__ == "__main__":
    run_all_tests()

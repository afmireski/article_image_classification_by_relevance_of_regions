import csv

import numpy as np

from sum_of_probabilities import export_to_csv


def test_export_to_csv_also_exports_sumprob_metrics(tmp_path):
    # Arrange: cria um arquivo de métricas de relevância com linhas de especialistas
    fieldnames = [
        "model",
        "accuracy (%)",
        "accuracy_std (+- %)",
        "f1_score (%)",
        "f1_score_std (+- %)",
        "recall (%)",
        "recall_std (+- %)",
        "precision (%)",
        "precision_std (+- %)",
    ]

    relevance_dir = tmp_path / "relevance" / "csv_exports"
    relevance_dir.mkdir(parents=True, exist_ok=True)
    relevance_metrics = relevance_dir / "test_model_metrics.csv"

    with open(relevance_metrics, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Global (relevância) - deve ser ignorado pelo SumProb
        writer.writerow(
            {
                "model": "test_model_relevance_global",
                "accuracy (%)": "10.0000",
                "accuracy_std (+- %)": "###",
                "f1_score (%)": "11.0000",
                "f1_score_std (+- %)": "###",
                "recall (%)": "12.0000",
                "recall_std (+- %)": "###",
                "precision (%)": "13.0000",
                "precision_std (+- %)": "###",
            }
        )
        # Especialistas (mean + folds) - devem ser copiados e retaggeados
        writer.writerow(
            {
                "model": "test_model_relevance_specialist_0_mean",
                "accuracy (%)": "90.0000",
                "accuracy_std (+- %)": "1.0000",
                "f1_score (%)": "91.0000",
                "f1_score_std (+- %)": "1.1000",
                "recall (%)": "92.0000",
                "recall_std (+- %)": "1.2000",
                "precision (%)": "93.0000",
                "precision_std (+- %)": "1.3000",
            }
        )
        writer.writerow(
            {
                "model": "test_model_relevance_specialist_0_fold1",
                "accuracy (%)": "80.0000",
                "accuracy_std (+- %)": "###",
                "f1_score (%)": "81.0000",
                "f1_score_std (+- %)": "###",
                "recall (%)": "82.0000",
                "recall_std (+- %)": "###",
                "precision (%)": "83.0000",
                "precision_std (+- %)": "###",
            }
        )

    # Arrange: dados mínimos para export_to_csv
    predicted_labels = {"img1": 0, "img2": 1}
    true_labels = {"img1": 0, "img2": 0}

    probabilities = {
        "img1": np.array([[0.9, 0.1], [0.8, 0.2]]),
        "img2": np.array([[0.2, 0.8], [0.3, 0.7]]),
    }

    probability_sums = {img_id: np.sum(probs, axis=0) for img_id, probs in probabilities.items()}

    # Global SumProb (0.5 -> 50.0000 no CSV de métricas)
    model_metrics = (0.5, 0.5, 0.5, 0.5)

    out_dir = tmp_path / "sum" / "csv_exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "test_model_results.csv"

    # Act
    csv_path = export_to_csv(
        predicted_labels=predicted_labels,
        true_labels=true_labels,
        probability_sums=probability_sums,
        probabilities=probabilities,
        model_metrics=model_metrics,
        output_filepath=str(out_csv),
        relevance_metrics_filepath=str(relevance_metrics),
    )

    # Assert: CSV principal criado
    assert csv_path.endswith(".csv")

    # Assert: CSV de métricas SumProb criado
    sumprob_metrics = out_dir / "test_model_sumprob_metrics.csv"
    assert sumprob_metrics.exists(), f"Arquivo de métricas não foi criado: {sumprob_metrics}"

    with open(sumprob_metrics, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # 1 global (SumProb) + 2 linhas copiadas (mean + fold1)
    assert len(rows) == 3

    assert rows[0]["model"] == "test_model_sumprob_global"
    assert rows[0]["accuracy (%)"] == "50.0000"
    assert rows[0]["accuracy_std (+- %)"] == "###"

    assert rows[1]["model"] == "test_model_sumprob_specialist_0_mean"
    assert rows[2]["model"] == "test_model_sumprob_specialist_0_fold1"

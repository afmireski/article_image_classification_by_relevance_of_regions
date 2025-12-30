import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

models = ["LBP", "GLCM", "LPQ", "LBP+GLCM", "LBP+LPQ", "GLCM+LPQ", "LBP+GLCM+LPQ"]

# Estrutura:
## Linha 0 = F1
## Linha 1 = Recall
## Linha 2 = Precision
## Linha 3 = Accuracy
## Linha 4 = Tempo (segundos)

baseline_knn = np.array([
    [75.25, 73.58, 77.83, 77.50, 79.58, 78.22, 81.07],  # F1-Score
    [75.15, 73.45, 77.80, 77.40, 79.50, 78.20, 81.00],  # Recall
    [75.40, 73.91, 78.00, 77.89, 79.74, 78.60, 81.43],  # Precision
    [75.15, 73.45, 77.80, 77.40, 79.50, 78.20, 81.00],  # Accuracy
    [1.23, 0.92, 7.33, 6.35, 8.01, 8.11, 8.65],  # Tempo (segundos)
])
relevance_knn = np.array([
    [83.58, 81.68, 85.57, 85.23, 87.16, 86.46, 88.26],  # F1-Score
    [83.65, 81.70, 85.60, 85.25, 87.20, 86.45, 88.25],  # Recall
    [83.74, 81.95, 85.75, 85.57, 87.32, 86.75, 88.51],  # Precision
    [83.65, 81.70, 85.60, 85.25, 87.20, 86.45, 88.25],  # Accuracy
    [62, 70, 223, 116, 286, 297, 347],  # Tempo (segundos): 1m02s, 1m10s, 3m43s, 1m56s, 4m46s, 4m57s, 5m47s
])

baseline_svm = np.array([
    [84.91, 85.12, 89.06, 87.71, 90.66, 90.46, 90.35],  # F1-Score
    [84.90, 85.10, 89.05, 87.70, 90.65, 90.45, 90.35],  # Recall
    [84.93, 85.20, 89.09, 87.74, 90.67, 90.51, 90.37],  # Precision
    [84.90, 85.10, 89.05, 87.70, 90.65, 90.45, 90.35],  # Accuracy
])
relevance_svm = np.array([
    [90.00, 88.04, 92.75, 90.65, 94.05, 93.70, 94.40],  # F1-Score
    [90.00, 88.05, 92.75, 90.65, 94.05, 93.70, 94.40],  # Recall
    [90.02, 88.04, 92.77, 90.66, 94.06, 93.71, 94.40],  # Precision
    [90.00, 88.05, 92.75, 90.65, 94.05, 93.70, 94.40],
])

sumproba_knn = np.array([
    [85.03, 82.59, 86.78, 86.35, 87.95, 86.75, 88.70],  # F1-Score
    [85.05, 82.60, 86.80, 86.35, 87.94, 86.75, 88.71],  # Recall
    [85.05, 82.60, 86.80, 86.35, 87.95, 86.75, 88.70],  # Precision
    [85.09, 82.81, 86.88, 86.69, 87.98, 86.16, 89.01],  # Accuracy
])
sumproba_svm = np.array([
    [90.35, 88.59, 93.15, 91.00, 94.15, 93.65, 94.30],  # F1-Score
    [90.35, 88.60, 93.15, 91.00, 94.15, 93.65, 94.30],  # Recall
    [90.35, 88.60, 93.15, 91.00, 94.15, 93.65, 94.30],  # Precision
    [90.36, 88.61, 93.15, 91.01, 94.15, 93.66, 94.30],  # Accuracy
])

def generate_comparison_graph(models, baseline_data, relevance_data, metrics, approachs, output_path="graphs/grapx_x.png"):
    # Data Preparation for Seaborn
    data = []

    for i, metric in enumerate(metrics):
        for j, model in enumerate(models):
            # Baseline
            data.append({
                "Modelo": model,
                "Metrica": metric,
                "Abordagem": approachs[0],
                "Valor": baseline_data[i, j]
            })
            # Relevance
            data.append({
                "Modelo": model,
                "Metrica": metric,
                "Abordagem": approachs[1],
                "Valor": relevance_data[i, j]
            })

    df = pd.DataFrame(data)

    # Plotting
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        sns.barplot(
            data=df[df["Metrica"] == metric],
            x="Modelo",
            y="Valor",
            hue="Abordagem",
            ax=ax,
            palette="muted"
        )
        ax.set_title(f"Comparação de {metric}", fontsize=18)
        ax.set_xlabel("Modelo", fontsize=16)
        
        # Adjust y-axis label and limits based on metric type
        if "Tempo" in metric:
            ax.set_ylabel(metric, fontsize=16)
            ax.set_ylim(0, max(baseline_data[i].max(), relevance_data[i].max()) * 1.1)
        else:
            ax.set_ylabel("Valor (%)", fontsize=16)
            ax.set_ylim(70, 100)
        
        ax.tick_params(axis='x', rotation=45)
        ax.legend(loc='lower right', fontsize=16, frameon=True, shadow=True)
    
    # Hide the last subplot (6th position) since we only have 5 metrics
    # axes[4].set_visible(False)

    plt.tight_layout()

    # Save the plot
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(output_path)
    print(f"Graph saved to {output_path}")

if __name__ == "__main__":
    metrics = ["F1-Score", "Recall", "Precisão", "Acurácia"]
    generate_comparison_graph(models, baseline_knn, relevance_knn, metrics, ["Baseline", "Relevância"], "graphs/knn_baseline_vs_relevance.png")

    generate_comparison_graph(models, baseline_svm, relevance_svm, metrics, ["Baseline", "Relevância"], "graphs/svm_baseline_vs_relevance.png")

    generate_comparison_graph(models, relevance_knn, sumproba_knn, metrics, ["Relevância", "Soma Probalidades"], "graphs/knn_relevance_vs_sumproba.png")

    generate_comparison_graph(models, relevance_svm, sumproba_svm, metrics, ["Relevância", "Soma Probalidades"], "graphs/svm_relevance_vs_sumproba.png")




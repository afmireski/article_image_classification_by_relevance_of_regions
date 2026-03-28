#!/usr/bin/env python3
"""Cross-validation Sum-of-Probabilities (SumProb) over relevance cross fold results.

This script consumes the per-fold results exported by the relevance cross pipeline
(`csv_exports/<base>/<base>_foldN_results.csv`), applies SumProb on the
`probabilidades` column, and produces:

- Per-fold SumProb results CSVs
- A global out-of-fold (OOF) SumProb results CSV
- A single metrics CSV (global + mean/std + folds) + copied specialist rows from relevance
- A global confusion matrix PNG (OOF)

Output layout (inside the experiment directory by default):
    <experiment_dir>/sum_cross/
        csv_exports/<base>/
            <base>_sumprob_results.csv
            <base>_sumprob_foldN_results.csv
            <base>_sumprob_metrics.csv
        confusion_matrixs/
            <base>_sumprob_confusion_matrix.png

We intentionally do NOT modify `sum_of_probabilities.py`; this is a separate cross workflow.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from tools.relevance import compute_metrics

# Reuse the same confusion-matrix drawing helper for consistent visuals.
from sum_of_probabilities import generate_confusion_matrix as _generate_confusion_matrix_png


_METRICS_FIELDNAMES: List[str] = [
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


@dataclass(frozen=True)
class FoldRun:
    fold_idx: int  # 1-based
    relevance_fold_results_csv: str


@dataclass
class SumProbRunOutputs:
    base_name: str
    output_dir: str
    results_global_csv: str
    metrics_csv: str
    confusion_matrix_png: Optional[str]
    results_by_fold_csv: Dict[int, str]


def _fmt_num(x: float) -> str:
    return f"{x:.4f}"


def _to_percent(x: float) -> float:
    return float(x) * 100.0


def _normalize_base_name(base_name: str) -> str:
    base = base_name.lower().replace(" ", "_").replace("-", "_").replace("+", "_")
    base = "".join(c for c in base if c.isalnum() or c == "_")
    return base


def _parse_probabilities(value: str, img_id: str) -> np.ndarray:
    try:
        prob_list = json.loads(value)
        arr = np.array(prob_list, dtype=float)
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"❌ Erro ao desserializar probabilidades de {img_id}: {e}") from e

    if arr.ndim != 2:
        raise ValueError(
            f"❌ probabilidades de {img_id} deveria ser 2D (pedaços x especialistas), mas veio {arr.shape}"
        )

    return arr


def read_fold_results_csv(filepath: str) -> Dict[str, Dict]:
    """Reads a relevance fold results CSV.

    Expected columns: nome_imagem, label_real, label_predito, probabilidades.
    The `probabilidades` field must be JSON representing a 2D matrix.

    Returns:
        dict img_id -> {label_real:int, probabilidades:np.ndarray}
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Arquivo não encontrado: {filepath}")

    data: Dict[str, Dict] = {}
    with open(filepath, "r", encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        required_columns = ["nome_imagem", "label_real", "label_predito", "probabilidades"]
        if reader.fieldnames is None:
            raise ValueError(f"❌ CSV vazio/malformado: {filepath}")

        for col in required_columns:
            if col not in reader.fieldnames:
                raise ValueError(f"❌ Coluna obrigatória ausente no CSV: {col}")

        for row in reader:
            img_id = (row.get("nome_imagem") or "").strip()
            if not img_id:
                continue

            label_real = int(row["label_real"])
            probs = _parse_probabilities(row["probabilidades"], img_id=img_id)

            data[img_id] = {
                "label_real": label_real,
                "probabilidades": probs,
            }

    return data


def calculate_probability_sums(probabilities: np.ndarray) -> np.ndarray:
    return np.sum(probabilities, axis=0)


def predict_label_by_sum(probability_sums: np.ndarray) -> int:
    return int(np.argmax(probability_sums))


def run_sumprob_on_fold(data: Dict[str, Dict]) -> Tuple[Dict[str, int], Dict[str, np.ndarray], Dict[str, int], Dict[str, np.ndarray]]:
    """Runs SumProb for one fold.

    Returns:
        predicted_labels, probability_sums, true_labels, probabilities
    """
    predicted_labels: Dict[str, int] = {}
    probability_sums: Dict[str, np.ndarray] = {}
    true_labels: Dict[str, int] = {}
    probabilities: Dict[str, np.ndarray] = {}

    for img_id, img_data in data.items():
        probs = img_data["probabilidades"]
        label_real = int(img_data["label_real"])

        sums = calculate_probability_sums(probs)
        pred = predict_label_by_sum(sums)

        predicted_labels[img_id] = pred
        probability_sums[img_id] = sums
        true_labels[img_id] = label_real
        probabilities[img_id] = probs

    return predicted_labels, probability_sums, true_labels, probabilities


def export_sumprob_results_csv(
    *,
    predicted_labels: Dict[str, int],
    true_labels: Dict[str, int],
    probability_sums: Dict[str, np.ndarray],
    probabilities: Dict[str, np.ndarray],
    model_metrics: Tuple[float, float, float, float],
    output_filepath: str,
) -> str:
    accuracy, f1, recall, precision = model_metrics
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    def serialize_array(arr: Optional[np.ndarray]) -> str:
        if arr is None:
            return "[]"
        arr_list = np.asarray(arr).tolist()

        def round_nested(obj):
            if isinstance(obj, list):
                return [round_nested(item) for item in obj]
            if isinstance(obj, float):
                return round(obj, 4)
            return obj

        return json.dumps(round_nested(arr_list))

    with open(output_filepath, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "nome_imagem",
            "label_predito",
            "label_real",
            "acuracia_global",
            "f1_global",
            "recall_global",
            "precision_global",
            "probabilidades",
            "somas_probabilidades",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for img_id in sorted(predicted_labels.keys()):
            writer.writerow(
                {
                    "nome_imagem": img_id,
                    "label_predito": predicted_labels[img_id],
                    "label_real": true_labels.get(img_id, -1),
                    "acuracia_global": round(accuracy, 4),
                    "f1_global": round(f1, 4),
                    "recall_global": round(recall, 4),
                    "precision_global": round(precision, 4),
                    "probabilidades": serialize_array(probabilities.get(img_id)),
                    "somas_probabilidades": serialize_array(probability_sums.get(img_id)),
                }
            )

    return output_filepath


def export_sumprob_cross_metrics_csv(
    *,
    base_name: str,
    oof_global_metrics: Tuple[float, float, float, float],
    fold_metrics: Dict[int, Tuple[float, float, float, float]],
    relevance_metrics_filepath: Optional[str],
    output_dir: str,
) -> str:
    """Exports a single metrics CSV for SumProb cross.

    Includes:
    - global OOF row
    - mean/std + fold rows for SumProb across folds
    - (optional) copied specialist rows from relevance metrics (supports old and new tag formats)
    """

    os.makedirs(output_dir, exist_ok=True)

    sumprob_tag_prefix = f"{base_name}_sumprob"
    metrics_filepath = os.path.join(output_dir, f"{sumprob_tag_prefix}_metrics.csv")

    accuracy_global, f1_global, recall_global, precision_global = oof_global_metrics

    # Prepare folds list (stable order)
    ordered_fold_idxs = sorted(fold_metrics.keys())
    fold_values = np.array([fold_metrics[i] for i in ordered_fold_idxs], dtype=float) if ordered_fold_idxs else np.zeros((0, 4))

    mean_vals = tuple(np.mean(fold_values, axis=0).tolist()) if len(ordered_fold_idxs) else (0.0, 0.0, 0.0, 0.0)
    std_vals = tuple(np.std(fold_values, axis=0).tolist()) if len(ordered_fold_idxs) else (0.0, 0.0, 0.0, 0.0)

    def make_row(model_tag: str, metrics: Tuple[float, float, float, float], std: Optional[Tuple[float, float, float, float]] = None) -> Dict[str, str]:
        acc, f1, rec, prec = metrics
        acc_std, f1_std, rec_std, prec_std = (std or (None, None, None, None))

        def std_or_hash(x: Optional[float]) -> str:
            return "###" if x is None else _fmt_num(_to_percent(x))

        return {
            "model": model_tag,
            "accuracy (%)": _fmt_num(_to_percent(acc)),
            "accuracy_std (+- %)": std_or_hash(acc_std),
            "f1_score (%)": _fmt_num(_to_percent(f1)),
            "f1_score_std (+- %)": std_or_hash(f1_std),
            "recall (%)": _fmt_num(_to_percent(rec)),
            "recall_std (+- %)": std_or_hash(rec_std),
            "precision (%)": _fmt_num(_to_percent(prec)),
            "precision_std (+- %)": std_or_hash(prec_std),
        }

    rows: List[Dict[str, str]] = []

    # Global OOF
    rows.append(make_row(f"{sumprob_tag_prefix}_global", (accuracy_global, f1_global, recall_global, precision_global)))

    # CV (mean/std + folds)
    if ordered_fold_idxs:
        rows.append(make_row(f"{sumprob_tag_prefix}_mean", mean_vals, std_vals))
        for fold_idx in ordered_fold_idxs:
            rows.append(make_row(f"{sumprob_tag_prefix}_fold{fold_idx}", fold_metrics[fold_idx]))

    # Copy specialist rows from relevance metrics (optional)
    if relevance_metrics_filepath and os.path.exists(relevance_metrics_filepath):
        with open(relevance_metrics_filepath, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"❌ CSV de métricas vazio/malformado: {relevance_metrics_filepath}")

            missing_cols = [c for c in _METRICS_FIELDNAMES if c not in reader.fieldnames]
            if missing_cols:
                raise ValueError(
                    "❌ CSV de métricas da relevância não tem as colunas esperadas. "
                    f"Faltando: {missing_cols}"
                )

            for row in reader:
                model_tag = (row.get("model") or "").strip()
                if not model_tag:
                    continue

                new_model_tag: Optional[str] = None

                # Legacy relevance specialist tags
                if model_tag.startswith(f"{base_name}_relevance_specialist_"):
                    new_model_tag = model_tag.replace(f"{base_name}_relevance", sumprob_tag_prefix, 1)

                # Current relevance specialist tags (no 'relevance' in specialist prefix)
                elif model_tag.startswith(f"{base_name}_specialist_"):
                    new_model_tag = model_tag.replace(f"{base_name}_", f"{sumprob_tag_prefix}_", 1)

                if new_model_tag is None:
                    continue

                rows.append(
                    {
                        "model": new_model_tag,
                        "accuracy (%)": row["accuracy (%)"],
                        "accuracy_std (+- %)": row["accuracy_std (+- %)"] ,
                        "f1_score (%)": row["f1_score (%)"],
                        "f1_score_std (+- %)": row["f1_score_std (+- %)"] ,
                        "recall (%)": row["recall (%)"],
                        "recall_std (+- %)": row["recall_std (+- %)"] ,
                        "precision (%)": row["precision (%)"],
                        "precision_std (+- %)": row["precision_std (+- %)"] ,
                    }
                )

    with open(metrics_filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=_METRICS_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return metrics_filepath


def _discover_relevance_fold_results(experiment_dir: str) -> Dict[str, List[FoldRun]]:
    """Discovers relevance cross fold results under <experiment_dir>/csv_exports/<base>/*_foldN_results.csv."""

    csv_exports_dir = os.path.join(experiment_dir, "csv_exports")
    if not os.path.isdir(csv_exports_dir):
        raise FileNotFoundError(
            "❌ Diretório csv_exports não encontrado. "
            f"Esperado: {csv_exports_dir}"
        )

    fold_re = re.compile(r"^(?P<base>.+)_fold(?P<fold>\d+)_results\.csv$")

    by_base: Dict[str, List[FoldRun]] = {}

    for base_folder in sorted(os.listdir(csv_exports_dir)):
        base_path = os.path.join(csv_exports_dir, base_folder)
        if not os.path.isdir(base_path):
            continue

        for filename in sorted(os.listdir(base_path)):
            match = fold_re.match(filename)
            if not match:
                continue

            base = match.group("base")
            fold_idx = int(match.group("fold"))
            fullpath = os.path.join(base_path, filename)

            by_base.setdefault(base, []).append(FoldRun(fold_idx=fold_idx, relevance_fold_results_csv=fullpath))

    # Ensure stable ordering
    for base, folds in list(by_base.items()):
        by_base[base] = sorted(folds, key=lambda x: x.fold_idx)

    return by_base


def run_sumprob_cross_for_base(
    *,
    experiment_dir: str,
    base_name: str,
    folds: Sequence[FoldRun],
    output_root_dir: Optional[str] = None,
    generate_confusion: bool = True,
) -> SumProbRunOutputs:
    """Runs SumProb cross for a single model base."""

    normalized_base = _normalize_base_name(base_name)
    sum_cross_dir = output_root_dir or os.path.join(experiment_dir, "sum_cross")

    out_csv_dir = os.path.join(sum_cross_dir, "csv_exports", normalized_base)
    os.makedirs(out_csv_dir, exist_ok=True)

    relevance_metrics = os.path.join(experiment_dir, "csv_exports", normalized_base, f"{normalized_base}_metrics.csv")
    relevance_metrics_filepath = relevance_metrics if os.path.exists(relevance_metrics) else None

    # Per-fold processing
    fold_outputs: Dict[int, str] = {}
    fold_metrics: Dict[int, Tuple[float, float, float, float]] = {}

    oof_pred: Dict[str, int] = {}
    oof_true: Dict[str, int] = {}
    oof_sums: Dict[str, np.ndarray] = {}
    oof_probs: Dict[str, np.ndarray] = {}

    for fold in folds:
        fold_data = read_fold_results_csv(fold.relevance_fold_results_csv)
        pred, sums, true, probs = run_sumprob_on_fold(fold_data)

        (_, _), metrics = compute_metrics(true, pred)
        fold_metrics[fold.fold_idx] = metrics

        fold_out_csv = os.path.join(out_csv_dir, f"{normalized_base}_sumprob_fold{fold.fold_idx}_results.csv")
        export_sumprob_results_csv(
            predicted_labels=pred,
            true_labels=true,
            probability_sums=sums,
            probabilities=probs,
            model_metrics=metrics,
            output_filepath=fold_out_csv,
        )
        fold_outputs[fold.fold_idx] = fold_out_csv

        # accumulate OOF
        for img_id in pred.keys():
            if img_id in oof_pred:
                raise ValueError(
                    f"❌ Imagem repetida no OOF ({img_id}). "
                    "Os folds deveriam particionar as imagens sem sobreposição."
                )
            oof_pred[img_id] = pred[img_id]
            oof_true[img_id] = true[img_id]
            oof_sums[img_id] = sums[img_id]
            oof_probs[img_id] = probs[img_id]

    # Global OOF
    (true_y, pred_y), oof_metrics = compute_metrics(oof_true, oof_pred)

    global_results_csv = os.path.join(out_csv_dir, f"{normalized_base}_sumprob_results.csv")
    export_sumprob_results_csv(
        predicted_labels=oof_pred,
        true_labels=oof_true,
        probability_sums=oof_sums,
        probabilities=oof_probs,
        model_metrics=oof_metrics,
        output_filepath=global_results_csv,
    )

    metrics_csv = export_sumprob_cross_metrics_csv(
        base_name=normalized_base,
        oof_global_metrics=oof_metrics,
        fold_metrics=fold_metrics,
        relevance_metrics_filepath=relevance_metrics_filepath,
        output_dir=out_csv_dir,
    )

    confusion_png: Optional[str] = None
    if generate_confusion:
        confusion_png = _generate_confusion_matrix_png(
            true_y=true_y,
            predicted_y=pred_y,
            output_dir=sum_cross_dir,
            model_name=f"{normalized_base}_sumprob",
            cmap="Blues",
            use_custom_dir=True,
        )

    return SumProbRunOutputs(
        base_name=normalized_base,
        output_dir=sum_cross_dir,
        results_global_csv=global_results_csv,
        metrics_csv=metrics_csv,
        confusion_matrix_png=confusion_png,
        results_by_fold_csv=fold_outputs,
    )


def run_sumprob_cross(
    *,
    experiment_dir: str,
    models: Optional[Sequence[str]] = None,
    output_root_dir: Optional[str] = None,
    generate_confusion: bool = True,
) -> List[SumProbRunOutputs]:
    by_base = _discover_relevance_fold_results(experiment_dir)

    selected_bases: Iterable[str]
    if models:
        selected_bases = [_normalize_base_name(m) for m in models]
    else:
        selected_bases = sorted(by_base.keys())

    outputs: List[SumProbRunOutputs] = []

    for base in selected_bases:
        folds = by_base.get(base)
        if not folds:
            raise ValueError(
                f"❌ Nenhum fold encontrado para modelo '{base}'. "
                "Verifique se existem arquivos *_foldN_results.csv em csv_exports/<modelo>/."
            )

        outputs.append(
            run_sumprob_cross_for_base(
                experiment_dir=experiment_dir,
                base_name=base,
                folds=folds,
                output_root_dir=output_root_dir,
                generate_confusion=generate_confusion,
            )
        )

    return outputs


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aplica Sum-of-Probabilities (SumProb) em resultados cross da técnica de relevância (por fold)."
        )
    )
    parser.add_argument(
        "experiment_dir",
        help="Diretório do experimento contendo csv_exports/<modelo>/*_foldN_results.csv",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Lista de bases de modelos para processar (padrão: todos encontrados)",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Diretório de saída (padrão: <experiment_dir>/sum_cross)",
    )
    parser.add_argument(
        "--no-confusion",
        action="store_true",
        help="Desativa a geração da matriz de confusão global (OOF)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    experiment_dir = os.path.abspath(args.experiment_dir)
    outputs = run_sumprob_cross(
        experiment_dir=experiment_dir,
        models=args.models,
        output_root_dir=args.output_root,
        generate_confusion=not args.no_confusion,
    )

    # Short summary for logs
    for out in outputs:
        print("=" * 80)
        print(f"✅ SumProb cross finalizado para: {out.base_name}")
        print(f"   📁 output_dir: {out.output_dir}")
        print(f"   📄 results (OOF): {out.results_global_csv}")
        print(f"   📊 metrics: {out.metrics_csv}")
        if out.confusion_matrix_png:
            print(f"   🧩 confusion: {out.confusion_matrix_png}")
        print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

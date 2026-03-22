import os
import csv
import glob

input_dir = "/home/afmireski/Documentos/BCC/artigos/relevancia/code/image_classification_by_relevance_of_regions_article/results/experiments/animals/experimento_20260118_064204_sum_rel/sum/csv_exports"
output_dir = os.path.join(input_dir, "tsv_exports")

def format_value(val):
    if val == "###":
        return ""
    try:
        float_val = float(val)
        return f"{val.replace('.', ',')}%"
    except ValueError:
        return val

def transform_model_name(name):
    if "global" in name:
        return "Global"
    
    parts = name.split("_")
    # Expected format: classifier_descriptors_relevance_specialist_X_type
    # Example: knn_lbp_relevance_specialist_0_mean
    # Or: svm_lbp_glcm_lpq_relevance_specialist_1_fold1
    
    specialist_idx = -1
    for i, p in enumerate(parts):
        if p == "specialist":
            specialist_idx = i
            break
            
    if specialist_idx != -1:
        e_num = parts[specialist_idx + 1]
        type_str = parts[specialist_idx + 2]
        
        if type_str == "mean":
            return f"E{e_num} Média"
        elif type_str.startswith("fold"):
            fold_num = type_str.replace("fold", "")
            return f"E{e_num} Fold{fold_num}"
            
    return name

files = glob.glob(os.path.join(input_dir, "**", "*_metrics.csv"), recursive=True)

for file_path in files:
    filename = os.path.basename(file_path)
    output_filename = filename.replace(".csv", ".tsv")
    output_path = os.path.join(output_dir, output_filename)
    
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = []
        for row in reader:
            if not row: continue
            new_row = [transform_model_name(row[0])]
            for val in row[1:]:
                new_row.append(format_value(val))
            rows.append(new_row)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
            
    with open(output_path, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)

print(f"Processados {len(files)} arquivos.")

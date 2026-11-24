
import pandas as pd
import numpy as np

from pathlib import Path
import os
from dotenv import load_dotenv

import splink.comparison_library as cl
from splink import Linker, DuckDBAPI, SettingsCreator, block_on

from data_preparation import prepare_splink_data

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 0. CONFIGURACIÓN
# ============================================================

SAMPLE_SIZE = 100_000
SEED = 42
THRESHOLD = 0.5
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]


N_THREADS = 4
MEMORY_LIMIT = "8GB"

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "data")
DATASETS_PATH = Path(DATA_DIR) / "cleaned" / "datasets"


# ============================================================
# 1. PREPARAR DATOS
# ============================================================

df_A_original = prepare_splink_data(
    parquet_path=DATASETS_PATH / "dataset_A.parquet",
    date_columns=["fecha_nacim"],
    sex_columns=["sexo"]
)

# Conservar ground truth por separado
df_truth = df_A_original[
    ["unique_id", "id_persona"]
].copy()

# Verificar identificadores
assert df_truth["unique_id"].is_unique, (
    "unique_id debe ser único."
)

assert df_truth["id_persona"].notna().all(), (
    "Hay registros sin id_persona en el ground truth."
)

# Eliminar información del ground truth
df_A = df_A_original.drop(
    columns=["id_registro", "id_persona"]
).copy()


# ============================================================
# 2. CREAR MUESTRA
# ============================================================

df_A_sample = df_A.sample(
    n=min(SAMPLE_SIZE, len(df_A)),
    random_state=SEED
).copy()

print(f"Registros en muestra: {len(df_A_sample):,}")


# ============================================================
# 3. CONFIGURAR COMPARACIONES
# ============================================================

comparisons_dic = {
    "sexo": cl.ExactMatch("sexo"),

    "cve_lengua_nom": cl.ExactMatch("cve_lengua_nom"),

    "nacesp_cve_nom": cl.ExactMatch("nacesp_cve_nom"),

    "fecha_nacim": cl.DateOfBirthComparison(
        "fecha_nacim",
        input_is_string=True,
        datetime_format="%Y-%m-%d",
        datetime_thresholds=[1, 1, 5],
        datetime_metrics=["day", "month", "year"],
    ),

    "nombre": cl.NameComparison("nombre"),

    "apellido_paterno": cl.NameComparison("apellido_paterno"),

    "apellido_materno": cl.NameComparison("apellido_materno"),

    "ent_resid_nom": cl.ExactMatch(
        "ent_resid_nom"
    ).configure(term_frequency_adjustments=True),

    "tloc_resid_nom": cl.JaroWinklerAtThresholds(
        "tloc_resid_nom",
        [0.95, 0.90, 0.80],
    ),
}


# ============================================================
# 4. REGLAS DE BLOQUEO
# ============================================================

blocking_rules = [
    block_on("fecha_nacim", "sexo"),
    block_on("apellido_paterno", "fecha_nacim"),
    block_on("apellido_paterno", "apellido_materno"),
]


# ============================================================
# 5. CONFIGURAR SPLINK Y DUCKDB
# ============================================================

comparisons = [
    comparisons_dic["sexo"],
    comparisons_dic["cve_lengua_nom"],
    comparisons_dic["nacesp_cve_nom"],
    comparisons_dic["fecha_nacim"],
    comparisons_dic["nombre"],
    comparisons_dic["apellido_paterno"],
    comparisons_dic["apellido_materno"],
    comparisons_dic["ent_resid_nom"],
    comparisons_dic["tloc_resid_nom"],
]

settings = SettingsCreator(
    link_type="dedupe_only",
    comparisons=comparisons,
    blocking_rules_to_generate_predictions=blocking_rules,
    retain_intermediate_calculation_columns=True,
)

db_api = DuckDBAPI()

db_api._con.execute(f"SET threads = {N_THREADS}")
db_api._con.execute("SET preserve_insertion_order = false")
db_api._con.execute(
    f"SET memory_limit = '{MEMORY_LIMIT}'"
)

df_sdf = db_api.register(df_A_sample)

linker = Linker(df_sdf, settings)


# ============================================================
# 6. ENTRENAMIENTO
# ============================================================

print("\nEstimando probabilidades u...")

linker.training.estimate_u_using_random_sampling(
    max_pairs=10_000_000,
    seed=SEED
)

print("\nEstimando probabilidades m...")

# Sesión 1: entrenamiento usando fecha de nacimiento
linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("fecha_nacim")
)

# Sesión 2: entrenamiento usando nombres y apellidos
linker.training.estimate_parameters_using_expectation_maximisation(
    block_on(
        "nombre",
        "apellido_paterno",
        "apellido_materno"
    )
)

# ============================================================
# 7. GENERAR PREDICCIONES
# ============================================================

THRESHOLDS = [
    0.00,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.99
]

print("\nGenerando predicciones...")

# Generar todas las parejas candidatas clasificadas por Splink
df_predictions = linker.inference.predict(
    threshold_match_probability=0
)

predictions_all = df_predictions.as_pandas_dataframe()

print(f"Parejas candidatas: {len(predictions_all):,}")


# ============================================================
# 8. CONSTRUIR GROUND TRUTH
# ============================================================

truth_map = df_truth.set_index("unique_id")["id_persona"]

# Recuperar id_persona de cada registro
predictions_all["id_persona_l"] = (
    predictions_all["unique_id_l"].map(truth_map)
)

predictions_all["id_persona_r"] = (
    predictions_all["unique_id_r"].map(truth_map)
)

assert predictions_all[
    ["id_persona_l", "id_persona_r"]
].notna().all().all(), (
    "Hay registros predichos sin correspondencia "
    "en el ground truth."
)

# Etiqueta verdadera de cada pareja candidata
predictions_all["y_true"] = (
    predictions_all["id_persona_l"]
    == predictions_all["id_persona_r"]
).astype(int)


# ============================================================
# 9. TOTAL DE PAREJAS VERDADERAS
# ============================================================

sample_truth = df_A_sample[["unique_id"]].merge(
    df_truth,
    on="unique_id",
    how="left",
    validate="one_to_one"
)["id_persona"]

assert sample_truth.notna().all(), (
    "Hay registros de la muestra sin correspondencia "
    "en el ground truth."
)

person_counts = sample_truth.value_counts()

total_true_pairs = int(
    (person_counts * (person_counts - 1) // 2).sum()
)


# ============================================================
# 10. EVALUAR DIFERENTES THRESHOLDS
# ============================================================

results_list = []

for threshold in THRESHOLDS:

    # Filtrar parejas que superan el umbral
    predictions = predictions_all.loc[
        predictions_all["match_probability"] >= threshold
    ]

    # Calcular TP y FP
    TP = int(predictions["y_true"].sum())
    FP = int((predictions["y_true"] == 0).sum())

    # Calcular FN
    FN = total_true_pairs - TP

    assert FN >= 0, (
        f"TP supera el total de parejas verdaderas "
        f"para threshold={threshold}."
    )

    # Métricas
    precision = (
        TP / (TP + FP)
        if TP + FP > 0
        else 0
    )

    recall = (
        TP / total_true_pairs
        if total_true_pairs > 0
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0
    )

    results_list.append({
        "sample_size": len(df_A_sample),
        "threshold": threshold,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "total_true_pairs": total_true_pairs,
        "predicted_pairs": len(predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1
    })


# ============================================================
# 11. REPORTE
# ============================================================

results = pd.DataFrame(results_list)

print("\n========== RESULTADOS ==========")

print(
    results.to_string(
        index=False,
        formatters={
            "precision": "{:.4f}".format,
            "recall": "{:.4f}".format,
            "f1": "{:.4f}".format,
        }
    )
)




plt.figure(figsize=(9, 5))

plt.plot(results["threshold"], results["precision"],
         marker="o", label="Precision")

plt.plot(results["threshold"], results["recall"],
         marker="o", label="Recall")

plt.plot(results["threshold"], results["f1"],
         marker="o", label="F1")

plt.xlabel("Threshold")
plt.ylabel("Métrica")
plt.title("Métricas según threshold")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


threshold = 0.99

pred = predictions_all["match_probability"] >= threshold

y_true = predictions_all["y_true"]
y_pred = pred.astype(int)

ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    display_labels=["No match", "Match"],
    values_format=","
)

plt.title(f"Matriz de confusión (threshold={threshold})")
plt.tight_layout()
plt.show()



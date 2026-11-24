"""
Paso 1: entrena Splink (tu código) sobre una instancia pequeña y guarda
  dedup_inputs/pairs.parquet    (i, j, P)         pares candidatos con match_probability
  dedup_inputs/records.parquet  (idx, unique_id, id_persona)   ground truth, SOLO validación

Cambios respecto a tu script:
  * La instancia se arma muestreando PERSONAS completas (todos los registros de cada
    id_persona), para conservar los duplicados. Un muestreo aleatorio de registros de una
    base grande deja casi todo como singletons. Usa TARGET_RECORDS ~ 2,000-5,000.
  * Se elimina la evaluación por umbrales (la hace nsga_dedup.py).
NO lo pude correr contra tu Splink ni tus datos; revisa los nombres de columnas
(unique_id_l, unique_id_r, match_probability) con predictions_all.columns.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import splink.comparison_library as cl
from splink import Linker, DuckDBAPI, SettingsCreator, block_on


import sys
sys.path.append(str(Path(__file__).parent.parent))
from data_preparation import prepare_splink_data

# ============================================================
# 0. CONFIGURACIÓN
# ============================================================
TARGET_RECORDS = 6000
print(f"TARGET_RECORDS = {TARGET_RECORDS} (para muestrear personas completas)")
SEED = 42
N_THREADS = 4
MEMORY_LIMIT = "8GB"
# TODO:  Se puede ajustar hacer análisis de sensibilidad sobre SUPPOSED_RECALL,
SUPPOSED_RECALL = 0.6 # prob de todos los matches verdaderos que se recuperan con las reglas determinísticas. 
print(f"SUPPOSED_RECALL = {SUPPOSED_RECALL} (para estimar p_m)")

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR", "data")
DATASETS_PATH = Path(DATA_DIR) / "cleaned" / "datasets"
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
OUTPUT_PATH = Path(OUTPUT_DIR) / "dedup_inputs"

# ============================================================
# 1. PREPARAR DATOS
# ============================================================
df_A_original = prepare_splink_data(
    parquet_path=DATASETS_PATH / "dataset_A.parquet",
    date_columns=["fecha_nacim"],
    sex_columns=["sexo"],
)
df_truth = df_A_original[["unique_id", "id_persona"]].copy()
assert df_truth["unique_id"].is_unique, "unique_id debe ser único."
assert df_truth["id_persona"].notna().all(), "Hay registros sin id_persona."

df_A = df_A_original.drop(columns=["id_registro", "id_persona"]).copy()

# ============================================================
# 2. INSTANCIA: muestrear personas completas
# ============================================================
rng = np.random.default_rng(SEED)
counts = df_truth["id_persona"].value_counts()
persons = counts.index.to_numpy().copy()
rng.shuffle(persons)
cum = counts.reindex(persons).cumsum()
chosen = cum.index[cum.to_numpy() <= TARGET_RECORDS]
ids = df_truth.loc[df_truth["id_persona"].isin(chosen), "unique_id"]
df_A_sample = df_A[df_A["unique_id"].isin(ids)].copy()

sz = df_truth[df_truth["unique_id"].isin(ids)]["id_persona"].value_counts()
print(f"Instancia: {len(df_A_sample):,} registros, {len(sz):,} personas, "
      f"{int((sz > 1).sum()):,} con duplicados, pares verdaderos = "
      f"{int((sz * (sz - 1) // 2).sum()):,}")
print("Distribución de registros por persona:\n", sz.value_counts().sort_index().head(10))

# ============================================================
# 3-6. SPLINK 
# ============================================================
comparisons = [
    cl.ExactMatch("sexo"),
    cl.ExactMatch("cve_lengua_nom"),
    cl.DateOfBirthComparison(
        "fecha_nacim", input_is_string=True, datetime_format="%Y-%m-%d",
        datetime_thresholds=[1, 1, 5], datetime_metrics=["day", "month", "year"]),
    cl.NameComparison("nombre"),
    cl.NameComparison("apellido_paterno"),
    cl.NameComparison("apellido_materno"),
    cl.ExactMatch("ent_resid_nom").configure(term_frequency_adjustments=True),
    cl.JaroWinklerAtThresholds("tloc_resid_nom", [0.95, 0.90, 0.80]),
]
blocking_rules = [
    # originales
    block_on("fecha_nacim", "sexo"),
    block_on("apellido_paterno", "fecha_nacim"),
    block_on("apellido_paterno", "apellido_materno"),
    # nuevas: cada una omite un campo distinto
    block_on("nombre", "apellido_paterno"),
    block_on("nombre", "fecha_nacim"),
    block_on("apellido_materno", "fecha_nacim"),
    # parcial: prefijos + año de nacimiento
    block_on("substr(nombre,1,3)", "substr(apellido_paterno,1,3)", "substr(fecha_nacim,1,4)"),
]
settings = SettingsCreator(
    link_type="dedupe_only",
    comparisons=comparisons,
    blocking_rules_to_generate_predictions=blocking_rules,
    retain_intermediate_calculation_columns=True,
)



deterministic_rules = [
    # Ambos apellidos y fecha de nacimiento exactos
    """
    l.apellido_paterno = r.apellido_paterno
    AND l.apellido_materno = r.apellido_materno
    AND l.fecha_nacim = r.fecha_nacim
    """,
]



db_api = DuckDBAPI()
db_api._con.execute(f"SET threads = {N_THREADS}")
db_api._con.execute("SET preserve_insertion_order = false")
db_api._con.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
linker = Linker(db_api.register(df_A_sample), settings)




linker.training.estimate_probability_two_random_records_match(deterministic_rules, recall=SUPPOSED_RECALL)



linker.training.estimate_u_using_random_sampling(max_pairs=10_000_000, seed=SEED)
linker.training.estimate_parameters_using_expectation_maximisation(block_on("fecha_nacim"))
linker.training.estimate_parameters_using_expectation_maximisation(
    block_on("nombre", "apellido_paterno", "apellido_materno"))

# ============================================================
# 7. PREDICCIONES (todas las parejas candidatas) Y GUARDADO
# ============================================================
pred = linker.inference.predict(threshold_match_probability=0).as_pandas_dataframe()
print(f"Parejas candidatas: {len(pred):,}")

records = (df_A_sample[["unique_id"]]
           .merge(df_truth, on="unique_id", how="left", validate="one_to_one")
           .reset_index(drop=True))
records["idx"] = np.arange(len(records))
idx_of = pd.Series(records["idx"].to_numpy(), index=records["unique_id"].to_numpy())

a = pred["unique_id_l"].map(idx_of)
b = pred["unique_id_r"].map(idx_of)
assert a.notna().all() and b.notna().all(), "Hay unique_id en predicciones sin registro."
pairs = pd.DataFrame({
    "i": np.minimum(a, b).astype(np.int64),
    "j": np.maximum(a, b).astype(np.int64),
    "P": pred["match_probability"].astype(float),
}).drop_duplicates(["i", "j"]).reset_index(drop=True)
assert (pairs["i"] != pairs["j"]).all()

OUTPUT_PATH.mkdir(exist_ok=True)
pairs.to_parquet(OUTPUT_PATH / "pairs.parquet", index=False)
records[["idx", "unique_id", "id_persona"]].to_parquet(OUTPUT_PATH / "records.parquet", index=False)
print(f"Guardado en {OUTPUT_PATH}/  ({len(pairs):,} pares, {len(records):,} registros)")
print("Siguiente paso: python nsga_dedup.py")
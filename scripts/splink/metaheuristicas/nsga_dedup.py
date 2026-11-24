#!/usr/bin/env python
"""
Deduplicación multiobjetivo: NSGA-II vs Random Search.

Entradas (producidas por extract_pairs.py):
  pairs.parquet    columnas i, j, P   (i<j, índices 0..N-1; P = match_probability de Splink)
  records.parquet  columnas idx, unique_id, id_persona (ground truth, SOLO para validar)

Problema
--------
Variable de decisión: vector de genes G (longitud N), codificación por adyacencia
(locus-based, estilo MOCK). G[j] = 0 -> el registro j apunta a sí mismo;
G[j] = k>0 -> apunta a su k-ésimo vecino candidato (pares con P >= p_min_gene).
Los clusters son las componentes conexas del grafo {j -- target(j)}.
TODO vector es válido (la partición siempre es válida y transitiva).

Objetivos (ambos a minimizar, en [0,1]); P_ij = 0 fuera del conjunto de pares E_P:
  n_C  = sum_k n_k (n_k - 1)/2           pares dentro de clusters
  ETP  = sum_{(i,j) in E_P} 1[Z_i=Z_j] P_ij
  f1   = FDR estimado = (n_C - ETP) / n_C       (0 si n_C = 0)
  f2   = FNR estimado = (sum P - ETP) / sum P

El ground truth NO se usa en la optimización; solo en la validación posterior.
"""
import argparse
import os
import platform
import time
from pathlib import Path
from dotenv import load_dotenv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import pymoo
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

load_dotenv()
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
DEDUP_INPUTS = Path(OUTPUT_DIR) / "dedup_inputs"
DEDUP_OUTPUTS = Path(OUTPUT_DIR) / "dedup_outputs"




# ----------------------------------------------------------------------------
# Hyperparameters
# ----------------------------------------------------------------------------
P_MIN_GENE = 1e-3

# Metricas
REF_POINT = np.array([0.1, 0.1]) # antes 1,1
HV_IND = HV(ref_point=REF_POINT)

# ----------------------------------------------------------------------------
# Instancia: evidencia fija + evaluación de soluciones
# ----------------------------------------------------------------------------
class Instance:
    def __init__(self, pairs, records, p_min_gene=1e-3, mut_smooth=0.05):
        self.N = len(records)
        self.i = pairs["i"].to_numpy(np.int64)
        self.j = pairs["j"].to_numpy(np.int64)
        self.P = np.clip(pairs["P"].to_numpy(float), 0.0, 1.0)
        self.sumP = float(self.P.sum())
        self.n_pairs = len(self.P)

        # Ground truth (solo validación)
        self.person = pd.factorize(records["id_persona"])[0].astype(np.int64)
        cnt = np.bincount(self.person)
        self.total_true_pairs = int((cnt * (cnt - 1) // 2).sum())

        # Adyacencia (ambas direcciones) con pares P >= p_min_gene
        m = self.P >= p_min_gene
        src = np.concatenate([self.i[m], self.j[m]])
        dst = np.concatenate([self.j[m], self.i[m]])
        w = np.concatenate([self.P[m], self.P[m]])
        o = np.lexsort((dst, src))
        self.src, self.dst, self.w = src[o], dst[o], w[o]
        self.deg = np.bincount(self.src, minlength=self.N).astype(np.int64)
        self.indptr = np.concatenate([[0], np.cumsum(self.deg)]).astype(np.int64)
        self.active = np.flatnonzero(self.deg > 0) # conjunto de registros con al menos un vecino candidato (pares con P >= p_min_gene)

        # Pesos acumulados para mutación sesgada (prob. ∝ P + mut_smooth)
        cum = np.concatenate([[0.0], np.cumsum(self.w + mut_smooth)])
        self.cum = cum[1:]
        self.cum_lo = cum[self.indptr[:-1]]
        self.cum_hi = cum[self.indptr[1:]]

    # -- decodificación ------------------------------------------------------
    def labels_from_genes(self, G):
        t = np.arange(self.N)
        nz = G > 0
        t[nz] = self.dst[self.indptr[:-1][nz] + G[nz] - 1]
        mv = t != np.arange(self.N)
        g = csr_matrix((np.ones(mv.sum(), np.int8), (np.arange(self.N)[mv], t[mv])),
                       shape=(self.N, self.N))
        return connected_components(g, directed=False)[1]

    def labels_from_threshold(self, thr):
        m = self.P >= thr
        g = csr_matrix((np.ones(m.sum(), np.int8), (self.i[m], self.j[m])),
                       shape=(self.N, self.N))
        return connected_components(g, directed=False)[1]

    # -- objetivos -----------------------------------------------------------
    def stats(self, lab):
        sizes = np.bincount(lab)
        nC = int((sizes * (sizes - 1) // 2).sum())
        same = lab[self.i] == lab[self.j]
        etp = float(self.P[same].sum())
        f1 = (nC - etp) / nC if nC > 0 else 0.0
        # print("Pares dentro de clusters:", nC, "ETP:", etp, "m", same.sum())
        f2 = (self.sumP - etp) / self.sumP if self.sumP > 0 else 1.0
        f1 = min(max(f1, 0.0), 1.0)
        f2 = min(max(f2, 0.0), 1.0)
        exp_f = 2 * etp / (nC + self.sumP) if (nC + self.sumP) > 0 else 0.0
        return f1, f2, nC, etp, exp_f

    def objectives(self, G):
        f1, f2, *_ = self.stats(self.labels_from_genes(G))
        return f1, f2

    def real_metrics(self, lab):
        """Métricas contra el ground truth (solo validación)."""
        sizes = np.bincount(lab)
        nC = int((sizes * (sizes - 1) // 2).sum())
        key = lab.astype(np.int64) * (self.person.max() + 1) + self.person
        c = np.unique(key, return_counts=True)[1]
        tp = int((c * (c - 1) // 2).sum())
        T = self.total_true_pairs
        return dict(real_fdr=1 - tp / nC if nC > 0 else 0.0,
                    real_fnr=1 - tp / T if T > 0 else 0.0,
                    real_f1=2 * tp / (nC + T) if (nC + T) > 0 else 0.0,
                    real_precision=tp / nC if nC > 0 else 0.0,
                    real_recall=tp / T if T > 0 else 0.0,
                    n_clusters=int(len(sizes)), n_pairs_in_clusters=nC, TP=tp)

    # -- muestreo sesgado de un gen -------------------------------------------
    def draw_genes(self, nodes, p_self):
        u = np.random.random(len(nodes))
        tgt = self.cum_lo[nodes] + u * (self.cum_hi[nodes] - self.cum_lo[nodes])
        e = np.searchsorted(self.cum, tgt, side="right")
        e = np.clip(e, self.indptr[nodes], self.indptr[nodes + 1] - 1)
        g = e - self.indptr[nodes] + 1
        g[np.random.random(len(nodes)) < p_self] = 0
        return g


def sample_structured(inst, n, rng, logit_range):
    """Muestreador estructurado (común a Random Search y a la población inicial de
    NSGA-II): umbral aleatorio t sobre P (log-odds ~ U(-L, L)); cada registro apunta
    a un vecino aleatorio entre las aristas con P >= t, o a sí mismo si no tiene."""
    G = np.zeros((n, inst.N), dtype=np.int64)
    for r in range(n):
        L = rng.uniform(-logit_range, logit_range)
        t = 1.0 / (1.0 + np.exp(-L))
        idx = np.flatnonzero(inst.w >= t)
        if idx.size == 0:
            continue
        s = inst.src[idx]
        o = np.lexsort((rng.random(idx.size), s))
        s_o, idx_o = s[o], idx[o]
        last = np.ones(len(s_o), bool)
        last[:-1] = s_o[:-1] != s_o[1:]
        node, edge = s_o[last], idx_o[last]
        G[r, node] = edge - inst.indptr[node] + 1
    return G


# ----------------------------------------------------------------------------
# Componentes de pymoo
# ----------------------------------------------------------------------------
class DedupProblem(Problem):
    def __init__(self, inst):
        super().__init__(n_var=inst.N, n_obj=2, xl=0, xu=inst.deg, vtype=int)
        self.inst = inst

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.array([self.inst.objectives(x) for x in X])


class StructuredSampling(Sampling):
    def __init__(self, inst, logit_range):
        super().__init__()
        self.inst, self.L = inst, logit_range

    def _do(self, problem, n_samples, **kwargs):
        return sample_structured(self.inst, n_samples, np.random, self.L)


class UniformCrossover(Crossover):
    """Cruza uniforme: cada gen es independiente, por lo que los hijos siempre son válidos."""
    def __init__(self, prob=0.9):
        super().__init__(2, 2, prob=prob)

    def _do(self, problem, X, **kwargs):
        _, n_m, n_var = X.shape
        mask = np.random.random((n_m, n_var)) < 0.5
        Y = np.empty_like(X)
        Y[0] = np.where(mask, X[0], X[1])
        Y[1] = np.where(mask, X[1], X[0])
        return Y


class NeighbourMutation(Mutation):
    """Reasigna k genes (k ~ Binomial(n_activos, rate)) a un vecino sesgado por P o a sí mismo."""
    def __init__(self, inst, rate, p_self=0.2):
        super().__init__()
        self.inst, self.rate, self.p_self = inst, rate, p_self

    def _do(self, problem, X, **kwargs):
        X = X.copy()
        act = self.inst.active
        for r in range(X.shape[0]):
            k = np.random.binomial(len(act), self.rate)
            if k == 0:
                continue
            nodes = np.random.choice(act, k, replace=False)
            X[r, nodes] = self.inst.draw_genes(nodes, self.p_self)
        return X


class HVCallback(Callback):
    def __init__(self):
        super().__init__()
        self.data["n_eval"], self.data["hv"] = [], []

    def notify(self, algorithm):
        self.data["n_eval"].append(algorithm.evaluator.n_eval)
        self.data["hv"].append(float(HV_IND(algorithm.opt.get("F"))))


# ----------------------------------------------------------------------------
# Métodos
# ----------------------------------------------------------------------------
def nondominated_unique(X, F):
    nd = NonDominatedSorting().do(F, only_non_dominated_front=True)
    X, F = X[nd], F[nd]
    _, u = np.unique(np.round(F, 12), axis=0, return_index=True)
    return X[u], F[u]


def run_nsga2(inst, budget, pop, seed, logit_range, mut_factor, p_self):
    np.random.seed(seed)
    random.seed(seed)
    rate = min(1.0, mut_factor / max(len(inst.active), 1))
    algo = NSGA2(pop_size=pop,
                 sampling=StructuredSampling(inst, logit_range),
                 crossover=UniformCrossover(),
                 mutation=NeighbourMutation(inst, rate, p_self),
                 eliminate_duplicates=True)
    cb = HVCallback()
    t0 = time.time()
    res = minimize(DedupProblem(inst), algo, termination=("n_eval", budget),
                   seed=seed, callback=cb, verbose=False)
    X, F = nondominated_unique(np.atleast_2d(res.X), np.atleast_2d(res.F))
    return dict(X=X, F=F, n_eval=res.algorithm.evaluator.n_eval, time=time.time() - t0,
                curve=(np.array(cb.data["n_eval"]), np.array(cb.data["hv"])))


def run_random_search(inst, budget, seed, logit_range, chunk=100):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    aX = aF = None
    n, ev, hv = 0, [], []
    while n < budget:
        m = min(chunk, budget - n)
        Xc = sample_structured(inst, m, rng, logit_range)
        Fc = np.array([inst.objectives(x) for x in Xc])
        n += m
        if aX is None:
            aX, aF = Xc, Fc
        else:
            aX, aF = np.vstack([aX, Xc]), np.vstack([aF, Fc])
        aX, aF = nondominated_unique(aX, aF)
        ev.append(n)
        hv.append(float(HV_IND(aF)))
    return dict(X=aX, F=aF, n_eval=n, time=time.time() - t0,
                curve=(np.array(ev), np.array(hv)))


def threshold_baseline(inst, n_grid=200):
    """Enfoque 'greedy'/determinista: componentes conexas a un umbral sobre P."""
    ts = 1 / (1 + np.exp(-np.linspace(-12, 12, n_grid)))
    ts = np.unique(np.concatenate([ts, [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0 + 1e-9]]))
    rows = []
    for t in ts:
        lab = inst.labels_from_threshold(t)
        f1, f2, nC, etp, ef = inst.stats(lab)
        rows.append(dict(threshold=t, f1=f1, f2=f2, exp_F=ef, **inst.real_metrics(lab)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Tablas, soluciones representativas y validación
# ----------------------------------------------------------------------------
def front_table(inst, X, F):
    rows = []
    for x, f in zip(X, F):
        lab = inst.labels_from_genes(x)
        f1, f2, nC, etp, ef = inst.stats(lab)
        rows.append(dict(f1=f1, f2=f2, exp_F=ef, **inst.real_metrics(lab)))
    return pd.DataFrame(rows)


def pick_representatives(tab, f2_cap):
    """Tres soluciones: (a) mín. f2 [agresiva], (b) mín. f1 con f2<=f2_cap [conservadora,
    excluye la solución trivial de todo-singletons], (c) máx. F-score esperado [compromiso]."""
    out = {}
    out["min_f2 (agresiva)"] = tab.sort_values(["f2", "f1"]).index[0]
    cand = tab[tab.f2 <= f2_cap]
    out[f"min_f1 con f2<={f2_cap} (conservadora)"] = (
        cand.sort_values(["f1", "f2"]).index[0] if len(cand) else out["min_f2 (agresiva)"])
    out["max_F_esperado (compromiso)"] = tab["exp_F"].idxmax()
    return out


def interp_curves(curves, grid):
    ys = np.array([np.interp(grid, ev, hv) for ev, hv in curves])
    return ys.mean(0), ys.std(0, ddof=1) if len(ys) > 1 else np.zeros_like(grid)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="pairs.parquet")
    ap.add_argument("--records", default="records.parquet")
    ap.add_argument("--budget", type=int, default=20000, help="evaluaciones por corrida")
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029, 2030])
    ap.add_argument("--logit-range", type=float, default=12.0)
    ap.add_argument("--mut-factor", type=float, default=5.0,
                    help="genes esperados a mutar por hijo = mut_factor")
    ap.add_argument("--p-self", type=float, default=0.2)
    ap.add_argument("--p-min-gene", type=float, default=1e-3)
    ap.add_argument("--f2-cap", type=float, default=0.5)
    a = ap.parse_args()


    pairs = pd.read_parquet(DEDUP_INPUTS / a.pairs)
    records = pd.read_parquet(DEDUP_INPUTS / a.records).sort_values("idx").reset_index(drop=True)
    inst = Instance(pairs, records, a.p_min_gene)
    print(f"N={inst.N}  pares E_P={inst.n_pairs:,}  sum(P)={inst.sumP:,.1f}  "
          f"genes activos={len(inst.active)}  pares verdaderos totales={inst.total_true_pairs:,}")
    print(f"Parámetros: budget={a.budget} pop={a.pop} logit_range={a.logit_range} "
          f"mut_factor={a.mut_factor} p_self={a.p_self} f2_cap={a.f2_cap}")
    # tiempo por evaluación
    G0 = sample_structured(inst, 20, np.random.default_rng(0), a.logit_range)
    t0 = time.time()
    for g in G0:
        inst.objectives(g)
    print(f"~{(time.time() - t0) / 20 * 1e3:.1f} ms por evaluación")

    base = threshold_baseline(inst)
    base.to_csv(DEDUP_OUTPUTS / "baseline_thresholds.csv", index=False)
    hv_base = float(HV_IND(base[["f1", "f2"]].to_numpy()))
    print(f"HV línea base (umbrales) = {hv_base:.5f}")

    results = {"NSGA-II": [], "Random Search": []}
    for s in a.seeds:
        r = run_nsga2(inst, a.budget, a.pop, s, a.logit_range, a.mut_factor, a.p_self)
        results["NSGA-II"].append(r)
        rs = run_random_search(inst, r["n_eval"], s, a.logit_range)  # mismo n de evaluaciones
        results["Random Search"].append(rs)
        print(f"seed {s}: NSGA-II HV={r['curve'][1][-1]:.5f} ({r['n_eval']} ev, {r['time']:.0f}s, "
              f"{len(r['F'])} sol.) | RS HV={rs['curve'][1][-1]:.5f} ({rs['n_eval']} ev, "
              f"{rs['time']:.0f}s, {len(rs['F'])} sol.)")

    # ---- resumen HV ----
    rows = []
    for m, lst in results.items():
        for s, r in zip(a.seeds, lst):
            rows.append(dict(method=m, seed=s, hv=r["curve"][1][-1], n_eval=r["n_eval"],
                             n_front=len(r["F"]), time_s=r["time"]))
    hv_df = pd.DataFrame(rows)
    hv_df.to_csv(DEDUP_OUTPUTS / "hv_per_seed.csv", index=False)
    summ = hv_df.groupby("method")["hv"].agg(["mean", "std", "min", "max"])
    print(f"\nHipervolumen (ref={REF_POINT} sobre {len(a.seeds)} semillas:\n", summ.round(5))
    summ.to_csv(DEDUP_OUTPUTS / "hv_summary.csv")

    # ---- representativas + validación (todas las semillas) ----
    rep_rows, fronts_real = [], {}
    for m, lst in results.items():
        for s, r in zip(a.seeds, lst):
            tab = front_table(inst, r["X"], r["F"])
            fronts_real[(m, s)] = tab
            for name, ix in pick_representatives(tab, a.f2_cap).items():
                rep_rows.append(dict(method=m, seed=s, solution=name, **tab.loc[ix].to_dict()))
    pd.DataFrame(rep_rows).to_csv(DEDUP_OUTPUTS / "representative_solutions.csv", index=False)
    rep = pd.DataFrame(rep_rows)
    print("\nSoluciones representativas (media sobre semillas):")
    print(rep.groupby(["method", "solution"])[["f1", "f2", "exp_F", "real_fdr", "real_fnr",
                                               "real_f1", "n_clusters"]].mean().round(4))

    # ---- Figura 1: frentes ----
    col = {"NSGA-II": "#1f77b4", "Random Search": "#ff7f0e"}

    # límites del zoom a partir de los frentes de ambos métodos
    allF = np.vstack([r["F"] for lst in results.values() for r in lst])
    px = 0.05 * (allF[:, 0].max() - allF[:, 0].min())
    py = 0.05 * (allF[:, 1].max() - allF[:, 1].min())

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    b = base.sort_values("f1")
    for ax in axs:
        for m, lst in results.items():
            for k, r in enumerate(lst):
                F = r["F"][np.argsort(r["F"][:, 0])]
                ax.plot(F[:, 0], F[:, 1], "o", ms=3, alpha=.5, color=col[m],
                        label=m if k == 0 else None)
        ax.plot(b.f1, b.f2, "-k", lw=1, label="Umbral + comp. conexas")
        for t, mk in [(0.5, "*"), (0.99, "s")]:
            r0 = base.iloc[(base.threshold - t).abs().argmin()]
            ax.plot(r0.f1, r0.f2, mk, color="k", ms=9, mfc="none", label=f"umbral {t}")
        ax.set_xlabel("f1 = FDR estimado (minimizar)")
        ax.set_ylabel("f2 = FNR estimado (minimizar)")
        ax.grid(alpha=.3)

    axs[0].set_title("Rango completo")
    axs[0].legend()
    axs[1].set_xlim(allF[:, 0].min() - px, allF[:, 0].max() + px)
    axs[1].set_ylim(allF[:, 1].min() - py, allF[:, 1].max() + py)
    axs[1].set_title("Ampliación sobre las soluciones de NSGA-II y RS")
    fig.suptitle(f"Aproximaciones del frente de Pareto ({len(a.seeds)} semillas)")
    fig.tight_layout(); fig.savefig(DEDUP_OUTPUTS / "fig_fronts.png", dpi=160); plt.close(fig)


    # ---- Figura 2: hipervolumen vs evaluaciones ----
    grid = np.linspace(0, results["NSGA-II"][0]["n_eval"], 200)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, lst in results.items():
        mu, sd = interp_curves([r["curve"] for r in lst], grid)
        ax.plot(grid, mu, color=col[m], label=m)
        ax.fill_between(grid, mu - sd, mu + sd, color=col[m], alpha=.2)
    ax.axhline(hv_base, ls="--", color="k", label="Línea base (umbrales)")
    ax.set_xlabel("Evaluaciones de la función objetivo")
    ax.set_ylabel(f"Hipervolumen (ref {REF_POINT} ; mayor es mejor)")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(DEDUP_OUTPUTS / "fig_hv_convergence.png", dpi=160); plt.close(fig)

    # ---- Figura 3: calibración sobre el frente de NSGA-II (semilla mediana por HV) ----
    hvs = np.array([r["curve"][1][-1] for r in results["NSGA-II"]])
    ms = a.seeds[np.argsort(hvs)[len(hvs) // 2]]
    tab = fronts_real[("NSGA-II", ms)]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (xe, ye, ttl) in zip(axs, [("f1", "real_fdr", "FDR"), ("f2", "real_fnr", "FNR"),
                                       ("exp_F", "real_f1", "F-score")]):
        ax.scatter(tab[xe], tab[ye], s=14, color=col["NSGA-II"])
        lim = [0, 1]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"{ttl} estimado (modelo)"); ax.set_ylabel(f"{ttl} real (ground truth)")
        ax.grid(alpha=.3)
    axs[1].set_title("FNR real incluye pares fuera de E_P (bloqueo)", fontsize=9)
    fig.suptitle(f"Calibración a lo largo del frente de NSGA-II (semilla {ms})")
    fig.tight_layout(); fig.savefig(DEDUP_OUTPUTS / "fig_calibration.png", dpi=160); plt.close(fig)

    # ---- metadatos de reproducibilidad ----
    with open(DEDUP_OUTPUTS / "run_info.txt", "w") as f:
        f.write(f"python {platform.python_version()} | pymoo {pymoo.__version__} | "
                f"numpy {np.__version__} | {platform.platform()} | {platform.processor()}\n")
        f.write(f"args: {vars(a)}\n")
        f.write(f"N={inst.N} E_P={inst.n_pairs} sumP={inst.sumP:.3f} "
                f"true_pairs={inst.total_true_pairs}\n")
    print(f"\nListo. Resultados en {DEDUP_OUTPUTS}/")


if __name__ == "__main__":
    main()
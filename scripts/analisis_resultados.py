# -*- coding: utf-8 -*-
"""
Análisis de resultados - TP Temas de NLP (Grupo 6)
Evaluación de traducción automática qom -> español.

Calcula:
  - IAA (Alpha de Krippendorff) para A (sin/con guía), B y C (Accuracy/Fluency).
  - Proporciones de acuerdo (3/3, 2/3, 0/3) y matrices de confusión.
  - Casos de mayor desacuerdo.
  - Métricas automáticas BLEU (sacreBLEU) y BERTScore (BETO) sobre C.
  - Correlación humano <-> métrica.

Salidas:
  - resultados/resultados.json   (todos los valores; fuente de verdad)
  - informe/tablas/*.tex         (fragmentos de tabla booktabs)
  - informe/figuras/*.pdf|*.png  (figuras)
"""
import os, re, json, sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import krippendorff
import sacrebleu
from scipy import stats

# ----------------------------------------------------------------------------- paths
ROOT = r"C:\Users\Juan Cruz\Desktop\GitHub\activos\temas_NLP_TP"
DATA = os.path.join(ROOT, "datasets_anotados")
OUT_JSON_DIR = os.path.join(ROOT, "resultados")
OUT_TAB = os.path.join(ROOT, "informe", "tablas")
OUT_FIG = os.path.join(ROOT, "informe", "figuras")
for d in (OUT_JSON_DIR, OUT_TAB, OUT_FIG):
    os.makedirs(d, exist_ok=True)

F_A_SIN = os.path.join(DATA, "dataset_A_sin_criterio.xlsx")
F_A_PROP = os.path.join(DATA, "dataset_A_criterio_propio.xlsx")
F_B_PROP = os.path.join(DATA, "dataset_B_criterio_propio.xlsx")
F_C_EXT = os.path.join(DATA, "dataset_C_criterio_externo.xlsx")

RATERS = ["Juan I Catania", "Juan Cruz Mendoza", "Camila"]  # orden canónico
LABELS3 = ["Incorrecta", "Parcialmente correcta", "Correcta"]  # 1,2,3
ANCHORS = [0, 5, 10, 15, 20]
SHARED_IDS = [40, 43, 56, 68, 74, 81, 107, 112, 113, 114, 115, 116, 126, 129, 130]

results = {}

# ----------------------------------------------------------------------------- helpers
def norm_label(s):
    """Etiqueta categórica -> código ordinal 1/2/3 (Incorrecta<Parcial<Correcta)."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    t = re.sub(r"\s+", " ", str(s)).strip().casefold()
    if t.startswith("parcial"):
        return 2
    if t.startswith("incorrect"):
        return 1
    if t.startswith("correct"):
        return 3
    return np.nan

def kalpha(matrix, level):
    """matrix: raters x items (np.nan = faltante). Devuelve alpha o None."""
    arr = np.array(matrix, dtype=float)
    try:
        a = krippendorff.alpha(reliability_data=arr, level_of_measurement=level)
        if a is None or (isinstance(a, float) and np.isnan(a)):
            return None
        return float(a)
    except Exception as e:
        return None

def agreement_props(cols_by_item):
    """cols_by_item: lista de tuplas (3 valores por ítem). Devuelve proporciones."""
    n = len(cols_by_item)
    una = maj = des = 0
    for vals in cols_by_item:
        d = len(set(vals))
        if d == 1:
            una += 1
        elif d == 2:
            maj += 1
        else:
            des += 1
    return {"n": n, "unanime_3de3": una, "mayoritario_2de3": maj,
            "desacuerdo_0de3": des,
            "p_unanime": una / n, "p_mayoritario": maj / n, "p_desacuerdo": des / n}

def confusion_pairwise(items_vals, categories):
    """Matriz de coincidencia simétrica (pares de anotadores agrupados por ítem)."""
    idx = {c: i for i, c in enumerate(categories)}
    M = np.zeros((len(categories), len(categories)), dtype=int)
    for vals in items_vals:
        for a in range(len(vals)):
            for b in range(a + 1, len(vals)):
                x, y = idx[vals[a]], idx[vals[b]]
                M[x, y] += 1
                M[y, x] += 1
    return M

def to_bucket(x):
    return int(min(ANCHORS, key=lambda a: abs(a - x)))

# ----------------------------------------------------------------------------- A sin guía
dfA0 = pd.read_excel(F_A_SIN, sheet_name="A")
ids_A = dfA0["Column 1"].astype(int).tolist()
mapA0 = {"Juan I Catania": "Juan I Catania",
         "Juan Cruz Mendoza": "Juan Cruz Mendoza",
         "Camila": "Camila Rodriguez"}
A0_codes = {r: [norm_label(v) for v in dfA0[mapA0[r]]] for r in RATERS}
A0_matrix = [A0_codes[r] for r in RATERS]              # 3 x 15
A0_items = list(zip(*A0_matrix))                       # 15 tuplas de 3
assert all(not np.isnan(v) for row in A0_matrix for v in row), "NaN en A sin guía"

a0_alpha = kalpha(A0_matrix, "ordinal")
a0_props = agreement_props(A0_items)
A0_labels = [[LABELS3[int(v) - 1] for v in row] for row in A0_matrix]
A0_items_lab = list(zip(*A0_labels))
a0_conf = confusion_pairwise(A0_items_lab, LABELS3)

results["A_sin_guia"] = {
    "escala": "ordinal (Incorrecta<Parcialmente correcta<Correcta)",
    "n_items": len(ids_A), "n_anotadores": 3,
    "krippendorff_alpha_ordinal": a0_alpha,
    "acuerdo": a0_props,
    "matriz_confusion_labels": LABELS3,
    "matriz_confusion": a0_conf.tolist(),
}

# ----------------------------------------------------------------------------- A/B con guía
def load_criterio(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet)
    cols = list(df.columns)
    ids = df["Column 1"].astype(int).tolist()
    overall = {}   # rater -> [codes]
    dims = {"Fidelidad": {}, "Naturalidad": {}, "Preservacion": {}}
    for r in RATERS:
        i = cols.index(r)
        overall[r] = [norm_label(v) for v in df[cols[i]]]
        dims["Fidelidad"][r] = [float(v) for v in df[cols[i + 1]]]
        dims["Naturalidad"][r] = [float(v) for v in df[cols[i + 2]]]
        dims["Preservacion"][r] = [float(v) for v in df[cols[i + 3]]]
    return df, ids, overall, dims

def analyze_criterio(name, path, sheet):
    df, ids, overall, dims = load_criterio(path, sheet)
    out = {"n_items": len(ids), "n_anotadores": 3,
           "escala_global": "ordinal (3 niveles)",
           "escala_dimensiones": "ordinal 1-4"}
    # global
    gmat = [overall[r] for r in RATERS]
    assert all(not np.isnan(v) for row in gmat for v in row), f"NaN global {name}"
    gitems = list(zip(*gmat))
    out["global"] = {
        "krippendorff_alpha_ordinal": kalpha(gmat, "ordinal"),
        "acuerdo": agreement_props(gitems),
    }
    glab = [[LABELS3[int(v) - 1] for v in row] for row in gmat]
    out["global"]["matriz_confusion_labels"] = LABELS3
    out["global"]["matriz_confusion"] = confusion_pairwise(list(zip(*glab)), LABELS3).tolist()
    # dimensiones
    out["dimensiones"] = {}
    for dim in ["Fidelidad", "Naturalidad", "Preservacion"]:
        dmat = [dims[dim][r] for r in RATERS]
        ditems = list(zip(*dmat))
        out["dimensiones"][dim] = {
            "krippendorff_alpha_ordinal": kalpha(dmat, "ordinal"),
            "acuerdo": agreement_props(ditems),
            "valores_observados": sorted({int(v) for row in dmat for v in row}),
        }
    return df, ids, overall, dims, out

dfAp, ids_Ap, A_overall, A_dims, res_Ap = analyze_criterio("A_prop", F_A_PROP, "A")
dfBp, ids_Bp, B_overall, B_dims, res_Bp = analyze_criterio("B_prop", F_B_PROP, "B")
results["A_con_guia"] = res_Ap
results["B_con_guia"] = res_Bp

# comparación A sin vs con guía (global)
results["comparacion_A"] = {
    "alpha_sin_guia": a0_alpha,
    "alpha_con_guia": res_Ap["global"]["krippendorff_alpha_ordinal"],
    "delta": res_Ap["global"]["krippendorff_alpha_ordinal"] - a0_alpha,
}

# chequeo consistencia regla dims->global (guía propia): Correcta=todas 4; Incorrecta= alguna 1 o Fidelidad=2; resto Parcial
def rule_label(fid, nat, pres):
    if fid == 1 or nat == 1 or pres == 1 or fid == 2:
        return 1
    if fid == 4 and nat == 4 and pres == 4:
        return 3
    return 2

def consistency(ids, overall, dims):
    total = mism = 0
    for r in RATERS:
        for k in range(len(ids)):
            pred = rule_label(dims["Fidelidad"][r][k], dims["Naturalidad"][r][k], dims["Preservacion"][r][k])
            obs = int(overall[r][k])
            total += 1
            if pred != obs:
                mism += 1
    return {"n": total, "mismatch": mism, "tasa_mismatch": mism / total}

results["A_con_guia"]["consistencia_regla"] = consistency(ids_Ap, A_overall, A_dims)
results["B_con_guia"]["consistencia_regla"] = consistency(ids_Bp, B_overall, B_dims)

# casos de mayor desacuerdo en A sin guía
dis_A = []
for k, vals in enumerate(A0_items):
    spread = max(vals) - min(vals)
    ndist = len(set(vals))
    dis_A.append((ndist, spread, ids_A[k], [LABELS3[int(v) - 1] for v in vals]))
dis_A.sort(key=lambda t: (t[0], t[1]), reverse=True)
top_dis_A = []
for ndist, spread, sid, labs in dis_A[:3]:
    row = dfA0[dfA0["Column 1"] == sid].iloc[0]
    top_dis_A.append({"id": sid, "n_distintas": ndist, "spread": spread,
                      "etiquetas": dict(zip(RATERS, labs)),
                      "qom": row["qom"], "ref": row["ref"], "mt": row["mt"]})
results["A_sin_guia"]["casos_mayor_desacuerdo"] = top_dis_A

# ----------------------------------------------------------------------------- C externo
dfC = pd.read_excel(F_C_EXT, sheet_name="C")
dfC = dfC[dfC["Anotador"].isin([1, 2, 3])].copy()
dfC["id"] = dfC["Column 1"].astype(int)

def parse_anot(s):
    m = re.search(r"Accuracy:\s*(\d+).*?Fluency:\s*(\d+)", str(s), re.DOTALL | re.IGNORECASE)
    if not m:
        return (np.nan, np.nan)
    a, f = int(m.group(1)), int(m.group(2))
    a = a if 0 <= a <= 20 else np.nan
    f = f if 0 <= f <= 20 else np.nan
    return (a, f)

dfC[["acc", "flu"]] = dfC["Anotación"].apply(lambda s: pd.Series(parse_anot(s)))
assert dfC["acc"].notna().all() and dfC["flu"].notna().all(), "Anotación C mal parseada"

# verificación set compartido
cnt = dfC["id"].value_counts()
shared = sorted(cnt[cnt == 3].index.tolist())
assert shared == sorted(SHARED_IDS), f"shared mismatch: {shared}"
assert len(shared) == 15

def c_dim_matrix(dim):
    M = []
    for an in [1, 2, 3]:
        sub = dfC[dfC["Anotador"] == an].set_index("id")
        M.append([float(sub.loc[sid, dim]) for sid in shared])
    return M  # 3 x 15

def analyze_C_dim(dim):
    M = c_dim_matrix(dim)
    items = list(zip(*M))
    buck = [[to_bucket(v) for v in row] for row in M]
    bitems = list(zip(*buck))
    within1 = sum(1 for vals in items if (max(vals) - min(vals)) <= 5) / len(items)
    return {
        "krippendorff_alpha_ordinal": kalpha(M, "ordinal"),
        "acuerdo_buckets": agreement_props(bitems),
        "p_acuerdo_pm1_ancla": within1,
        "matriz_confusion_anclas": ANCHORS,
        "matriz_confusion": confusion_pairwise(bitems, ANCHORS).tolist(),
        "media_global": float(np.mean([v for row in M for v in row])),
    }

res_C = {"escala": "ordinal/continua 0-20 (Direct Assessment, 5 anclas)",
         "n_items_compartidos": 15, "n_anotadores": 3,
         "Accuracy": analyze_C_dim("acc"), "Fluency": analyze_C_dim("flu")}

# casos de mayor desacuerdo en C (por rango)
disC = []
for sid in shared:
    sub = dfC[dfC["id"] == sid]
    av = sub["acc"].tolist(); fv = sub["flu"].tolist()
    ra, rf = max(av) - min(av), max(fv) - min(fv)
    disC.append((max(ra, rf), ra, rf, sid, av, fv))
disC.sort(reverse=True)
topC = []
for mx, ra, rf, sid, av, fv in disC[:3]:
    row = dfC[dfC["id"] == sid].iloc[0]
    topC.append({"id": sid, "rango_acc": ra, "rango_flu": rf,
                 "acc": av, "flu": fv, "ref": row["ref"], "mt": row["mt"], "qom": row["qom"]})
res_C["casos_mayor_desacuerdo"] = topC
results["C_externo"] = res_C

# ----------------------------------------------------------------------------- métricas automáticas (C)
uni = dfC.drop_duplicates(subset="id").copy()
uni["ref_s"] = uni["ref"].astype(str).str.strip()
uni["mt_s"] = uni["mt"].astype(str).str.strip()
valid = uni[(uni["ref_s"] != "") & (uni["mt_s"] != "")].copy()
n_excl = len(uni) - len(valid)
refs = valid["ref_s"].tolist()
hyps = valid["mt_s"].tolist()
print(f"Segmentos únicos: {len(uni)}  válidos: {len(valid)}  excluidos: {n_excl}")

from sacrebleu.metrics import BLEU
bleu_metric = BLEU()
bleu_corpus = bleu_metric.corpus_score(hyps, [refs])
bleu_signature = str(bleu_metric.get_signature())
valid["sent_bleu"] = [sacrebleu.sentence_bleu(h, [r], smooth_method="exp").score
                      for h, r in zip(hyps, refs)]

print("Calculando BERTScore (BETO)...")
from bert_score import score as bertscore
BETO = "dccuchile/bert-base-spanish-wwm-cased"
try:
    P, R, F1 = bertscore(hyps, refs, model_type=BETO, num_layers=9,
                         lang="es", rescale_with_baseline=False, verbose=False)
    bert_model_used = f"{BETO} (capa 9)"
except Exception as e:
    print("Fallback a num_layers default:", e)
    P, R, F1 = bertscore(hyps, refs, model_type=BETO, lang="es",
                         rescale_with_baseline=False, verbose=False)
    bert_model_used = f"{BETO} (capa por defecto)"
valid["bert_f1"] = F1.numpy()
valid["bert_p"] = P.numpy()
valid["bert_r"] = R.numpy()

results["metricas"] = {
    "n_segmentos_unicos": int(len(uni)),
    "n_evaluados": int(len(valid)),
    "n_excluidos_vacios": int(n_excl),
    "BLEU": {
        "corpus_bleu": float(bleu_corpus.score),
        "signature": bleu_signature,
        "sent_bleu_media": float(valid["sent_bleu"].mean()),
        "sent_bleu_mediana": float(valid["sent_bleu"].median()),
        "sent_bleu_p_cero": float((valid["sent_bleu"] == 0).mean()),
        "smoothing": "exp", "tokenizer": "13a",
        "sacrebleu_version": sacrebleu.__version__,
    },
    "BERTScore": {
        "modelo": bert_model_used,
        "F1_media": float(valid["bert_f1"].mean()),
        "F1_mediana": float(np.median(valid["bert_f1"])),
        "P_media": float(valid["bert_p"].mean()),
        "R_media": float(valid["bert_r"].mean()),
        "rescale_with_baseline": False,
    },
}

# ----------------------------------------------------------------------------- correlación humano <-> métrica
hum = dfC.groupby("id").agg(h_acc=("acc", "mean"), h_flu=("flu", "mean")).reset_index()
hum["h_over"] = (hum["h_acc"] + hum["h_flu"]) / 2
m = valid.merge(hum, on="id", how="left")

def corr_block(x, y):
    pr, pp = stats.pearsonr(x, y)
    return {"pearson_r": float(pr), "pearson_p": float(pp), "n": int(len(x))}

corr = {}
for met, mc in [("BLEU", "sent_bleu"), ("BERTScore_F1", "bert_f1")]:
    corr[met] = {}
    for hh, hc in [("Accuracy", "h_acc"), ("Fluency", "h_flu"), ("Overall", "h_over")]:
        corr[met][hh] = corr_block(m[mc].values, m[hc].values)
results["correlacion"] = {"todos_unicos": corr}

# robustez: solo 15 compartidos (medias de 3)
ms = m[m["id"].isin(shared)]
corr_sh = {}
for met, mc in [("BLEU", "sent_bleu"), ("BERTScore_F1", "bert_f1")]:
    corr_sh[met] = {}
    for hh, hc in [("Accuracy", "h_acc"), ("Fluency", "h_flu"), ("Overall", "h_over")]:
        corr_sh[met][hh] = corr_block(ms[mc].values, ms[hc].values)
results["correlacion"]["solo_compartidos_n15"] = corr_sh

# ----------------------------------------------------------------------------- timing crudo (para Metodología; no se usa en Resultados)
try:
    raw = pd.read_excel(F_C_EXT, sheet_name="C")
    tcol = [c for c in raw.columns if str(c).startswith("Unnamed")]
    times = {}
    for _, rr in raw.iterrows():
        for c in tcol:
            val = rr[c]
            if isinstance(val, str) and "Tiempo en minutos" in val:
                # buscar valor en la misma fila en otra unnamed col
                for c2 in tcol:
                    v2 = rr[c2]
                    if v2 is not None and not (isinstance(v2, float) and np.isnan(v2)) and not (isinstance(v2, str)):
                        times[val.strip()] = str(v2)
    results["timing_crudo_C"] = times
except Exception as e:
    results["timing_crudo_C"] = {"error": str(e)}

# ----------------------------------------------------------------------------- guardar JSON
with open(os.path.join(OUT_JSON_DIR, "resultados.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# guardar tabla por-segmento (trazabilidad)
m_out = m[["id", "h_acc", "h_flu", "h_over", "sent_bleu", "bert_f1", "ref_s", "mt_s"]]
m_out.to_csv(os.path.join(OUT_JSON_DIR, "por_segmento_C.csv"), index=False, encoding="utf-8")

print("\n==== RESUMEN ====")
print("alpha A sin guía (ordinal):", round(a0_alpha, 3))
print("alpha A con guía (global):", round(res_Ap["global"]["krippendorff_alpha_ordinal"], 3))
print("alpha B con guía (global):", round(res_Bp["global"]["krippendorff_alpha_ordinal"], 3))
print("alpha C Accuracy (ordinal):", round(res_C["Accuracy"]["krippendorff_alpha_ordinal"], 3))
print("alpha C Fluency (ordinal):", round(res_C["Fluency"]["krippendorff_alpha_ordinal"], 3))
print("BLEU corpus:", round(bleu_corpus.score, 2))
print("BERTScore F1 media:", round(float(valid["bert_f1"].mean()), 4))
for met in corr:
    for hh in corr[met]:
        c = corr[met][hh]
        print(f"corr {met} vs {hh}: r={c['pearson_r']:.3f} (p={c['pearson_p']:.3g}) N={c['n']}")
print("OK -> resultados/resultados.json")

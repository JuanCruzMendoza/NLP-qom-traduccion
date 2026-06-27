# -*- coding: utf-8 -*-
"""Baseline por azar (permutación) para BLEU y BERTScore sobre el conjunto C."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
import sacrebleu
from sacrebleu.metrics import BLEU
from bert_score import score as bertscore

RES = r"C:\Users\Juan Cruz\Desktop\GitHub\activos\temas_NLP_TP\resultados\por_segmento_C.csv"
df = pd.read_csv(RES)
refs = df["ref_s"].astype(str).str.strip().tolist()
hyps = df["mt_s"].astype(str).str.strip().tolist()
n = len(refs)
BETO = "dccuchile/bert-base-spanish-wwm-cased"

# --- real ---
bleu = BLEU()
real_bleu = bleu.corpus_score(hyps, [refs]).score
_, _, F1 = bertscore(hyps, refs, model_type=BETO, num_layers=9, lang="es",
                     rescale_with_baseline=False, verbose=False)
real_bert = float(F1.mean())

# --- baseline por permutación (K barajados) ---
rng = np.random.default_rng(0)
K = 5
sh_bleu, sh_bert = [], []
for k in range(K):
    perm = rng.permutation(n)
    # evitar emparejamientos identidad accidentales
    for i in range(n):
        if perm[i] == i:
            perm[i] = (perm[i] + 1) % n
    refs_sh = [refs[i] for i in perm]
    sh_bleu.append(bleu.corpus_score(hyps, [refs_sh]).score)
    _, _, f = bertscore(hyps, refs_sh, model_type=BETO, num_layers=9, lang="es",
                        rescale_with_baseline=False, verbose=False)
    sh_bert.append(float(f.mean()))

b_bleu = float(np.mean(sh_bleu))
b_bert = float(np.mean(sh_bert))

# --- piso identidad para BLEU: copiar la fuente qom como "traducción" ---
qom = pd.read_excel(r"C:\Users\Juan Cruz\Desktop\GitHub\activos\temas_NLP_TP\datasets_anotados\dataset_C_criterio_externo.xlsx",
                    sheet_name="C")
qom = qom.drop_duplicates("Column 1")
qom_hyp = qom["qom"].astype(str).str.strip().tolist()
qom_ref = qom["ref"].astype(str).str.strip().tolist()
copy_bleu = bleu.corpus_score(qom_hyp, [qom_ref]).score

def rescale(raw, base):
    return (raw - base) / (1 - base)

print(f"N = {n}\n")
print(f"{'':26}{'real':>8}{'azar':>8}{'reescalado':>12}")
print(f"{'BLEU (corpus)':26}{real_bleu:>8.2f}{b_bleu:>8.2f}{rescale(real_bleu/100, b_bleu/100)*100:>12.2f}")
print(f"{'BERTScore F1 (media)':26}{real_bert:>8.3f}{b_bert:>8.3f}{rescale(real_bert, b_bert):>12.3f}")
print(f"\nBLEU copiando la fuente qom como hipótesis (piso 'idioma equivocado'): {copy_bleu:.2f}")
print(f"BLEU por permutación, cada corrida: {[round(x,2) for x in sh_bleu]}")
print(f"BERTScore por permutación, cada corrida: {[round(x,3) for x in sh_bert]}")

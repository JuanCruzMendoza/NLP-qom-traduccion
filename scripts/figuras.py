# -*- coding: utf-8 -*-
"""Genera las figuras del informe a partir de resultados.json y por_segmento_C.csv."""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raíz del repo
RES = os.path.join(ROOT, "resultados")
FIG = os.path.join(ROOT, "informe", "figuras")
os.makedirs(FIG, exist_ok=True)

with open(os.path.join(RES, "resultados.json"), encoding="utf-8") as f:
    R = json.load(f)
seg = pd.read_csv(os.path.join(RES, "por_segmento_C.csv"))

plt.rcParams.update({"font.size": 9, "font.family": "serif",
                     "axes.titlesize": 9, "figure.dpi": 150})

def heatmap(ax, M, labels, title):
    M = np.array(M)
    im = ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
    thr = M.max() * 0.6 if M.max() > 0 else 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(M[i, j]), ha="center", va="center",
                    color="white" if M[i, j] > thr else "black", fontsize=8)
    ax.set_title(title)
    return im

# --------- Figura 1: matrices de confusión ----------
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
heatmap(axes[0], R["A_sin_guia"]["matriz_confusion"],
        ["Inc.", "Parc.", "Corr."], "(a) A sin guía (global)")
heatmap(axes[1], R["C_externo"]["Accuracy"]["matriz_confusion"],
        ["0", "5", "10", "15", "20"], "(b) C – Precisión")
im = heatmap(axes[2], R["C_externo"]["Fluency"]["matriz_confusion"],
             ["0", "5", "10", "15", "20"], "(c) C – Fluidez")
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="pares de anotadores")
fig.suptitle("Matrices de confusión (coincidencia por pares de anotadores)", y=1.02)
fig.savefig(os.path.join(FIG, "confusion.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(FIG, "confusion.png"), bbox_inches="tight")
plt.close(fig)

# --------- Figura 2: dispersión métrica vs juicio humano (Precisión) ----------
def scatter(ax, x, y, xlab, ylab, r):
    ax.scatter(x, y, s=18, alpha=0.55, edgecolor="none", color="#1f4e79")
    z = np.polyfit(x, y, 1); xs = np.linspace(min(x), max(x), 50)
    ax.plot(xs, np.polyval(z, xs), color="#c0392b", lw=1.3)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.text(0.04, 0.92, f"r = {r:.2f}", transform=ax.transAxes,
            fontsize=9, va="top")

rB = R["correlacion"]["todos_unicos"]["BLEU"]["Accuracy"]["pearson_r"]
rT = R["correlacion"]["todos_unicos"]["BERTScore_F1"]["Accuracy"]["pearson_r"]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
scatter(axes[0], seg["h_acc"], seg["sent_bleu"], "Precisión humana (0–20)", "BLEU (oración)", rB)
scatter(axes[1], seg["h_acc"], seg["bert_f1"], "Precisión humana (0–20)", "BERTScore F1", rT)
fig.suptitle("Métricas automáticas vs. juicio humano de Precisión (conjunto C, N=120)", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "scatter.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(FIG, "scatter.png"), bbox_inches="tight")
plt.close(fig)

print("Figuras generadas en", FIG)

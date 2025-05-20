#!/usr/bin/env python
# compare_rag_vs_no_rag.py
"""
Compare deux CSV d’évaluation (RAG vs. no-RAG) en agrégeant
les métriques par document (10 lignes / doc) puis en traçant
un histogramme côte-à-côte pour chaque métrique.

• Entrées  : rag_eval_results.csv, no_rag_eval_results.csv
• Sorties  : histogrammes PNG dans ./plots/
"""

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ───────────────────────────
# Chargement
# ───────────────────────────
rag_path     = Path(__file__).resolve().parent / "rag_eval_results.csv"
no_rag_path  = Path(__file__).resolve().parent / "no_rag_eval_results.csv"

df_rag    = pd.read_csv(rag_path)
df_norag  = pd.read_csv(no_rag_path)


# ───────────────────────────
# Sélection des métriques numériques
# ───────────────────────────
exclude = {"question", "answer", "prediction", "doc_name"}
metrics = [c for c in df_rag.columns
           if c not in exclude and pd.api.types.is_numeric_dtype(df_rag[c])]

# ───────────────────────────
# Agrégation par document
# ───────────────────────────
rag_mean   = df_rag.groupby("doc_name")[metrics].mean().add_suffix("_rag")
norag_mean = df_norag.groupby("doc_name")[metrics].mean().add_suffix("_norag")

combined = rag_mean.join(norag_mean)

# ──────────────────────────
# Traçage des histogrammes (version horizontale)
# ───────────────────────────
plot_dir = Path("tests/plots")
plot_dir.mkdir(exist_ok=True)

for metric in metrics:
    cols = [f"{metric}_rag", f"{metric}_norag"]

    # hauteur dynamique : ~0,45 po par document
    fig_height = max(4, 0.45 * len(combined))

    ax = (
        combined[cols]
        .plot.barh(
            figsize=(8, fig_height),
            title=f"Moyenne par document – {metric}",
            xlabel=metric
        )
    )

    ax.legend(["RAG", "no-RAG"])
    plt.tight_layout()                  # ou bbox_inches="tight" dans savefig
    out = plot_dir / f"{metric}_compare_h.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] {out}")

print("Tous les graphiques horizontaux sont disponibles dans le dossier 'plots/'.")
# ───────────────────────────
# Rassembler les valeurs tracées dans un DataFrame et les sauver
# ───────────────────────────
records = []

for metric in metrics:
    # Sous-ensemble à deux colonnes + index (doc_name)
    subset = combined[[f"{metric}_rag", f"{metric}_norag"]].copy()
    subset = subset.reset_index()                                    # doc_name redevient colonne
    subset = subset.rename(columns={
        f"{metric}_rag":  "rag",
        f"{metric}_norag": "no_rag"
    })
    subset["metric"] = metric                                        # nouvelle colonne
    records.append(subset)

plot_df = pd.concat(records, ignore_index=True)[
    ["metric", "doc_name", "rag", "no_rag"]
]

# Chemin de sortie (on réutilise ./plots/)
csv_path = Path(__file__).resolve().parent / "plot_values.csv"
plot_df.to_csv(csv_path, index=False)
print(f"[✓] Valeurs agrégées sauvegardées dans {csv_path}")



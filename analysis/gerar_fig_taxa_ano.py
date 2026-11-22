"""Taxa de baixo peso por ano (texto do TCC), a partir do resultados.json do modelo v4.

Mesmo estilo da figura do notebook 10, com os anos inclinados (na figura do notebook, os 14 rótulos se sobrepõem)
e os períodos de treino, validação e teste indicados. Uso: python gerar_fig_taxa_ano.py (precisa de matplotlib).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

AQUI = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(AQUI, "..", "results", "model_v4", "resultados.json"), encoding="utf-8"))
pa = R["por_ano"]
anos = [str(a["ano_nasc"]) for a in pa]
# taxa calculada das contagens: o valor gravado já vem arredondado a 2 casas (7,55), e reformatá-lo com 1 casa
# dava 7,5 em vez de 7,6 (duplo arredondamento)
taxas = [a["baixo_peso"] / a["nascimentos"] * 100 for a in pa]
cores = ["#4C72B0" if a["ano_nasc"] <= 2021 else ("#8FA9D3" if a["ano_nasc"] <= 2023 else "#C44E52") for a in pa]

plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.bar(anos, taxas, color=cores)
ax.set_ylabel("Baixo peso (%)")
ax.set_xlabel("Ano de nascimento")
ax.set_ylim(0, max(taxas) * 1.3)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}".replace(".", ",")))
for i, v in enumerate(taxas):
    ax.text(i, v + 0.12, f"{v:.1f}".replace(".", ","), ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(len(anos)))
ax.set_xticklabels([a + ("*" if a == "2025" else "") for a in anos], rotation=45, ha="right")
ax.legend(handles=[Patch(color="#4C72B0", label="Treino"), Patch(color="#8FA9D3", label="Validação"),
                   Patch(color="#C44E52", label="Teste")], loc="upper left", ncol=3, fontsize=8, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(AQUI, "..", "results", "figures", "fig_taxa_ano.png"), facecolor="white")
print("ok")

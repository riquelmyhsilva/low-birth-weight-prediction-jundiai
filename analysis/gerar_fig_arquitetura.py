"""Figura da arquitetura do pipeline (texto do TCC). Uso: python gerar_fig_arquitetura.py (precisa de matplotlib)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

AQUI = os.path.dirname(os.path.abspath(__file__))

# (título, notebook, controle) por etapa; cores: preparação dos arquivos em cinza, camadas medalhão em tons próprios
LINHA1 = [
    ("DATASUS\n(FTP)", "", "arquivos .dbc\nanuais por UF", "#e8e8e8"),
    ("Zona de pouso\n(.dbc)", "01_01", "manifesto: tamanho\ne data no FTP", "#e8e8e8"),
    ("Descompactação\n(.dbf)", "01_02", "tamanho = cabeçalho\ndo DBF", "#e8e8e8"),
    ("Conversão\n(.csv)", "01_03", "linhas = registros\nativos do DBF", "#e8e8e8"),
]
LINHA2 = [
    ("Bronze\n(Delta, texto)", "02", "contagem =\nmanifesto", "#e9d3bd"),
    ("Silver\n(tipada, todas\nas UFs e anos)", "03", "contagem por partição;\nlinhagem do arquivo", "#dcdfe3"),
    ("Gold de ML\n(recorte do estudo)", "06", "regras da população;\nrelatório de ausência", "#f1e2a6"),
    ("Modelo\n(Random Forest)", "10", "imputação, treino,\ncalibração, avaliação", "#cfe0d4"),
]

fig, ax = plt.subplots(figsize=(10, 4.7), dpi=200)
ax.set_xlim(0, 10)
ax.set_ylim(0.45, 5.15)
ax.axis("off")
L, A = 1.9, 1.05  # largura e altura das caixas
xs = [0.25 + i * 2.5 for i in range(4)]


def caixa(x, y, titulo, nb, controle, cor):
    ax.add_patch(FancyBboxPatch((x, y), L, A, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=cor, ec="#333333", lw=1))
    ax.text(x + L / 2, y + A / 2 + (0.08 if nb else 0), titulo, ha="center", va="center", fontsize=9.5)
    if nb:
        ax.text(x + L / 2, y + 0.1, f"notebook {nb}", ha="center", va="bottom", fontsize=7.5, color="#444444",
                style="italic")
    ax.text(x + L / 2, y - 0.12, controle, ha="center", va="top", fontsize=7.5, color="#333333")


def seta(x0, y0, x1, y1, estilo="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=estilo, mutation_scale=12, lw=1.1, color="#333333",
                                 connectionstyle="arc3,rad=0"))


y1, y2 = 3.75, 1.2
for i, (t, nb, c, cor) in enumerate(LINHA1):
    caixa(xs[i], y1, t, nb, c, cor)
    if i:
        seta(xs[i - 1] + L, y1 + A / 2, xs[i], y1 + A / 2)
for i, (t, nb, c, cor) in enumerate(LINHA2):
    caixa(xs[i], y2, t, nb, c, cor)
    if i:
        seta(xs[i - 1] + L, y2 + A / 2, xs[i], y2 + A / 2)
# da conversão (fim da linha 1) para a Bronze (início da linha 2)
y_meio = 2.95
ax.plot([xs[3] + L / 2, xs[3] + L / 2, xs[0] + L / 2], [y1 - 0.62, y_meio, y_meio], color="#333333", lw=1.1)
seta(xs[0] + L / 2, y_meio, xs[0] + L / 2, y2 + A + 0.02)
ax.text(5.0, 4.98, "Preparação dos arquivos", ha="center", fontsize=9, weight="bold")
ax.text(5.0, 2.47, "Arquitetura medalhão (Databricks, tabelas Delta) e modelagem", ha="center", fontsize=9, weight="bold")
fig.savefig(os.path.join(AQUI, "..", "results", "figures", "fig_arquitetura.png"), bbox_inches="tight", facecolor="white")
print("ok")

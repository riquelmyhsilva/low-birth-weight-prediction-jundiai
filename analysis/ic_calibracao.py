"""Calibração com IC95% (bootstrap, 1000 reamostras, semente 42), com e sem a calibração de Platt.

Roda localmente sobre `predicoes_validacao_teste.csv` (gravado pelo notebook 10, v4). O notebook calcula só as
estimativas pontuais; este script acrescenta:
- os IC95% do intercepto, da inclinação e da razão O/E da probabilidade calibrada (pedidos pelo TRIPOD);
- as mesmas medidas, a média prevista e o escore de Brier do escore bruto do Random Forest (sem Platt), para
  comparar as duas versões;
- o IC95% da AUC-ROC observada com 1000 reamostras nos dois períodos, para que a Tabela de case-mix use o mesmo
  intervalo da tabela de métricas (no teste, a reamostragem reproduz a do notebook).
A inclinação é estimada por Newton-Raphson (máxima verossimilhança exata). Até 22/11/2026 usava-se a
LogisticRegression do scikit-learn, que parava antes da convergência completa (0,6991 em vez de 0,7012 no teste).
Uso: python ic_calibracao.py [caminho_do_csv]
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.metrics import average_precision_score, roc_auc_score


def logit(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def inclinacao_mv(x, y, iteracoes=50):
    """Regressão logística de y em x (com intercepto) por Newton-Raphson; devolve o coeficiente de x."""
    X = np.column_stack([np.ones_like(x), x])
    b = np.zeros(2)
    for _ in range(iteracoes):
        p = 1 / (1 + np.exp(-X @ b))
        passo = np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]), X.T @ (y - p))
        b += passo
        if np.abs(passo).max() < 1e-10:
            break
    return b[1]


def calibracao(y, q):
    """Mesmas definições do notebook 10: intercepto com logit(q) como offset; inclinação; observado/esperado."""
    lq = logit(q)
    intercepto = brentq(lambda a: np.sum(y - 1 / (1 + np.exp(-(a + lq)))), -5, 5)
    return intercepto, inclinacao_mv(lq, y), y.mean() / q.mean()


def bloco(est, reamostras):
    lo, hi = np.percentile(np.array(reamostras), [2.5, 97.5], axis=0)
    return {nome: {"estimativa": round(float(est[k]), 4), "ic95": [round(float(lo[k]), 4), round(float(hi[k]), 4)]}
            for k, nome in enumerate(["intercepto", "inclinacao", "observado_esperado"])}


def ic_auc(y, p, B=1000, semente=42):
    """Mesmo procedimento do bootstrap do notebook 10 (gerador novo com a semente, reamostras sem um só desfecho
    descartadas)."""
    rng = np.random.default_rng(semente)
    aucs = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        aucs.append(roc_auc_score(y[i], p[i]))
    return [round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    csv = sys.argv[1] if len(sys.argv) > 1 else "predicoes_validacao_teste.csv"
    p = pd.read_csv(csv)
    saida = {}
    for periodo in ["validacao", "teste"]:
        d = p[p.periodo == periodo]
        y, q, s = d.baixo_peso.to_numpy(), d.prob_calibrada.to_numpy(), d.escore_bruto.to_numpy()
        est_q, est_s = calibracao(y, q), calibracao(y, s)
        rng = np.random.default_rng(42)
        rq, rs = [], []
        for _ in range(1000):
            i = rng.integers(0, len(y), len(y))
            rq.append(calibracao(y[i], q[i]))
            rs.append(calibracao(y[i], s[i]))
        saida[periodo] = bloco(est_q, rq)
        saida[periodo]["media_prevista"] = round(float(q.mean()), 4)
        saida[periodo]["brier"] = round(float(np.mean((q - y) ** 2)), 5)
        saida[periodo]["bruto"] = bloco(est_s, rs)
        saida[periodo]["bruto"]["media_prevista"] = round(float(s.mean()), 4)
        saida[periodo]["bruto"]["brier"] = round(float(np.mean((s - y) ** 2)), 5)
        saida[periodo]["prevalencia"] = round(float(y.mean()), 4)
        saida[periodo]["auc_roc"] = round(float(roc_auc_score(y, s)), 4)
        saida[periodo]["auc_roc_ic95_bootstrap"] = ic_auc(y, s)
        print(periodo, "| AUC-ROC", saida[periodo]["auc_roc"], saida[periodo]["auc_roc_ic95_bootstrap"],
              "| AUC-PR", round(average_precision_score(y, s), 4))
        for rot, b in (("Platt", saida[periodo]), ("bruto", saida[periodo]["bruto"])):
            print(f"  {rot}: média {b['media_prevista']:.4f} | Brier {b['brier']:.5f} | " + " | ".join(
                f"{k} {b[k]['estimativa']:.4f} ({b[k]['ic95'][0]:.4f} a {b[k]['ic95'][1]:.4f})"
                for k in ("intercepto", "inclinacao", "observado_esperado")))
    # Lido pelo gerador do texto (ml_tcc/gerar_esboco_tcc.py)
    with open(os.path.join(os.path.dirname(os.path.abspath(csv)), "ic_calibracao.json"), "w", encoding="utf-8") as f:
        json.dump({"bootstrap": 1000, "semente": 42, "metodo_inclinacao": "Newton-Raphson", **saida}, f,
                  ensure_ascii=False, indent=2)

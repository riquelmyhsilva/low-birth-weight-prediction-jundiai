# Databricks notebook source
# MAGIC %md
# MAGIC # TCC — Random Forest para baixo peso ao nascer na RMJ (versão 4)
# MAGIC
# MAGIC - **Fonte:** Gold de ML (`03_gold/ml_baixo_peso_rmj`, gerada pelo notebook de engenharia `06_load_gold_ml`). A tabela já traz o recorte (fonte por ano, 7 municípios da Região Metropolitana de Jundiaí, 2012–2025, sem duplicatas), os atributos tratados, as flags da população e o alvo. O relatório da população (fluxo de exclusões, auditoria de domínios, ausência) vem de `03_gold/_control/estado_gold_ml.json`.
# MAGIC - **População:** linhas com `populacao_estudo` (gestação única, peso registrado e ≥ 500 g).
# MAGIC - **Alvo:** `baixo_peso` (peso < 2.500 g).
# MAGIC - **Atributos:** somente maternos e de histórico obstétrico compatíveis com o início do pré-natal.
# MAGIC - **Divisão temporal:** treino até 2021, validação 2022–2023 (hiperparâmetros, calibração e limiares), teste 2024–2025.
# MAGIC
# MAGIC - **Dados ausentes (TRIPOD, Moons et al., 2015, Box D):** valores fora da faixa ou do domínio e o código 9 (ignorado) são tratados como ausentes e preenchidos por **imputação múltipla** (equações encadeadas, M = 5), ajustada só no treino. O modelo final é a média das M florestas. Não há categoria "ignorado" nem indicador de ausência.
# MAGIC - **Relato (TRIPOD):** tamanho e eventos por período, comparação dos atributos entre períodos, ausência por período, calibração (intercepto, inclinação e curva), efeito de *case-mix* na AUC (Vergouwe; Moons; Steyerberg, 2010) e probabilidades previstas gravadas.
# MAGIC
# MAGIC Este notebook só faz o que aprende com os dados (imputação, codificação, treino, calibração, avaliação). Para mudar o recorte ou a população, altere os parâmetros do `06_load_gold_ml`. Resultados gravados na pasta `saida`.

# COMMAND ----------

import os, json, time
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import sklearn
from scipy.optimize import brentq
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 (habilita o IterativeImputer)
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, confusion_matrix, brier_score_loss)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Parâmetros

# COMMAND ----------

PADRAO = {
    "tabela_ml": "/Volumes/workspace/sinasc/03_gold/ml_baixo_peso_rmj",
    "estado_gold_ml": "/Volumes/workspace/sinasc/03_gold/_control/estado_gold_ml.json",
    "ano_fim_treino": "2021",
    "ano_fim_validacao": "2023",
    "n_bootstrap": "1000",
    "n_repeticoes_permutacao": "30",
    "n_imputacoes": "5",
    "n_sorteios_casemix": "500",
    "saida": "/Volumes/workspace/sinasc/03_gold/_tcc_ml/v4_tripod/",
}
for k, v in PADRAO.items():
    dbutils.widgets.text(k, v)
P = {k: dbutils.widgets.get(k).strip() for k in PADRAO}

# Estado da Gold de ML: parâmetros do recorte e relatório da população
with open(P["estado_gold_ml"], encoding="utf-8") as f:
    GOLD = json.load(f)
versao_atual = spark.sql(f"DESCRIBE HISTORY delta.`{P['tabela_ml']}` LIMIT 1").collect()[0]["version"]
assert versao_atual == GOLD["versao_tabela"], (
    f"Tabela na versão {versao_atual}, relatório da versão {GOLD['versao_tabela']}: rode o 06_load_gold_ml")
G = GOLD["parametros"]
ANO_INI, ANO_FIM = int(G["ano_inicio"]), int(G["ano_fim"])
PESO_MIN = int(G["peso_min_g"])
FIM_TREINO, FIM_VAL = int(P["ano_fim_treino"]), int(P["ano_fim_validacao"])
N_BOOT, N_PERM = int(P["n_bootstrap"]), int(P["n_repeticoes_permutacao"])
M_IMP, B_CASEMIX = int(P["n_imputacoes"]), int(P["n_sorteios_casemix"])
SAIDA = P["saida"].rstrip("/") + "/"
os.makedirs(SAIDA, exist_ok=True)
SEED = 42

# Atributos (já tratados na Gold de ML: numéricos fora da faixa = nulo; categóricos fora do domínio = "9")
NUM = GOLD["colunas_atributos"]["numericos"]
CAT = GOLD["colunas_atributos"]["categoricos"]
HIST = NUM[1:]
ROTULO = {"idade_mae": "Idade da mãe", "qtd_gestacoes_anteriores": "Gestações anteriores",
          "qtd_partos_normais": "Partos vaginais anteriores", "qtd_partos_cesareos": "Cesáreas anteriores",
          "qtd_filhos_vivos": "Filhos vivos", "qtd_perdas_fetais_abortos": "Perdas fetais/abortos",
          "id_raca_cor_mae": "Raça/cor da mãe", "id_escolaridade_mae_2010": "Escolaridade da mãe",
          "id_estado_civil_mae": "Situação conjugal",
          "historico_obstetrico": "Histórico obstétrico (5 variáveis em bloco)"}
ROT_CAT = {"id_raca_cor_mae": {"1": "Branca", "2": "Preta", "3": "Amarela", "4": "Parda", "5": "Indígena", "9": "Ignorado"},
           "id_escolaridade_mae_2010": {"0": "Sem escolaridade", "1": "Fundamental I", "2": "Fundamental II",
                                        "3": "Médio", "4": "Superior incompleto", "5": "Superior completo", "9": "Ignorado"},
           "id_estado_civil_mae": {"1": "Solteira", "2": "Casada", "3": "Viúva", "4": "Separada/divorciada",
                                   "5": "União estável", "9": "Ignorado"}}
print(P)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Leitura da Gold de ML

# COMMAND ----------

df = spark.read.format("delta").load(P["tabela_ml"]).toPandas()
# Ordem determinística: o Random Forest depende da ordem das linhas (bootstrap), e o Spark não garante
# a ordem de leitura. id_linha é atribuído na Gold pela ordem das colunas brutas.
df = df.sort_values("id_linha", kind="mergesort").reset_index(drop=True)

REL = GOLD["relatorio"]
fonte_por_ano = REL["fonte_por_ano"]
fonte = {r["ano"]: r["uf_fonte"] for r in fonte_por_ano}
print(fonte_por_ano)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. População do estudo (flags da Gold) e relatório de exclusões e domínios

# COMMAND ----------

m_peso_reg = df["peso_registrado"].astype(bool)
m_peso_ok = df["peso_minimo_ok"].astype(bool)
m_unica = df["gestacao_unica"].astype(bool)
m_pop = df["populacao_estudo"].astype(bool)

fluxo, excluidos = REL["fluxo_populacao"], REL["excluidos"]
auditoria, ausencia = REL["auditoria_dominios"], REL["ausencia"]

def preparar(d):
    """Matriz do modelo. Numéricos: nulo onde ausente (a Gold já anulou valores fora da faixa).
    Categóricos: código numérico; o código 9 (ignorado ou fora do domínio, na Gold) vira ausente (NaN),
    a ser preenchido pela imputação múltipla. Não há categoria "ignorado" nem indicador de ausência."""
    X = d[NUM].astype(float).copy()
    for c in CAT:
        X[c] = pd.to_numeric(d[c], errors="coerce").where(d[c] != "9")
    return X

pop = df[m_pop].copy()
Xpop = pop[NUM + CAT].copy()  # valores como registrados, só para a análise descritiva
n_pop = len(pop)
assert n_pop == GOLD["linhas_populacao_estudo"] == fluxo[-1]["n"], (n_pop, GOLD["linhas_populacao_estudo"])
print(fluxo); print(excluidos); print(auditoria); print(ausencia)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Perfil descritivo da população (por ano, município e atributo)

# COMMAND ----------

por_ano = (pop.groupby("ano_nasc").agg(nascimentos=("baixo_peso", "size"), baixo_peso=("baixo_peso", "sum")).reset_index())
por_ano["taxa_baixo_peso_pct"] = (por_ano.baixo_peso / por_ano.nascimentos * 100).round(2)
por_ano["uf_fonte"] = por_ano.ano_nasc.map(fonte)
por_mun = (pop.groupby("municipio").agg(nascimentos=("baixo_peso", "size"), baixo_peso=("baixo_peso", "sum")).reset_index())
por_mun["taxa_baixo_peso_pct"] = (por_mun.baixo_peso / por_mun.nascimentos * 100).round(2)

def faixa(v, cortes, rot):
    out = pd.Series("Ignorado", index=v.index)
    for (lo, hi), r in zip(cortes, rot):
        out[(v >= lo) & (v <= hi)] = r
    return out

desc_cols = {
    "idade_mae": faixa(Xpop.idade_mae, [(10, 19), (20, 34), (35, 55)], ["10 a 19", "20 a 34", "35 ou mais"]),
    "qtd_gestacoes_anteriores": faixa(Xpop.qtd_gestacoes_anteriores, [(0, 0), (1, 1), (2, 2), (3, 30)], ["0", "1", "2", "3 ou mais"]),
    "qtd_partos_normais": faixa(Xpop.qtd_partos_normais, [(0, 0), (1, 1), (2, 30)], ["0", "1", "2 ou mais"]),
    "qtd_partos_cesareos": faixa(Xpop.qtd_partos_cesareos, [(0, 0), (1, 1), (2, 30)], ["0", "1", "2 ou mais"]),
    "qtd_filhos_vivos": faixa(Xpop.qtd_filhos_vivos, [(0, 0), (1, 1), (2, 2), (3, 30)], ["0", "1", "2", "3 ou mais"]),
    "qtd_perdas_fetais_abortos": faixa(Xpop.qtd_perdas_fetais_abortos, [(0, 0), (1, 1), (2, 30)], ["0", "1", "2 ou mais"]),
    **{c: Xpop[c].map(ROT_CAT[c]) for c in CAT},
}
descritiva = []
for c, cat in desc_cols.items():
    g = pd.DataFrame({"cat": cat, "y": pop.baixo_peso}).groupby("cat", sort=False).y.agg(["size", "sum"])
    ordem = [k for k in (list(ROT_CAT[c].values()) if c in CAT else []) if k in g.index] or None
    if ordem is None:
        ordem = [k for k in g.index if k != "Ignorado"]
        ordem = sorted(ordem, key=lambda s: int(s.split()[0])) + (["Ignorado"] if "Ignorado" in g.index else [])
    for k in ordem:
        descritiva.append({"variavel": c, "categoria": k, "n": int(g.loc[k, "size"]),
                           "pct": round(float(g.loc[k, "size"] / n_pop * 100), 2),
                           "baixo_peso": int(g.loc[k, "sum"]),
                           "taxa_baixo_peso_pct": round(float(g.loc[k, "sum"] / g.loc[k, "size"] * 100), 2)})
print(por_ano.to_string()); print(por_mun.to_string()); print(pd.DataFrame(descritiva).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Imputação múltipla e modelo (ajustados só no treino)
# MAGIC
# MAGIC Cada uma das M imputações é estocástica e usa semente própria. Numéricos: equações encadeadas com regressão bayesiana e sorteio da distribuição posterior, arredondados e limitados ao intervalo observado no treino. Categóricos: regressão logística multinomial sobre os demais atributos, com a categoria sorteada pelas probabilidades previstas. Para cada imputação treina-se uma floresta; a probabilidade prevista é a média das M florestas. Validação e teste são imputados com os modelos de imputação ajustados no treino.

# COMMAND ----------

class ImputadorMultiplo(BaseEstimator, TransformerMixin):
    """Uma das M imputações estocásticas (equações encadeadas), ajustada só no treino."""

    def __init__(self, semente=0):
        self.semente = semente

    def _cat_preenchidos(self, X):
        # categóricos com a moda do treino no lugar dos ausentes (só para servir de preditores)
        return X[CAT].fillna(pd.Series(self.moda_)).astype(int).astype(str)

    def _base_num(self, X):
        return np.hstack([X[NUM].to_numpy(float), self.oh_.transform(self._cat_preenchidos(X))])

    def _num(self, X):
        # reinicia o sorteio a cada chamada: a mesma entrada recebe sempre a mesma imputação (reprodutível)
        self.imp_num_.random_state_ = np.random.RandomState(self.semente)
        v = pd.DataFrame(self.imp_num_.transform(self._base_num(X))[:, :len(NUM)], columns=NUM, index=X.index)
        for c in NUM:
            lo, hi = self.lim_[c]
            v[c] = X[c].where(X[c].notna(), v[c].clip(lo, hi).round())
        return v

    def _pred_cat(self, num, catp, c):
        outras = [k for k in CAT if k != c]
        return np.hstack([num.to_numpy(float), self.oh_outras_[c].transform(catp[outras])])

    def fit(self, X, y=None):
        self.moda_ = {c: X[c].mode().iloc[0] for c in CAT}
        self.lim_ = {c: (X[c].min(), X[c].max()) for c in NUM}
        catp = self._cat_preenchidos(X)
        self.oh_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(catp)
        self.imp_num_ = IterativeImputer(estimator=BayesianRidge(), sample_posterior=True, max_iter=10,
                                         random_state=self.semente).fit(self._base_num(X))
        num = self._num(X)
        self.oh_outras_, self.mod_cat_ = {}, {}
        for c in CAT:
            outras = [k for k in CAT if k != c]
            self.oh_outras_[c] = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(catp[outras])
            obs = X[c].notna().to_numpy()
            Z = self._pred_cat(num, catp, c)
            self.mod_cat_[c] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(
                Z[obs], X[c].to_numpy()[obs].astype(int))
        return self

    def transform(self, X):
        rng = np.random.default_rng(self.semente)
        num = self._num(X)
        catp = self._cat_preenchidos(X)
        out = num.copy()
        for c in CAT:
            v = X[c].to_numpy(float).copy()
            falta = np.isnan(v)
            if falta.any():
                pr = self.mod_cat_[c].predict_proba(self._pred_cat(num[falta], catp[falta], c))
                sorteio = (rng.random(len(pr))[:, None] > pr.cumsum(axis=1)).sum(axis=1)
                v[falta] = self.mod_cat_[c].classes_[np.minimum(sorteio, pr.shape[1] - 1)]
            out[c] = v.astype(int).astype(str)
        return out


def ajustar_imputadores(X, m):
    """Os M imputadores (sementes SEED..SEED+m-1) ajustados em X. Não dependem dos hiperparâmetros da floresta."""
    return [ImputadorMultiplo(SEED + k).fit(X) for k in range(m)]


class ModeloImputacaoMultipla(BaseEstimator, ClassifierMixin):
    """M pipelines (imputação estocástica → one-hot → Random Forest); a probabilidade é a média das M florestas.

    `imputadores`: imputadores já ajustados no MESMO X passado ao fit (reaproveitados na busca de
    hiperparâmetros, porque a imputação não depende deles). Sem eles, o fit ajusta os seus. O resultado é o
    mesmo nos dois casos: o imputador é determinístico dada a semente e os dados de treino."""

    def __init__(self, max_depth=6, min_samples_leaf=50, class_weight=None, m=5, imputadores=None):
        self.max_depth, self.min_samples_leaf, self.class_weight, self.m = max_depth, min_samples_leaf, class_weight, m
        self.imputadores = imputadores

    def fit(self, X, y):
        imps = self.imputadores if self.imputadores is not None else ajustar_imputadores(X, self.m)
        assert len(imps) == self.m
        self.modelos_ = []
        for imp in imps:
            Xi = imp.transform(X)
            pre = ColumnTransformer([("num", "passthrough", NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
            Z = pre.fit_transform(Xi)
            rf = RandomForestClassifier(n_estimators=300, max_features="sqrt", max_depth=self.max_depth,
                                        min_samples_leaf=self.min_samples_leaf, class_weight=self.class_weight,
                                        n_jobs=-1, random_state=SEED).fit(Z, y)
            self.modelos_.append(Pipeline([("imp", imp), ("pre", pre), ("rf", rf)]))
        self.classes_ = self.modelos_[0].named_steps["rf"].classes_
        return self

    def predict_proba(self, X):
        return np.mean([mdl.predict_proba(X) for mdl in self.modelos_], axis=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1 Divisão temporal, tamanho por período e comparação dos períodos (TRIPOD, itens 8 e 13)

# COMMAND ----------

def montar(mask):
    d = df[mask]
    return preparar(d), d["baixo_peso"].values, d["ano_nasc"].values, d["municipio"].values

def particoes(ano):
    return ano <= FIM_TREINO, (ano > FIM_TREINO) & (ano <= FIM_VAL), ano > FIM_VAL

def novo_modelo(max_depth, min_samples_leaf, class_weight, imputadores=None):
    return ModeloImputacaoMultipla(max_depth, min_samples_leaf, class_weight, M_IMP, imputadores)

X, y, ano, mun = montar(m_pop)
ids = df.loc[m_pop, "id_linha"].values
tr, va, te = particoes(ano)
divisao = {k: {"anos": f"{a0}–{a1}", "n": int(m.sum()), "baixo_peso": int(y[m].sum()),
               "prevalencia_pct": round(float(y[m].mean() * 100), 2)}
           for k, a0, a1, m in [("treino", ANO_INI, FIM_TREINO, tr), ("validacao", FIM_TREINO + 1, FIM_VAL, va),
                                ("teste", FIM_VAL + 1, ANO_FIM, te)]}
print(divisao)

PERIODOS = ["treino", "validacao", "teste"]
periodo = pd.Series(np.select([tr, va], ["treino", "validacao"], "teste"), index=pop.index)
comparacao_periodos = []
for c, cat in desc_cols.items():
    tab = pd.crosstab(cat, periodo, normalize="columns") * 100
    for k in tab.index:
        comparacao_periodos.append({"variavel": c, "categoria": k,
                                    **{p: round(float(tab.loc[k, p]), 2) for p in PERIODOS if p in tab.columns}})
ausencia_por_periodo = {
    c: {p: {"n": int(X.loc[(periodo == p).values, c].isna().sum()),
            "pct": round(float(X.loc[(periodo == p).values, c].isna().mean() * 100), 3)} for p in PERIODOS}
    for c in NUM + CAT}
print(pd.DataFrame(comparacao_periodos).to_string()); print(ausencia_por_periodo)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Busca de hiperparâmetros na validação (critério: AUC-PR)

# COMMAND ----------

PROF = [3, 4, 6, 8, 12]
FOLHA = [50, 100, 200, 400, 800]
PESO_CLASSE = [None, "balanced_subsample"]
# Imputadores ajustados uma vez no treino e reaproveitados em todas as configurações
t0 = time.time()
imputadores_treino = ajustar_imputadores(X[tr], M_IMP)
seg_imputadores = round(time.time() - t0, 1)
print("imputadores ajustados em", seg_imputadores, "s")
busca = []
for cw in PESO_CLASSE:
    for d in PROF:
        for l in FOLHA:
            t0 = time.time()
            m = novo_modelo(d, l, cw, imputadores_treino).fit(X[tr], y[tr])
            pv = m.predict_proba(X[va])[:, 1]
            busca.append({"max_depth": d, "min_samples_leaf": l, "class_weight": cw or "None",
                          "auc_roc_val": round(roc_auc_score(y[va], pv), 4),
                          "auc_pr_val": round(average_precision_score(y[va], pv), 4),
                          "brier_val": round(brier_score_loss(y[va], pv), 4),
                          "seg": round(time.time() - t0, 1)})
            print(busca[-1])
melhor = max(busca, key=lambda r: r["auc_pr_val"])
params = {"max_depth": melhor["max_depth"], "min_samples_leaf": melhor["min_samples_leaf"],
          "class_weight": None if melhor["class_weight"] == "None" else melhor["class_weight"]}
na_borda = {"max_depth": params["max_depth"] in (min(PROF), max(PROF)),
            "min_samples_leaf": params["min_samples_leaf"] in (min(FOLHA), max(FOLHA))}
# Diferença entre o melhor e os demais: quantas configurações ficam a menos de 0,002 de AUC-PR do melhor
quase_empate = sum(1 for r in busca if melhor["auc_pr_val"] - r["auc_pr_val"] < 0.002)
print("melhor:", melhor, "borda:", na_borda, "quase empates:", quase_empate)

modelo = novo_modelo(**params, imputadores=imputadores_treino).fit(X[tr], y[tr])
p_va = modelo.predict_proba(X[va])[:, 1]
p_te = modelo.predict_proba(X[te])[:, 1]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Calibração (Platt, ajustada na validação) e métricas probabilísticas no teste

# COMMAND ----------

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)

platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p_va), y[va])
calibrar = lambda p: platt.predict_proba(logit(p))[:, 1]
q_va, q_te = calibrar(p_va), calibrar(p_te)

prev_treino = float(y[tr].mean())
prob = {
    "auc_roc": round(roc_auc_score(y[te], p_te), 4),
    "auc_pr": round(average_precision_score(y[te], p_te), 4),
    "prevalencia_teste": round(float(y[te].mean()), 4),
    "brier_sem_calibracao": round(brier_score_loss(y[te], p_te), 4),
    "brier_calibrado": round(brier_score_loss(y[te], q_te), 4),
    "brier_referencia_prevalencia_treino": round(brier_score_loss(y[te], np.full(te.sum(), prev_treino)), 4),
    "prevalencia_treino": round(prev_treino, 4),
    "prob_calibrada_min_max": [round(float(q_te.min()), 4), round(float(q_te.max()), 4)],
}
# Curva de calibração por decis da probabilidade calibrada
dec = pd.qcut(q_te, 10, labels=False, duplicates="drop")
calib = (pd.DataFrame({"dec": dec, "q": q_te, "p": p_te, "y": y[te]}).groupby("dec")
           .agg(prob_calibrada_media=("q", "mean"), escore_bruto_medio=("p", "mean"), observado=("y", "mean"), n=("y", "size"))
           .round(4).reset_index().to_dict("records"))

def inclinacao_mv(x, yv, iteracoes=50):
    """Regressão logística de y em x (com intercepto) por Newton-Raphson (máxima verossimilhança exata),
    o mesmo método de resultados_v4_tripod/ic_calibracao.py; devolve o coeficiente de x."""
    X = np.column_stack([np.ones_like(x), x])
    b = np.zeros(2)
    for _ in range(iteracoes):
        pr = 1 / (1 + np.exp(-X @ b))
        passo = np.linalg.solve(X.T @ (X * (pr * (1 - pr))[:, None]), X.T @ (yv - pr))
        b += passo
        if np.abs(passo).max() < 1e-10:
            break
    return b[1]

def medidas_calibracao(yv, qv):
    """Intercepto (calibração geral, com o logit previsto como offset), inclinação e razão observado/esperado."""
    lq = logit(qv).ravel()
    intercepto = brentq(lambda a: float((yv - 1 / (1 + np.exp(-(a + lq)))).sum()), -10, 10)
    inclinacao = inclinacao_mv(lq, yv.astype(float))
    return {"intercepto": round(float(intercepto), 4), "inclinacao": round(float(inclinacao), 4),
            "razao_observado_esperado": round(float(yv.mean() / qv.mean()), 4)}

calibracao_teste = medidas_calibracao(y[te], q_te)
print(prob); print(calibracao_teste)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Pontos de operação (limiares definidos na validação, aplicados ao teste)

# COMMAND ----------

def ponto(yv, pv, lim):
    yh = (pv >= lim).astype(int)
    tn, fp, fn, tp = confusion_matrix(yv, yh, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    vpp = tp / (tp + fp) if tp + fp else 0.0
    return {"sinalizadas_pct": round((tp + fp) / len(yv) * 100, 2),
            "sensibilidade": round(sens, 4), "especificidade": round(tn / (tn + fp), 4) if tn + fp else None,
            "vpp": round(vpp, 4), "f1": round(2 * sens * vpp / (sens + vpp), 4) if sens + vpp else 0.0,
            "lift": round(vpp / yv.mean(), 3),
            "matriz": {"vp": int(tp), "fp": int(fp), "fn": int(fn), "vn": int(tn)}}

pontos = []
for k in (0.05, 0.10, 0.20, 0.30):
    lim = float(np.quantile(p_va, 1 - k))
    pontos.append({"criterio": f"{int(k * 100)}% de maior escore", "limiar_escore": round(lim, 4), **ponto(y[te], p_te, lim)})
pr_v, rc_v, th_v = precision_recall_curve(y[va], p_va)
f1v = 2 * pr_v[:-1] * rc_v[:-1] / np.clip(pr_v[:-1] + rc_v[:-1], 1e-9, None)
lim_f1 = float(th_v[np.argmax(f1v)])
pontos.append({"criterio": "máximo F1 na validação", "limiar_escore": round(lim_f1, 4), **ponto(y[te], p_te, lim_f1)})
pontos.append({"criterio": "referência: todas sinalizadas", "limiar_escore": None, **ponto(y[te], p_te, -np.inf)})
print(pd.DataFrame(pontos).drop(columns="matriz").to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Intervalos de confiança (bootstrap no teste) e desempenho por ano e município

# COMMAND ----------

rng = np.random.default_rng(SEED)
lim10 = float(np.quantile(p_va, 0.90))

def bootstrap(yv, pv, qv, B):
    n, out = len(yv), {"auc_roc": [], "auc_pr": [], "brier_calibrado": [], "sens_top10": [], "vpp_top10": []}
    for _ in range(B):
        i = rng.integers(0, n, n)
        yb, pb = yv[i], pv[i]
        if yb.min() == yb.max():
            continue
        out["auc_roc"].append(roc_auc_score(yb, pb))
        out["auc_pr"].append(average_precision_score(yb, pb))
        out["brier_calibrado"].append(brier_score_loss(yb, qv[i]))
        s = pb >= lim10
        out["sens_top10"].append(yb[s].sum() / yb.sum())
        out["vpp_top10"].append(yb[s].mean() if s.any() else 0.0)
    return {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for k, v in out.items()}

ic95 = bootstrap(y[te], p_te, q_te, N_BOOT)

def subgrupo(rotulos):
    linhas = []
    for g in sorted(set(rotulos)):
        s = rotulos == g
        yy, pp = y[te][s], p_te[s]
        lin = {"grupo": str(g), "n": int(s.sum()), "baixo_peso": int(yy.sum()), "prevalencia_pct": round(float(yy.mean() * 100), 2)}
        if 0 < yy.sum() < len(yy):
            lin.update({"auc_roc": round(roc_auc_score(yy, pp), 4), "auc_pr": round(average_precision_score(yy, pp), 4)})
            aucs = []
            for _ in range(500):
                i = rng.integers(0, len(yy), len(yy))
                if 0 < yy[i].sum() < len(yy):
                    aucs.append(roc_auc_score(yy[i], pp[i]))
            lin["auc_roc_ic95"] = [round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)]
            lin["sens_top10"] = round(float(yy[pp >= lim10].sum() / yy.sum()), 4)
        linhas.append(lin)
    return linhas

por_ano_teste = subgrupo(ano[te])
por_mun_teste = subgrupo(mun[te])
print(ic95); print(pd.DataFrame(por_ano_teste).to_string()); print(pd.DataFrame(por_mun_teste).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8.1 Efeito de *case-mix* na AUC (Vergouwe; Moons; Steyerberg, 2010; Debray et al., 2015)
# MAGIC
# MAGIC A AUC "corrigida para case-mix" é a AUC esperada se o modelo calibrado estivesse correto naquela amostra: o desfecho é sorteado a partir da probabilidade calibrada (B vezes) e a AUC é calculada contra o escore. Ela depende só da dispersão dos riscos previstos. Se a AUC corrigida cai da validação para o teste tanto quanto a observada, a queda é compatível com uma população mais homogênea; se a corrigida fica estável e a observada cai, há indício de mudança real nas associações.

# COMMAND ----------

def resumo_casemix(yv, pv, qv, B):
    r = np.random.default_rng(SEED)
    corr = [roc_auc_score(r.random(len(qv)) < qv, pv) for _ in range(B)]
    boot = []
    for _ in range(B):
        i = r.integers(0, len(yv), len(yv))
        if 0 < yv[i].sum() < len(yv):
            boot.append(roc_auc_score(yv[i], pv[i]))
    lq = logit(qv).ravel()
    return {"n": int(len(yv)), "auc_observada": round(float(roc_auc_score(yv, pv)), 4),
            "auc_observada_ic95": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
            "auc_corrigida_casemix": round(float(np.mean(corr)), 4),
            "auc_corrigida_ic95": [round(float(np.percentile(corr, 2.5)), 4), round(float(np.percentile(corr, 97.5)), 4)],
            "prob_calibrada_media": round(float(qv.mean()), 4), "prob_calibrada_dp": round(float(qv.std()), 4),
            "logit_medio": round(float(lq.mean()), 4), "logit_dp": round(float(lq.std()), 4)}

case_mix = {"validacao": resumo_casemix(y[va], p_va, q_va, B_CASEMIX),
            "teste": resumo_casemix(y[te], p_te, q_te, B_CASEMIX)}
print(case_mix)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Importância por permutação em bloco (variável original inteira, no teste)

# COMMAND ----------

GRUPOS = {c: [c] for c in NUM + CAT}
GRUPOS["historico_obstetrico"] = HIST

def importancia_permutacao(mdl, Xv, yv, grupos, n_rep):
    base_pr = average_precision_score(yv, mdl.predict_proba(Xv)[:, 1])
    base_roc = roc_auc_score(yv, mdl.predict_proba(Xv)[:, 1])
    r = np.random.default_rng(SEED)
    res = {}
    for g, cols in grupos.items():
        d_pr, d_roc = [], []
        for _ in range(n_rep):
            Xp = Xv.copy()
            idx = r.permutation(len(Xp))
            for c in cols:                       # mesma permutação para todas as colunas do grupo
                Xp[c] = Xv[c].values[idx]
            pp = mdl.predict_proba(Xp)[:, 1]
            d_pr.append(base_pr - average_precision_score(yv, pp))
            d_roc.append(base_roc - roc_auc_score(yv, pp))
        res[g] = {"queda_auc_pr_media": round(float(np.mean(d_pr)), 5), "queda_auc_pr_dp": round(float(np.std(d_pr, ddof=1)), 5),
                  "queda_auc_roc_media": round(float(np.mean(d_roc)), 5), "queda_auc_roc_dp": round(float(np.std(d_roc, ddof=1)), 5)}
    return res

imp_perm = importancia_permutacao(modelo, X[te], y[te], GRUPOS, N_PERM)

# Gini (impureza) agregada por variável original, média das M florestas (as colunas do one-hot são as mesmas)
nomes = modelo.modelos_[0].named_steps["pre"].get_feature_names_out()
assert all(list(mm.named_steps["pre"].get_feature_names_out()) == list(nomes) for mm in modelo.modelos_)
gini_medio = np.mean([mm.named_steps["rf"].feature_importances_ for mm in modelo.modelos_], axis=0)
def origem(nome):
    cand = [v for v in NUM + CAT if v in nome]
    return max(cand, key=len) if cand else nome
imp_gini = (pd.DataFrame({"col": nomes, "imp": gini_medio})
              .assign(variavel=lambda d: d.col.map(origem)).groupby("variavel").imp.sum()
              .sort_values(ascending=False).round(5).to_dict())
print(pd.DataFrame(imp_perm).T.sort_values("queda_auc_pr_media", ascending=False).to_string()); print(imp_gini)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Análises de sensibilidade da população (mesmos hiperparâmetros)

# COMMAND ----------

def avaliar_populacao(mask, descricao):
    Xs, ys, anos_s, _ = montar(mask)
    a, _, c = particoes(anos_s)
    m = novo_modelo(**params).fit(Xs[a], ys[a])
    ps = m.predict_proba(Xs[c])[:, 1]
    return {"populacao": descricao, "n_total": int(len(ys)), "n_teste": int(c.sum()),
            "prevalencia_teste_pct": round(float(ys[c].mean() * 100), 2),
            "auc_roc": round(roc_auc_score(ys[c], ps), 4), "auc_pr": round(average_precision_score(ys[c], ps), 4)}

sensibilidade_pop = [
    {"populacao": "principal", "n_total": int(len(y)), "n_teste": int(te.sum()),
     "prevalencia_teste_pct": round(float(y[te].mean() * 100), 2), "auc_roc": prob["auc_roc"], "auc_pr": prob["auc_pr"]},
    avaliar_populacao(m_peso_reg & m_unica, "gestação única, todos os pesos registrados (> 0 g)"),
    avaliar_populacao(m_peso_ok, f"inclui gestações múltiplas e ignoradas (peso ≥ {PESO_MIN} g)"),
]
print(pd.DataFrame(sensibilidade_pop).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Figuras (vírgula decimal) e gravação

# COMMAND ----------

virg = lambda v, casas=3: f"{v:.{casas}f}".replace(".", ",")
fmt_eixo = FuncFormatter(lambda x, _: ("%g" % round(x, 4)).replace(".", ","))
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
COR, COR2 = "#4C72B0", "#C44E52"

fig, ax = plt.subplots(figsize=(6, 3.2))
ax.bar(por_ano.ano_nasc.astype(str), por_ano.taxa_baixo_peso_pct, color=COR)
ax.set_ylabel("Baixo peso (%)"); ax.set_xlabel("Ano"); ax.set_ylim(0, por_ano.taxa_baixo_peso_pct.max() * 1.25)
for i, v in enumerate(por_ano.taxa_baixo_peso_pct):
    ax.text(i, v + 0.15, virg(v, 1), ha="center", fontsize=8)
ax.yaxis.set_major_formatter(fmt_eixo)
fig.tight_layout(); fig.savefig(SAIDA + "fig_taxa_ano.png"); plt.close(fig)

fpr, tpr, _ = roc_curve(y[te], p_te)
fig, ax = plt.subplots(figsize=(4.2, 4))
ax.plot(fpr, tpr, color=COR, label=f"Random Forest (AUC-ROC = {virg(prob['auc_roc'])})")
ax.plot([0, 1], [0, 1], "--", color="gray", label="Classificação aleatória")
ax.set_xlabel("1 − Especificidade"); ax.set_ylabel("Sensibilidade"); ax.legend(loc="lower right", fontsize=8)
ax.xaxis.set_major_formatter(fmt_eixo); ax.yaxis.set_major_formatter(fmt_eixo)
fig.tight_layout(); fig.savefig(SAIDA + "fig_roc.png"); plt.close(fig)

pr_t, rc_t, _ = precision_recall_curve(y[te], p_te)
fig, ax = plt.subplots(figsize=(4.2, 4))
ax.plot(rc_t, pr_t, color=COR, label=f"Random Forest (AUC-PR = {virg(prob['auc_pr'])})")
ax.axhline(y[te].mean(), ls="--", color="gray", label=f"Prevalência no teste ({virg(y[te].mean())})")
topo = float(np.max(pr_t[rc_t >= 0.02])) if (rc_t >= 0.02).any() else 1.0
ax.set_ylim(0, min(1.0, round(topo * 1.15 + 0.005, 2)))
ax.set_xlabel("Sensibilidade"); ax.set_ylabel("Precisão (VPP)"); ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(fmt_eixo); ax.yaxis.set_major_formatter(fmt_eixo)
fig.tight_layout(); fig.savefig(SAIDA + "fig_pr.png"); plt.close(fig)

cdf = pd.DataFrame(calib)
fig, ax = plt.subplots(figsize=(4.2, 4))
lim_ax = max(cdf.prob_calibrada_media.max(), cdf.observado.max()) * 1.15
ax.plot([0, lim_ax], [0, lim_ax], "--", color="gray", label="Calibração perfeita")
ax.plot(cdf.prob_calibrada_media, cdf.observado, "o-", color=COR, label="Modelo calibrado (decis)")
ax.set_xlim(0, lim_ax); ax.set_ylim(0, lim_ax)
ax.set_xlabel("Probabilidade prevista"); ax.set_ylabel("Proporção observada de baixo peso"); ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(fmt_eixo); ax.yaxis.set_major_formatter(fmt_eixo)
fig.tight_layout(); fig.savefig(SAIDA + "fig_calibracao.png"); plt.close(fig)

ip = pd.DataFrame(imp_perm).T.sort_values("queda_auc_pr_media")
fig, ax = plt.subplots(figsize=(6.4, 3.8))
cores = [COR2 if g == "historico_obstetrico" else COR for g in ip.index]
ax.barh([ROTULO[g] for g in ip.index], ip.queda_auc_pr_media, xerr=ip.queda_auc_pr_dp, color=cores, capsize=2)
ax.axvline(0, color="black", lw=0.6)
ax.set_xlabel("Queda média na AUC-PR ao permutar (teste)")
ax.xaxis.set_major_formatter(fmt_eixo)
fig.tight_layout(); fig.savefig(SAIDA + "fig_importancia.png"); plt.close(fig)

resultado = {
    "versao": 4, "gerado_em": time.strftime("%Y-%m-%d %H:%M:%S"), "sklearn": sklearn.__version__,
    "tratamento_ausentes": {"metodo": "imputação múltipla por equações encadeadas, ajustada só no treino; média das florestas",
                            "n_imputacoes": M_IMP, "codigo_9_como_ausente": True, "indicador_de_ausencia": False,
                            "seg_ajuste_imputadores_treino": seg_imputadores},
    "ausencia_por_periodo": ausencia_por_periodo, "comparacao_periodos": comparacao_periodos,
    "calibracao_teste": calibracao_teste, "case_mix": case_mix, "n_sorteios_casemix": B_CASEMIX,
    "parametros": {**G, **P},
    "gold_ml": {"tabela": GOLD["tabela"], "versao_tabela": GOLD["versao_tabela"],
                "versao_silver": GOLD["versao_silver"], "gerado_em": GOLD["gerado_em"]},
    "fonte_por_ano": fonte_por_ano, "fluxo_populacao": fluxo, "excluidos": excluidos,
    "auditoria_dominios": auditoria, "ausencia": ausencia,
    "por_ano": por_ano.to_dict("records"), "por_municipio": por_mun.to_dict("records"), "descritiva": descritiva,
    "divisao": divisao, "n_colunas_apos_preprocessamento": int(len(nomes)),
    "busca": busca, "params": {**params, "class_weight": params["class_weight"] or "None",
                               "n_estimators": 300, "max_features": "sqrt"},
    "melhor_na_borda_da_grade": na_borda, "configuracoes_a_menos_de_0002_do_melhor": quase_empate,
    "teste": prob, "ic95_bootstrap_teste": ic95, "n_bootstrap": N_BOOT, "calibracao_decis_teste": calib,
    "pontos_operacao_teste": pontos, "por_ano_teste": por_ano_teste, "por_municipio_teste": por_mun_teste,
    "importancia_permutacao": imp_perm, "n_repeticoes_permutacao": N_PERM, "importancia_gini": imp_gini,
    "sensibilidade_populacao": sensibilidade_pop,
}
predicoes = pd.concat([
    pd.DataFrame({"id_linha": ids[m], "periodo": nome, "ano_nasc": ano[m], "municipio": mun[m], "baixo_peso": y[m],
                  "escore_bruto": pv, "prob_calibrada": qv})
    for nome, m, pv, qv in [("validacao", va, p_va, q_va), ("teste", te, p_te, q_te)]])
predicoes.to_csv(SAIDA + "predicoes_validacao_teste.csv", index=False)
with open(SAIDA + "resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultado, f, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
dbutils.notebook.exit(json.dumps({"auc_roc": prob["auc_roc"], "auc_pr": prob["auc_pr"], "params": resultado["params"]}))

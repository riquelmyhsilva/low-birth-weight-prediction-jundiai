# Databricks notebook source
# MAGIC %md
# MAGIC # TCC — Teste de atributos adicionais do SINASC (registro, não entra no modelo principal)
# MAGIC
# MAGIC Pergunta: atributos maternos do início da gestação que ficaram fora do modelo acrescentariam desempenho?
# MAGIC
# MAGIC - **Testados:** naturalidade da mãe (`NATURALMAE` → `cod_naturalidade_mae`) e ocupação da mãe
# MAGIC   (`CODOCUPMAE` → `cod_cbo_ocupacao_mae`, agrupada pelo grande grupo da CBO, com categoria de ausente).
# MAGIC - **Só medidos (justificativa de exclusão):** preenchimento por ano de idade do pai, escolaridade agregada
# MAGIC   (`ESCMAEAGR1`), UF e município de naturalidade, na mesma população do estudo.
# MAGIC
# MAGIC A população é reconstruída da Silver com as mesmas regras do `06_load_gold_ml` (fonte por ano, 7 municípios,
# MAGIC 2012–2025, duplicatas, faixas e domínios, gestação única, peso ≥ 500 g) e conferida contra a Gold de ML.
# MAGIC Mesma divisão temporal do modelo principal. Cada variante escolhe hiperparâmetros numa grade pequena
# MAGIC pela AUC-PR na validação; o teste é só para relato, com bootstrap pareado contra a referência reconstruída.
# MAGIC Somente leitura; resultados na pasta `saida`.

# COMMAND ----------

import os, json, time
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# COMMAND ----------

PADRAO = {
    "estado_gold_ml": "/Volumes/workspace/sinasc/03_gold/_control/estado_gold_ml.json",
    "ano_fim_treino": "2021",
    "ano_fim_validacao": "2023",
    "n_bootstrap": "500",
    "n_repeticoes_permutacao": "30",
    "saida": "/Volumes/workspace/sinasc/03_gold/_tcc_ml/teste_atributos_extras/",
}
for k, v in PADRAO.items():
    dbutils.widgets.text(k, v)
P = {k: dbutils.widgets.get(k).strip() for k in PADRAO}
with open(P["estado_gold_ml"], encoding="utf-8") as f:
    GOLD = json.load(f)
G = GOLD["parametros"]
MUN6 = [m.strip() for m in G["municipios_ibge6"].split(",") if m.strip()]
ANO_INI, ANO_FIM, PESO_MIN = int(G["ano_inicio"]), int(G["ano_fim"]), int(G["peso_min_g"])
FIM_TREINO, FIM_VAL = int(P["ano_fim_treino"]), int(P["ano_fim_validacao"])
N_BOOT, N_PERM = int(P["n_bootstrap"]), int(P["n_repeticoes_permutacao"])
SAIDA = P["saida"].rstrip("/") + "/"
os.makedirs(SAIDA, exist_ok=True)
SEED = 42
SILVER = "/Volumes/workspace/sinasc/02_silver/"

NUM = ["idade_mae", "qtd_gestacoes_anteriores", "qtd_partos_normais",
       "qtd_partos_cesareos", "qtd_filhos_vivos", "qtd_perdas_fetais_abortos"]
CAT = ["id_raca_cor_mae", "id_escolaridade_mae_2010", "id_estado_civil_mae"]
assert NUM == GOLD["colunas_atributos"]["numericos"] and CAT == GOLD["colunas_atributos"]["categoricos"]
FAIXA = {"idade_mae": (10, 55), **{c: (0, 30) for c in NUM[1:]}}
DOMINIO = {"id_raca_cor_mae": {1, 2, 3, 4, 5}, "id_escolaridade_mae_2010": {0, 1, 2, 3, 4, 5},
           "id_estado_civil_mae": {1, 2, 3, 4, 5}}
COLS_ORDEM = ["uf_nasc", "ano_nasc", "cod_municipio_residencia", "peso_gramas", "id_tipo_gravidez", *NUM, *CAT]
EXTRAS = ["cod_naturalidade_mae", "cod_cbo_ocupacao_mae", "idade_pai", "id_escolaridade_mae_agregada",
          "cod_uf_naturalidade_mae", "cod_municipio_naturalidade_mae"]
print(P)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. População reconstruída da Silver (regras do 06) com os atributos extras

# COMMAND ----------

silver = spark.read.format("delta").load(SILVER)
faltam = [c for c in EXTRAS if c not in silver.columns]
assert not faltam, f"colunas ausentes na Silver: {faltam}"
fonte = {r["ano"]: r["uf_fonte"] for r in GOLD["relatorio"]["fonte_por_ano"]}
cond = None
for a, uf in fonte.items():
    c = (F.col("uf_nasc") == uf) & (F.col("ano_nasc") == a)
    cond = c if cond is None else (cond | c)
META = {"uf_nasc", "id_sequencial_registro", "_src_file", "_load_time", "_bronze_load_time", "_silver_load_time"}
conteudo = [c for c in silver.columns if c not in META]
regional = (silver.filter(cond).filter(F.substring("cod_municipio_residencia", 1, 6).isin(MUN6))
                  .dropDuplicates(conteudo).select(*COLS_ORDEM, *EXTRAS)
                  .select(*COLS_ORDEM, *[F.col(c).cast("string").alias(c) for c in EXTRAS]))
df = regional.toPandas().sort_values(COLS_ORDEM, na_position="last", kind="mergesort").reset_index(drop=True)
assert len(df) == GOLD["linhas"], (len(df), GOLD["linhas"])

peso = df["peso_gramas"].astype(float)
m_pop = peso.notna() & (peso > 0) & (peso < 9999) & (peso >= PESO_MIN) & (df["id_tipo_gravidez"].astype(float) == 1)
pop = df[m_pop].copy()
assert len(pop) == GOLD["linhas_populacao_estudo"], (len(pop), GOLD["linhas_populacao_estudo"])
pop["baixo_peso"] = (pop["peso_gramas"].astype(float) < 2500).astype(int)

def texto(s):
    """Código como texto limpo; ausente vira "" (evita NA em máscaras booleanas)."""
    s = s.astype(object).where(pd.notna(s), "").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return s.where(~s.str.lower().isin(["nan", "none", "<na>"]), "")

# Naturalidade: brasileira = "8" + código da UF (IBGE); demais códigos = país estrangeiro.
REGIAO = {"1": "Norte", "2": "Nordeste", "3": "Sudeste (exceto SP)", "4": "Sul", "5": "Centro-Oeste"}
def naturalidade(s):
    s = texto(s)
    out = pd.Series("Ignorado", index=s.index, dtype=object)
    br = s.str.fullmatch(r"8\d\d")
    out[br & (s == "835")] = "SP"
    for d, nome in REGIAO.items():
        out[br & (s != "835") & (s.str[1:2] == d)] = nome
    estr = ~br & s.str.fullmatch(r"\d+") & ~s.isin(["0", "00", "000", "99", "999"])
    out[estr] = "Estrangeira"
    return out

# Ocupação (CBO 2002, 6 dígitos): grande grupo = 1º dígito; códigos especiais 99999x mantidos à parte.
def ocupacao(s):
    s = texto(s)
    s = s.where(~s.str.fullmatch(r"\d{5}"), s.str.zfill(6))
    out = pd.Series("Ignorado", index=s.index, dtype=object)
    especial = s.str.fullmatch(r"99999\d")
    out[especial] = "Especial " + s[especial]
    cbo = s.str.fullmatch(r"\d{6}") & ~especial
    out[cbo] = "GG " + s[cbo].str[0]
    return out

X = pd.DataFrame(index=pop.index)
for c in NUM:
    v = pop[c].astype(float); lo, hi = FAIXA[c]
    X[c] = v.where((v >= lo) & (v <= hi))
for c in CAT:
    v = pop[c].astype(float).fillna(-1).astype(int)
    X[c] = v.where(v.isin(DOMINIO[c]), 9).astype(str)
X["naturalidade_mae"] = naturalidade(pop["cod_naturalidade_mae"])
X["ocupacao_mae_gg"] = ocupacao(pop["cod_cbo_ocupacao_mae"])
y, ano = pop["baixo_peso"].values, pop["ano_nasc"].astype(int).values
tr, va, te = ano <= FIM_TREINO, (ano > FIM_TREINO) & (ano <= FIM_VAL), ano > FIM_VAL
print({k: int(m.sum()) for k, m in [("treino", tr), ("validacao", va), ("teste", te)]})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Perfil dos atributos: códigos brutos, categorias, taxa de baixo peso e preenchimento por ano

# COMMAND ----------

def top_codigos(c, k=15):
    s = texto(pop[c]).replace("", "<vazio>")
    vc = s.value_counts()
    return [{"codigo": i, "n": int(n), "pct": round(n / len(s) * 100, 2)} for i, n in vc.head(k).items()]

def perfil_categoria(col):
    d = pd.DataFrame({"cat": X[col], "y": y})
    g = d.groupby("cat").y.agg(["size", "sum"]).sort_values("size", ascending=False)
    g_tr = pd.DataFrame({"cat": X[col][tr], "y": y[tr]}).groupby("cat").y.agg(["size", "mean"])
    return [{"categoria": i, "n": int(r["size"]), "pct": round(r["size"] / len(d) * 100, 2),
             "taxa_baixo_peso_pct": round(r["sum"] / r["size"] * 100, 2),
             "n_treino": int(g_tr.loc[i, "size"]) if i in g_tr.index else 0} for i, r in g.iterrows()]

def preenchimento_por_ano():
    linhas = []
    for a in sorted(set(ano)):
        s = pop[ano == a]
        ip = pd.to_numeric(s["idade_pai"], errors="coerce")
        ea = pd.to_numeric(s["id_escolaridade_mae_agregada"], errors="coerce")
        linhas.append({
            "ano": int(a), "n": int(len(s)),
            "naturalidade_valida_pct": round(float((X.loc[s.index, "naturalidade_mae"] != "Ignorado").mean() * 100), 2),
            "ocupacao_cbo_valida_pct": round(float(X.loc[s.index, "ocupacao_mae_gg"].str.startswith("GG").mean() * 100), 2),
            "ocupacao_especial_99999x_pct": round(float(X.loc[s.index, "ocupacao_mae_gg"].str.startswith("Especial").mean() * 100), 2),
            "idade_pai_valida_10a99_pct": round(float(ip.between(10, 98).mean() * 100), 2),
            "escolaridade_agregada_valida_pct": round(float((ea.notna() & (ea >= 0) & (ea != 99)).mean() * 100), 2),
            "uf_naturalidade_preenchida_pct": round(float((texto(s["cod_uf_naturalidade_mae"]) != "").mean() * 100), 2),
            "mun_naturalidade_preenchido_pct": round(float((texto(s["cod_municipio_naturalidade_mae"]) != "").mean() * 100), 2),
        })
    return linhas

perfil = {
    "codigos_brutos_naturalidade": top_codigos("cod_naturalidade_mae"),
    "codigos_brutos_ocupacao": top_codigos("cod_cbo_ocupacao_mae", 25),
    "categorias_naturalidade": perfil_categoria("naturalidade_mae"),
    "categorias_ocupacao": perfil_categoria("ocupacao_mae_gg"),
    "preenchimento_por_ano": preenchimento_por_ano(),
}
print(json.dumps(perfil, ensure_ascii=False, indent=1)[:6000])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Modelos: referência e variantes com os atributos extras

# COMMAND ----------

def modelo(num, cat, max_depth, min_samples_leaf):
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    return Pipeline([("pre", pre), ("rf", RandomForestClassifier(
        n_estimators=300, max_features="sqrt", max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        n_jobs=-1, random_state=SEED))])

def metricas(yv, pv):
    return {"auc_roc": round(roc_auc_score(yv, pv), 4), "auc_pr": round(average_precision_score(yv, pv), 4),
            "brier": round(brier_score_loss(yv, pv), 4)}

GRADE = [{"max_depth": d, "min_samples_leaf": l} for d in [6, 8] for l in [50, 100]]
VARIANTES = {
    "referencia": CAT,
    "naturalidade": CAT + ["naturalidade_mae"],
    "ocupacao": CAT + ["ocupacao_mae_gg"],
    "naturalidade_ocupacao": CAT + ["naturalidade_mae", "ocupacao_mae_gg"],
}
cols_all = NUM + CAT + ["naturalidade_mae", "ocupacao_mae_gg"]
res, p_te, modelos = {}, {}, {}
for nome, cat in VARIANTES.items():
    cols = NUM + cat
    linhas, melhor = [], None
    for g in GRADE:
        m = modelo(NUM, cat, **g).fit(X.loc[tr, cols], y[tr])
        pv = m.predict_proba(X.loc[va, cols])[:, 1]
        lin = {**g, **{f"{k}_val": v for k, v in metricas(y[va], pv).items()}}
        linhas.append(lin)
        if melhor is None or lin["auc_pr_val"] > melhor[0]["auc_pr_val"]:
            melhor = (lin, m)
    best, mdl = melhor
    p_te[nome] = mdl.predict_proba(X.loc[te, cols])[:, 1]
    modelos[nome] = (mdl, cols)
    res[nome] = {"grade": linhas, "melhor": best, "teste": metricas(y[te], p_te[nome])}
    print(nome, best, res[nome]["teste"])

# Referência fixa do modelo principal (prof. 6, folha 50) para conferir a reconstrução
m_ref = modelo(NUM, CAT, 6, 50).fit(X.loc[tr, NUM + CAT], y[tr])
conferencia_ref = metricas(y[te], m_ref.predict_proba(X.loc[te, NUM + CAT])[:, 1])
print("referência prof. 6, folha 50 (modelo principal: AUC-ROC 0,5668, AUC-PR 0,0948):", conferencia_ref)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Bootstrap pareado no teste e importância por permutação dos atributos extras (validação)

# COMMAND ----------

rng = np.random.default_rng(SEED)
yt, n = y[te], int(te.sum())
idx = [rng.integers(0, n, n) for _ in range(N_BOOT)]
diferencas = {}
for k in VARIANTES:
    if k == "referencia":
        continue
    d_roc, d_pr = [], []
    for i in idx:
        yb = yt[i]
        if yb.min() == yb.max():
            continue
        d_roc.append(roc_auc_score(yb, p_te[k][i]) - roc_auc_score(yb, p_te["referencia"][i]))
        d_pr.append(average_precision_score(yb, p_te[k][i]) - average_precision_score(yb, p_te["referencia"][i]))
    diferencas[k] = {"delta_auc_roc": round(float(np.mean(d_roc)), 4),
                     "delta_auc_roc_ic95": [round(float(np.percentile(d_roc, q)), 4) for q in (2.5, 97.5)],
                     "delta_auc_pr": round(float(np.mean(d_pr)), 4),
                     "delta_auc_pr_ic95": [round(float(np.percentile(d_pr, q)), 4) for q in (2.5, 97.5)]}
print(json.dumps(diferencas, indent=1))

mdl, cols = modelos["naturalidade_ocupacao"]
Xv, yv = X.loc[va, cols], y[va]
base_pr = average_precision_score(yv, mdl.predict_proba(Xv)[:, 1])
base_roc = roc_auc_score(yv, mdl.predict_proba(Xv)[:, 1])
r = np.random.default_rng(SEED)
importancia = {}
for c in cols:
    d_pr, d_roc = [], []
    for _ in range(N_PERM):
        Xp = Xv.copy()
        Xp[c] = Xv[c].values[r.permutation(len(Xp))]
        pp = mdl.predict_proba(Xp)[:, 1]
        d_pr.append(base_pr - average_precision_score(yv, pp))
        d_roc.append(base_roc - roc_auc_score(yv, pp))
    importancia[c] = {"queda_auc_pr_media": round(float(np.mean(d_pr)), 5), "queda_auc_pr_dp": round(float(np.std(d_pr, ddof=1)), 5),
                      "queda_auc_roc_media": round(float(np.mean(d_roc)), 5), "queda_auc_roc_dp": round(float(np.std(d_roc, ddof=1)), 5)}
print(pd.DataFrame(importancia).T.sort_values("queda_auc_pr_media", ascending=False).to_string())

resultado = {"gerado_em": time.strftime("%Y-%m-%d %H:%M:%S"), "sklearn": sklearn.__version__, "parametros": P,
             "versao_gold_ml_conferida": GOLD["versao_tabela"], "linhas_populacao": int(len(pop)),
             "perfil": perfil, "variantes": res, "conferencia_referencia_prof6_folha50": conferencia_ref,
             "diferenca_vs_referencia_teste": diferencas, "n_bootstrap": N_BOOT,
             "importancia_permutacao_validacao_modelo_com_extras": importancia, "n_repeticoes_permutacao": N_PERM}
with open(SAIDA + "resultados_atributos_extras.json", "w", encoding="utf-8") as f:
    json.dump(resultado, f, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
dbutils.notebook.exit(json.dumps({"conferencia_ref": conferencia_ref, "diferencas": diferencas}))

# Databricks notebook source
# MAGIC %md
# MAGIC # Gold de ML — população do estudo de baixo peso ao nascer (RMJ)
# MAGIC
# MAGIC Tabela de consumo do modelo do TCC (`tcc_ml/10_tcc_rf_baixo_peso_rmj`). Uma linha por nascido vivo com residência materna nos municípios do recorte, com as regras determinísticas já aplicadas:
# MAGIC
# MAGIC | Etapa | Regra |
# MAGIC |---|---|
# MAGIC | Fonte por ano | partição da UF preferida (`SP`); se o ano não existir nela, a UF reserva (`BR`, arquivo nacional) |
# MAGIC | Recorte | residência materna nos municípios informados (6 primeiros dígitos do código IBGE), anos `ano_inicio` a `ano_fim` |
# MAGIC | Duplicatas | remoção de linhas com conteúdo idêntico |
# MAGIC | Atributos | faixa plausível (numéricos) e domínio do dicionário SINASC (categóricos); fora deles vira nulo ou `9` (ignorado) |
# MAGIC | População | flags `peso_registrado`, `peso_minimo_ok`, `gestacao_unica` e `populacao_estudo` |
# MAGIC | Alvo | `baixo_peso` = peso registrado menor que 2.500 g |
# MAGIC
# MAGIC O modelo usa as linhas com `populacao_estudo = true`. As demais ficam na tabela para as análises de sensibilidade (por exemplo, incluir gestações múltiplas) e para a contagem das exclusões.
# MAGIC
# MAGIC Fica **fora** desta tabela tudo o que aprende com os dados (imputação, codificação one-hot, divisão em treino, validação e teste). Isso é ajustado só no treino, dentro do notebook do modelo, para não vazar informação do teste.
# MAGIC
# MAGIC ## Saídas
# MAGIC - `03_gold/ml_baixo_peso_rmj` (Delta) e `workspace.sinasc.ml_baixo_peso_rmj` (catálogo);
# MAGIC - `03_gold/_control/estado_gold_ml.json`: parâmetros, versão da Silver usada, versão da tabela e relatório da população (fonte por ano, fluxo de exclusões, auditoria de domínios e ausência por atributo).
# MAGIC
# MAGIC ## Quando é refeita
# MAGIC Só quando a Silver (versão Delta) ou os parâmetros mudam. Use `forcar = sim` para refazer mesmo assim.

# COMMAND ----------

import os
from datetime import datetime, timezone
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql import Window

# COMMAND ----------

# MAGIC %run ./00_02_schema_contracts

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parâmetros do recorte e regras dos atributos

# COMMAND ----------

PADRAO = {
    "uf_preferida": "SP",
    "uf_reserva": "BR",
    "municipios_ibge6": "350840,350960,352400,352520,352590,352730,355650",
    "ano_inicio": "2012",   # 1º ano com as 9 variáveis (formulário da DN de 2011)
    "ano_fim": "2025",      # 2025 vem da pasta PRELIM do DATASUS (preliminar); 2026 incompleto fica fora
    "peso_min_g": "500",
    "somente_gestacao_unica": "sim",
}
for k, v in PADRAO.items():
    dbutils.widgets.text(k, v)
dbutils.widgets.dropdown("forcar", "nao", ["nao", "sim"])
P = {k: dbutils.widgets.get(k).strip() for k in PADRAO}
FORCAR = dbutils.widgets.get("forcar") == "sim"

UF_PREF, UF_RES = P["uf_preferida"], P["uf_reserva"]
MUN6 = [m.strip() for m in P["municipios_ibge6"].split(",") if m.strip()]
ANO_INI, ANO_FIM = int(P["ano_inicio"]), int(P["ano_fim"])
PESO_MIN = int(P["peso_min_g"])
SO_UNICA = P["somente_gestacao_unica"].lower() in ("sim", "s", "true", "1")

SILVER = "/Volumes/workspace/sinasc/02_silver/"
DST = "/Volumes/workspace/sinasc/03_gold/ml_baixo_peso_rmj"
TABELA = "workspace.sinasc.ml_baixo_peso_rmj"
ESTADO = "/Volumes/workspace/sinasc/03_gold/_control/estado_gold_ml.json"

NOME_MUN = {"350840": "Cabreúva", "350960": "Campo Limpo Paulista", "352400": "Itupeva",
            "352520": "Jarinu", "352590": "Jundiaí", "352730": "Louveira", "355650": "Várzea Paulista"}

# Atributos (nomes da Silver) e regras de validade.
NUM = ["idade_mae", "qtd_gestacoes_anteriores", "qtd_partos_normais",
       "qtd_partos_cesareos", "qtd_filhos_vivos", "qtd_perdas_fetais_abortos"]
CAT = ["id_raca_cor_mae", "id_escolaridade_mae_2010", "id_estado_civil_mae"]
HIST = NUM[1:]
# Faixa plausível; fora dela (inclui 99 = ignorado e -1 = nulo na Silver) vira nulo.
FAIXA = {"idade_mae": (10, 55), **{c: (0, 30) for c in HIST}}
# Domínio do dicionário SINASC (BRASIL, 2019); fora dele (inclui 9 e -1) vira categoria "9" (ignorado).
DOMINIO = {"id_raca_cor_mae": [1, 2, 3, 4, 5],
           "id_escolaridade_mae_2010": [0, 1, 2, 3, 4, 5],
           "id_estado_civil_mae": [1, 2, 3, 4, 5]}
# Colunas brutas que definem a ordem das linhas (id_linha)
COLS_ORDEM = ["uf_nasc", "ano_nasc", "cod_municipio_residencia", "peso_gramas", "id_tipo_gravidez", *NUM, *CAT]
print(P)

# COMMAND ----------

# MAGIC %md
# MAGIC ## A Silver ou os parâmetros mudaram desde a última carga?

# COMMAND ----------

versao_silver = spark.sql(f"DESCRIBE HISTORY delta.`{SILVER}` LIMIT 1").collect()[0]["version"]
estado = ler_json(ESTADO)
existe = os.path.exists(DST + "/_delta_log") and spark.catalog.tableExists(TABELA)

if (not FORCAR and existe and estado and estado.get("versao_silver") == versao_silver
        and estado.get("parametros") == P):
    print(f"Silver na versão {versao_silver} e mesmos parâmetros da última carga: nada a fazer.")
    dbutils.notebook.exit(json.dumps({"status": "skip", "versao_silver": versao_silver}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Fonte por ano, recorte dos municípios e remoção de duplicatas

# COMMAND ----------

silver = spark.read.format("delta").load(SILVER)
anos = list(range(ANO_INI, ANO_FIM + 1))

def anos_disponiveis(uf):
    return {r["ano_nasc"] for r in (silver.filter((F.col("uf_nasc") == uf) & F.col("ano_nasc").between(ANO_INI, ANO_FIM))
                                     .select("ano_nasc").distinct().collect())}

disp_pref = anos_disponiveis(UF_PREF)
faltantes = [a for a in anos if a not in disp_pref]
disp_res = anos_disponiveis(UF_RES) if faltantes else set()
sem_fonte = [a for a in faltantes if a not in disp_res]
assert not sem_fonte, f"Anos sem dados em {UF_PREF} nem {UF_RES}: {sem_fonte}"
fonte = {a: (UF_PREF if a in disp_pref else UF_RES) for a in anos}

cond = None
for uf in sorted(set(fonte.values())):
    c = (F.col("uf_nasc") == uf) & F.col("ano_nasc").isin([a for a in anos if fonte[a] == uf])
    cond = c if cond is None else (cond | c)

META = {"uf_nasc", "id_sequencial_registro", "_src_file", "_load_time", "_bronze_load_time", "_silver_load_time"}
conteudo = [c for c in silver.columns if c not in META]
regional = silver.filter(cond).filter(F.substring("cod_municipio_residencia", 1, 6).isin(MUN6))
n_por_ano_bruto = {r["ano_nasc"]: r["count"] for r in regional.groupBy("ano_nasc").count().collect()}
regional = regional.dropDuplicates(conteudo)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Atributos tratados, flags da população e alvo
# MAGIC
# MAGIC `id_linha` numera as linhas em ordem determinística (pelas colunas brutas). O Random Forest depende da ordem das linhas (bootstrap), e o Spark não garante a mesma ordem de leitura quando a tabela é regravada; o modelo ordena por `id_linha`. Linhas empatadas em todas essas colunas são idênticas para o modelo.

# COMMAND ----------

peso = F.col("peso_gramas").cast("double")
peso_registrado = peso.isNotNull() & (peso > 0) & (peso < 9999)
peso_minimo_ok = peso_registrado & (peso >= PESO_MIN)
gestacao_unica = F.coalesce(F.col("id_tipo_gravidez").cast("double") == 1, F.lit(False))
populacao = (peso_minimo_ok & gestacao_unica) if SO_UNICA else peso_minimo_ok

def num_tratado(c):
    v = F.col(c).cast("double")
    lo, hi = FAIXA[c]
    return F.when(v.between(lo, hi), v).alias(c)

def cat_tratado(c):
    v = F.coalesce(F.col(c).cast("double"), F.lit(-1.0)).cast("int")
    return F.when(v.isin(DOMINIO[c]), v.cast("string")).otherwise(F.lit("9")).alias(c)

mapa_mun = F.create_map(*[F.lit(x) for kv in NOME_MUN.items() for x in kv])
cod6 = F.substring("cod_municipio_residencia", 1, 6)
ordem = Window.orderBy(*[F.col(c).asc_nulls_last() for c in COLS_ORDEM])

gold_ml = (regional
    .withColumn("id_linha", F.row_number().over(ordem).cast("long"))
    .select(
        "id_linha",
        "ano_nasc",
        F.col("uf_nasc").alias("uf_fonte"),
        "cod_municipio_residencia",
        F.coalesce(mapa_mun[cod6], cod6).alias("municipio"),
        "peso_gramas",
        "id_tipo_gravidez",
        peso_registrado.alias("peso_registrado"),
        peso_minimo_ok.alias("peso_minimo_ok"),
        gestacao_unica.alias("gestacao_unica"),
        populacao.alias("populacao_estudo"),
        (peso_registrado & (peso < 2500)).cast("int").alias("baixo_peso"),
        *[num_tratado(c) for c in NUM],
        *[cat_tratado(c) for c in CAT],
        "_src_file",
    ))

gold_ml.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(DST)
tab = spark.read.format("delta").load(DST)
tab.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(TABELA)
versao_tabela = spark.sql(f"DESCRIBE HISTORY delta.`{DST}` LIMIT 1").collect()[0]["version"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Validação

# COMMAND ----------

n_regional = regional.count()
n_tab = tab.count()
n_ids = tab.select("id_linha").distinct().count()
n_catalogo = spark.table(TABELA).count()
n_pop = tab.filter("populacao_estudo").count()
print(f"recorte sem duplicatas: {n_regional} | tabela: {n_tab} | id_linha distintos: {n_ids} | "
      f"catálogo: {n_catalogo} | população do estudo: {n_pop}")
if not (n_regional == n_tab == n_ids == n_catalogo) or n_pop == 0:
    raise RuntimeError("Validação da Gold de ML falhou")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Relatório da população (fonte por ano, exclusões, auditoria de domínios e ausência)
# MAGIC
# MAGIC Calculado sobre os valores brutos da Silver (auditoria) e sobre a tabela gravada (ausência após o tratamento). O notebook do modelo copia este relatório para o `resultados.json`.

# COMMAND ----------

df = tab.toPandas().sort_values("id_linha").reset_index(drop=True)
bruto = (regional.select(*COLS_ORDEM).toPandas()
         .sort_values(COLS_ORDEM, na_position="last", kind="mergesort").reset_index(drop=True))

fonte_por_ano = [{"ano": a, "uf_fonte": fonte[a], "registros": int(n_por_ano_bruto.get(a, 0)),
                  "duplicatas_removidas": int(n_por_ano_bruto.get(a, 0) - (df.ano_nasc == a).sum())} for a in anos]

m_reg, m_ok, m_unica, m_pop = (df[c].astype(bool) for c in ["peso_registrado", "peso_minimo_ok", "gestacao_unica", "populacao_estudo"])
peso_pd = df["peso_gramas"].astype(float)
tipo = df["id_tipo_gravidez"]
fluxo = [
    {"etapa": "Registros com residência na RMJ (fonte por ano)", "n": int(sum(n_por_ano_bruto.values()))},
    {"etapa": "Após remoção de duplicatas exatas", "n": int(len(df))},
    {"etapa": "Com peso registrado (> 0 e < 9.999 g)", "n": int(m_reg.sum())},
    {"etapa": f"Com peso ≥ {PESO_MIN} g", "n": int(m_ok.sum())},
]
if SO_UNICA:
    fluxo.append({"etapa": "Gestação única (exclui dupla, tripla ou mais e ignorado)", "n": int(m_pop.sum())})
excluidos = {
    "peso_ausente_ou_invalido": int((~m_reg).sum()),
    f"peso_menor_{PESO_MIN}g": int((m_reg & ~m_ok).sum()),
    f"pesos_menores_{PESO_MIN}g_valores": sorted(peso_pd[m_reg & ~m_ok].astype(int).tolist()),
    "gestacao_multipla": int((m_ok & tipo.isin([2, 3])).sum()),
    "gestacao_ignorada": int((m_ok & ~tipo.isin([1, 2, 3])).sum()),
    "baixo_peso_entre_multiplas_pct": round(float(df.loc[m_ok & tipo.isin([2, 3]), "baixo_peso"].mean() * 100), 2),
}

# Auditoria de domínios: valores brutos da Silver, na população do estudo
bp = bruto["peso_gramas"].astype(float)
b_reg = bp.notna() & (bp > 0) & (bp < 9999)
b_pop = b_reg & (bp >= PESO_MIN) & ((bruto["id_tipo_gravidez"].astype(float) == 1) if SO_UNICA else True)
pop_bruta = bruto[b_pop]
assert len(pop_bruta) == n_pop, (len(pop_bruta), n_pop)
auditoria = {}
for c in NUM:
    v = pop_bruta[c].astype(float)
    lo, hi = FAIXA[c]
    auditoria[c] = {"nulo_ou_menos1": int((v.isna() | (v == -1)).sum()), "codigo_99": int((v == 99).sum()),
                    "fora_da_faixa": int((v.notna() & (v != -1) & (v != 99) & ((v < lo) | (v > hi))).sum()),
                    "faixa_aceita": [lo, hi], "min_valido": float(v[(v >= lo) & (v <= hi)].min()),
                    "max_valido": float(v[(v >= lo) & (v <= hi)].max())}
for c in CAT:
    v = pop_bruta[c].astype(float).fillna(-1).astype(int)
    dom = set(DOMINIO[c])
    auditoria[c] = {"nulo_ou_menos1": int((v == -1).sum()), "codigo_9": int((v == 9).sum()),
                    "fora_do_dominio": int((~v.isin(dom | {9, -1})).sum()),
                    "valores_fora": sorted(set(v[~v.isin(dom | {9, -1})].tolist()))}

pop = df[m_pop]
incoer = pop.qtd_partos_normais + pop.qtd_partos_cesareos > pop.qtd_gestacoes_anteriores
auditoria["incoerencia_partos_maior_que_gestacoes"] = int(incoer.sum())
ausencia = {c: {"n": int(pop[c].isna().sum()) if c in NUM else int((pop[c] == "9").sum()),
                "pct": round(float((pop[c].isna() if c in NUM else (pop[c] == "9")).mean() * 100), 4)}
            for c in NUM + CAT}
print(fonte_por_ano); print(fluxo); print(excluidos); print(auditoria); print(ausencia)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Registro do estado

# COMMAND ----------

estado_novo = {
    "versao_silver": versao_silver, "parametros": P, "tabela": TABELA, "caminho": DST,
    "versao_tabela": versao_tabela, "linhas": n_tab, "linhas_populacao_estudo": n_pop,
    "colunas_atributos": {"numericos": NUM, "categoricos": CAT},
    "gerado_em": datetime.now(timezone.utc).isoformat(),
    "relatorio": {"fonte_por_ano": fonte_por_ano, "fluxo_populacao": fluxo, "excluidos": excluidos,
                  "auditoria_dominios": auditoria, "ausencia": ausencia},
}
gravar_json(ESTADO, estado_novo)
dbutils.notebook.exit(json.dumps({"status": "ok", "versao_silver": versao_silver, "versao_tabela": versao_tabela,
                                  "linhas": n_tab, "linhas_populacao_estudo": n_pop}))

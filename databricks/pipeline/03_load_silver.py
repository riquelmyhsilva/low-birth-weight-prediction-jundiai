# Databricks notebook source
# MAGIC %md
# MAGIC # Processamento e carga na Silver
# MAGIC
# MAGIC Neste notebook ocorre o processo de limpeza e carga de dados na camada Silver: uma tabela Delta única, particionada por `uf_nasc` (UF do arquivo; `BR` = arquivo nacional) e `ano_nasc` (ano do arquivo).
# MAGIC
# MAGIC ## Quando um arquivo é processado de novo
# MAGIC
# MAGIC No modo `incremental` (padrão), um arquivo da Bronze **não** é reprocessado quando a partição correspondente na Silver já tem o mesmo número de linhas e veio da mesma carga da Bronze (`_bronze_load_time` igual ao `_load_time` da Bronze). Assim, só entram arquivos novos ou recarregados na Bronze.
# MAGIC
# MAGIC O modo `reconstruir` refaz a tabela inteira a partir da Bronze (a versão anterior continua no histórico Delta e pode ser restaurada com `RESTORE`). Use quando o contrato de schema mudar.
# MAGIC
# MAGIC ## Correções em relação à versão anterior
# MAGIC - Um arquivo com problema é registrado e o loop **continua**; o notebook falha no final listando os erros (antes, `raise e` interrompia tudo — foi assim que SP 1997–2001, SP 2018 e BR 2011–2012 ficaram fora da Silver).
# MAGIC - `try_cast`/`try_to_date` via SQL (`F.expr`), que existem também no ambiente serverless de jobs.
# MAGIC - Nomes de coluna normalizados para maiúsculas (ex.: `contador` → `CONTADOR`, antes perdido em alguns anos).
# MAGIC - `SEXO` com letras (`M`/`F`/`I`, ex.: 2014) convertido para `1`/`2`/`0` antes do cast (antes virava `-1`).
# MAGIC - Linhagem preservada: `_src_file` e `_bronze_load_time`.
# MAGIC - `ano_nasc` = ano do arquivo (a partição nunca diverge do `replaceWhere`); registros cuja data de nascimento cai em outro ano são contados no resumo.
# MAGIC - Conferência de contagem Bronze × Silver depois de cada gravação.
# MAGIC
# MAGIC ![image_1778612686302.png](./img/image_1778612686302.png "image_1778612686302.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bibliotecas

# COMMAND ----------

import os
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

# COMMAND ----------

# MAGIC %run ./00_02_schema_contracts

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parâmetros, fonte e destino

# COMMAND ----------

dbutils.widgets.dropdown("modo", "incremental", ["incremental", "reconstruir"])
dbutils.widgets.text("arquivos", "")  # vazio = todos; ou lista: DNSP2018,DNSP2025
MODO = dbutils.widgets.get("modo")
FILTRO = {a.strip().upper() for a in dbutils.widgets.get("arquivos").split(",") if a.strip()}

src_bronze = "/Volumes/workspace/sinasc/01_bronze/"
dst_silver = "/Volumes/workspace/sinasc/02_silver/"
LIMITE_DTNASC_NULO_PCT = 50.0  # acima disso o arquivo é rejeitado (formato de data incompatível)

# COMMAND ----------

# MAGIC %md
# MAGIC # Função

# COMMAND ----------

def transform_to_silver(df_bronze, uf: str, ano: int):
    """
    Recebe DataFrame Bronze (tudo String) e retorna Silver com:
    - Nomes de coluna em maiúsculas (alguns arquivos trazem 'contador' em minúsculas)
    - SEXO M/F/I → 1/2/0
    - Cast explícito de tipos (Integer, Date) com try_cast/try_to_date
    - Colunas de partição: uf_nasc e ano_nasc (do arquivo), mes_nasc (de DTNASC)
    - Extração de hora inteira de HORANASC (HHMM → HH)
    - Trim em códigos geográficos
    - Renomeação em massa por projeção única
    - Linhagem: _src_file, _bronze_load_time, _silver_load_time
    """
    df = df_bronze.toDF(*[c if c.startswith("_") else c.upper() for c in df_bronze.columns])

    if "SEXO" in df.columns:
        sexo = F.upper(F.trim(F.col("SEXO")))
        expr_sexo = F.col("SEXO")
        for letra, codigo in SEXO_LETRAS.items():
            expr_sexo = F.when(sexo == letra, F.lit(codigo)).otherwise(expr_sexo)
        df = df.withColumn("SEXO", expr_sexo)

    # 1. Cast de inteiros — strings vazias, nulas ou inválidas viram -1
    for col_name in INT_COLS:
        if col_name in df.columns:
            df = df.withColumn(col_name, F.coalesce(F.expr(f"try_cast(`{col_name}` AS INT)"), F.lit(-1)))

    # 2. Parse de datas DDMMYYYY → DateType
    for col_name in DATE_COLS_DDMMYYYY:
        if col_name in df.columns:
            df = df.withColumn(col_name, F.expr(f"try_to_date(`{col_name}`, 'ddMMyyyy')"))

    # 3. Colunas de partição: UF e ano vêm do nome do arquivo; mês da data de nascimento
    df = df \
        .withColumn("ano_nasc", F.lit(ano)) \
        .withColumn("mes_nasc", F.month(F.col("DTNASC"))) \
        .withColumn("uf_nasc", F.lit(uf))

    # 4. Extração de hora inteira: "2000" → 20 (malformado → null)
    if "HORANASC" in df.columns:
        df = df.withColumn(
            "hora_nascimento_int",
            F.when(F.length(F.trim(F.col("HORANASC"))) == 4,
                   F.expr("try_cast(substring(HORANASC, 1, 2) AS INT)")).otherwise(F.lit(None).cast(IntegerType())))

    # 5. Trim em campos de código geográfico
    for col_name in STRING_COLS:
        if col_name in df.columns:
            df = df.withColumn(col_name, F.coalesce(
                F.when(F.trim(F.col(col_name)) == "", None).otherwise(F.trim(F.col(col_name))), F.lit("N/A")))

    # 6. Linhagem
    df = df.withColumnRenamed("_load_time", "_bronze_load_time") \
           .withColumn("_silver_load_time", F.current_timestamp())

    # 7. Renomeação em massa por projeção única
    system_cols = ["ano_nasc", "mes_nasc", "uf_nasc", "hora_nascimento_int",
                   "_src_file", "_bronze_load_time", "_silver_load_time"]
    selected_cols = [F.col(c).alias(RENAME_MAP[c]) for c in df.columns if c in RENAME_MAP] + \
                    [F.col(c) for c in system_cols if c in df.columns]
    return df.select(selected_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Leitura da bronze e estado atual da Silver

# COMMAND ----------

bronze_files = []
for f in dbutils.fs.ls(src_bronze):
    ident = identificar_arquivo(f.name)
    if f.name.startswith("_") or ident is None:
        if not f.name.startswith("_"):
            print(f"Ignorado (nome fora do padrão): {f.name}")
        continue
    if FILTRO and f.name.rstrip("/").upper() not in FILTRO:
        continue
    bronze_files.append((f.name.rstrip("/"), f.path, *ident))
print(f"{len(bronze_files)} arquivos da Bronze selecionados")

# estado da Silver: (uf, ano) -> (linhas, _bronze_load_time)
estado_silver = {}
silver_existe = os.path.exists(dst_silver + "_delta_log")
if MODO == "incremental" and silver_existe:
    df_s = spark.read.format("delta").load(dst_silver)
    if "_bronze_load_time" in df_s.columns:
        for r in (df_s.groupBy("uf_nasc", "ano_nasc")
                      .agg(F.count("*").alias("n"), F.max("_bronze_load_time").alias("t"))
                      .collect()):
            estado_silver[(r["uf_nasc"], r["ano_nasc"])] = (r["n"], r["t"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loop de carga

# COMMAND ----------

resumo = []
primeira_gravacao_reconstrucao = MODO == "reconstruir"

for nome, path, uf, ano in bronze_files:
    r = {"arquivo": nome, "uf": uf, "ano": ano}
    try:
        df_bronze = spark.read.format("delta").load(path)
        info = df_bronze.agg(F.count("*").alias("n"), F.max("_load_time").alias("t")).collect()[0]
        r["n_bronze"] = info["n"]

        if MODO == "incremental" and estado_silver.get((uf, ano)) == (info["n"], info["t"]):
            r["status"] = "skip"
            resumo.append(r)
            continue

        df_silver = transform_to_silver(df_bronze, uf, ano)
        q = df_silver.agg(
            F.sum(F.col("data_nascimento").isNull().cast("int")).alias("dt_nulo"),
            F.sum((F.year("data_nascimento") != ano).cast("int")).alias("ano_divergente"),
        ).collect()[0]
        r["dtnasc_nulo_pct"] = round(q["dt_nulo"] / info["n"] * 100, 2) if info["n"] else 100.0
        r["dtnasc_outro_ano"] = q["ano_divergente"] or 0

        if r["dtnasc_nulo_pct"] > LIMITE_DTNASC_NULO_PCT:
            r["status"] = "rejeitado_datas_invalidas"
            print(r)
            resumo.append(r)
            continue

        writer = df_silver.write.format("delta").partitionBy("uf_nasc", "ano_nasc")
        if primeira_gravacao_reconstrucao:
            writer.mode("overwrite").option("overwriteSchema", "true").save(dst_silver)
            primeira_gravacao_reconstrucao = False
        else:
            (writer.mode("overwrite").option("mergeSchema", "true")
                   .option("replaceWhere", f"uf_nasc = '{uf}' AND ano_nasc = {ano}").save(dst_silver))

        r["n_silver"] = spark.read.format("delta").load(dst_silver) \
            .filter((F.col("uf_nasc") == uf) & (F.col("ano_nasc") == ano)).count()
        r["status"] = "ok" if r["n_silver"] == info["n"] else "erro_contagem_divergente"

    except Exception as e:
        r["status"] = "erro"
        r["erro"] = f"{type(e).__name__}: {str(e)[:500]}"
    print(r)
    resumo.append(r)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumo

# COMMAND ----------

contagem = {}
for r in resumo:
    contagem[r["status"]] = contagem.get(r["status"], 0) + 1
print("RESUMO:", contagem)
for r in resumo:
    if r["status"] != "skip":
        print(r)

erros = [r for r in resumo if r["status"].startswith("erro")]
if erros:
    raise RuntimeError(f"Arquivos com erro na Silver: {erros}")
dbutils.notebook.exit(json.dumps({"modo": MODO, "contagem": contagem,
                                  "processados": [r["arquivo"] for r in resumo if r["status"] == "ok"],
                                  "rejeitados": [r["arquivo"] for r in resumo if r["status"].startswith("rejeitado")]},
                                 default=str))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validação (opcional)

# COMMAND ----------

# MAGIC %skip
# MAGIC df_silver = spark.read.format("delta").load(dst_silver)
# MAGIC df_silver.groupBy("uf_nasc", "ano_nasc").count().orderBy("uf_nasc", "ano_nasc").show(200)

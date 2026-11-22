# Databricks notebook source
# MAGIC %md
# MAGIC # Carga na Bronze
# MAGIC
# MAGIC Neste notebook ocorre o processo de persistir os arquivos `.delta` na bronze. Os arquivos `.csv` são convertidos para `.delta` em seu estado bruto (todas as colunas como texto, nomes originais), com duas colunas de linhagem: `_src_file` e `_load_time`.
# MAGIC
# MAGIC ## Quando um arquivo é carregado de novo
# MAGIC
# MAGIC Um CSV **não** é recarregado quando:
# MAGIC 1. a **última carga com sucesso** registrada no log (`01_bronze/_control/file_control.delta`) tem o mesmo tamanho do CSV atual;
# MAGIC 2. a tabela Delta do arquivo existe na Bronze; e
# MAGIC 3. o número de linhas dessa tabela é igual ao de linhas do CSV registrado no manifesto da conversão (`02_processado/_manifestos/<ARQUIVO>.json`).
# MAGIC
# MAGIC Antes, o log era lido como um dicionário sem ordem: com duas entradas para o mesmo arquivo, valia uma qualquer, e cargas com erro também contavam como feitas. Depois de gravar, a contagem da tabela é conferida com o manifesto.
# MAGIC
# MAGIC ![image_1778612524952.png](./img/image_1778612524952.png "image_1778612524952.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bibliotecas

# COMMAND ----------

import os
from datetime import datetime
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %run ./00_02_schema_contracts

# COMMAND ----------

# MAGIC %md
# MAGIC ## Caminhos de fonte, destino e log

# COMMAND ----------

src = "/Volumes/workspace/sinasc/00_landingzone/02_processado/"
dst = "/Volumes/workspace/sinasc/01_bronze/"
log_path = "/Volumes/workspace/sinasc/01_bronze/_control/file_control.delta"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Última carga com sucesso por arquivo

# COMMAND ----------

ultima_ok = {}
if os.path.exists(log_path + "/_delta_log"):
    df_control = spark.read.format("delta").load(log_path)
    linhas_log = (df_control
                  .filter(F.lower(F.col("status")).isin("ok", "success"))
                  .select("file", "size", "end")
                  .collect())
    for r in sorted(linhas_log, key=lambda r: r["end"]):
        ultima_ok[r["file"]] = r  # a mais recente fica por último

# COMMAND ----------

# MAGIC %md
# MAGIC ## Seleção dos arquivos

# COMMAND ----------

todos = [f for f in dbutils.fs.ls(src) if not f.name.endswith("/")]
csv_files = [f for f in todos if f.name.lower().endswith(".csv") and identificar_arquivo(f.name)]
ignorados = sorted(f.name for f in todos if f.name.lower().endswith(".csv") and not identificar_arquivo(f.name))
if ignorados:
    print(f"[ INFO ] Ignorados (fora do padrão DNUFAAAA.csv): {ignorados}")


def linhas_esperadas(nome_csv: str):
    m = ler_json(os.path.join(src, "_manifestos", os.path.splitext(nome_csv)[0].upper() + ".json"))
    return m.get("linhas_emitidas") if m and m.get("status") == "ok" else None


def bronze_atualizada(f) -> tuple:
    """Retorna (atualizada: bool, motivo: str)."""
    saida = dst + os.path.splitext(f.name)[0]
    reg = ultima_ok.get(f.name)
    if reg is None:
        return False, "sem carga com sucesso no log"
    if reg["size"] != f.size:
        return False, f"CSV mudou de tamanho ({reg['size']} → {f.size})"
    if not os.path.exists(saida + "/_delta_log"):
        return False, "tabela Delta ausente"
    esperado = linhas_esperadas(f.name)
    if esperado is None:
        return False, "CSV sem manifesto de conversão ok"
    n = spark.read.format("delta").load(saida).count()
    if n != esperado:
        return False, f"Bronze com {n} linhas; manifesto indica {esperado}"
    return True, "atualizada"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carga dos arquivos CSV em Delta na Bronze com Schema-on-read

# COMMAND ----------

log = []
for f in csv_files:
    atualizada, motivo = bronze_atualizada(f)
    if atualizada:
        continue

    name_delta = os.path.splitext(f.name)[0]
    output = dst + name_delta
    log_start = datetime.now()
    try:
        esperado = linhas_esperadas(f.name)
        if esperado is None:
            raise ValueError("CSV sem manifesto de conversão com status ok; rode o 01_03 antes")

        # lê o arquivo sem inferir schema
        df = spark.read.option("header", "true").option("inferSchema", "false").csv(f.path)
        lines = df.count()
        if lines != esperado:
            raise ValueError(f"CSV lido com {lines} linhas; manifesto indica {esperado}")

        (df.withColumn("_src_file", F.lit(f.name))
           .withColumn("_load_time", F.current_timestamp())
           .write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(output))

        gravadas = spark.read.format("delta").load(output).count()
        if gravadas != esperado:
            raise ValueError(f"Bronze gravada com {gravadas} linhas; esperado {esperado}")

        log_end = datetime.now()
        log.append({"file": f.name, "size": f.size, "status": "OK", "lines": lines, "error": None,
                    "motivo": motivo, "start": log_start, "end": log_end,
                    "duration_sec": (log_end - log_start).seconds})
        print(f"[  OK  ] {f.name}: {lines} linhas ({motivo})")

    except Exception as e:
        log_end = datetime.now()
        log.append({"file": f.name, "size": f.size, "status": "err", "lines": None,
                    "error": f"{type(e).__name__}: {str(e)[:1000]}", "motivo": motivo,
                    "start": log_start, "end": log_end, "duration_sec": (log_end - log_start).seconds})
        print(f"[ERROR] {f.name}: {e}")

# persiste o log se algo foi alterado
if log:
    schema_log = ("file string, size long, status string, lines long, error string, motivo string, "
                  "start timestamp, `end` timestamp, duration_sec long")
    (spark.createDataFrame(log, schema_log)
          .write.format("delta").mode("append").option("mergeSchema", "true").save(log_path))
else:
    print("Nenhum arquivo novo ou alterado.")

erros = [r for r in log if r["status"] != "OK"]
if erros:
    raise RuntimeError(f"Cargas com erro: {[(r['file'], r['error']) for r in erros]}")
dbutils.notebook.exit(json.dumps({"csv_avaliados": len(csv_files), "carregados": [r["file"] for r in log]}))

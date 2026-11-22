# Databricks notebook source
# MAGIC %md
# MAGIC # Setup
# MAGIC
# MAGIC Estabelecendo Schema, Volumes e diretórios no ambiente de Catalog do Databricks. Idempotente: pode rodar a cada execução do pipeline.
# MAGIC
# MAGIC **Catálogo**: workspace
# MAGIC
# MAGIC **Schema**: sinasc
# MAGIC
# MAGIC **Volumes**: 00_landingzone, 01_bronze, 02_silver, 03_gold
# MAGIC
# MAGIC **Diretórios (os mesmos usados pelas etapas seguintes):**
# MAGIC
# MAGIC | Diretório | Conteúdo | Etapa que escreve |
# MAGIC |---|---|---|
# MAGIC | `00_landingzone/00_descompactar` | `.dbc` baixados do FTP | 01_01 |
# MAGIC | `00_landingzone/01_processar` | `.dbf` descompactados | 01_02 |
# MAGIC | `00_landingzone/02_processado` | `.csv` convertidos + manifestos | 01_03 |
# MAGIC | `01_bronze/<arquivo>` e `01_bronze/_control` | Delta bruto por arquivo + log de carga | 02 |
# MAGIC | `02_silver` | Delta único particionado por `uf_nasc`/`ano_nasc` | 03 |
# MAGIC | `03_gold/<tabela>` e `03_gold/_control` | fato, dimensões e estado da última carga | 04 e 05 |
# MAGIC
# MAGIC ![image_1778604745237.png](./img/image_1778604745237.png "image_1778604745237.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Criação do schema, volumes e diretórios

# COMMAND ----------

# Schema
spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.sinasc COMMENT 'Schema SINASC'")

# Volumes dentro do schema (Unity Catalog)
spark.sql("CREATE VOLUME IF NOT EXISTS workspace.sinasc.00_landingzone COMMENT '00 Landingzone volume'")
spark.sql("CREATE VOLUME IF NOT EXISTS workspace.sinasc.01_bronze COMMENT '01 Bronze volume'")
spark.sql("CREATE VOLUME IF NOT EXISTS workspace.sinasc.02_silver COMMENT '02 Silver volume'")
spark.sql("CREATE VOLUME IF NOT EXISTS workspace.sinasc.03_gold COMMENT '03 Gold volume'")

# Diretórios específicos (antes eram criados 00_processar/01_processado, que nenhuma etapa usava)
for d in [
    "00_landingzone/00_descompactar",
    "00_landingzone/01_processar",
    "00_landingzone/02_processado",
    "01_bronze/_control",
    "03_gold/_control",
]:
    dbutils.fs.mkdirs(f"/Volumes/workspace/sinasc/{d}")

# O curso usava DBFS mounts, mas implementei com Unity Catalog Volumes que é o padrão atual
# As montagens de DBFS raiz e DBFS são preteridas e não recomendadas pelo Databricks. Novas contas são provisionadas sem acesso a esses recursos.
# Em vez disso, o Databricks recomenda usar volumes do Unity Catalog, locais externos ou arquivos de espaço de trabalho.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verificando schema (manutenção)

# COMMAND ----------

# MAGIC %sql
# MAGIC /*SHOW VOLUMES in sinasc;*/

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deletar schema (manutenção)

# COMMAND ----------

# DELETAR SCHEMA
# # spark.sql("DROP SCHEMA IF EXISTS workspace.sinasc # CASCADE")
# CASCADE deleta o schema e tudo dentro dele

# OU

# Apagando Volumes
# spark.sql("DROP VOLUME workspace.sinasc.00_landingzone")
# spark.sql("DROP VOLUME workspace.sinasc.01_bronze")
# spark.sql("DROP VOLUME workspace.sinasc.02_silver")
# spark.sql("DROP VOLUME workspace.sinasc.03_gold")

# Apagando Schema
# spark.sql("DROP SCHEMA  workspace.sinasc")

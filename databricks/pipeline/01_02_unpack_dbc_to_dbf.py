# Databricks notebook source
# MAGIC %md
# MAGIC # Descompactação
# MAGIC
# MAGIC Este notebook tem o papel de descompactar os arquivos `.dbc` baixados em `00_descompactar`, transformando-os em arquivos `.dbf` e carregando-os no diretório `01_processar`.
# MAGIC
# MAGIC ## Quando um arquivo é descompactado de novo
# MAGIC
# MAGIC Um `.dbf` existente **é mantido** quando as duas condições valem:
# MAGIC 1. **DBF íntegro:** o tamanho do arquivo é exatamente `tamanho do cabeçalho + nº de registros × tamanho do registro` (valores lidos do próprio cabeçalho DBF, com ou sem o byte final `0x1A`). Um DBF truncado ou corrompido falha nessa conta.
# MAGIC 2. **Mesma fonte:** o manifesto (`01_processar/_manifestos/<arquivo>.json`) registra o tamanho e a data do `.dbc` que gerou o `.dbf`, e o `.dbc` atual é o mesmo. Se o `.dbc` foi baixado de novo (ex.: nova versão do preliminar), o `.dbf` é refeito.
# MAGIC
# MAGIC DBFs antigos sem manifesto são aceitos se estiverem íntegros e forem mais novos que o `.dbc`; nesse caso o manifesto é criado. A heurística anterior ("DBF com pelo menos 4× o tamanho do DBC") foi removida: ela não comprovava a integridade.
# MAGIC
# MAGIC A descompactação grava em `<arquivo>.dbf.partial` e só renomeia depois de validar o cabeçalho.
# MAGIC
# MAGIC ![image_1778609243292.png](./img/image_1778609243292.png "image_1778609243292.png")

# COMMAND ----------

# instala a biblioteca (já disponível no ambiente do workspace)
#%pip install dbc_to_dbf

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bibliotecas

# COMMAND ----------

import os
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from dbctodbf import DBCDecompress # biblioteca para descompactar o arquivo DBC

# COMMAND ----------

# MAGIC %run ./00_02_schema_contracts

# COMMAND ----------

# MAGIC %md
# MAGIC ## Caminho de fonte e destino

# COMMAND ----------

src = "/Volumes/workspace/sinasc/00_landingzone/00_descompactar/"
dst = "/Volumes/workspace/sinasc/00_landingzone/01_processar/"
dir_manifestos = dst + "_manifestos/"
os.makedirs(dir_manifestos, exist_ok=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Funções

# COMMAND ----------

def verificar_dbf(caminho: str) -> dict:
    """
    Confere a integridade estrutural de um DBF pelo cabeçalho:
    tamanho real == cabeçalho + registros × tamanho do registro (+ 1 byte opcional de fim de arquivo).
    """
    with open(caminho, "rb") as f:
        cab = f.read(32)
    if len(cab) < 32:
        return {"integro": False, "motivo": "arquivo menor que o cabeçalho DBF"}
    n_registros = int.from_bytes(cab[4:8], "little")
    tam_cabecalho = int.from_bytes(cab[8:10], "little")
    tam_registro = int.from_bytes(cab[10:12], "little")
    tam_real = os.path.getsize(caminho)
    esperado = tam_cabecalho + n_registros * tam_registro
    integro = tam_registro > 0 and n_registros > 0 and tam_real in (esperado, esperado + 1)
    return {"integro": integro, "registros": n_registros, "tamanho": tam_real, "tamanho_esperado": esperado,
            "motivo": None if integro else f"tamanho {tam_real} difere do esperado {esperado}"}


def decompress_single_file(name: str, path: str, dst: str) -> dict:
    """Descompacta um .dbc, pulando o que já está íntegro e veio da mesma versão do .dbc."""
    path_real = path.replace("dbfs:", "")
    base = os.path.splitext(name)[0]
    name_dbf = base + ".dbf"
    output = os.path.join(dst, name_dbf)
    parcial = output + ".partial"
    manifesto_path = dir_manifestos + base.upper() + ".json"
    fonte = assinatura_arquivo(path_real)

    if os.path.exists(output):
        dbf = verificar_dbf(output)
        manifesto = ler_json(manifesto_path)
        if dbf["integro"] and manifesto and manifesto.get("fonte_dbc") == fonte:
            return {"file": name_dbf, "status": "skip", "registros": dbf["registros"]}
        if dbf["integro"] and manifesto is None and os.stat(output).st_mtime >= fonte["mtime"]:
            gravar_json(manifesto_path, {"dbc": name, "dbf": name_dbf, "fonte_dbc": fonte, "dbf_verificado": dbf,
                                         "adotado_sem_descompactar": True,
                                         "registrado_em": datetime.now(timezone.utc).isoformat()})
            return {"file": name_dbf, "status": "skip (manifesto criado)", "registros": dbf["registros"]}
        motivo = dbf["motivo"] or "o .dbc mudou desde a última descompactação"
    else:
        motivo = "dbf ausente"

    try:
        if os.path.exists(parcial):
            os.remove(parcial)
        DBCDecompress().decompressFile(path_real, parcial)
        dbf = verificar_dbf(parcial)
        if not dbf["integro"]:
            raise ValueError(f"DBF gerado inválido: {dbf['motivo']}")
        os.replace(parcial, output)
        gravar_json(manifesto_path, {"dbc": name, "dbf": name_dbf, "fonte_dbc": fonte, "dbf_verificado": dbf,
                                     "descompactado_em": datetime.now(timezone.utc).isoformat()})
        return {"file": name_dbf, "status": "ok", "motivo": motivo, "registros": dbf["registros"]}
    except Exception as e:
        if os.path.exists(parcial):
            os.remove(parcial)
        return {"file": name_dbf, "status": "error", "error": f"{type(e).__name__}: {e}"}

# COMMAND ----------

def batch_decompress_parallel(files: list, dst: str, max_workers: int = 4) -> list:
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(decompress_single_file, n, p, dst) for n, p in files]
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            if r["status"] != "skip":
                print(r)

    resumo = {}
    for r in results:
        resumo[r["status"]] = resumo.get(r["status"], 0) + 1
    print(f"\n[RESUMO] Total: {len(results)} | {resumo}")
    return results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Seleção dos arquivos e descompactação

# COMMAND ----------

todos = [(f.name, f.path) for f in dbutils.fs.ls(src) if not f.name.endswith("/")]
files = [(n, p) for n, p in todos if n.lower().endswith(".dbc") and identificar_arquivo(n)]
selecionados = {n for n, _ in files}
ignorados = sorted(n for n, _ in todos if n not in selecionados and not n.endswith(".partial"))
if ignorados:
    print(f"[ INFO ] Ignorados (fora do padrão DNUFAAAA.dbc): {ignorados}")

resultados = batch_decompress_parallel(files, dst)
erros = [r for r in resultados if r["status"] == "error"]
if erros:
    raise RuntimeError(f"Falha ao descompactar: {erros}")
dbutils.notebook.exit(json.dumps({"total": len(resultados),
                                  "descompactados": [r["file"] for r in resultados if r["status"] == "ok"]}))

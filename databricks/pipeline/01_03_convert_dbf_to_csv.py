# Databricks notebook source
# MAGIC %md
# MAGIC # Conversão DBF → CSV
# MAGIC
# MAGIC Converte os `.dbf` de `01_processar` em `.csv` em `02_processado`, com um manifesto por arquivo em `02_processado/_manifestos/<ARQUIVO>.json` (manifestos antigos, gravados como `<arquivo>.manifest.json` ao lado do CSV, são movidos para lá automaticamente).
# MAGIC
# MAGIC ## Quando um arquivo é convertido de novo
# MAGIC
# MAGIC Um `.csv` existente **é mantido** quando:
# MAGIC 1. o manifesto está com `status = ok`;
# MAGIC 2. o `.dbf` de origem é o mesmo registrado no manifesto (tamanho, data e hash do 1º MiB);
# MAGIC 3. o `.csv` tem o mesmo tamanho registrado no manifesto (a escrita é atômica, então tamanho igual = arquivo completo).
# MAGIC
# MAGIC CSVs antigos, sem manifesto, são **auditados uma vez**: se forem mais novos que o `.dbf` e tiverem exatamente uma linha por registro ativo do DBF, ganham manifesto e são mantidos; senão são convertidos de novo. A versão anterior deste notebook pulava qualquer `.csv` que existisse — foi assim que o `DNBR2014` truncado (303.582 de 2.979.259 linhas) passou despercebido.
# MAGIC
# MAGIC ## Garantias da conversão
# MAGIC - Seleção estrita: só `DNUFAAAA.dbf` (qualquer caixa); destinos duplicados abortam antes de começar.
# MAGIC - Escrita atômica em `<arquivo>.csv.partial`, publicada como `.csv` só depois da validação.
# MAGIC - Contagem independente dos marcadores do DBF (ativos/excluídos/inválidos) comparada com as linhas emitidas e com o CSV gravado.

# COMMAND ----------

# MAGIC %pip install dbfread

# COMMAND ----------

import os
import csv
import hashlib
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from dbfread import DBF

csv.field_size_limit(10**9)

# COMMAND ----------

# MAGIC %run ./00_02_schema_contracts

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parâmetros

# COMMAND ----------

dbutils.widgets.text("src", "/Volumes/workspace/sinasc/00_landingzone/01_processar/")
dbutils.widgets.text("dst", "/Volumes/workspace/sinasc/00_landingzone/02_processado/")
dbutils.widgets.text("arquivos", "")  # vazio = todos os DNUFAAAA.dbf; ou lista: DNSP2025.dbf,DNSP2026.dbf
dbutils.widgets.text("max_workers", "4")
SRC = dbutils.widgets.get("src")
DST = dbutils.widgets.get("dst")
FILTRO = {a.strip().upper() for a in dbutils.widgets.get("arquivos").split(",") if a.strip()}
MAX_WORKERS = int(dbutils.widgets.get("max_workers"))

ENCODING_DBF = "latin-1"
VERSAO_CONVERSOR = "2026-11-21.1"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Funções

# COMMAND ----------

def nome_destino(nome_fonte: str) -> str:
    """Regra única de nome de destino."""
    return os.path.splitext(nome_fonte)[0] + ".csv"


def caminho_manifesto(dst: str, nome_fonte: str) -> str:
    """Manifesto em <dst>/_manifestos/<ARQUIVO>.json (mesmo padrão das etapas 01_01 e 01_02)."""
    return os.path.join(dst, "_manifestos", os.path.splitext(nome_fonte)[0].upper() + ".json")


def migrar_manifesto_antigo(dst: str, nome_fonte: str) -> None:
    """Versões anteriores gravavam <ARQUIVO>.manifest.json ao lado do CSV; move para _manifestos/."""
    antigo = os.path.join(dst, os.path.splitext(nome_fonte)[0] + ".manifest.json")
    if not os.path.exists(antigo):
        return
    novo = caminho_manifesto(dst, nome_fonte)
    if os.path.exists(novo):
        os.remove(antigo)
    else:
        os.makedirs(os.path.dirname(novo), exist_ok=True)
        os.replace(antigo, novo)


def selecionar_entradas(arquivos: list, filtro: set | None = None) -> list:
    """
    arquivos: lista de (nome, caminho). Mantém só DNUFAAAA.dbf (qualquer caixa),
    aplica o filtro opcional e aborta se dois arquivos gerarem o mesmo destino.
    """
    selecionados = [
        (n, p.replace("dbfs:", "")) for n, p in arquivos
        if n.lower().endswith(".dbf") and identificar_arquivo(n) and (not filtro or n.upper() in filtro)
    ]
    destinos = {}
    for n, _ in selecionados:
        destinos.setdefault(nome_destino(n).upper(), []).append(n)
    conflitos = {d: ns for d, ns in destinos.items() if len(ns) > 1}
    if conflitos:
        raise ValueError(f"Destinos conflitantes: {conflitos}")
    if filtro:
        faltando = filtro - {n.upper() for n, _ in selecionados}
        if faltando:
            raise FileNotFoundError(f"Arquivos do filtro não encontrados: {sorted(faltando)}")
    return selecionados


def contar_marcadores_dbf(caminho: str, bloco_registros: int = 100_000) -> dict:
    """
    Contagem independente da dbfread: lê o cabeçalho e o 1º byte de cada
    posição de registro. ' ' = ativo, '*' = excluído, outro = inválido.
    """
    with open(caminho, "rb") as f:
        cab = f.read(32)
        n_registros = int.from_bytes(cab[4:8], "little")
        tam_cabecalho = int.from_bytes(cab[8:10], "little")
        tam_registro = int.from_bytes(cab[10:12], "little")
        tam_arquivo = os.fstat(f.fileno()).st_size

        ativos = excluidos = invalidos = 0
        primeiro_invalido = None
        f.seek(tam_cabecalho)
        lidos = 0
        while lidos < n_registros:
            n = min(bloco_registros, n_registros - lidos)
            dados = f.read(n * tam_registro)
            completos = len(dados) // tam_registro
            for i in range(completos):
                m = dados[i * tam_registro]
                if m == 0x20:
                    ativos += 1
                elif m == 0x2A:
                    excluidos += 1
                else:
                    invalidos += 1
                    if primeiro_invalido is None:
                        primeiro_invalido = lidos + i
            lidos += completos
            if completos < n:  # arquivo truncado
                break

    return {
        "registros_cabecalho": n_registros, "posicoes_lidas": lidos,
        "ativos": ativos, "excluidos": excluidos, "invalidos": invalidos,
        "primeiro_invalido": primeiro_invalido, "tamanho_cabecalho": tam_cabecalho,
        "tamanho_registro": tam_registro, "tamanho_arquivo": tam_arquivo,
        "tamanho_esperado": tam_cabecalho + n_registros * tam_registro,
    }


def assinatura_fonte(caminho: str) -> dict:
    st = os.stat(caminho)
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        h.update(f.read(1 << 20))  # 1º MiB: identifica sem reler o arquivo todo
    return {"tamanho": st.st_size, "mtime": int(st.st_mtime), "sha256_1mib": h.hexdigest()}


def contar_linhas_csv(caminho: str) -> int:
    with open(caminho, "r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.reader(f)) - 1  # menos o cabeçalho


def validar_csv(csv_path: str, colunas_esperadas: set, linhas_esperadas: int | None = None) -> dict:
    """Validação estrutural com nomes normalizados e comparação com a fonte."""
    esperadas = {c.upper() for c in colunas_esperadas}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        originais = list(reader.fieldnames or [])
        total = sum(1 for _ in reader)
    reais = {c.upper() for c in originais}
    return {
        "total_linhas": total,
        "colunas_originais": originais,
        "colunas_ausentes": sorted(esperadas - reais),
        "colunas_extras": sorted(reais - esperadas),
        "contagem_ok": linhas_esperadas is None or total == linhas_esperadas,
    }


def situacao_existente(fonte: str, saida: str, manifesto_path: str) -> tuple:
    """Decide se o CSV atual pode ser mantido. Retorna (manter: bool, motivo: str)."""
    if not os.path.exists(saida):
        return False, "csv ausente"
    m = ler_json(manifesto_path)
    if m is not None:
        if m.get("status") != "ok":
            return False, f"manifesto com status {m.get('status')}"
        if m.get("fonte") != assinatura_fonte(fonte):
            return False, "o .dbf mudou desde a conversão"
        tam_csv = os.path.getsize(saida)
        if m.get("csv_tamanho") == tam_csv:
            return True, "manifesto ok"
        if m.get("csv_tamanho") is None and contar_linhas_csv(saida) == m.get("linhas_emitidas"):
            m["csv_tamanho"] = tam_csv  # manifestos antigos não guardavam o tamanho do CSV
            gravar_json(manifesto_path, m)
            return True, "manifesto ok (tamanho do csv registrado agora)"
        return False, "csv diferente do registrado no manifesto"

    # CSV legado sem manifesto: auditoria única
    if os.stat(saida).st_mtime < os.stat(fonte).st_mtime:
        return False, "csv legado mais antigo que o .dbf"
    marc = contar_marcadores_dbf(fonte)
    linhas = contar_linhas_csv(saida)
    if marc["posicoes_lidas"] == marc["registros_cabecalho"] and marc["invalidos"] == 0 and linhas == marc["ativos"]:
        gravar_json(manifesto_path, {
            "arquivo_fonte": os.path.basename(fonte), "arquivo_saida": os.path.basename(saida),
            "versao_conversor": VERSAO_CONVERSOR, "status": "ok", "adotado_sem_converter": True,
            "fonte": assinatura_fonte(fonte), "marcadores_dbf": marc, "linhas_emitidas": linhas,
            "csv_tamanho": os.path.getsize(saida), "registrado_em": datetime.now(timezone.utc).isoformat(),
        })
        return True, "csv legado auditado e aceito"
    return False, f"csv legado divergente ({linhas} linhas para {marc['ativos']} registros ativos)"


def converter_dbf(entrada: tuple, dst: str, colunas_esperadas: set) -> dict:
    nome, caminho = entrada
    nome_csv = nome_destino(nome)
    saida = os.path.join(dst, nome_csv)
    parcial = saida + ".partial"
    manifesto_path = caminho_manifesto(dst, nome)

    try:
        migrar_manifesto_antigo(dst, nome)
        manter, motivo = situacao_existente(caminho, saida, manifesto_path)
    except Exception as e:
        manter, motivo = False, f"falha ao verificar o existente: {type(e).__name__}: {e}"
    if manter:
        return {"arquivo": nome_csv, "status": "skip", "motivo": motivo}

    manifesto = {
        "arquivo_fonte": nome, "caminho_fonte": caminho, "arquivo_saida": nome_csv,
        "versao_conversor": VERSAO_CONVERSOR, "motivo_conversao": motivo,
        "inicio": datetime.now(timezone.utc).isoformat(), "status": "em_andamento",
    }
    try:
        manifesto["fonte"] = assinatura_fonte(caminho)
        marc = contar_marcadores_dbf(caminho)
        manifesto["marcadores_dbf"] = marc
        gravar_json(manifesto_path, manifesto)

        if os.path.exists(parcial):
            os.remove(parcial)

        emitidas = 0
        with open(parcial, "w", newline="", encoding="utf-8") as f:
            tabela = DBF(caminho, encoding=ENCODING_DBF)
            writer = csv.DictWriter(f, fieldnames=tabela.field_names)
            writer.writeheader()
            for rec in tabela:
                writer.writerow(rec)
                emitidas += 1
        manifesto["linhas_emitidas"] = emitidas

        problemas = []
        if marc["posicoes_lidas"] != marc["registros_cabecalho"]:
            problemas.append("DBF truncado: menos posições que o cabeçalho declara")
        if marc["invalidos"]:
            problemas.append(f"{marc['invalidos']} posições com marcador inválido (primeira: {marc['primeiro_invalido']})")
        if emitidas != marc["ativos"]:
            problemas.append(f"linhas emitidas ({emitidas}) != registros ativos ({marc['ativos']})")

        val = validar_csv(parcial, colunas_esperadas, linhas_esperadas=marc["ativos"])
        manifesto["validacao"] = val
        if not val["contagem_ok"]:
            problemas.append("contagem do CSV gravado difere dos registros ativos")

        manifesto["fim"] = datetime.now(timezone.utc).isoformat()
        if problemas:
            manifesto["status"] = "invalido"
            manifesto["problemas"] = problemas
            gravar_json(manifesto_path, manifesto)
            return {"arquivo": nome_csv, "status": "invalido", "problemas": problemas}  # .partial fica para inspeção

        os.replace(parcial, saida)  # publicação só após validação
        manifesto["csv_tamanho"] = os.path.getsize(saida)
        manifesto["status"] = "ok"
        gravar_json(manifesto_path, manifesto)
        return {"arquivo": nome_csv, "status": "ok", "motivo": motivo, "linhas": emitidas,
                "colunas_ausentes": val["colunas_ausentes"]}

    except Exception as e:
        manifesto.update(status="erro", erro=f"{type(e).__name__}: {e}", fim=datetime.now(timezone.utc).isoformat())
        gravar_json(manifesto_path, manifesto)
        return {"arquivo": nome_csv, "status": "erro", "erro": manifesto["erro"]}


def converter_lote(entradas: list, dst: str, colunas_esperadas: set, max_workers: int) -> list:
    os.makedirs(dst, exist_ok=True)
    resultados = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futuros = [ex.submit(converter_dbf, e, dst, colunas_esperadas) for e in entradas]
        for fut in as_completed(futuros):
            r = fut.result()
            resultados.append(r)
            print(r)
    contagem = {}
    for r in resultados:
        contagem[r["status"]] = contagem.get(r["status"], 0) + 1
    print(f"\n[RESUMO] Total: {len(resultados)} | {contagem}")
    return resultados

# COMMAND ----------

# MAGIC %md
# MAGIC ## Execução

# COMMAND ----------

COLUNAS_ESPERADAS = set(INT_COLS + DATE_COLS_DDMMYYYY + STRING_COLS)
arquivos = [(f.name, f.path) for f in dbutils.fs.ls(SRC) if not f.name.endswith("/")]
entradas = selecionar_entradas(arquivos, FILTRO or None)
ignorados = sorted(n for n, _ in arquivos if n not in {e[0] for e in entradas} and not n.endswith(".partial"))
if ignorados and not FILTRO:
    print(f"[ INFO ] Ignorados (fora do padrão DNUFAAAA.dbf): {ignorados}")

resultados = converter_lote(entradas, DST, COLUNAS_ESPERADAS, MAX_WORKERS)
falhas = [r for r in resultados if r["status"] in ("erro", "invalido")]
if falhas:
    raise RuntimeError(f"Conversões com falha: {falhas}")
dbutils.notebook.exit(json.dumps({"total": len(resultados),
                                  "convertidos": [r["arquivo"] for r in resultados if r["status"] == "ok"]}))

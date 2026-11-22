# Databricks notebook source
# MAGIC %md
# MAGIC # Importação
# MAGIC
# MAGIC Neste notebook ocorre o processo de importação dos dados do DataSUS para o Databricks, que chegam na pasta `00_descompactar` do volume `00_landingzone` (`/00_landingzone/00_descompactar`).
# MAGIC
# MAGIC ## Processo
# MAGIC
# MAGIC O processo é feito através da biblioteca `ftplib`, que baixa os arquivos do servidor `ftp` do DataSUS. Esses arquivos são alocados diretamente no diretório `00_descompactar` no formato `.dbc` (formato comprimido dos dados) e ficam lá até o momento de serem descomprimidos.
# MAGIC
# MAGIC Pastas do SINASC no FTP (`/dissemin/publicos/SINASC/`):
# MAGIC - `1996_/Dados/DNRES` — dados **definitivos** por UF de residência (`DNUFAAAA.dbc`), de 1996 em diante.
# MAGIC - `PRELIM/DNRES` — dados **preliminares** dos anos ainda não fechados (ex.: 2025 e 2026). São atualizados pelo DATASUS ao longo do ano.
# MAGIC
# MAGIC ## Quando um arquivo é baixado de novo
# MAGIC
# MAGIC Um arquivo **não** é baixado se o arquivo local tem o mesmo tamanho do FTP **e** a data de modificação no FTP (`MDTM`) é a mesma registrada no manifesto do último download (`00_descompactar/_manifestos/<arquivo>.json`). Arquivos antigos sem manifesto e com o mesmo tamanho são aceitos e ganham manifesto. Assim, uma nova versão de um arquivo preliminar é baixada mesmo que o tamanho não mude.
# MAGIC
# MAGIC ## Uso
# MAGIC
# MAGIC 1. Escolha a pasta, as UFs e os anos nos widgets (`anos` aceita `todos`, `2018`, `2012-2024` ou `2018,2025`).
# MAGIC 2. Com `executar = nao`, o notebook só lista o que existe no FTP e compara com a landing.
# MAGIC 3. Com `executar = sim`, baixa o que falta ou está desatualizado.
# MAGIC
# MAGIC ![image_1778609185008.png](./img/image_1778609185008.png "image_1778609185008.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bibliotecas

# COMMAND ----------

import os
import time
import ftplib # biblioteca para puxar dados de um servidor ftp (ja vem no ambiente databricks)
from datetime import datetime, timezone

# COMMAND ----------

# MAGIC %run ./00_02_schema_contracts

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parâmetros

# COMMAND ----------

FTP_HOST = "ftp.datasus.gov.br"
FTP_BASE = "/dissemin/publicos/SINASC/"
PASTAS = ["1996_/Dados/DNRES", "PRELIM/DNRES"]

dbutils.widgets.dropdown("pasta", PASTAS[0], PASTAS, "Pasta no FTP")
dbutils.widgets.text("ufs", "SP", "UFs (vírgula)")
dbutils.widgets.text("anos", "todos", "Anos (todos | 2018 | 2012-2024 | 2018,2025)")
dbutils.widgets.dropdown("executar", "nao", ["nao", "sim"], "Baixar?")

pasta = dbutils.widgets.get("pasta")
ufs_sel = [u.strip().upper() for u in dbutils.widgets.get("ufs").split(",") if u.strip()]
anos_txt = dbutils.widgets.get("anos").strip().lower()
executar = dbutils.widgets.get("executar") == "sim"

dst = "/Volumes/workspace/sinasc/00_landingzone/00_descompactar/"
dir_manifestos = dst + "_manifestos/"
os.makedirs(dir_manifestos, exist_ok=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Funções

# COMMAND ----------

# MAGIC %md
# MAGIC ### Função de Conexão

# COMMAND ----------

def connect_datasus_ftp(path: str = FTP_BASE + PASTAS[0], timeout: int = 60):
    """
    Estabelece uma conexão com o servidor FTP do DataSUS em modo binário.

    Args:
        path: diretório no servidor. O padrão é a pasta de dados definitivos (DNRES).
        timeout: tempo limite da conexão em segundos, para o processo não ficar travado.

    Returns:
        ftplib.FTP | None: conexão aberta, ou None em caso de erro.
    """
    try:
        ftp = ftplib.FTP(FTP_HOST, timeout=timeout)
        ftp.login()
        ftp.cwd(path)
        ftp.voidcmd("TYPE I")  # modo binário: necessário para SIZE e RETR de .dbc
        print(f"[  OK  ] Conectado ao FTP do DataSUS no diretório: {path}")
        return ftp
    except Exception as e:
        print(f"[ ERROR] Erro ao conectar ao FTP do DataSUS: {e}")
        return None

# COMMAND ----------

# MAGIC %md
# MAGIC ### Funções de listagem

# COMMAND ----------

def listar_arquivos_dn(ftp) -> dict:
    """
    Lista os arquivos DNUFAAAA.dbc da pasta atual, ignorando a caixa da extensão
    (o FTP mistura .dbc e .DBC). Retorna {(uf, ano): nome_original_no_ftp}.
    """
    arquivos = {}
    for nome in ftp.nlst():
        ident = identificar_arquivo(nome)
        if ident and nome.lower().endswith(".dbc"):
            arquivos[ident] = nome
    return arquivos


def mdtm(ftp, nome: str) -> str:
    """Data de modificação do arquivo no FTP (AAAAMMDDhhmmss)."""
    try:
        return ftp.sendcmd(f"MDTM {nome}").split()[-1]
    except Exception:
        return None


def interpretar_anos(texto: str, disponiveis: list) -> list:
    """'todos' | '2018' | '2012-2024' | '2018,2025' → lista de anos."""
    if texto in ("", "todos"):
        return sorted(disponiveis)
    anos = set()
    for parte in texto.split(","):
        parte = parte.strip()
        if "-" in parte:
            a, b = (int(x) for x in parte.split("-"))
            anos.update(range(a, b + 1))
        elif parte:
            anos.add(int(parte))
    return sorted(anos)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Função de download

# COMMAND ----------

def download_datasus_file(
    target_file: str,
    output: str,
    ftp_path: str,
    max_retries: int = 3,
    retry_delay: int = 15,
) -> dict | None:
    """
    Baixa um arquivo do FTP do DataSUS com nova conexão a cada tentativa.

    Grava em `<output>.partial` e só renomeia para o nome final depois de conferir o tamanho,
    para que um download interrompido nunca pareça um arquivo completo.

    Returns:
        dict com tamanho e MDTM do arquivo baixado, ou None se falhou após todas as tentativas.
    """
    parcial = output + ".partial"
    for attempt in range(1, max_retries + 1):
        ftp = connect_datasus_ftp(ftp_path)
        if ftp is None:
            print(f"[RETRY {attempt}/{max_retries}] Falha na conexão, aguardando {retry_delay}s...")
            time.sleep(retry_delay)
            continue
        try:
            size_ftp = ftp.size(target_file)
            data_ftp = mdtm(ftp, target_file)
            print(f"[STATUS] Baixando {target_file} ({size_ftp / 1e6:.1f} MB)...")
            start = time.time()
            with open(parcial, "wb") as f:
                ftp.retrbinary(f"RETR {target_file}", f.write)

            actual_size = os.path.getsize(parcial)
            if actual_size != size_ftp:
                raise ValueError(f"Tamanho divergente: esperado {size_ftp}, recebido {actual_size}")

            os.replace(parcial, output)
            print(f"[  OK  ] Baixado: {target_file} | {actual_size / 1e6:.1f} MB | {time.time() - start:.1f}s")
            return {"tamanho": actual_size, "mdtm_ftp": data_ftp}

        except Exception as e:
            print(f"[RETRY {attempt}/{max_retries}] Erro em {target_file}: {e}")
            if os.path.exists(parcial):
                os.remove(parcial)
            if attempt < max_retries:
                time.sleep(retry_delay)
        finally:
            try:
                ftp.quit()
            except Exception:
                pass

    print(f"[ ERROR] {target_file} falhou após {max_retries} tentativas.")
    return None

# COMMAND ----------

# MAGIC %md
# MAGIC ### Função de download em lote

# COMMAND ----------

def batch_download_sequential(downloads: list, ftp_path: str) -> list:
    """
    Executa downloads sequencialmente com intervalo de 5s entre arquivos,
    para o servidor do DataSUS não interpretar reconexões sucessivas como abuso.
    `downloads`: lista de (nome_no_ftp, caminho_local). Retorna a lista de arquivos que falharam.
    """
    falhas = []
    for i, (target_file, output) in enumerate(downloads, 1):
        print(f"\n[  {i}/{len(downloads)} ] {target_file}")
        info = download_datasus_file(target_file, output, ftp_path)
        if info is None:
            falhas.append(target_file)
        else:
            gravar_json(dir_manifestos + os.path.splitext(target_file)[0].upper() + ".json",
                        {"arquivo": os.path.basename(output), "pasta_ftp": ftp_path, **info,
                         "baixado_em": datetime.now(timezone.utc).isoformat()})
        if i < len(downloads):
            time.sleep(5)
    return falhas

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extrai arquivos disponíveis no FTP do DataSUS

# COMMAND ----------

ftp_path = FTP_BASE + pasta
conn = connect_datasus_ftp(ftp_path)
if conn is None:
    raise ConnectionError(f"Sem conexão com {FTP_HOST}{ftp_path}")

disponiveis = listar_arquivos_dn(conn)
available_ufs = sorted({uf for uf, _ in disponiveis})
available_years = sorted({ano for _, ano in disponiveis})
anos_sel = interpretar_anos(anos_txt, available_years)
info_ftp = {nome: (conn.size(nome), mdtm(conn, nome))
            for (uf, ano), nome in disponiveis.items() if uf in ufs_sel and ano in anos_sel}
conn.quit()

print(f"Pasta: {ftp_path}")
print(f"{len(disponiveis)} arquivos DN | UFs: {available_ufs}")
print(f"Anos: {available_years[0]}–{available_years[-1]} ({len(available_years)} anos)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Seleção e comparação com a landing

# COMMAND ----------

# arquivos locais indexados pelo nome em maiúsculas (o mesmo arquivo pode estar como .dbc ou .DBC)
locais = {f.name.upper(): f for f in dbutils.fs.ls(dst) if not f.name.endswith("/")}

downloads, situacao = [], []
for uf in ufs_sel:
    for ano in anos_sel:
        nome = disponiveis.get((uf, ano))
        if nome is None:
            situacao.append((f"DN{uf}{ano}", "não existe no FTP", None, None))
            continue
        tam_ftp, data_ftp = info_ftp[nome]
        local = locais.get(nome.upper())
        caminho_local = dst + (local.name if local else nome)  # reaproveita o nome local existente
        manifesto_path = dir_manifestos + os.path.splitext(nome)[0].upper() + ".json"
        manifesto = ler_json(manifesto_path)

        if local is None:
            estado = "ausente na landing"
        elif local.size != tam_ftp:
            estado = "tamanho diferente do FTP"
        elif manifesto is None:
            estado = "ok (mesmo tamanho; manifesto criado agora)"
            gravar_json(manifesto_path, {"arquivo": local.name, "pasta_ftp": ftp_path, "tamanho": tam_ftp,
                                         "mdtm_ftp": data_ftp, "adotado_sem_download": True,
                                         "registrado_em": datetime.now(timezone.utc).isoformat()})
        elif data_ftp and manifesto.get("mdtm_ftp") and manifesto["mdtm_ftp"] != data_ftp:
            estado = f"versão nova no FTP ({manifesto['mdtm_ftp']} → {data_ftp})"
        else:
            estado = "ok (mesmo tamanho e mesma versão)"

        if not estado.startswith("ok"):
            downloads.append((nome, caminho_local))
        situacao.append((nome, estado, tam_ftp, local.size if local else None))

display(spark.createDataFrame(situacao, "arquivo string, situacao string, tamanho_ftp long, tamanho_landing long"))
print(f"{len(downloads)} arquivo(s) a baixar: {[n for n, _ in downloads]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download em lote
# MAGIC
# MAGIC Só roda com `executar = sim`.

# COMMAND ----------

if executar and downloads:
    falhas = batch_download_sequential(downloads, ftp_path)
    if falhas:
        raise RuntimeError(f"Downloads com falha: {falhas}")
    dbutils.notebook.exit(json.dumps({"baixados": [n for n, _ in downloads]}))
elif executar:
    dbutils.notebook.exit(json.dumps({"baixados": []}))
else:
    print("Execução desligada (widget 'executar' = nao): apenas listagem.")

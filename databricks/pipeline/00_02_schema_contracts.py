# Databricks notebook source
# MAGIC %md
# MAGIC # Contratos de schema e regras comuns
# MAGIC
# MAGIC Incluído pelos demais notebooks com `%run ./00_02_schema_contracts`. Centraliza:
# MAGIC - tipos de cada campo do SINASC (inteiro, data `DDMMYYYY`, texto);
# MAGIC - o mapa de renomeação para a Silver;
# MAGIC - a regra de nome de arquivo aceita pelo pipeline (`DN` + UF + ano, ex.: `DNSP2018`);
# MAGIC - funções de manifesto (JSON) usadas para decidir se uma etapa precisa ser refeita.

# COMMAND ----------

import os
import re
import json

# campos que recebem cast para int
INT_COLS = [
    "ORIGEM",
    "LOCNASC",    "IDADEMAE",   "ESTCIVMAE",  "ESCMAE",
    "QTDFILVIVO", "QTDFILMORT", "GESTACAO",   "GRAVIDEZ",
    "PARTO",      "CONSULTAS",  "SEXO",       "APGAR1",
    "APGAR5",     "RACACOR",    "PESO",       "IDANOMAL",
    "DIFDATA",    "ESCMAE2010", "SERIESCMAE", "RACACORMAE",
    "QTDGESTANT", "QTDPARTNOR", "QTDPARTCES", "IDADEPAI",
    "SEMAGESTAC", "TPMETESTIM", "CONSPRENAT", "MESPRENAT",
    "TPAPRESENT", "STTRABPART", "STCESPARTO", "TPNASCASSI",
    "TPFUNCRESP", "TPDOCRESP",  "ESCMAEAGR1", "STDNEPIDEM",
    "STDNNOVA",   "TPROBSON",   "PARIDADE",   "KOTELCHUCK",
    "CONTADOR",
]

# campos de data tipo DDMMYYYY
DATE_COLS_DDMMYYYY = [
    "DTNASC",    "DTCADASTRO", "DTRECEBIM",
    "DTNASCMAE", "DTULTMENST", "DTDECLARAC", "DTRECORIGA",
]

# campos string com trim (códigos com zeros à esquerda)
STRING_COLS = [
    "CODMUNNASC", "CODMUNRES",  "CODESTAB",
    "CODUFNATU",  "CODMUNNATU", "CODPAISRES",
    "CODOCUPMAE", "NATURALMAE", "CODANOMAL",
    "NUMEROLOTE", "VERSAOSIST", "HORANASC"
]

# SEXO usa letras em alguns anos (ex.: 2014: M/F/I) e números nos demais (1/2/0).
# As letras são convertidas para o código numérico antes do cast.
SEXO_LETRAS = {"M": "1", "F": "2", "I": "0"}

# COMMAND ----------

# mapa de renomeação
RENAME_MAP = {

    # identificação / origem
    "ORIGEM":     "id_origem_registro",
    "CONTADOR":   "id_sequencial_registro",
    "NUMEROLOTE": "num_lote_envio",
    "VERSAOSIST": "versao_sistema",
    "STDNEPIDEM": "status_epidemiologico",
    "STDNNOVA":   "flag_dn_nova",               # "Status de DN Nova" (1 = sim; 0 = não) — BRASIL, 2019

    # estabelecimento / responsáveis
    "CODESTAB":   "cod_cnes_estabelecimento",
    "LOCNASC":    "id_local_nascimento",
    "TPNASCASSI": "id_assistente_parto",
    "TPFUNCRESP": "id_funcao_responsavel",
    "TPDOCRESP":  "id_documento_responsavel",

    # localização geográfica
    "CODMUNNASC": "cod_municipio_nascimento",
    "CODMUNRES":  "cod_municipio_residencia",
    "CODMUNNATU": "cod_municipio_naturalidade_mae",
    "CODUFNATU":  "cod_uf_naturalidade_mae",
    "CODPAISRES": "cod_pais_residencia",
    "NATURALMAE": "cod_naturalidade_mae",

    # mãe e pai
    "IDADEMAE":   "idade_mae",
    "ESTCIVMAE":  "id_estado_civil_mae",
    "ESCMAE":     "id_escolaridade_mae",
    "ESCMAE2010": "id_escolaridade_mae_2010",
    "SERIESCMAE": "serie_escolar_mae",
    "ESCMAEAGR1": "id_escolaridade_mae_agregada",
    "CODOCUPMAE": "cod_cbo_ocupacao_mae",
    "RACACORMAE": "id_raca_cor_mae",
    "IDADEPAI":   "idade_pai",
    "DTNASCMAE":  "data_nascimento_mae",
    "DTULTMENST": "data_ultima_menstruacao",

    # gestação
    "QTDFILVIVO": "qtd_filhos_vivos",
    "QTDFILMORT": "qtd_perdas_fetais_abortos",  # "Número de perdas fetais e abortos" — BRASIL, 2019
    "QTDGESTANT": "qtd_gestacoes_anteriores",
    "QTDPARTNOR": "qtd_partos_normais",
    "QTDPARTCES": "qtd_partos_cesareos",
    "GESTACAO":   "id_semanas_gestacao_faixa",
    "GRAVIDEZ":   "id_tipo_gravidez",
    "SEMAGESTAC": "qtd_semanas_gestacao",
    "TPMETESTIM": "id_metodo_estimativa_gestacao",
    "CONSPRENAT": "qtd_consultas_prenatal",
    "MESPRENAT":  "mes_inicio_prenatal",
    "KOTELCHUCK": "id_indice_kotelchuck",

    # parto
    "PARTO":      "id_tipo_parto",
    "CONSULTAS":  "id_consultas_prenatal_faixa",
    "TPAPRESENT": "id_apresentacao_fetal",
    "STTRABPART": "flag_trabalho_parto_induzido",
    "STCESPARTO": "flag_cesarea_trabalho_parto",
    "PARIDADE":   "id_paridade",
    "TPROBSON":   "id_grupo_robson",

    # recém-nascido
    "DTNASC":     "data_nascimento",
    "HORANASC":   "hora_nascimento",
    "SEXO":       "id_sexo",
    "APGAR1":     "apgar_1_minuto",
    "APGAR5":     "apgar_5_minutos",
    "RACACOR":    "id_raca_cor",
    "PESO":       "peso_gramas",
    "IDANOMAL":   "flag_anomalia_congenita",
    "CODANOMAL":  "cod_cid10_anomalia",

    # administrativo / controle
    "DTCADASTRO": "data_cadastro",
    "DTRECEBIM":  "data_recebimento",
    "DTRECORIGA": "data_recebimento_original",
    "DTDECLARAC": "data_declaracao",
    "DIFDATA":    "qtd_dias_diferenca_datas",
}

# COMMAND ----------

# Regra única de nome de arquivo aceita em todas as etapas: DN + UF (2 letras) + ano (4 dígitos),
# com qualquer extensão e qualquer caixa (o FTP mistura .dbc e .DBC).
# Arquivos fora do padrão (ex.: DBSE2022.dbc, cópias com extensão trocada) são ignorados e listados.
_RE_DN = re.compile(r"^DN([A-Z]{2})(\d{4})$", flags=re.IGNORECASE)


def identificar_arquivo(nome: str):
    """'DNSP2018.DBC' | 'DNSP2018.csv' | 'DNSP2018/' → ('SP', 2018); fora do padrão → None."""
    base = os.path.splitext(nome.rstrip("/"))[0]
    m = _RE_DN.match(base)
    return (m.group(1).upper(), int(m.group(2))) if m else None


def ler_json(caminho: str):
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def gravar_json(caminho: str, dados: dict) -> None:
    """Gravação atômica: escreve em .tmp e renomeia."""
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    tmp = caminho + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, caminho)


def assinatura_arquivo(caminho: str) -> dict:
    """Tamanho e data de modificação: identificam a versão de um arquivo sem relê-lo."""
    st = os.stat(caminho)
    return {"tamanho": st.st_size, "mtime": int(st.st_mtime)}

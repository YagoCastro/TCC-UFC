# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                 TCC PLD — FASE 01: ZOO DE MODELOS + MÉTRICAS                  ║
# ║      Treinamento Comparativo, Coleta Automática e Tabela Real x Previsto     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  IDENTIFICAÇÃO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Autor       : Yago
#  Instituição : Universidade Federal do Ceará
#  Curso       : Engenharia Elétrica
#  Data        : Julho / 2026
#  Versão      : 1.0 — Arquivo Único (Zoo de Modelos + Coleta de Métricas unificados)
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DESCRIÇÃO / FUNÇÃO DO SCRIPT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Este script implementa a Fase 01 da modelagem preditiva do PLD (submercado
#  Nordeste) do TCC, unificando em um único arquivo Colab-friendly:
#
#    (A) ZOO DE MODELOS — treina e avalia 10 algoritmos de regressão (do
#        linear ao gradient boosting), em dois cenários de composição do
#        treino:
#          • CASO 1 : Cenário A completo (todos os anos históricos)
#          • CASO 2 : Cenário A sem os anos de 2022 e 2023
#        Em ambos os casos, Janeiro e Fevereiro/2026 são incorporados ao
#        TREINO, e o TESTE é sempre Março e Abril/2026 — período crítico de
#        transição seca→chuvosa no Nordeste, usado como teste cego.
#
#    (B) COLETA AUTOMÁTICA DE MÉTRICAS — ao final da execução, varre as
#        variáveis em memória (y_true / y_pred_teste do modelo campeão) e
#        grava/atualiza a linha correspondente à "Parte 1" no arquivo
#        metricas_recalculadas.csv, com deduplicação por (Parte, Tipo).
#
#    (C) TABELA REAL x PREVISTO — monta e imprime uma tabela (amostrada) com
#        data, hora, valor real, valor previsto, erro e erro percentual para
#        o período de teste, tanto para o melhor modelo de cada caso quanto
#        para o campeão geral, exportando este último em CSV.
#
#  INTEGRAÇÃO ENTRE AS PARTES:
#    O coletor de métricas (B) foi originalmente escrito para ler variáveis
#    soltas (y_true, y_pred_teste, etc.) diretamente do namespace global. No
#    Zoo de Modelos (A), essas variáveis só existem dentro de run_caso(), por
#    modelo. Para que a união funcione de fato, ao final da Fase 01 o script:
#      1. Guarda as predições de TODOS os modelos, em AMBOS os casos.
#      2. Elege o "campeão geral" (menor MAE entre Caso 1 e Caso 2).
#      3. Expõe y_true / y_pred_teste desse campeão no escopo global.
#      4. Executa a célula de coleta de métricas, que grava a linha "Parte 1"
#         em metricas_recalculadas.csv usando exatamente esse campeão.
#      5. Salva um CSV com REAL x PREVISTO (data/hora incluídos) do campeão.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENTRADAS (INPUTS)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  O script não requer entradas manuais do usuário. Os parâmetros de controle
#  estão definidos nas constantes globais (logo após o tema do Rich):
#
#    BASE_DIR / INPUT_DIR : caminho dos dados normalizados
#                           (.../10_dados_normalizados/<cenário>/<estratégia>/)
#    CENARIO_DEFAULT       : "cenario_A_todos_anos"
#    STRATEGY_DEFAULT      : "HYBRID_AGGRESSIVE"
#    TESTE_ANO / TESTE_MESES : 2026 / [3, 4]  → período de teste fixo (Mar/Abr)
#    TREINO_EXTRA_MESES    : [1, 2]  → Jan/Fev/2026 movidos para o treino
#    ANOS_EXCLUIR_CASO_2   : {2022, 2023}  → removidos do treino no Caso 2
#
#  Arquivos de entrada esperados:
#    • INPUT_DIR / CENARIO_DEFAULT / STRATEGY_DEFAULT / TREINO_NORM.parquet
#    • INPUT_DIR / CENARIO_DEFAULT / STRATEGY_DEFAULT / TESTE_NORM.parquet
#      → dados já normalizados, contendo a coluna-alvo (PLD Nordeste), as
#        colunas-chave temporais (KEY_ANO, KEY_MES, KEY_DIA, KEY_HORA) e,
#        opcionalmente, a coluna REGIME_PRECO (excluída por vazamento).
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SAÍDAS (OUTPUTS)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. CSV de métricas → PASTA_SAIDA_METRICAS / metricas_recalculadas.csv
#     Uma linha por (Parte, Tipo), com métricas de regressão (RMSE, R², MAE,
#     MAPE) e, se aplicável, de classificação (ROC-AUC, PR-AUC, F1, etc.).
#     Deduplicado automaticamente a cada execução (Parte=1, Tipo="regressao").
#
#  2. CSV real x previsto → PASTA_SAIDA_METRICAS /
#     real_vs_previsto_parte{PARTE_TCC}_{modelo_campeão}.csv
#     Contém: data, hora, modelo, real, previsto, erro, erro_% — para todo o
#     período de teste (Mar/Abr 2026) do modelo campeão geral.
#
#  3. Saída no terminal (Rich):
#     • Cabeçalho de fase (Painel) por caso executado
#     • Tabela de métricas pontuais (MAE, RMSE, MAPE, R², bias, Spike MAE)
#     • Tabela de métricas probabilísticas aproximadas (Pinball, CRPS, Cov 80%)
#     • Ranking final (top 3) por caso, com recomendação para a Fase 02
#     • Tabela real x previsto (amostrada) do melhor modelo de cada caso
#     • Comparação lado a lado entre Caso 1 e Caso 2 (Δ MAE)
#     • Anúncio do campeão geral + tabela real x previsto completa
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MODELOS TESTADOS (ZOO — 10 algoritmos)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  N°  Modelo             Família                 Necessita Scaling
#  01  LinearRegression   Linear                  Sim
#  02  Ridge              Linear regularizado L2  Sim
#  03  Lasso              Linear regularizado L1  Sim
#  04  ElasticNet         Linear regularizado L1+L2  Sim
#  05  KNN                Baseado em distância    Sim
#  06  DecisionTree       Árvore única            Não
#  07  RandomForest       Ensemble de árvores     Não
#  08  XGBoost            Gradient Boosting       Não  (se disponível)
#  09  LightGBM           Gradient Boosting       Não  (se disponível)
#  10  CatBoost           Gradient Boosting       Não  (se disponível)
#
#  Os três modelos de boosting (08–10) são opcionais: o script tenta instalar
#  e importar cada biblioteca; se indisponível, o modelo é simplesmente
#  omitido do zoo, sem interromper a execução.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ETAPAS DO PROCESSO (resumo)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  N°  Etapa                       Estratégia                    Descrição
#  01  Carregamento                read_parquet (1x por sessão)  Lê TREINO_NORM e TESTE_NORM
#  02  Detecção automática         regex + tokens                Localiza TARGET, KEY_* e REGIME_PRECO
#  03  Split treino/teste          Regras de mês/ano fixas        Jan/Fev→treino, Mar/Abr→teste, exclui anos (Caso 2)
#  04  Engenharia temporal         TemporalFE (22 features)       Cíclicas (seno/cosseno), calendário, feriados
#  05  Padronização                StandardScaler                 Aplicado somente aos modelos que exigem escala
#  06  Treinamento                 Loop pelo zoo de modelos        Treina e cronometra cada um dos 10 modelos
#  07  Avaliação pontual            MAE/RMSE/MAPE/R²/bias/Spike     Métricas de regressão + foco em picos de preço
#  08  Avaliação probabilística    Bootstrap de resíduos            Pinball loss, CRPS aproximado, cobertura 80%
#  09  Ranking e comparação        Ordenação por MAE                Top 3 por caso + comparação Caso 1 vs Caso 2
#  10  Eleição do campeão geral    Menor MAE entre os casos         Expõe y_true/y_pred no escopo global
#  11  Real x Previsto             DataFrame ordenado por data/hora Tabela amostrada + CSV completo do campeão
#  12  Coleta de métricas          Varredura de arrays no globals() Grava/atualiza metricas_recalculadas.csv
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONSIDERAÇÕES INICIAIS E OBSERVAÇÕES TÉCNICAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#  1. AMBIENTE DE EXECUÇÃO
#     O script foi desenvolvido e testado no Google Colab (Python 3.10+).
#     Instala automaticamente (via pip, silenciosamente) as bibliotecas
#     xgboost, lightgbm, catboost, holidays e rich caso ainda não estejam
#     presentes no ambiente.
#
#  2. PERÍODO DE TESTE FIXO NO CÓDIGO
#     TESTE_ANO = 2026, TESTE_MESES = [3, 4]  →  Março e Abril/2026 são
#     sempre o conjunto de teste cego. Janeiro e Fevereiro/2026 (que vêm no
#     dataset de teste bruto) são redirecionados para o TREINO antes do
#     corte, via TREINO_EXTRA_MESES.
#
#  3. DOIS CENÁRIOS DE TREINO (CASO 1 x CASO 2)
#     O Caso 2 remove 2022 e 2023 do histórico de treino (ANOS_EXCLUIR_CASO_2)
#     para avaliar se anos atípicos prejudicam a generalização do modelo no
#     período de teste. Os dois casos compartilham o mesmo zoo de modelos e
#     o mesmo período de teste, permitindo comparação direta por MAE.
#
#  4. DETECÇÃO AUTOMÁTICA DE COLUNAS
#     detectar_target(), detectar_key_cols() e detectar_regime_col() tentam
#     localizar a coluna-alvo, as colunas-chave temporais e a coluna de
#     regime de preço por nome canônico e, em caso de falha, por tokens
#     (ex.: "nordeste" + "target"). REGIME_PRECO é sempre excluída das
#     features por representar vazamento de informação (data leakage).
#
#  5. ENGENHARIA TEMPORAL (TemporalFE)
#     Gera 22 features cíclicas e de calendário: seno/cosseno de hora, dia
#     da semana, dia do mês, dia do ano e mês; flags de fim de semana,
#     horário de pico, madrugada, período seco/chuvoso do Nordeste, ano
#     crítico (2014, 2015, 2021, 2024) e feriados nacionais (via biblioteca
#     `holidays`, se disponível).
#
#  6. MÉTRICAS COM FOCO EM PICOS
#     Além das métricas padrão de regressão, calcula MAE e bias específicos
#     para observações com PLD acima de SPIKE_ALTO_LIMIAR (R$ 500), já que
#     picos de preço são o principal desafio de previsão no Nordeste.
#
#  7. MÉTRICAS PROBABILÍSTICAS APROXIMADAS
#     Como o zoo é composto por modelos pontuais (não nativamente
#     probabilísticos), a incerteza é aproximada via bootstrap gaussiano dos
#     resíduos de treino, gerando quantis P10/P50/P90 e métricas de Pinball
#     Loss, CRPS aproximado e cobertura do intervalo de 80%.
#
#  8. TABELA REAL x PREVISTO
#     Para tabelas grandes (> N_LINHAS_AMOSTRA_REAL_PREVISTO, padrão 20), o
#     console exibe uma amostra aleatória ordenada por data/hora, mas o
#     resumo estatístico do erro (bias, erro absoluto médio, erro máximo,
#     % de horas com erro > 10%) é sempre calculado sobre o período de teste
#     completo. O CSV exportado do campeão geral contém todas as horas.
#
#  9. COLETA AUTOMÁTICA DE MÉTRICAS (PARTE B)
#     Varre o namespace global por arrays numéricos com ao menos 10
#     elementos, tentando casar automaticamente uma variável de "verdade"
#     (y_true, y_test, y_real...) com uma de "previsão" (y_pred_teste,
#     y_pred, preds...). Detecta ainda um possível caso de classificação
#     binária (score de probabilidade + rótulo 0/1), mas isso normalmente
#     não se aplica a esta fase, que é puramente de regressão.
#
#  10. LIMITES CONHECIDOS
#      - Os três modelos de boosting (XGBoost, LightGBM, CatBoost) usam
#        parâmetros padrão razoáveis, sem tuning de hiperparâmetros — a
#        otimização fica para uma fase posterior do TCC.
#      - A coleta de métricas (Parte B) depende de nomes de variáveis
#        convencionais no escopo global; renomear as variáveis-ponte
#        (y_true, y_pred_teste, y_pred) pode quebrar a detecção automática.
#
#  11. REPRODUTIBILIDADE
#      RANDOM_STATE = 42 é usado em todos os modelos, no bootstrap de
#      resíduos e na amostragem da tabela real x previsto, garantindo
#      resultados determinísticos entre execuções com os mesmos dados.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DEPENDÊNCIAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Biblioteca        Versão mínima  Finalidade
#  numpy             1.23           Operações numéricas e amostragem (bootstrap)
#  pandas            1.5            Manipulação de DataFrames e Parquet
#  scikit-learn      1.2            Modelos lineares, KNN, árvores, métricas
#  xgboost           —              Modelo de gradient boosting (opcional)
#  lightgbm          —              Modelo de gradient boosting (opcional)
#  catboost          —              Modelo de gradient boosting (opcional)
#  holidays          —              Feriados nacionais para engenharia temporal
#  rich              13.0           Tabelas, painéis e barras de progresso
#
#  Instalação automática (executada no início do script, se necessário):
#      pip install -q xgboost lightgbm catboost holidays rich
#  (numpy, pandas e scikit-learn já estão disponíveis no Colab por padrão)
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ESTRUTURA DE ARQUIVOS ENVOLVIDA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TCC_PLD_Project/
#  └── 09-ESCRITA_TCC/
#      └── PARTES_TCC/
#          └── codigos/
#              ├── dados/
#              │   └── 10_dados_normalizados/
#              │       └── cenario_A_todos_anos/
#              │           └── HYBRID_AGGRESSIVE/
#              │               ├── TREINO_NORM.parquet   (entrada)
#              │               └── TESTE_NORM.parquet    (entrada)
#              └── codigos_modelagem/
#                  ├── metricas_recalculadas.csv                          (saída)
#                  └── real_vs_previsto_parte1_<modelo_campeão>.csv       (saída)
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  COMO EXECUTAR (Google Colab)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Célula 1 — Montar o Drive:
#      from google.colab import drive
#      drive.mount('/content/drive')
#
#  Célula 2 — Executar o script:
#      exec(open('fase01_zoo_modelos_coleta_metricas.py').read())
#   ou simplesmente executar este arquivo como módulo principal.
#   As dependências (xgboost, lightgbm, catboost, holidays, rich) são
#   instaladas automaticamente na primeira execução, se necessário.
#
# ══════════════════════════════════════════════════════════════════════════════

import subprocess, sys
def _ensure(pkg, imp=None):
    try:
        __import__(imp or pkg)
    except ImportError:
        print(f"⏬ Instalando {pkg}...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

_ensure("xgboost"); _ensure("lightgbm"); _ensure("catboost")
_ensure("holidays"); _ensure("rich")

import warnings
warnings.filterwarnings("ignore")

import re
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error,
)

try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    import lightgbm as lgb
    LGB_OK = True
except ImportError:
    LGB_OK = False

try:
    import catboost as cb
    CAT_OK = True
except ImportError:
    CAT_OK = False

try:
    import holidays
    HOLIDAYS_OK = True
except ImportError:
    HOLIDAYS_OK = False

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TimeElapsedColumn, MofNCompleteColumn,
)
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# ==============================================================================
THEME = Theme({
    "info"       : "bold cyan",
    "warning"    : "bold yellow",
    "error"      : "bold red",
    "success"    : "bold green",
    "header_v2"  : "bold white on purple4",
    "highlight"  : "bold magenta",
    "muted"      : "dim white",
    "metric_best": "bold green",
    "metric_bad" : "bold red",
    "acerto"     : "bold green",
    "data"       : "bold yellow",
})
console = Console(theme=THEME)

# ==============================================================================
BASE_DIR  = Path("/content/drive/MyDrive/TCC_PLD_Project/09-ESCRITA_TCC/PARTES_TCC/codigos/dados")
INPUT_DIR = BASE_DIR / "10_dados_normalizados"

CENARIO_DEFAULT   = "cenario_A_todos_anos"
STRATEGY_DEFAULT  = "HYBRID_AGGRESSIVE"
TARGET_CANONICAL  = "01_CCEE_NORDESTE_TARGET_PLD_HORA_NORDESTE"
_TARGET_TOKENS    = ["nordeste", "target"]
SPIKE_ALTO_LIMIAR = 500.0
TESTE_MESES       = [3, 4]          # Março e Abril de 2026  <<< PERÍODO DE TESTE
TESTE_ANO         = 2026
TREINO_EXTRA_MESES = [1, 2]         # Janeiro e Fevereiro vão para o treino
RANDOM_STATE      = 42

# Caso 2: anos que devem ser REMOVIDOS do treino (mesmos dados do Cenário A)
ANOS_EXCLUIR_CASO_2 = {2022, 2023}

# Configuração da coleta de métricas (parte B)
PARTE_TCC = 1
PASTA_SAIDA_METRICAS = r"/content/drive/MyDrive/TCC_PLD_Project/09-ESCRITA_TCC/PARTES_TCC/codigos/codigos_modelagem"

# NOVO: quantas linhas mostrar na amostra da tabela real x previsto no console
N_LINHAS_AMOSTRA_REAL_PREVISTO = 20

# ==============================================================================
def safe_round(v, decimals=4):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    return round(float(v), decimals)

def detectar_target(df):
    if TARGET_CANONICAL in df.columns:
        return TARGET_CANONICAL
    cols_lower = {c: c.lower() for c in df.columns}
    for tokens in [_TARGET_TOKENS, ["target", "nordeste"], ["pld", "nordeste"]]:
        cands = [c for c, cl in cols_lower.items() if all(t in cl for t in tokens)]
        if cands:
            return cands[0]
    raise KeyError("TARGET não encontrado")

def detectar_key_cols(df):
    mapping = {}
    for key in ["KEY_ANO", "KEY_MES", "KEY_DIA", "KEY_HORA"]:
        if key in df.columns:
            mapping[key] = key
        else:
            found = [c for c in df.columns if key.lower() in c.lower()]
            mapping[key] = found[0] if found else None
    return mapping

def detectar_regime_col(df):
    cands = [c for c in df.columns if "REGIME_PRECO" in c.upper()]
    return cands[0] if cands else None

# ==============================================================================
def load_parquet(path, label=""):
    console.print(f"   [info]📂 {label}: {path.name}[/]")
    df = pd.read_parquet(path)
    df.columns = [re.sub(r'[":{},\[\]]', "_", str(c)) for c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mb = df.memory_usage(deep=True).sum() / 1024**2
    console.print(f"   [info]   Shape: {df.shape} | {mb:.1f} MB[/]")
    return df

def split_treino_teste(df_treino_raw, df_teste_raw, key_cols,
                       anos_excluir_treino=None):
    """
    Monta treino/teste.

    Em todos os casos:
      - Jan/Fev 2026 (do dataset de teste) → vão para o TREINO.
      - Mar/Abr 2026 → TESTE.  <<< PERÍODO DE TESTE FINAL

    Caso 2:
      - anos_excluir_treino={2022, 2023} remove esses anos do TREINO final.
    """
    k_ano, k_mes = key_cols.get("KEY_ANO"), key_cols.get("KEY_MES")

    # Janeiro e Fevereiro/2026 do dataset de teste → TREINO
    mask_jan_fev = (
        (df_teste_raw[k_ano] == TESTE_ANO) &
        (df_teste_raw[k_mes].isin(TREINO_EXTRA_MESES))
    )
    df_jan_fev = df_teste_raw[mask_jan_fev]
    console.print(f"   [info]📅 Janeiro+Fevereiro/2026: {len(df_jan_fev):,} → TREINO[/]")

    # Março e Abril/2026 → TESTE
    mask_teste = (
        (df_teste_raw[k_ano] == TESTE_ANO) &
        (df_teste_raw[k_mes].isin(TESTE_MESES))
    )
    df_teste_final  = df_teste_raw[mask_teste].reset_index(drop=True)
    df_treino_final = pd.concat([df_treino_raw, df_jan_fev], ignore_index=True)

    # Caso 2: remover anos específicos do TREINO (após o concat)
    if anos_excluir_treino:
        anos_excluir_treino = set(anos_excluir_treino)
        n_antes = len(df_treino_final)
        df_treino_final = df_treino_final[
            ~df_treino_final[k_ano].astype(int).isin(anos_excluir_treino)
        ].reset_index(drop=True)
        n_removidos = n_antes - len(df_treino_final)
        console.print(
            f"   [warning]🗑  Removendo anos {sorted(anos_excluir_treino)} do TREINO: "
            f"-{n_removidos:,} registros[/]"
        )

    df_treino_final = df_treino_final.reset_index(drop=True)
    console.print(f"   [success]✔  TREINO final : {len(df_treino_final):,}[/]")
    console.print(f"   [success]✔  TESTE  final : {len(df_teste_final):,} (Mar/Abr 2026)[/]")
    return df_treino_final, df_teste_final

# ==============================================================================
class TemporalFE:
    ANOS_CRITICOS    = {2014, 2015, 2021, 2024}
    HORA_PICO        = (18, 21)
    HORA_MADRUGADA   = (0, 5)
    MESES_CHUVOSOS_NE = {12, 1, 2, 3}
    MESES_SECOS_NE    = {6, 7, 8, 9, 10, 11}

    def __init__(self, use_holidays=True):
        self.use_holidays = use_holidays and HOLIDAYS_OK
        self.br_holidays = (
            holidays.Brazil(years=range(2018, 2031)) if self.use_holidays else None
        )

    def _safe_date(self, a, m, d):
        try:
            return date(int(a), int(m), int(d))
        except (ValueError, TypeError):
            return None

    def transform(self, df, key_cols):
        df_out = df.copy()
        k_ano  = key_cols["KEY_ANO"]
        k_mes  = key_cols["KEY_MES"]
        k_dia  = key_cols["KEY_DIA"]
        k_hora = key_cols["KEY_HORA"]

        ano  = df_out[k_ano].astype(int).values
        mes  = df_out[k_mes].astype(int).values
        dia  = df_out[k_dia].astype(int).values
        hora = df_out[k_hora].astype(int).values

        datas_safe = [self._safe_date(a, m, d) or date(1900, 1, 1)
                      for a, m, d in zip(ano, mes, dia)]
        dia_semana = np.array([d.weekday() for d in datas_safe], dtype=int)
        dia_do_ano = np.array([d.timetuple().tm_yday for d in datas_safe], dtype=int)

        df_out["CYC_hora_sin"]    = np.sin(2 * np.pi * hora / 24.0)
        df_out["CYC_hora_cos"]    = np.cos(2 * np.pi * hora / 24.0)
        df_out["CYC_dia_sem_sin"] = np.sin(2 * np.pi * dia_semana / 7.0)
        df_out["CYC_dia_sem_cos"] = np.cos(2 * np.pi * dia_semana / 7.0)
        df_out["CYC_dia_mes_sin"] = np.sin(2 * np.pi * (dia - 1) / 31.0)
        df_out["CYC_dia_mes_cos"] = np.cos(2 * np.pi * (dia - 1) / 31.0)
        df_out["CYC_mes_sin"]     = np.sin(2 * np.pi * (mes - 1) / 12.0)
        df_out["CYC_mes_cos"]     = np.cos(2 * np.pi * (mes - 1) / 12.0)
        df_out["CYC_dia_ano_sin"] = np.sin(2 * np.pi * (dia_do_ano - 1) / 365.0)
        df_out["CYC_dia_ano_cos"] = np.cos(2 * np.pi * (dia_do_ano - 1) / 365.0)

        df_out["CAL_is_fim_semana"]   = (dia_semana >= 5).astype(int)
        df_out["CAL_is_horario_pico"] = (
            (hora >= self.HORA_PICO[0]) & (hora <= self.HORA_PICO[1])
        ).astype(int)
        df_out["CAL_is_madrugada"]    = (
            (hora >= self.HORA_MADRUGADA[0]) & (hora <= self.HORA_MADRUGADA[1])
        ).astype(int)
        df_out["CAL_is_periodo_seco_NE"]    = np.isin(mes, list(self.MESES_SECOS_NE)).astype(int)
        df_out["CAL_is_periodo_chuvoso_NE"] = np.isin(mes, list(self.MESES_CHUVOSOS_NE)).astype(int)
        df_out["CAL_is_ano_critico"]        = np.isin(ano, list(self.ANOS_CRITICOS)).astype(int)

        if self.use_holidays:
            df_out["CAL_is_feriado"] = np.array(
                [1 if d in self.br_holidays else 0 for d in datas_safe], dtype=int
            )
        else:
            df_out["CAL_is_feriado"] = 0

        df_out["CAL_is_dia_util"] = (
            (df_out["CAL_is_fim_semana"] == 0) & (df_out["CAL_is_feriado"] == 0)
        ).astype(int)

        df_out["TEMP_trimestre"]       = ((mes - 1) // 3 + 1).astype(int)
        df_out["TEMP_dia_do_ano"]      = dia_do_ano
        df_out["TEMP_ano_normalizado"] = (ano - ano.min()).astype(int)
        df_out["TEMP_progresso_ano"]   = (dia_do_ano - 1) / 365.0
        df_out["TEMP_progresso_mes"]   = (dia - 1) / 31.0
        df_out["TEMP_weekend_pico"]    = (
            df_out["CAL_is_fim_semana"] * df_out["CAL_is_horario_pico"]
        )

        df_out["_DIA_SEMANA_NUM"] = dia_semana
        df_out["_DATA"] = pd.to_datetime({
            "year": ano, "month": mes, "day": dia,
        }, errors="coerce").dt.date

        return df_out

def get_feature_cols(df, target_col, key_cols, regime_col=None):
    exclude = set([target_col])
    exclude.update(key_cols.values())
    if regime_col:
        exclude.add(regime_col)
    exclude.update([c for c in df.columns if c.startswith("_")])
    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

# ==============================================================================
def metricas_pontuais(y_true, y_pred, model_name=""):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2   = r2_score(y_true, y_pred)
    resid = y_true - y_pred
    me    = float(np.mean(resid))
    smape = float(np.mean(
        2 * np.abs(resid) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    ) * 100)
    nrmse = rmse / (np.mean(y_true) + 1e-8) * 100

    spike_mask = y_true > SPIKE_ALTO_LIMIAR
    if spike_mask.sum() > 5:
        spike_mae  = float(mean_absolute_error(y_true[spike_mask], y_pred[spike_mask]))
        spike_bias = float(np.mean(y_true[spike_mask] - y_pred[spike_mask]))
    else:
        spike_mae, spike_bias = np.nan, np.nan

    return {
        "model"     : model_name,
        "MAE"       : safe_round(mae, 4),
        "RMSE"      : safe_round(rmse, 4),
        "MAPE_%"    : safe_round(mape, 4),
        "sMAPE_%"   : safe_round(smape, 4),
        "R2"        : safe_round(r2, 6),
        "ME_bias"   : safe_round(me, 4),
        "NRMSE_%"   : safe_round(nrmse, 4),
        "Spike_MAE" : safe_round(spike_mae, 4),
        "Spike_Bias": safe_round(spike_bias, 4),
    }

def pinball_loss(y_true, y_pred_q, q):
    e = y_true - y_pred_q
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))

def coverage_score(y_true, lower, upper):
    return float(np.mean((y_true >= lower) & (y_true <= upper)))

def metricas_probabilisticas_aproximadas(
    y_true, y_pred, residuos_treino, model_name="", n_bootstrap=200,
    random_state=RANDOM_STATE,
):
    rng = np.random.default_rng(random_state)
    n = len(y_pred)
    sigma = np.std(residuos_treino)
    noise = rng.normal(0, sigma, size=(n_bootstrap, n))
    samples = y_pred[None, :] + noise
    p10 = np.quantile(samples, 0.10, axis=0)
    p50 = np.quantile(samples, 0.50, axis=0)
    p90 = np.quantile(samples, 0.90, axis=0)

    pin_q10 = pinball_loss(y_true, p10, 0.10)
    pin_q50 = pinball_loss(y_true, p50, 0.50)
    pin_q90 = pinball_loss(y_true, p90, 0.90)
    crps_aprox = (pin_q10 + pin_q50 + pin_q90) * 2.0 / 3.0
    cov80 = coverage_score(y_true, p10, p90)
    largura = float(np.mean(p90 - p10))

    return {
        "model"          : model_name,
        "Pinball_Q10"    : safe_round(pin_q10, 4),
        "Pinball_Q50"    : safe_round(pin_q50, 4),
        "Pinball_Q90"    : safe_round(pin_q90, 4),
        "CRPS_aprox"     : safe_round(crps_aprox, 4),
        "Coverage_80%"   : safe_round(cov80, 4),
        "Largura_P10_P90": safe_round(largura, 4),
    }

# ==============================================================================
def print_header_fase(numero, titulo, descricao="", caso_info=""):
    console.print(Panel(
        Text.assemble(
            (f"🧪 FASE {numero:02d} — {titulo}\n", "bold green"),
            (f"   {caso_info}\n", "highlight") if caso_info else ("", "muted"),
            (f"   {descricao}\n", "muted") if descricao else ("", "muted"),
            (f"   Cenário: {CENARIO_DEFAULT} | Strategy: {STRATEGY_DEFAULT}\n", "muted"),
        ),
        title="[header_v2]  TCC PLD — EXPERIMENTO ITERATIVO  [/header_v2]",
        border_style="purple4", padding=(1, 4),
    ))

def print_tabela_pontual(metrics_list, titulo="MÉTRICAS DE REGRESSÃO"):
    if not metrics_list:
        return
    t = Table(title=titulo, box=box.DOUBLE_EDGE, border_style="cyan",
              show_lines=True, min_width=120)
    t.add_column("Modelo",     style="bold",    min_width=22)
    t.add_column("MAE",        justify="right", min_width=10)
    t.add_column("RMSE",       justify="right", min_width=10)
    t.add_column("MAPE %",     justify="right", min_width=10)
    t.add_column("R²",         justify="right", min_width=10)
    t.add_column("ME Bias",    justify="right", min_width=10)
    t.add_column("Spike MAE",  justify="right", min_width=10)
    t.add_column("Tempo (s)",  justify="right", min_width=10)

    df_m = pd.DataFrame(metrics_list)
    best_mae  = df_m["MAE"].min()  if "MAE"  in df_m else None
    best_rmse = df_m["RMSE"].min() if "RMSE" in df_m else None
    best_r2   = df_m["R2"].max()   if "R2"   in df_m else None

    for m in metrics_list:
        mae_v, rmse_v, r2_v = m.get("MAE"), m.get("RMSE"), m.get("R2")
        mae_st  = "metric_best" if mae_v == best_mae else "white"
        rmse_st = "metric_best" if rmse_v == best_rmse else "white"
        r2_st   = "metric_best" if r2_v == best_r2 else "white"
        t.add_row(
            str(m.get("model", "?")),
            f"[{mae_st}]{mae_v:.3f}[/{mae_st}]" if mae_v is not None else "—",
            f"[{rmse_st}]{rmse_v:.3f}[/{rmse_st}]" if rmse_v is not None else "—",
            f"{m.get('MAPE_%', 0):.2f}" if m.get("MAPE_%") is not None else "—",
            f"[{r2_st}]{r2_v:.4f}[/{r2_st}]" if r2_v is not None else "—",
            f"{m.get('ME_bias', 0):.2f}" if m.get("ME_bias") is not None else "—",
            f"{m.get('Spike_MAE', 0):.2f}" if m.get("Spike_MAE") is not None else "—",
            f"{m.get('tempo_s', 0):.1f}" if m.get("tempo_s") is not None else "—",
        )
    console.print(t)

def print_tabela_probabilistica(metrics_list, titulo="MÉTRICAS PROBABILÍSTICAS APROXIMADAS"):
    if not metrics_list:
        return
    t = Table(title=titulo, box=box.ROUNDED, border_style="magenta",
              show_lines=True, min_width=110)
    t.add_column("Modelo",     style="bold",    min_width=22)
    t.add_column("Pin Q10",    justify="right", min_width=10)
    t.add_column("Pin Q50",    justify="right", min_width=10)
    t.add_column("Pin Q90",    justify="right", min_width=10)
    t.add_column("CRPS aprox", justify="right", min_width=11)
    t.add_column("Cov 80%",    justify="right", min_width=10)
    t.add_column("Largura",    justify="right", min_width=10)

    df_m = pd.DataFrame(metrics_list)
    best_crps = df_m["CRPS_aprox"].min() if "CRPS_aprox" in df_m else None

    for m in metrics_list:
        crps_v = m.get("CRPS_aprox")
        crps_st = "metric_best" if crps_v == best_crps else "white"
        cov = m.get("Coverage_80%", 0) or 0
        cov_str = f"{cov*100:.1f}%"
        if abs(cov - 0.80) < 0.05:
            cov_str = f"[acerto]{cov_str}[/acerto]"
        elif abs(cov - 0.80) > 0.20:
            cov_str = f"[metric_bad]{cov_str}[/metric_bad]"
        t.add_row(
            str(m.get("model", "?")),
            f"{m.get('Pinball_Q10', 0):.2f}" if m.get('Pinball_Q10') is not None else "—",
            f"{m.get('Pinball_Q50', 0):.2f}" if m.get('Pinball_Q50') is not None else "—",
            f"{m.get('Pinball_Q90', 0):.2f}" if m.get('Pinball_Q90') is not None else "—",
            f"[{crps_st}]{crps_v:.2f}[/{crps_st}]" if crps_v is not None else "—",
            cov_str,
            f"{m.get('Largura_P10_P90', 0):.1f}" if m.get('Largura_P10_P90') is not None else "—",
        )
    console.print(t)

def print_ranking_final(metrics_list, top_n=3, criterio="MAE"):
    df = pd.DataFrame(metrics_list).sort_values(criterio).reset_index(drop=True)
    t = Table(title=f"🏆 RANKING FINAL (por {criterio}, top {top_n} destacados)",
              box=box.DOUBLE_EDGE, border_style="yellow",
              show_lines=True, min_width=80)
    t.add_column("#",       justify="right", min_width=4)
    t.add_column("Modelo",  style="bold",    min_width=24)
    t.add_column("MAE",     justify="right", min_width=12)
    t.add_column("R²",      justify="right", min_width=10)
    t.add_column("Status",  justify="center", min_width=10)

    for i, row in df.iterrows():
        pos = i + 1
        if pos <= top_n:
            status = f"[acerto]🥇 TOP {pos}[/acerto]"
            style  = "bold green"
        else:
            status = "[muted]—[/muted]"
            style  = "white"
        t.add_row(
            f"{pos}",
            f"[{style}]{row['model']}[/{style}]",
            f"{row['MAE']:.3f}",
            f"{row['R2']:.4f}",
            status,
        )
    console.print(t)
    return df.head(top_n)["model"].tolist()

def print_comparacao_casos(resultados_por_caso, criterio="MAE"):
    """
    Compara o mesmo modelo entre os casos (mesma 'zoo'), por um critério.
    resultados_por_caso: dict {nome_caso: [lista de métricas pontuais]}
    """
    nomes_casos = list(resultados_por_caso.keys())
    if len(nomes_casos) < 2:
        return

    # mapa: caso -> {modelo: valor_do_criterio}
    mapas = {
        caso: {m["model"]: m.get(criterio) for m in res}
        for caso, res in resultados_por_caso.items()
    }
    todos_modelos = sorted(set().union(*[set(mp.keys()) for mp in mapas.values()]))

    t = Table(
        title=f"⚖️  COMPARAÇÃO ENTRE CASOS (por {criterio}; menor é melhor)",
        box=box.DOUBLE_EDGE, border_style="green",
        show_lines=True, min_width=100,
    )
    t.add_column("Modelo", style="bold", min_width=22)
    for caso in nomes_casos:
        t.add_column(caso, justify="right", min_width=14)
    t.add_column("Δ (C2−C1)", justify="right", min_width=12)
    t.add_column("Melhor", justify="center", min_width=12)

    for modelo in todos_modelos:
        vals = [mapas[caso].get(modelo) for caso in nomes_casos]
        validos = [v for v in vals if v is not None]
        menor = min(validos) if validos else None

        cells = []
        for v in vals:
            st = "metric_best" if (v is not None and v == menor) else "white"
            cells.append(f"[{st}]{v:.3f}[/{st}]" if v is not None else "—")

        if len(vals) >= 2 and vals[0] is not None and vals[1] is not None:
            delta = vals[1] - vals[0]
            delta_str = f"{delta:+.3f}"
            melhor = nomes_casos[0] if vals[0] <= vals[1] else nomes_casos[1]
        else:
            delta_str = "—"
            melhor = "—"

        t.add_row(modelo, *cells, delta_str, melhor)
    console.print(t)

# ==============================================================================
# TABELA (E DATAFRAME) DE REAL x PREVISTO
# ==============================================================================
def montar_df_real_previsto(df_teste, key_cols, y_teste, y_pred, model_name=""):
    """
    Monta um DataFrame com data, hora, valor real, valor previsto, erro e erro%
    para o período de teste (Mar/Abr 2026), usando as colunas já geradas por
    TemporalFE (_DATA) e a coluna de hora detectada em key_cols.
    """
    k_hora = key_cols.get("KEY_HORA")
    df_rp = pd.DataFrame({
        "data"     : df_teste["_DATA"].values,
        "hora"     : df_teste[k_hora].values.astype(int),
        "modelo"   : model_name,
        "real"     : np.asarray(y_teste, dtype=float),
        "previsto" : np.asarray(y_pred, dtype=float),
    })
    df_rp["erro"] = df_rp["real"] - df_rp["previsto"]
    with np.errstate(divide="ignore", invalid="ignore"):
        df_rp["erro_%"] = np.where(
            df_rp["real"].abs() > 1e-6,
            (df_rp["erro"] / df_rp["real"]) * 100,
            np.nan,
        )
    df_rp = df_rp.sort_values(["data", "hora"]).reset_index(drop=True)
    return df_rp

def print_tabela_real_previsto(df_rp, model_name="", n_linhas=N_LINHAS_AMOSTRA_REAL_PREVISTO,
                                titulo=None, amostrar=True):
    """
    Imprime uma tabela (amostra ordenada por data/hora, se o df for grande)
    com real x previsto x erro.
    """
    if df_rp is None or df_rp.empty:
        return

    if amostrar and len(df_rp) > n_linhas:
        amostra = df_rp.sample(n=n_linhas, random_state=RANDOM_STATE).sort_values(["data", "hora"])
        subtitulo_extra = f" (amostra de {n_linhas} de {len(df_rp)} horas)"
    else:
        amostra = df_rp.sort_values(["data", "hora"])
        subtitulo_extra = f" ({len(df_rp)} horas)"

    t = Table(
        title=(titulo or f"📋 REAL x PREVISTO — {model_name}") + subtitulo_extra,
        box=box.SIMPLE_HEAVY, border_style="cyan",
        show_lines=False, min_width=90,
    )
    t.add_column("Data",      min_width=12)
    t.add_column("Hora",      justify="right", min_width=6)
    t.add_column("Real",      justify="right", min_width=10)
    t.add_column("Previsto",  justify="right", min_width=10)
    t.add_column("Erro",      justify="right", min_width=10)
    t.add_column("Erro %",    justify="right", min_width=9)

    for _, row in amostra.iterrows():
        erro_abs = abs(row["erro"])
        erro_style = "metric_bad" if erro_abs > SPIKE_ALTO_LIMIAR * 0.10 else "white"
        erro_pct_str = f"{row['erro_%']:+.1f}%" if pd.notna(row["erro_%"]) else "—"
        t.add_row(
            str(row["data"]),
            f"{int(row['hora']):02d}h",
            f"{row['real']:.2f}",
            f"{row['previsto']:.2f}",
            f"[{erro_style}]{row['erro']:+.2f}[/{erro_style}]",
            erro_pct_str,
        )
    console.print(t)

    # Resumo estatístico do erro no período completo (não só na amostra)
    resumo = Table(title="📈 Resumo do erro no período de teste completo",
                   box=box.MINIMAL, border_style="muted", min_width=60)
    resumo.add_column("Métrica", min_width=20)
    resumo.add_column("Valor", justify="right", min_width=15)
    resumo.add_row("Erro médio (bias)",  f"{df_rp['erro'].mean():+.2f}")
    resumo.add_row("Erro absoluto médio", f"{df_rp['erro'].abs().mean():.2f}")
    resumo.add_row("Erro máximo (abs)",   f"{df_rp['erro'].abs().max():.2f}")
    resumo.add_row("Horas com erro > 10%", f"{(df_rp['erro_%'].abs() > 10).sum()} / {len(df_rp)}")
    console.print(resumo)

# ==============================================================================
def get_modelos_zoo():
    zoo = [
        ("LinearRegression",
         lambda: LinearRegression(), True),
        ("Ridge",
         lambda: Ridge(alpha=1.0, random_state=RANDOM_STATE), True),
        ("Lasso",
         lambda: Lasso(alpha=0.1, random_state=RANDOM_STATE, max_iter=5000), True),
        ("ElasticNet",
         lambda: ElasticNet(alpha=0.1, l1_ratio=0.5,
                            random_state=RANDOM_STATE, max_iter=5000), True),
        ("KNN",
         lambda: KNeighborsRegressor(n_neighbors=10, n_jobs=-1), True),
        ("DecisionTree",
         lambda: DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE), False),
        ("RandomForest",
         lambda: RandomForestRegressor(n_estimators=200, max_depth=15,
                                       n_jobs=-1, random_state=RANDOM_STATE), False),
    ]
    if XGB_OK:
        zoo.append(("XGBoost",
                    lambda: xgb.XGBRegressor(
                        n_estimators=300, max_depth=6, learning_rate=0.05,
                        random_state=RANDOM_STATE, n_jobs=-1,
                        tree_method="hist", verbosity=0), False))
    if LGB_OK:
        zoo.append(("LightGBM",
                    lambda: lgb.LGBMRegressor(
                        n_estimators=300, max_depth=6, learning_rate=0.05,
                        num_leaves=63, random_state=RANDOM_STATE,
                        n_jobs=-1, verbose=-1), False))
    if CAT_OK:
        zoo.append(("CatBoost",
                    lambda: cb.CatBoostRegressor(
                        iterations=300, depth=6, learning_rate=0.05,
                        random_seed=RANDOM_STATE, verbose=False), False))
    return zoo

# ==============================================================================
def carregar_dados_base():
    """Carrega TREINO_NORM e TESTE_NORM uma única vez (parquet é caro)."""
    console.print("\n   [info]📂 Carregando dados (uma única vez)...[/]")
    treino_path = INPUT_DIR / CENARIO_DEFAULT / STRATEGY_DEFAULT / "TREINO_NORM.parquet"
    teste_path  = INPUT_DIR / CENARIO_DEFAULT / STRATEGY_DEFAULT / "TESTE_NORM.parquet"
    df_treino_raw = load_parquet(treino_path, "TREINO_NORM")
    df_teste_raw  = load_parquet(teste_path,  "TESTE_NORM")
    return df_treino_raw, df_teste_raw

def run_caso(df_treino_raw, df_teste_raw, caso=1, anos_excluir_treino=None,
             titulo_caso="", descricao_caso=""):
    """
    Executa o zoo de modelos para um caso específico.

    caso=1 -> anos_excluir_treino=None  (Cenário A completo)
    caso=2 -> anos_excluir_treino={2022, 2023}  (Cenário A sem 2022/2023)

    Guarda em memória, por modelo:
      - y_teste (mesmo para todos os modelos deste caso)
      - y_pred_teste de cada modelo (preds_por_modelo)
    Isso permite, ao final, eleger o melhor modelo entre os casos, expor
    y_true / y_pred_teste no escopo global para o coletor de métricas, e
    montar a tabela de real x previsto.
    """
    print_header_fase(
        numero=1,
        titulo="ZOO DE MODELOS (10 algoritmos)",
        caso_info=titulo_caso or f"CASO {caso}",
        descricao=descricao_caso or "Testando do linear ao gradient boosting, defaults razoáveis",
    )

    target_col = detectar_target(df_treino_raw)
    key_cols   = detectar_key_cols(df_treino_raw)
    regime_col = detectar_regime_col(df_treino_raw)
    console.print(f"   [info]🎯 TARGET: {target_col}[/]")
    if regime_col:
        console.print(f"   [info]🚨 REGIME_PRECO removido (leakage)[/]")

    df_treino, df_teste = split_treino_teste(
        df_treino_raw, df_teste_raw, key_cols,
        anos_excluir_treino=anos_excluir_treino,
    )

    console.print("\n   [info]🗓  Engenharia temporal (22 features cíclicas)...[/]")
    fe = TemporalFE(use_holidays=True)
    df_treino = fe.transform(df_treino, key_cols)
    df_teste  = fe.transform(df_teste,  key_cols)

    feature_cols = get_feature_cols(df_treino, target_col, key_cols, regime_col)
    console.print(f"   [info]🔢 N features: {len(feature_cols)}[/]")

    X_treino = df_treino[feature_cols].values.astype(np.float32)
    y_treino = df_treino[target_col].values.astype(np.float32)
    X_teste  = df_teste[feature_cols].values.astype(np.float32)
    y_teste  = df_teste[target_col].values.astype(np.float32)
    console.print(f"   [info]🏋️  X_treino: {X_treino.shape} | X_teste: {X_teste.shape}[/]")

    scaler = StandardScaler()
    X_treino_scaled = scaler.fit_transform(X_treino)
    X_teste_scaled  = scaler.transform(X_teste)

    zoo = get_modelos_zoo()
    console.print(f"\n   [info]🚀 Treinando {len(zoo)} modelos...[/]\n")

    results_pontuais = []
    results_probabilist = []
    preds_por_modelo = {}   # guarda y_pred_teste de cada modelo

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task(f"Zoo de modelos (Caso {caso})", total=len(zoo))

        for name, factory, requer_scaling in zoo:
            t0 = time.perf_counter()
            X_tr = X_treino_scaled if requer_scaling else X_treino
            X_te = X_teste_scaled  if requer_scaling else X_teste

            try:
                model = factory()
                model.fit(X_tr, y_treino)
                y_pred_treino = model.predict(X_tr)
                y_pred_teste  = model.predict(X_te)
                elapsed = time.perf_counter() - t0

                m_pont = metricas_pontuais(y_teste, y_pred_teste, name)
                m_pont["tempo_s"] = round(elapsed, 2)
                results_pontuais.append(m_pont)
                preds_por_modelo[name] = y_pred_teste

                residuos_treino = y_treino - y_pred_treino
                m_prob = metricas_probabilisticas_aproximadas(
                    y_true=y_teste, y_pred=y_pred_teste,
                    residuos_treino=residuos_treino, model_name=name,
                )
                results_probabilist.append(m_prob)

                console.print(
                    f"   [success]✔  {name:22s} "
                    f"MAE={m_pont['MAE']:7.3f} | "
                    f"R²={m_pont['R2']:7.4f} | "
                    f"CRPS={m_prob['CRPS_aprox']:6.2f} | "
                    f"[{elapsed:5.1f}s][/]"
                )
            except Exception as e:
                console.print(f"   [error]✘  {name}: {str(e)[:80]}[/]")

            prog.advance(task)

    console.print()
    print_tabela_pontual(
        results_pontuais,
        titulo=f"📊 MÉTRICAS PONTUAIS — FASE 01 — CASO {caso}",
    )
    console.print()
    print_tabela_probabilistica(
        results_probabilist,
        titulo=f"📊 MÉTRICAS PROBABILÍSTICAS (BOOTSTRAP APROXIMADO) — FASE 01 — CASO {caso}",
    )
    console.print()
    top3 = print_ranking_final(results_pontuais, top_n=3, criterio="MAE")

    console.print()
    console.print(f"   [success]🎯 RECOMENDAÇÃO PARA FASE 02 (Caso {caso}):[/]")
    for i, m in enumerate(top3, 1):
        console.print(f"   [success]   {i}. {m}[/]")

    # ---------- tabela real x previsto do melhor modelo deste caso ----------
    melhor_modelo_caso = top3[0] if top3 else None
    df_rp_caso = None
    if melhor_modelo_caso is not None:
        console.print()
        df_rp_caso = montar_df_real_previsto(
            df_teste, key_cols, y_teste, preds_por_modelo[melhor_modelo_caso],
            model_name=melhor_modelo_caso,
        )
        print_tabela_real_previsto(
            df_rp_caso, model_name=melhor_modelo_caso,
            titulo=f"📋 REAL x PREVISTO — Caso {caso} — Melhor modelo: {melhor_modelo_caso}",
        )

    return {
        "caso"                : caso,
        "results_pontuais"    : results_pontuais,
        "results_probabilist" : results_probabilist,
        "top3_models"         : top3,
        "feature_cols"        : feature_cols,
        "y_teste"             : y_teste,
        "preds_por_modelo"    : preds_por_modelo,
        "df_teste"            : df_teste,      # guarda datas/horas p/ tabela real x previsto
        "key_cols"            : key_cols,
        "df_rp_melhor_modelo" : df_rp_caso,
    }

def run_todos_os_casos():
    """
    Orquestra os dois casos:
      - CASO 1: Cenário A completo (todos os anos) + Jan/Fev 2026 no treino.
      - CASO 2: Cenário A SEM 2022 e 2023 + Jan/Fev 2026 no treino.
    Teste sempre = Março/2026 e Abril/2026.
    """
    df_treino_raw, df_teste_raw = carregar_dados_base()

    casos_cfg = [
        dict(
            caso=1,
            anos_excluir_treino=None,
            titulo_caso="CASO 1 — Cenário A completo (todos os anos)",
            descricao_caso="Treino com todos os anos + Jan/Fev 2026 | Teste: Mar/Abr 2026",
        ),
        dict(
            caso=2,
            anos_excluir_treino=ANOS_EXCLUIR_CASO_2,
            titulo_caso="CASO 2 — Cenário A SEM 2022 e 2023",
            descricao_caso="Treino sem 2022/2023 + Jan/Fev 2026 | Teste: Mar/Abr 2026",
        ),
    ]

    saidas = {}
    resultados_por_caso = {}
    for cfg in casos_cfg:
        console.print()
        out = run_caso(df_treino_raw, df_teste_raw, **cfg)
        nome = f"Caso {cfg['caso']}"
        saidas[nome] = out
        resultados_por_caso[nome] = out["results_pontuais"]

    # Comparação final entre os casos
    console.print()
    print_comparacao_casos(resultados_por_caso, criterio="MAE")

    return saidas

# ==============================================================================
fase_01_saidas = run_todos_os_casos()

# ==============================================================================
# 🏆 ELEIÇÃO DO CAMPEÃO GERAL (entre os dois casos) — ponte para a coleta
#     automática de métricas, que precisa de y_true/y_pred_teste no global.
# ==============================================================================
console.print()
console.print(Panel(
    "[bold]Elegendo o melhor modelo entre Caso 1 e Caso 2 (menor MAE) "
    "para alimentar o coletor de métricas...[/]",
    title="[header_v2]  PONTE FASE 01 → COLETA DE MÉTRICAS  [/header_v2]",
    border_style="purple4",
))

_melhor_caso, _melhor_modelo, _melhor_mae = None, None, np.inf
for _nome_caso, _out in fase_01_saidas.items():
    for _m in _out["results_pontuais"]:
        if _m["MAE"] is not None and _m["MAE"] < _melhor_mae:
            _melhor_mae   = _m["MAE"]
            _melhor_caso  = _nome_caso
            _melhor_modelo = _m["model"]

if _melhor_caso is not None:
    console.print(
        f"   [success]🏆 Campeão geral: {_melhor_modelo} "
        f"({_melhor_caso}) — MAE={_melhor_mae:.3f}[/]"
    )
    # Variáveis expostas no escopo global para o coletor de métricas (Parte B)
    y_true       = fase_01_saidas[_melhor_caso]["y_teste"]
    y_pred_teste = fase_01_saidas[_melhor_caso]["preds_por_modelo"][_melhor_modelo]
    y_pred       = y_pred_teste  # alias, caso o coletor procure por este nome

    # ---------- tabela + CSV de real x previsto do CAMPEÃO GERAL ----------
    console.print()
    df_rp_campeao = montar_df_real_previsto(
        fase_01_saidas[_melhor_caso]["df_teste"],
        fase_01_saidas[_melhor_caso]["key_cols"],
        y_true, y_pred_teste, model_name=_melhor_modelo,
    )
    print_tabela_real_previsto(
        df_rp_campeao, model_name=_melhor_modelo,
        titulo=f"🏆 REAL x PREVISTO — CAMPEÃO GERAL ({_melhor_caso} — {_melhor_modelo})",
    )

    try:
        _csv_rp = f"{PASTA_SAIDA_METRICAS}/real_vs_previsto_parte{PARTE_TCC}_{_melhor_modelo}.csv"
        df_rp_campeao.to_csv(_csv_rp, index=False, encoding="utf-8-sig")
        console.print(f"   [success]✅ CSV completo real x previsto salvo em: {_csv_rp}[/]")
    except Exception as _e:
        console.print(f"   [warning]⚠  Não consegui salvar o CSV de real x previsto: {_e}[/]")
else:
    console.print("   [error]✘  Nenhum modelo produziu métricas válidas — "
                  "coleta de métricas será pulada.[/]")

# ==============================================================================
# 🧾 COLETA_METRICAS_TCC_AUTO  (não editar — célula gerada automaticamente)
# ==============================================================================
# Coleta de métricas finais desta parte. Não altera o modelo; só lê o que está
# em memória (y_true/y_pred_teste do campeão eleito acima) e grava em
# metricas_recalculadas.csv.
import numpy as _np, pandas as _pd, os as _os
PARTE = PARTE_TCC
PASTA_SAIDA = PASTA_SAIDA_METRICAS

def _arr(v):
    try:
        a = _np.asarray(v, dtype=float).ravel()
        return a if a.size else None
    except Exception:
        return None

_g = globals()
_cands = {}
for _k, _v in list(_g.items()):
    if _k.startswith("_"):
        continue
    _a = _arr(_v)
    if _a is not None and _a.size >= 10:
        _cands[_k] = _a

_linhas = []

# ---------- REGRESSÃO ----------
_truth = [n for n in ("y_true","y_test","y_real","y_teste","y_val","y") if n in _cands]
_preds = [n for n in ("y_pred_teste","y_pred_te","y_pred_test","y_pred_best",
                      "y_pred_mean","y_pred_q","y_pred","y_prev","pred","preds") if n in _cands]
_done = False
for _tn in _truth:
    _yt = _cands[_tn]
    for _pn in _preds:
        _yp = _cands[_pn]
        if _yp.size != _yt.size or _tn == _pn:
            continue
        _m = ~(_np.isnan(_yt) | _np.isnan(_yp))
        _a, _b = _yt[_m], _yp[_m]
        # pula se a "verdade" parecer binária (isso é classificação, não regressão)
        if _a.size == 0 or set(_np.unique(_a)) <= {0.0, 1.0}:
            continue
        from sklearn.metrics import mean_squared_error as _mse, r2_score as _r2, mean_absolute_error as _mae
        _nz = _np.abs(_a) > 1e-6
        _linhas.append({"Parte": PARTE, "Tipo": "regressao", "N_Features": None,
            "RMSE": round(float(_np.sqrt(_mse(_a,_b))),4), "R2": round(float(_r2(_a,_b)),4),
            "MAE": round(float(_mae(_a,_b)),4),
            "MAPE": (round(float(_np.mean(_np.abs((_a[_nz]-_b[_nz])/_a[_nz]))*100),2) if _nz.any() else None),
            "var_real": _tn, "var_pred": _pn, "n": int(_a.size)})
        print(f"[REG] Parte {PARTE}: RMSE={_linhas[-1]['RMSE']} R2={_linhas[-1]['R2']} "
              f"MAE={_linhas[-1]['MAE']} (usou {_tn} vs {_pn})")
        _done = True
        break
    if _done:
        break
if not _done:
    print(f"[REG] Parte {PARTE}: não encontrei par real/predito de regressão. "
          f"Arrays: {[(k,v.size) for k,v in _cands.items()][:12]}")

# ---------- CLASSIFICAÇÃO (se houver score de probabilidade + verdade binária) ----------
try:
    from sklearn.metrics import (roc_auc_score as _auc, average_precision_score as _ap,
                                 f1_score as _f1, recall_score as _rc, precision_score as _pr)
    _yb = None; _ybn = None
    for _n in ("y_true_clf","y_bin","y_spike","y_class","y_true","y_test"):
        if _n in _cands and set(_np.unique(_cands[_n][~_np.isnan(_cands[_n])])) <= {0.0,1.0}:
            _yb = _cands[_n]; _ybn = _n; break
    _score = None; _scn = None
    for _n in ("y_score","y_proba","y_prob","proba","scores","prob"):
        if _n in _cands:
            _s = _cands[_n]
            if _yb is not None and _s.size == _yb.size and _np.nanmin(_s) >= 0 and _np.nanmax(_s) <= 1:
                _score = _s; _scn = _n; break
    if _yb is not None and _score is not None:
        _pred01 = (_score >= 0.5).astype(int)
        _linhas.append({"Parte": PARTE, "Tipo": "classificacao", "N_Features": None,
            "ROC_AUC": round(float(_auc(_yb,_score)),4), "PR_AUC": round(float(_ap(_yb,_score)),4),
            "F1": round(float(_f1(_yb,_pred01)),4), "Recall": round(float(_rc(_yb,_pred01)),4),
            "Precision": round(float(_pr(_yb,_pred01, zero_division=0)),4),
            "var_real": _ybn, "var_pred": _scn, "n": int(_yb.size)})
        print(f"[CLF] Parte {PARTE}: ROC-AUC={_linhas[-1]['ROC_AUC']} F1={_linhas[-1]['F1']} (usou {_ybn} vs {_scn})")
    else:
        print(f"[CLF] Parte {PARTE}: sem score de probabilidade binário (ok se a parte for só regressão).")
except Exception as _e:
    print(f"[CLF] Parte {PARTE}: pulado ({type(_e).__name__}).")

# ---------- grava/atualiza CSV (dedupe por Parte+Tipo) ----------
if _linhas:
    _csv = _os.path.join(PASTA_SAIDA, "metricas_recalculadas.csv")
    _df = _pd.read_csv(_csv) if _os.path.exists(_csv) else _pd.DataFrame()
    _novo = _pd.DataFrame(_linhas)
    if not _df.empty:
        _chaves = set(zip(_novo["Parte"], _novo["Tipo"]))
        _df = _df[~_df.apply(lambda r: (r.get("Parte"), r.get("Tipo")) in _chaves, axis=1)]
    _df = _pd.concat([_df, _novo], ignore_index=True).sort_values(["Parte","Tipo"])
    _df.to_csv(_csv, index=False, encoding="utf-8-sig")
    print(f"✅ metricas_recalculadas.csv atualizado (+{len(_linhas)} linha(s)).")

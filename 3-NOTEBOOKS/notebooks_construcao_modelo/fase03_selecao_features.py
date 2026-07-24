# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                  TCC PLD — FASE 03: VARREDURA DE FEATURES                    ║
# ║      Seleção de N_Features (10–80), Ranking de Importância e Métricas       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  IDENTIFICAÇÃO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Autor       : Yago
#  Instituição : Universidade Federal do Ceará
#  Curso       : Engenharia Elétrica
#  Data        : Julho / 2026
#  Versão      : 1.0 — Arquivo Único (Varredura de Features + Coleta de Métricas)
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DESCRIÇÃO / FUNÇÃO DO SCRIPT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Este script implementa a Fase 03 da modelagem preditiva do PLD (submercado
#  Nordeste) do TCC, unificando em um único arquivo Colab-friendly:
#
#    (A) VARREDURA DE FEATURES — testa 8 quantidades de features (10, 20, 30,
#        40, 50, 60, 70, 80) contra 3 modelos (XGBoost, LightGBM, RandomForest),
#        totalizando 24 experimentos. Um conjunto de features "obrigatórias"
#        (colunas-chave temporais + 22 derivadas cíclicas/calendário) é sempre
#        incluído; o restante do orçamento de N_features é preenchido pelas
#        variáveis mais importantes segundo um ranking via LightGBM.
#
#    (B) COLETA AUTOMÁTICA DE MÉTRICAS — ao final da execução, varre as
#        variáveis em memória (y_true / y_pred_teste da melhor combinação) e
#        grava/atualiza a linha correspondente à "Parte 3" no arquivo
#        metricas_recalculadas.csv, com deduplicação por (Parte, Tipo).
#
#  SPLIT ANTI-LEAKAGE:
#    O treino é sempre o histórico completo + Janeiro/Fevereiro de 2026; o
#    teste é sempre Março e Abril de 2026. Diferente de um simples filtro por
#    mês, a função split_treino_teste() ativamente remove do parquet de
#    treino qualquer linha igual ou posterior a março/2026 (mesmo que ela já
#    estivesse presente no arquivo bruto), deduplica períodos que aparecem em
#    ambos os parquets (mantendo a versão de teste) e faz uma checagem final
#    de sanidade, emitindo um alerta explícito caso detecte vazamento.
#
#  INTEGRAÇÃO ENTRE AS PARTES:
#    O coletor de métricas (B) foi originalmente escrito para ler variáveis
#    soltas (y_true, y_pred_teste) diretamente do namespace global. Na Fase 03,
#    cada experimento (modelo × n_features) roda dentro de run_fase_03() e
#    suas predições não sobrevivem fora da função. Para a união funcionar de
#    fato, este script:
#      1. Guarda y_pred_teste de CADA experimento (24 combinações), indexado
#         por (modelo, n_features), além do y_teste de cada n_features.
#      2. Após eleger a melhor combinação global (menor MAE, o próprio Top 1
#         já calculado na varredura), expõe y_true / y_pred_teste dessa
#         combinação no escopo global.
#      3. Executa a célula de coleta de métricas, que grava a linha "Parte 3"
#         em metricas_recalculadas.csv usando exatamente essa combinação.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENTRADAS (INPUTS)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  O script não requer entradas manuais do usuário. Os parâmetros de controle
#  estão definidos nas constantes globais (logo após o tema do Rich):
#
#    BASE_DIR / INPUT_DIR    : caminho dos dados normalizados
#    OUTPUT_DIR              : "11_resultados_fase03" — onde a documentação
#                              das features da melhor combinação é salva
#    CENARIO_DEFAULT         : "cenario_A_todos_anos"
#    STRATEGY_DEFAULT        : "HYBRID_AGGRESSIVE"
#    TESTE_ANO / TESTE_MESES : 2026 / [3, 4]  → período de teste fixo (Mar/Abr)
#    TREINO_EXTRA_MESES      : [1, 2]  → Jan/Fev/2026 movidos para o treino
#    GRID_N_FEATURES         : [10, 20, 30, 40, 50, 60, 70, 80]
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
#     Uma linha por (Parte, Tipo) — aqui Parte=3, Tipo="regressao" — com
#     RMSE, R², MAE e MAPE da melhor combinação (modelo + n_features).
#     Deduplicado automaticamente a cada execução.
#
#  2. Documentação das features da melhor combinação → OUTPUT_DIR /
#     • features_melhor_combo.csv         — lista de features usadas (com tipo e importância)
#     • ranking_importancia_completo.csv  — importância de todos os candidatos extras
#     • features_melhor_combo.md          — versão legível, pronta para colar no texto do TCC
#
#  3. Saída no terminal (Rich):
#     • Cabeçalho de fase (Painel)
#     • 8 matrizes de resultados (N_features × modelo) para MAE, RMSE, R²,
#       sMAPE %, NRMSE %, CRPS aprox., Spike MAE e Spike Bias
#     • Curvas "elbow" de MAE × N_features, por modelo
#     • Tabela Top 5 global (melhores combinações modelo + n_features)
#     • Painel de recomendação para a Fase 04 (tuning)
#     • Tabela das features da melhor combinação (obrigatórias + extras)
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MODELOS E GRADE TESTADOS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Modelo         Papel no ranking (Fase 01)     N_features testados
#  XGBoost        🥇 (melhor da Fase 01)          10, 20, 30, 40, 50, 60, 70, 80
#  LightGBM       🥈                              10, 20, 30, 40, 50, 60, 70, 80
#  RandomForest   🥉                              10, 20, 30, 40, 50, 60, 70, 80
#
#  Total: 3 modelos × 8 quantidades de features = 24 experimentos por execução.
#  O ranking de importância das features extras (além das obrigatórias) é
#  calculado uma única vez por um LightGBM rápido (rank_features_lgb), e
#  reaproveitado para montar o subconjunto de cada N_features testado.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ETAPAS DO PROCESSO (resumo)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  N°  Etapa                        Estratégia                       Descrição
#  01  Carregamento                 read_parquet                     Lê TREINO_NORM e TESTE_NORM
#  02  Detecção automática          regex + tokens                    Localiza TARGET, KEY_* e REGIME_PRECO
#  03  Split anti-leakage           Filtro por AAAAMM + dedupe        Treino < mar/2026, teste = mar/abr/2026, com checagem de sanidade
#  04  Engenharia temporal          TemporalFE (22 features)          Cíclicas, calendário, feriados — sempre obrigatórias
#  05  Ranking de importância       LightGBM rápido (200 árvores)     Ordena candidatos extras por importância
#  06  Seleção por N_features       Obrigatórias + top-N do ranking   Monta o subconjunto de cada ponto da grade
#  07  Varredura (24 experimentos)  Loop N_features × modelo          Treina, prediz e mede tempo de cada combinação
#  08  Avaliação pontual            MAE/RMSE/MAPE/sMAPE/R²/NRMSE       Métricas completas de regressão
#  09  Avaliação de picos           Spike_MAE / Spike_Bias             Foco no erro em observações de PLD alto
#  10  Avaliação probabilística     Bootstrap de resíduos              Pinball loss, CRPS aproximado, cobertura 80%
#  11  Matrizes e curvas            Rich Table + pivot_table            Visualiza cada métrica por N_features × modelo
#  12  Eleição da melhor combinação Menor MAE global (Top 5)            Recomendação para a Fase 04 (tuning)
#  13  Documentação de features     CSV + Markdown                      Salva a composição da melhor combinação para o TCC
#  14  Coleta de métricas           Varredura de arrays no globals()    Grava/atualiza metricas_recalculadas.csv
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONSIDERAÇÕES INICIAIS E OBSERVAÇÕES TÉCNICAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#  1. AMBIENTE DE EXECUÇÃO
#     O script foi desenvolvido e testado no Google Colab (Python 3.10+).
#     Instala automaticamente (via pip, silenciosamente) xgboost, lightgbm,
#     holidays e rich caso ainda não estejam presentes no ambiente. Depende
#     dos arquivos TREINO_NORM.parquet e TESTE_NORM.parquet já normalizados
#     por etapas anteriores do pipeline.
#
#  2. TRAVA ANTI-LEAKAGE (PONTO CRÍTICO DESTA VERSÃO)
#     split_treino_teste() não confia apenas no filtro de mês do parquet de
#     treino: remove explicitamente qualquer linha com (ano×100+mês) ≥
#     corte_treino (março/2026) do parquet de treino bruto, mesmo que ele já
#     contivesse dados futuros por engano. Em seguida deduplica linhas de
#     Jan/Fev 2026 que possam existir tanto no parquet de treino quanto no de
#     teste (mantendo a versão do parquet de teste) e roda uma checagem final
#     que emite `🚨 LEAKAGE DETECTADO` caso alguma linha de Mar/Abr 2026 ainda
#     esteja presente no treino.
#
#  3. FEATURES OBRIGATÓRIAS x EXTRAS
#     As features obrigatórias (colunas-chave + as 22 derivadas cíclicas e de
#     calendário geradas por TemporalFE) sempre entram, independentemente do
#     N_features testado. Quando N_features é maior que o número de
#     obrigatórias, o espaço restante é preenchido pelas variáveis mais bem
#     ranqueadas (selecionar_top_n), garantindo comparabilidade entre pontos
#     da grade.
#
#  4. RANKING DE IMPORTÂNCIA
#     rank_features_lgb() treina um único LightGBM rápido (200 estimadores)
#     sobre obrigatórias + candidatos extras, e usa feature_importances_ para
#     ordenar os candidatos. Esse ranking é calculado uma única vez por
#     execução e reaproveitado em todos os 24 experimentos, evitando custo
#     computacional redundante.
#
#  5. MÉTRICAS COM FOCO EM PICOS E ASSIMETRIA
#     Além de MAE, RMSE, MAPE e R², o script calcula sMAPE (evita explosão
#     quando o valor real é próximo de zero), NRMSE (RMSE normalizado pela
#     média do real, útil para comparar janelas de teste diferentes) e,
#     principalmente, Spike_MAE e Spike_Bias para observações com PLD acima
#     de SPIKE_ALTO_LIMIAR (R$ 500). Spike_Bias positivo indica que o modelo
#     SUBESTIMA os picos — o cenário mais arriscado para quem toma decisão de
#     exposição no mercado livre no Nordeste.
#
#  6. MATRIZ DE BIAS COM LÓGICA PRÓPRIA
#     print_matriz_bias() trata Spike_Bias separadamente de métricas comuns:
#     como o valor pode ser positivo ou negativo, "melhor" é o mais próximo
#     de zero (menor valor absoluto), não o mínimo algébrico — por isso não
#     reaproveita a lógica de print_matriz_resultados().
#
#  7. MÉTRICAS PROBABILÍSTICAS APROXIMADAS
#     Como os 3 modelos são pontuais (não nativamente probabilísticos), a
#     incerteza é aproximada via bootstrap gaussiano dos resíduos de treino,
#     gerando quantis P10/P50/P90 e métricas de Pinball Loss, CRPS aproximado
#     e cobertura do intervalo de 80%.
#
#  8. DOCUMENTAÇÃO DAS FEATURES PARA O TCC
#     documentar_e_salvar_features() não apenas exibe no console a composição
#     da melhor combinação (obrigatórias + extras, com importância), como
#     também persiste três artefatos em disco (CSV da combinação, CSV do
#     ranking completo e um Markdown pronto para colar no texto do TCC).
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
#      - Os 3 modelos usam hiperparâmetros padrão razoáveis, sem tuning — a
#        otimização fina fica para a Fase 04 (Optuna).
#      - O ranking de importância é calculado uma única vez (LightGBM com
#        parâmetros fixos); ele não é recalculado por modelo nem por ponto
#        da grade, o que é intencional para isolar o efeito de N_features.
#
#  11. REPRODUTIBILIDADE
#      RANDOM_STATE = 42 é usado em todos os modelos, no ranking de
#      importância e no bootstrap de resíduos, garantindo resultados
#      determinísticos entre execuções com os mesmos dados.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DEPENDÊNCIAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Biblioteca        Versão mínima  Finalidade
#  numpy             1.23           Operações numéricas e amostragem (bootstrap)
#  pandas            1.5            Manipulação de DataFrames, Parquet e pivot tables
#  scikit-learn      1.2            RandomForest e métricas de regressão
#  xgboost           —              Modelo de gradient boosting (obrigatório nesta fase)
#  lightgbm          —              Modelo de gradient boosting + ranking de importância
#  holidays          —              Feriados nacionais para engenharia temporal
#  rich              13.0           Tabelas, painéis e barras de progresso
#
#  Instalação automática (executada no início do script, se necessário):
#      pip install -q xgboost lightgbm holidays rich
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
#              │   ├── 10_dados_normalizados/
#              │   │   └── cenario_A_todos_anos/
#              │   │       └── HYBRID_AGGRESSIVE/
#              │   │           ├── TREINO_NORM.parquet   (entrada)
#              │   │           └── TESTE_NORM.parquet    (entrada)
#              │   └── 11_resultados_fase03/             (saída)
#              │       ├── features_melhor_combo.csv
#              │       ├── ranking_importancia_completo.csv
#              │       └── features_melhor_combo.md
#              └── codigos_modelagem/
#                  └── metricas_recalculadas.csv          (saída)
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  COMO EXECUTAR (Google Colab)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Célula 1 — Montar o Drive:
#      from google.colab import drive
#      drive.mount('/content/drive')
#
#  Célula 2 — Executar o script:
#      exec(open('fase03_varredura_features_coleta_metricas.py').read())
#   ou simplesmente executar este arquivo como módulo principal.
#   As dependências (xgboost, lightgbm, holidays, rich) são instaladas
#   automaticamente na primeira execução, se necessário.
#
# ══════════════════════════════════════════════════════════════════════════════

import subprocess, sys
def _ensure(pkg, imp=None):
    try:
        __import__(imp or pkg)
    except ImportError:
        print(f"⏬ Instalando {pkg}...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

_ensure("xgboost"); _ensure("lightgbm")
_ensure("holidays"); _ensure("rich")

import warnings
warnings.filterwarnings("ignore")

import re
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error,
)

import xgboost as xgb
import lightgbm as lgb

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
    "info": "bold cyan", "warning": "bold yellow", "error": "bold red",
    "success": "bold green", "header_v2": "bold white on purple4",
    "highlight": "bold magenta", "muted": "dim white",
    "metric_best": "bold green", "metric_bad": "bold red",
    "acerto": "bold green", "data": "bold yellow",
})
console = Console(theme=THEME)

# ==============================================================================
BASE_DIR   = Path("/content/drive/MyDrive/TCC_PLD_Project/09-ESCRITA_TCC/PARTES_TCC/codigos/dados")
INPUT_DIR  = BASE_DIR / "10_dados_normalizados"
OUTPUT_DIR = BASE_DIR / "11_resultados_fase03"   # onde a documentação das features é salva

CENARIO_DEFAULT  = "cenario_A_todos_anos"
STRATEGY_DEFAULT = "HYBRID_AGGRESSIVE"
TARGET_CANONICAL = "01_CCEE_NORDESTE_TARGET_PLD_HORA_NORDESTE"
SPIKE_ALTO_LIMIAR = 500.0

# ── Datas de corte ──────────────────────────────────────────────────────────
TESTE_ANO          = 2026
TREINO_EXTRA_MESES = [1, 2]   # jan e fev de 2026 vão para treino
TESTE_MESES        = [3, 4]   # mar e abr de 2026 são o teste
# ────────────────────────────────────────────────────────────────────────────

RANDOM_STATE = 42

# Varredura de N features (incluindo obrigatórias)
GRID_N_FEATURES = [10, 20, 30, 40, 50, 60, 70, 80]

# Configuração da coleta de métricas (parte B)
PARTE_TCC = 3
PASTA_SAIDA_METRICAS = r"/content/drive/MyDrive/TCC_PLD_Project/09-ESCRITA_TCC/PARTES_TCC/codigos/codigos_modelagem"

# 3 modelos selecionados: XGBoost (🥇), LightGBM (🥈), RandomForest (🥉)
def get_modelos_fase03():
    return [
        ("XGBoost",
         lambda: xgb.XGBRegressor(
             n_estimators=300, max_depth=6, learning_rate=0.05,
             random_state=RANDOM_STATE, n_jobs=-1,
             tree_method="hist", verbosity=0), False),
        ("LightGBM",
         lambda: lgb.LGBMRegressor(
             n_estimators=300, max_depth=6, learning_rate=0.05,
             num_leaves=63, random_state=RANDOM_STATE,
             n_jobs=-1, verbose=-1), False),
        ("RandomForest",
         lambda: RandomForestRegressor(
             n_estimators=200, max_depth=15,
             n_jobs=-1, random_state=RANDOM_STATE), False),
    ]

# ==============================================================================
def safe_round(v, decimals=4):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    return round(float(v), decimals)

def detectar_target(df):
    if TARGET_CANONICAL in df.columns:
        return TARGET_CANONICAL
    for c in df.columns:
        cl = c.lower()
        if "nordeste" in cl and "target" in cl:
            return c
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

def load_parquet(path, label=""):
    console.print(f"   [info]📂 {label}: {path.name}[/]")
    df = pd.read_parquet(path)
    df.columns = [re.sub(r'[":{},\[\]]', "_", str(c)) for c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df

def split_treino_teste(df_treino_raw, df_teste_raw, key_cols):
    """
    GARANTE:
        TREINO = tudo ATÉ fevereiro/2026  (histórico + jan/fev 2026)
        TESTE  = março + abril / 2026

    Robustez (correção principal):
      - Remove do parquet de TREINO qualquer linha do período de teste ou
        posterior (>= março/2026), evitando LEAKAGE caso o parquet de treino
        já contenha dados de 2026.
      - Remove jan/fev 2026 duplicados (parquet de treino x parquet de teste),
        mantendo a versão do parquet de teste.
      - Faz uma checagem final de sanidade contra leakage.
    """
    k_ano  = key_cols.get("KEY_ANO")
    k_mes  = key_cols.get("KEY_MES")
    k_dia  = key_cols.get("KEY_DIA")
    k_hora = key_cols.get("KEY_HORA")

    # Chave numérica ano*100+mes para comparar períodos com segurança.
    # corte_treino = 202603 -> treino deve conter apenas (ano*100+mes) < 202603
    corte_treino = TESTE_ANO * 100 + min(TESTE_MESES)

    def _ano_mes(df):
        return df[k_ano].astype(int) * 100 + df[k_mes].astype(int)

    # 1) TESTE = mar + abr / 2026 (sempre do dataset de teste)
    mask_teste = (
        (df_teste_raw[k_ano] == TESTE_ANO) &
        (df_teste_raw[k_mes].isin(TESTE_MESES))
    )
    df_teste_final = df_teste_raw[mask_teste].reset_index(drop=True)

    # 2) EXTRA = jan + fev / 2026 (do dataset de teste) -> vão para o TREINO
    mask_extra = (
        (df_teste_raw[k_ano] == TESTE_ANO) &
        (df_teste_raw[k_mes].isin(TREINO_EXTRA_MESES))
    )
    df_extra = df_teste_raw[mask_extra]

    # 3) TREINO base: manter SOMENTE período < março/2026 (anti-leakage).
    #    Isto preserva todo o histórico + jan/fev 2026 e descarta mar/abr 2026
    #    (e qualquer mês posterior) que por acaso esteja no parquet de treino.
    df_treino_base = df_treino_raw[_ano_mes(df_treino_raw) < corte_treino]
    n_removidos_leak = len(df_treino_raw) - len(df_treino_base)
    if n_removidos_leak > 0:
        console.print(
            f"   [warning]⚠  Removidas {n_removidos_leak:,} linhas >= mar/2026 "
            f"do parquet de TREINO (proteção contra leakage)[/]"
        )

    # 4) Concatena treino base + jan/fev 2026
    df_treino_final = pd.concat([df_treino_base, df_extra], ignore_index=True)

    # 5) Deduplica por chave temporal (jan/fev 2026 pode existir nos dois parquets).
    #    keep="last" -> prioriza a versão vinda do parquet de teste (df_extra).
    chaves = [c for c in [k_ano, k_mes, k_dia, k_hora] if c]
    if chaves:
        antes = len(df_treino_final)
        df_treino_final = (
            df_treino_final
            .drop_duplicates(subset=chaves, keep="last")
            .reset_index(drop=True)
        )
        dup = antes - len(df_treino_final)
        if dup > 0:
            console.print(
                f"   [warning]⚠  Removidas {dup:,} linhas duplicadas no TREINO "
                f"(chave {chaves})[/]"
            )

    console.print(
        f"   [success]✔  TREINO: {len(df_treino_final):,} linhas "
        f"(histórico até fev/2026) | "
        f"TESTE: {len(df_teste_final):,} linhas (mar/abr 2026)[/]"
    )

    # 6) Sanidade: garante que NENHUM dado de mar/abr 2026 vazou para o TREINO
    leak = df_treino_final[
        (df_treino_final[k_ano] == TESTE_ANO) &
        (df_treino_final[k_mes].isin(TESTE_MESES))
    ]
    if len(leak) > 0:
        console.print(
            f"   [error]🚨 LEAKAGE DETECTADO: {len(leak):,} linhas de mar/abr "
            f"2026 no treino — verifique os parquets![/]"
        )
    else:
        console.print("   [success]🔒 Sem leakage: TREINO não contém mar/abr 2026[/]")

    if len(df_teste_final) == 0:
        console.print(
            "   [error]🚨 TESTE vazio: não há mar/abr 2026 no parquet de teste![/]"
        )

    return df_treino_final, df_teste_final

# ==============================================================================
class TemporalFE:
    ANOS_CRITICOS = {2014, 2015, 2021, 2024}
    HORA_PICO = (18, 21)
    HORA_MADRUGADA = (0, 5)
    MESES_CHUVOSOS_NE = {12, 1, 2, 3}
    MESES_SECOS_NE = {6, 7, 8, 9, 10, 11}

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
        k_ano, k_mes  = key_cols["KEY_ANO"], key_cols["KEY_MES"]
        k_dia, k_hora = key_cols["KEY_DIA"], key_cols["KEY_HORA"]
        ano  = df_out[k_ano].astype(int).values
        mes  = df_out[k_mes].astype(int).values
        dia  = df_out[k_dia].astype(int).values
        hora = df_out[k_hora].astype(int).values

        datas_safe = [self._safe_date(a, m, d) or date(1900, 1, 1)
                      for a, m, d in zip(ano, mes, dia)]
        dia_semana = np.array([d.weekday() for d in datas_safe], dtype=int)
        dia_do_ano = np.array([d.timetuple().tm_yday for d in datas_safe], dtype=int)

        df_out["CYC_hora_sin"]    = np.sin(2*np.pi*hora/24.0)
        df_out["CYC_hora_cos"]    = np.cos(2*np.pi*hora/24.0)
        df_out["CYC_dia_sem_sin"] = np.sin(2*np.pi*dia_semana/7.0)
        df_out["CYC_dia_sem_cos"] = np.cos(2*np.pi*dia_semana/7.0)
        df_out["CYC_dia_mes_sin"] = np.sin(2*np.pi*(dia-1)/31.0)
        df_out["CYC_dia_mes_cos"] = np.cos(2*np.pi*(dia-1)/31.0)
        df_out["CYC_mes_sin"]     = np.sin(2*np.pi*(mes-1)/12.0)
        df_out["CYC_mes_cos"]     = np.cos(2*np.pi*(mes-1)/12.0)
        df_out["CYC_dia_ano_sin"] = np.sin(2*np.pi*(dia_do_ano-1)/365.0)
        df_out["CYC_dia_ano_cos"] = np.cos(2*np.pi*(dia_do_ano-1)/365.0)

        df_out["CAL_is_fim_semana"]         = (dia_semana >= 5).astype(int)
        df_out["CAL_is_horario_pico"]       = ((hora >= 18) & (hora <= 21)).astype(int)
        df_out["CAL_is_madrugada"]          = ((hora >= 0) & (hora <= 5)).astype(int)
        df_out["CAL_is_periodo_seco_NE"]    = np.isin(mes, [6,7,8,9,10,11]).astype(int)
        df_out["CAL_is_periodo_chuvoso_NE"] = np.isin(mes, [12,1,2,3]).astype(int)
        df_out["CAL_is_ano_critico"]        = np.isin(ano, [2014,2015,2021,2024]).astype(int)

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

        df_out["_DATA"] = pd.to_datetime(
            {"year": ano, "month": mes, "day": dia}, errors="coerce"
        ).dt.date
        return df_out

    def list_obrigatorias(self, key_cols):
        """KEY_* + derivadas temporais = features obrigatórias."""
        keys = [v for v in key_cols.values() if v]
        derivadas = [
            "CYC_hora_sin","CYC_hora_cos","CYC_dia_sem_sin","CYC_dia_sem_cos",
            "CYC_dia_mes_sin","CYC_dia_mes_cos","CYC_mes_sin","CYC_mes_cos",
            "CYC_dia_ano_sin","CYC_dia_ano_cos",
            "CAL_is_fim_semana","CAL_is_horario_pico","CAL_is_madrugada",
            "CAL_is_periodo_seco_NE","CAL_is_periodo_chuvoso_NE",
            "CAL_is_ano_critico","CAL_is_feriado","CAL_is_dia_util",
            "TEMP_trimestre","TEMP_dia_do_ano","TEMP_ano_normalizado",
            "TEMP_progresso_ano","TEMP_progresso_mes","TEMP_weekend_pico",
        ]
        return keys + derivadas

# ==============================================================================
# MÉTRICAS — versão completa (igual à Fase 01)
# ==============================================================================
def metricas_pontuais(y_true, y_pred, model_name=""):
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2   = r2_score(y_true, y_pred)
    resid = y_true - y_pred
    me    = float(np.mean(resid))

    # sMAPE: simétrico, evita explosão quando y_true ~ 0 (comum em MAPE puro)
    smape = float(np.mean(
        2 * np.abs(resid) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    ) * 100)

    # NRMSE: RMSE normalizado pela média do real, útil para comparar entre
    # janelas de teste com magnitudes diferentes
    nrmse = rmse / (np.mean(y_true) + 1e-8) * 100

    spike_mask = y_true > SPIKE_ALTO_LIMIAR
    if spike_mask.sum() > 5:
        spike_mae  = float(mean_absolute_error(y_true[spike_mask], y_pred[spike_mask]))
        # Spike_Bias: sinal do erro nos picos -> positivo = modelo SUBESTIMA o pico
        #             (perigoso no Nordeste: subestimar spike é o pior cenário
        #             para quem toma decisão de exposição no mercado livre)
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
    return float(np.mean(np.maximum(q*e, (q-1)*e)))

def coverage_score(y_true, lower, upper):
    return float(np.mean((y_true >= lower) & (y_true <= upper)))

def metricas_prob_aproximadas(y_true, y_pred, residuos_treino, n_bootstrap=150,
                               random_state=RANDOM_STATE):
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

    return {
        "Pinball_Q10"    : safe_round(pin_q10, 4),
        "Pinball_Q50"    : safe_round(pin_q50, 4),
        "Pinball_Q90"    : safe_round(pin_q90, 4),
        "CRPS_aprox"     : safe_round(crps_aprox, 4),
        "Coverage_80%"   : safe_round(coverage_score(y_true, p10, p90), 4),
        "Largura_P10_P90": safe_round(float(np.mean(p90 - p10)), 4),
    }

# ==============================================================================
def rank_features_lgb(X, y, candidates, random_state=RANDOM_STATE):
    """Ranqueia features por importância usando LightGBM rápido."""
    ranker = lgb.LGBMRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        num_leaves=31, min_child_samples=20,
        random_state=random_state, n_jobs=-1, verbose=-1,
    )
    ranker.fit(X[candidates].values, y)
    return pd.DataFrame({
        "feature": candidates,
        "importance": ranker.feature_importances_,
    }).sort_values("importance", ascending=False)

def selecionar_top_n(obrigatorias, ranking_extras, n_total):
    """Garante obrigatórias + completa com top do ranking até n_total."""
    if n_total <= len(obrigatorias):
        return list(obrigatorias)
    n_extras = n_total - len(obrigatorias)
    extras = ranking_extras.head(n_extras)["feature"].tolist()
    return list(obrigatorias) + extras

# ==============================================================================
def print_header(numero, titulo, descricao=""):
    console.print(Panel(
        Text.assemble(
            (f"🧪 FASE {numero:02d} — {titulo}\n", "bold green"),
            (f"   {descricao}\n", "muted") if descricao else ("", "muted"),
            (f"   Cenário: {CENARIO_DEFAULT} | Strategy: {STRATEGY_DEFAULT}\n", "muted"),
            (f"   Treino: histórico + jan/fev 2026 | Teste: mar/abr 2026\n", "muted"),
        ),
        title="[header_v2]  TCC PLD — EXPERIMENTO ITERATIVO  [/header_v2]",
        border_style="purple4", padding=(1, 4),
    ))

# Métricas onde MENOR é melhor (usadas para colorir/eleger vencedor na matriz)
_METRICAS_MENOR_MELHOR = {"MAE", "RMSE", "CRPS_aprox", "Spike_MAE",
                          "sMAPE_%", "NRMSE_%", "MAPE_%"}

def print_matriz_resultados(rows, metric="MAE", titulo=""):
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="n_features", columns="model", values=metric)
    pivot = pivot.reindex(sorted(pivot.index))

    t = Table(title=f"📊 {titulo} | métrica: {metric}",
              box=box.DOUBLE_EDGE, border_style="cyan",
              show_lines=True, min_width=90)
    t.add_column("N feats", justify="right", style="bold", min_width=8)
    for col in pivot.columns:
        t.add_column(col, justify="right", min_width=14)
    t.add_column("Melhor", justify="center", min_width=14)

    menor_melhor = metric in _METRICAS_MENOR_MELHOR
    global_best = pivot.min().min() if menor_melhor else pivot.max().max()

    for idx, row in pivot.iterrows():
        if menor_melhor:
            best_in_row = row.min()
            winner = row.idxmin()
        else:
            best_in_row = row.max()
            winner = row.idxmax()

        cells = []
        for col in pivot.columns:
            v = row[col]
            if pd.isna(v):
                cells.append("—")
            elif v == global_best:
                cells.append(f"[acerto]★ {v:.3f}[/acerto]")
            elif v == best_in_row:
                cells.append(f"[metric_best]{v:.3f}[/metric_best]")
            else:
                cells.append(f"{v:.3f}")
        t.add_row(str(idx), *cells, f"[bold]{winner}[/bold]")
    console.print(t)

def print_matriz_bias(rows, metric="Spike_Bias", titulo=""):
    """
    Matriz específica para métricas de BIAS (podem ser negativas ou positivas).
    Aqui "melhor" = mais próximo de zero, não o mínimo algébrico — por isso
    não reaproveita a lógica de print_matriz_resultados (que assume min/max).
    """
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="n_features", columns="model", values=metric)
    pivot = pivot.reindex(sorted(pivot.index))

    t = Table(title=f"📊 {titulo} | métrica: {metric} (ideal: próximo de 0)",
              box=box.DOUBLE_EDGE, border_style="cyan",
              show_lines=True, min_width=90)
    t.add_column("N feats", justify="right", style="bold", min_width=8)
    for col in pivot.columns:
        t.add_column(col, justify="right", min_width=16)
    t.add_column("Melhor (|.| mín)", justify="center", min_width=16)

    abs_pivot = pivot.abs()
    global_best_abs = abs_pivot.min().min()

    for idx, row in pivot.iterrows():
        abs_row = row.abs()
        best_in_row_abs = abs_row.min()
        winner = abs_row.idxmin()

        cells = []
        for col in pivot.columns:
            v = row[col]
            if pd.isna(v):
                cells.append("—")
            elif abs(v) == global_best_abs:
                cells.append(f"[acerto]★ {v:+.3f}[/acerto]")
            elif abs(v) == best_in_row_abs:
                cells.append(f"[metric_best]{v:+.3f}[/metric_best]")
            else:
                sinal_style = "metric_bad" if v > 0 else "white"
                cells.append(f"[{sinal_style}]{v:+.3f}[/{sinal_style}]")
        t.add_row(str(idx), *cells, f"[bold]{winner}[/bold]")
    console.print(t)
    console.print(
        "   [muted]Nota: Spike_Bias positivo = modelo SUBESTIMA os picos "
        "(cenário mais arriscado no Nordeste); negativo = superestima.[/]"
    )

def print_curva_elbow(rows, model_name, metric="MAE"):
    df = pd.DataFrame(rows)
    df_m = df[df["model"] == model_name].sort_values("n_features")
    if df_m.empty:
        return
    vals = df_m[metric].values
    ns   = df_m["n_features"].values

    t = Table(title=f"📉 CURVA {metric} × N_features — {model_name}",
              box=box.ROUNDED, border_style="magenta",
              show_lines=False, min_width=60)
    t.add_column("N feats",      justify="right", style="bold")
    t.add_column(metric,         justify="right")
    t.add_column("Δ vs anterior",justify="right")
    t.add_column("Barra",        justify="left")

    vmin, vmax = vals.min(), vals.max()
    rng_v = vmax - vmin if vmax > vmin else 1.0
    prev = None
    for n, v in zip(ns, vals):
        delta   = f"{v - prev:+.3f}" if prev is not None else "—"
        bar_len = int(30 * (vmax - v) / rng_v)
        bar     = "█" * bar_len if v == vmin else "▓" * bar_len
        style   = "acerto" if v == vmin else "white"
        t.add_row(str(n), f"[{style}]{v:.3f}[/{style}]", delta, bar)
        prev = v

    console.print(t)

# ==============================================================================
def documentar_e_salvar_features(best, features_best, obrigatorias, ranking,
                                 output_dir=OUTPUT_DIR):
    """
    Mostra no console e SALVA em disco quais features compõem a MELHOR
    combinação (modelo + n_features), para documentação do TCC.

    Gera 3 arquivos em output_dir:
      - features_melhor_combo.csv        (lista usada pela melhor combinação)
      - ranking_importancia_completo.csv (importância de todos os extras)
      - features_melhor_combo.md         (legível, pronto para colar no texto)
    """
    set_obrig = set(obrigatorias)
    imp_map = dict(zip(ranking["feature"], ranking["importance"]))

    # Monta a tabela: tipo (Obrigatória/Extra) + importância
    linhas = []
    for f in features_best:
        linhas.append({
            "feature": f,
            "tipo": "Obrigatória" if f in set_obrig else "Extra",
            "importancia": imp_map.get(f, np.nan),
        })
    df_feat = pd.DataFrame(linhas)

    # Ordena: obrigatórias primeiro; entre os extras, por importância desc.
    df_feat["_ord_tipo"] = (df_feat["tipo"] == "Extra").astype(int)
    df_feat = (
        df_feat.sort_values(["_ord_tipo", "importancia"], ascending=[True, False])
        .drop(columns="_ord_tipo")
        .reset_index(drop=True)
    )
    df_feat.insert(0, "ordem", range(1, len(df_feat) + 1))
    df_feat["modelo"]     = best["model"]
    df_feat["n_features"] = int(best["n_features"])
    df_feat["MAE"]        = best["MAE"]

    n_obrig = int((df_feat["tipo"] == "Obrigatória").sum())
    n_extra = int((df_feat["tipo"] == "Extra").sum())

    # --- mostra no console ---
    t = Table(
        title=(f"🧬 FEATURES DA MELHOR COMBINAÇÃO — {best['model']} "
               f"({int(best['n_features'])} feats | MAE={best['MAE']:.3f})"),
        box=box.DOUBLE_EDGE, border_style="green",
        show_lines=False, min_width=90,
    )
    t.add_column("#",           justify="right", min_width=4)
    t.add_column("Feature",     style="bold",    min_width=42)
    t.add_column("Tipo",        justify="center", min_width=14)
    t.add_column("Importância", justify="right",  min_width=14)
    for _, r in df_feat.iterrows():
        tipo_str = (f"[data]{r['tipo']}[/data]" if r["tipo"] == "Obrigatória"
                    else f"[highlight]{r['tipo']}[/highlight]")
        imp_str = "—" if pd.isna(r["importancia"]) else f"{r['importancia']:.1f}"
        t.add_row(str(r["ordem"]), r["feature"], tipo_str, imp_str)
    console.print(t)
    console.print(
        f"   [muted]Total: {len(df_feat)} features "
        f"({n_obrig} obrigatórias + {n_extra} extras)[/]"
    )

    # --- salva em disco ---
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        destino = output_dir
    except Exception:
        destino = Path("/content")
        console.print(f"   [warning]⚠  Sem acesso a {output_dir}; salvando em {destino}[/]")

    salvos = []

    p_csv = destino / "features_melhor_combo.csv"
    df_feat.to_csv(p_csv, index=False)
    salvos.append(p_csv)

    p_rank = destino / "ranking_importancia_completo.csv"
    ranking.reset_index(drop=True).to_csv(p_rank, index=False)
    salvos.append(p_rank)

    # Markdown legível para colar no TCC
    md = [
        "# Features da melhor combinação — Fase 03",
        "",
        f"- **Modelo:** {best['model']}",
        f"- **N features:** {int(best['n_features'])}",
        f"- **MAE (mar/abr 2026):** {best['MAE']:.3f}",
        f"- **Composição:** {n_obrig} obrigatórias + {n_extra} extras",
        "",
        "| # | Feature | Tipo | Importância |",
        "|---|---------|------|-------------|",
    ]
    for _, r in df_feat.iterrows():
        imp_str = "—" if pd.isna(r["importancia"]) else f"{r['importancia']:.1f}"
        md.append(f"| {r['ordem']} | `{r['feature']}` | {r['tipo']} | {imp_str} |")
    p_md = destino / "features_melhor_combo.md"
    p_md.write_text("\n".join(md), encoding="utf-8")
    salvos.append(p_md)

    console.print("\n   [success]💾 Documentação das features salva em:[/]")
    for p in salvos:
        console.print(f"   [success]   • {p}[/]")

    return df_feat, [str(p) for p in salvos]

# ==============================================================================
def run_fase_03():
    print_header(
        numero=3,
        titulo="VARREDURA DE FEATURES (10–80)",
        descricao="3 modelos × 8 quantidades de features = 24 experimentos",
    )

    # 1. Dados
    console.print("\n   [info]📂 Carregando dados...[/]")
    treino_path = INPUT_DIR / CENARIO_DEFAULT / STRATEGY_DEFAULT / "TREINO_NORM.parquet"
    teste_path  = INPUT_DIR / CENARIO_DEFAULT / STRATEGY_DEFAULT / "TESTE_NORM.parquet"
    df_treino_raw = load_parquet(treino_path, "TREINO_NORM")
    df_teste_raw  = load_parquet(teste_path,  "TESTE_NORM")

    target_col = detectar_target(df_treino_raw)
    key_cols   = detectar_key_cols(df_treino_raw)
    regime_col = detectar_regime_col(df_treino_raw)
    console.print(f"   [info]🎯 TARGET: {target_col}[/]")
    if regime_col:
        console.print(f"   [info]🚨 REGIME_PRECO removido (leakage)[/]")

    # 2. Split: treino até fev/2026, teste = mar+abr/2026 (com trava anti-leakage)
    df_treino, df_teste = split_treino_teste(df_treino_raw, df_teste_raw, key_cols)

    # Validação dos períodos (robusta a treino sem 2026)
    k_ano, k_mes = key_cols["KEY_ANO"], key_cols["KEY_MES"]
    treino_2026 = df_treino[df_treino[k_ano] == TESTE_ANO]
    ultimo_mes_2026_treino = int(treino_2026[k_mes].max()) if len(treino_2026) else None
    console.print(
        f"   [muted]Treino  → ano range: "
        f"{int(df_treino[k_ano].min())}–{int(df_treino[k_ano].max())} | "
        f"último mês de 2026 no treino: {ultimo_mes_2026_treino}[/]"
    )
    meses_teste_2026 = sorted(
        df_teste[df_teste[k_ano] == TESTE_ANO][k_mes].unique().tolist()
    )
    console.print(f"   [muted]Teste   → meses 2026: {meses_teste_2026}[/]")

    # 3. Feature engineering
    console.print("\n   [info]🗓  Engenharia temporal...[/]")
    fe = TemporalFE(use_holidays=True)
    df_treino = fe.transform(df_treino, key_cols)
    df_teste  = fe.transform(df_teste,  key_cols)

    obrigatorias = fe.list_obrigatorias(key_cols)
    obrigatorias = [f for f in obrigatorias if f in df_treino.columns]
    n_keys  = sum(1 for v in key_cols.values() if v and v in df_treino.columns)
    n_deriv = len(obrigatorias) - n_keys
    console.print(
        f"   [info]🔒 Obrigatórias: {len(obrigatorias)} "
        f"({n_keys} KEY + {n_deriv} derivadas)[/]"
    )

    # 4. Candidatos extras
    exclude = set([target_col])
    if regime_col:
        exclude.add(regime_col)
    exclude.update([c for c in df_treino.columns if c.startswith("_")])
    exclude.update(obrigatorias)
    candidates_extras = [
        c for c in df_treino.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df_treino[c])
    ]
    console.print(f"   [info]🎲 Candidatos extras: {len(candidates_extras)}[/]")

    # 5. Ranking de importância
    console.print("\n   [info]📊 Ranqueando features por importância (LightGBM)...[/]")
    ranking = rank_features_lgb(
        df_treino[obrigatorias + candidates_extras],
        df_treino[target_col].values,
        candidates_extras,
    )

    # 6. Loop: 3 modelos × 8 N_features = 24 experimentos
    modelos = get_modelos_fase03()
    grid    = GRID_N_FEATURES
    total_exp = len(modelos) * len(grid)
    console.print(f"\n   [info]🚀 Rodando {total_exp} experimentos "
                  f"({len(modelos)} modelos × {len(grid)} N_features)...[/]\n")

    all_results = []
    features_por_n = {}       # n_real -> lista de features usadas (para documentar a melhor)
    preds_por_experimento = {}  # (model, n_real) -> y_pred_teste
    y_teste_por_n = {}           # n_real -> y_teste

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("Varredura", total=total_exp)

        for n_feats in grid:
            features_subset = selecionar_top_n(obrigatorias, ranking, n_feats)
            n_real = len(features_subset)
            features_por_n[n_real] = features_subset
            console.print(f"\n   [highlight]── N={n_feats} (real: {n_real} features) ──[/]")

            X_tr = df_treino[features_subset].values.astype(np.float32)
            y_tr = df_treino[target_col].values.astype(np.float32)
            X_te = df_teste[features_subset].values.astype(np.float32)
            y_te = df_teste[target_col].values.astype(np.float32)
            y_teste_por_n[n_real] = y_te

            for name, factory, _ in modelos:
                t0 = time.perf_counter()
                try:
                    model = factory()
                    model.fit(X_tr, y_tr)
                    y_pred_tr = model.predict(X_tr)
                    y_pred_te = model.predict(X_te)
                    elapsed   = time.perf_counter() - t0

                    m_pont = metricas_pontuais(y_te, y_pred_te, name)
                    residuos = y_tr - y_pred_tr
                    m_prob = metricas_prob_aproximadas(y_te, y_pred_te, residuos)

                    row = {
                        "model": name,
                        "n_features": n_real,
                        "tempo_s": round(elapsed, 2),
                        **m_pont, **m_prob,
                    }
                    all_results.append(row)
                    preds_por_experimento[(name, n_real)] = y_pred_te

                    console.print(
                        f"      [success]✔  {name:14s} "
                        f"MAE={m_pont['MAE']:7.3f} | R²={m_pont['R2']:7.4f} | "
                        f"sMAPE={m_pont['sMAPE_%']:6.2f}% | "
                        f"Spike_Bias={m_pont['Spike_Bias']:+7.2f} | "
                        f"CRPS={m_prob['CRPS_aprox']:6.2f} | [{elapsed:5.1f}s][/]"
                    )
                except Exception as e:
                    console.print(f"      [error]✘  {name}: {str(e)[:80]}[/]")
                prog.advance(task)

    # 7. Relatórios consolidados
    console.print()
    print_matriz_resultados(all_results, metric="MAE",        titulo="VARREDURA — MAE")
    console.print()
    print_matriz_resultados(all_results, metric="RMSE",       titulo="VARREDURA — RMSE")
    console.print()
    print_matriz_resultados(all_results, metric="R2",         titulo="VARREDURA — R²")
    console.print()
    print_matriz_resultados(all_results, metric="sMAPE_%",    titulo="VARREDURA — sMAPE %")
    console.print()
    print_matriz_resultados(all_results, metric="NRMSE_%",    titulo="VARREDURA — NRMSE %")
    console.print()
    print_matriz_resultados(all_results, metric="CRPS_aprox", titulo="VARREDURA — CRPS aprox")
    console.print()
    print_matriz_resultados(all_results, metric="Spike_MAE",  titulo="VARREDURA — Spike MAE")
    console.print()
    print_matriz_bias(all_results, metric="Spike_Bias",       titulo="VARREDURA — Spike Bias")

    # 8. Curvas elbow por modelo
    console.print()
    for m_name in ["XGBoost", "LightGBM", "RandomForest"]:
        print_curva_elbow(all_results, m_name, metric="MAE")
        console.print()

    # 9. Top 5 global
    df_all = pd.DataFrame(all_results).sort_values("MAE").reset_index(drop=True)
    top5   = df_all.head(5)

    t = Table(title="🏆 TOP 5 GLOBAL — (modelo, n_features) por menor MAE",
              box=box.DOUBLE_EDGE, border_style="yellow",
              show_lines=True, min_width=100)
    t.add_column("#",          justify="right",  min_width=4)
    t.add_column("Modelo",     style="bold",     min_width=16)
    t.add_column("N feats",    justify="right",  min_width=10)
    t.add_column("MAE",        justify="right",  min_width=10)
    t.add_column("RMSE",       justify="right",  min_width=10)
    t.add_column("R²",         justify="right",  min_width=10)
    t.add_column("Spike_Bias", justify="right",  min_width=12)
    t.add_column("CRPS aprox", justify="right",  min_width=12)
    t.add_column("Tempo (s)",  justify="right",  min_width=10)

    for i, row in top5.iterrows():
        pos   = i + 1
        style = "bold green" if pos == 1 else "white"
        t.add_row(
            f"{pos}",
            f"[{style}]{row['model']}[/{style}]",
            f"{int(row['n_features'])}",
            f"[acerto]{row['MAE']:.3f}[/acerto]" if pos == 1 else f"{row['MAE']:.3f}",
            f"{row['RMSE']:.3f}",
            f"{row['R2']:.4f}",
            f"{row['Spike_Bias']:+.2f}" if pd.notna(row['Spike_Bias']) else "—",
            f"{row['CRPS_aprox']:.2f}",
            f"{row['tempo_s']:.1f}",
        )
    console.print(t)

    # 10. Recomendação Fase 04
    best = top5.iloc[0]
    console.print()
    console.print(Panel(
        Text.assemble(
            ("🎯 RECOMENDAÇÃO PARA FASE 04 (TUNING):\n\n", "bold green"),
            ("   Melhor combinação: ", "muted"),
            (f"{best['model']} com {int(best['n_features'])} features\n", "bold"),
            ("   MAE atual: ", "muted"), (f"{best['MAE']:.3f}\n", "bold"),
            ("   RMSE atual: ", "muted"), (f"{best['RMSE']:.3f}\n", "bold"),
            ("   Próximo passo: ", "muted"),
            ("Optuna tuning nos top-3 modelos com esses N_features\n", "bold"),
            ("\n   Treino usado: histórico completo + jan/fev 2026\n", "muted"),
            ("   Teste avaliado: mar/abr 2026\n", "muted"),
        ),
        border_style="green", padding=(1, 4),
    ))

    # 11. Documentar/salvar as features da melhor combinação (para o TCC)
    n_best = int(best["n_features"])
    features_best = features_por_n.get(
        n_best, selecionar_top_n(obrigatorias, ranking, n_best)
    )
    console.print()
    df_features_best, arquivos_salvos = documentar_e_salvar_features(
        best, features_best, obrigatorias, ranking
    )

    return {
        "all_results"          : all_results,
        "best_combo"           : {
            "model"     : best["model"],
            "n_features": int(best["n_features"]),
            "MAE"       : best["MAE"],
        },
        "ranking_features"     : ranking,
        "obrigatorias"         : obrigatorias,
        "features_melhor"      : features_best,
        "df_features_melhor"   : df_features_best,
        "arquivos_salvos"      : arquivos_salvos,
        "preds_por_experimento": preds_por_experimento,
        "y_teste_por_n"        : y_teste_por_n,
    }

# ==============================================================================
fase_03_output = run_fase_03()

# ==============================================================================
# 🏆 EXPOSIÇÃO DO MELHOR MODELO (best_combo) — ponte para a coleta
#     automática de métricas, que precisa de y_true/y_pred_teste no global.
# ==============================================================================
console.print()
console.print(Panel(
    "[bold]Expondo y_true / y_pred_teste da melhor combinação "
    "(modelo + n_features) para o coletor de métricas...[/]",
    title="[header_v2]  PONTE FASE 03 → COLETA DE MÉTRICAS  [/header_v2]",
    border_style="purple4",
))

_best = fase_03_output["best_combo"]
_chave_best = (_best["model"], _best["n_features"])

if _chave_best in fase_03_output["preds_por_experimento"]:
    console.print(
        f"   [success]🏆 Melhor combinação: {_best['model']} "
        f"({_best['n_features']} features) — MAE={_best['MAE']:.3f}[/]"
    )
    # Variáveis expostas no escopo global para o coletor de métricas (Parte B)
    y_true       = fase_03_output["y_teste_por_n"][_best["n_features"]]
    y_pred_teste = fase_03_output["preds_por_experimento"][_chave_best]
    y_pred       = y_pred_teste  # alias, caso o coletor procure por este nome
else:
    console.print("   [error]✘  Não foi possível localizar as predições da "
                  "melhor combinação — coleta de métricas será pulada.[/]")

# ==============================================================================
# 🧾 COLETA_METRICAS_TCC_AUTO  (não editar — célula gerada automaticamente)
# ==============================================================================
# Coleta de métricas finais desta parte. Não altera o modelo; só lê o que está
# em memória (y_true/y_pred_teste da melhor combinação da Fase 03) e grava em
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

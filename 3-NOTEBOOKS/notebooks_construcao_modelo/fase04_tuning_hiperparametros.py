# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              TCC PLD — FASE 04: TUNING DE HIPERPARÂMETROS                     ║
# ║        Otimização Bayesiana (Optuna) das 3 Melhores Combinações da Fase 03    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  IDENTIFICAÇÃO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Autor       : Yago
#  Instituição : Universidade Federal do Ceará
#  Curso       : Engenharia Elétrica
#  Data        : Julho / 2026
#  Versão      : 1.0 — Arquivo Único (Tuning Optuna + Coleta de Métricas)
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DESCRIÇÃO / FUNÇÃO DO SCRIPT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Este script implementa a Fase 04 da modelagem preditiva do PLD (submercado
#  Nordeste) do TCC, unificando em um único arquivo Colab-friendly:
#
#    (A) TUNING DE HIPERPARÂMETROS — pega as 3 melhores combinações
#        (modelo × n_features) eleitas na varredura da Fase 03 e executa, para
#        cada uma, uma otimização bayesiana com Optuna (TPESampler). O objetivo
#        minimizado NÃO é o MAE do conjunto de teste (isso seria vazamento de
#        informação do período avaliado), e sim um MAE de validação cruzada
#        temporal calculada apenas dentro do treino.
#
#    (B) COLETA AUTOMÁTICA DE MÉTRICAS — ao final da execução, varre as
#        variáveis em memória, tenta casar um par (verdade, predição) e
#        grava/atualiza a linha correspondente à "Parte 4" no arquivo
#        metricas_recalculadas.csv, com deduplicação por (Parte, Tipo).
#
#  POR QUE 3 CONFIGURAÇÕES E NÃO 3 MODELOS:
#    A Fase 03 mostrou que o RandomForest venceu em DOIS pontos da grade de
#    features (50 e 70). Como o melhor resultado seguinte era um LightGBM com
#    60 features, o Top-3 real contém o RandomForest repetido. Por isso as
#    configurações são uma LISTA de experimentos, cada um com um rótulo único
#    (label = "Modelo_NNf"), e não mais um dicionário indexado pelo nome do
#    modelo — um dict silenciosamente descartaria a segunda entrada do RF.
#
#  RESULTADOS HERDADOS DA FASE 03 (baselines a serem batidos):
#      1º  RandomForest → 50 features   MAE = 46.524
#      2º  RandomForest → 70 features   MAE = 48.568
#      3º  LightGBM     → 60 features   MAE = 55.571
#
#  SPLIT ANTI-LEAKAGE:
#    Idêntico ao da Fase 03, para que a comparação baseline × tuning seja
#    justa. O treino é o histórico completo + Janeiro/Fevereiro de 2026; o
#    teste é Março e Abril de 2026. A função split_treino_teste() remove
#    ativamente do parquet de treino qualquer linha igual ou posterior a
#    março/2026, deduplica períodos presentes nos dois parquets (mantendo a
#    versão de teste) e faz uma checagem final de sanidade, emitindo alerta
#    explícito caso detecte vazamento.
#
#  CONTINUIDADE COM A FASE 03:
#    Feature engineering (TemporalFE), ranking de importância
#    (rank_features_lgb), seleção por N_features (selecionar_top_n), métricas
#    pontuais e métricas probabilísticas são deliberadamente os MESMOS da
#    Fase 03. Assim, qualquer diferença de MAE observada entre as duas fases
#    é atribuível exclusivamente aos hiperparâmetros, e não a mudanças no
#    pré-processamento ou no conjunto de features.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENTRADAS (INPUTS)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  O script não requer entradas manuais do usuário. Os parâmetros de controle
#  estão definidos nas constantes globais (logo após o tema do Rich):
#
#    BASE_DIR / INPUT_DIR    : caminho dos dados normalizados
#    CENARIO_DEFAULT         : "cenario_A_todos_anos"
#    STRATEGY_DEFAULT        : "HYBRID_AGGRESSIVE"
#    TESTE_ANO / TESTE_MESES : 2026 / [3, 4]  → período de teste fixo (Mar/Abr)
#    TREINO_EXTRA_MESES      : [1, 2]  → Jan/Fev/2026 movidos para o treino
#    CONFIGS_TUNING          : as 3 combinações herdadas da Fase 03
#    N_TRIALS                : 30  → tentativas do Optuna por configuração
#    CV_FOLDS                : 3   → dobras da validação temporal interna
#    N_CV_SAMPLES            : 15.000 → amostras mais recentes usadas na CV
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
#  1. Hiperparâmetros vencedores → BASE_DIR / fase_04_best_params.json
#     JSON chaveado pelo LABEL do experimento (e não pelo nome do modelo,
#     justamente porque o RandomForest aparece duas vezes). Este arquivo é o
#     insumo direto da Fase 05 (ensemble/stacking).
#
#  2. CSV de métricas → PASTA_SAIDA / metricas_recalculadas.csv
#     Uma linha por (Parte, Tipo) — aqui Parte=4 — com RMSE, R², MAE e MAPE.
#     Deduplicado automaticamente a cada execução.
#
#  3. Saída no terminal (Rich):
#     • Cabeçalho de fase (Painel)
#     • Progresso do Optuna por configuração, com melhor MAE_CV a cada 10 trials
#     • Tabela de melhores hiperparâmetros de cada experimento
#     • Tabela Top 5 trials de cada estudo (útil para discutir sensibilidade)
#     • Tabela comparativa Fase 03 (baseline) × Fase 04 (tuning), com Δ MAE
#     • Tabela de métricas probabilísticas pós-tuning
#     • Ranking final e painel de recomendação para a Fase 05
#
#  4. Objeto retornado por run_fase_04() (disponível em fase_04_output):
#     results_pontuais, results_prob, best_params_all, studies_all e os campos
#     best_label / best_model / best_n_features / best_MAE.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONFIGURAÇÕES LEVADAS AO TUNING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Label                 Modelo         N_features   MAE baseline (Fase 03)
#  RandomForest_50f      RandomForest   50           46.524
#  RandomForest_70f      RandomForest   70           48.568
#  LightGBM_60f          LightGBM       60           55.571
#
#  Total: 3 experimentos × 30 trials = 90 ajustes de modelo com CV interna.
#  O espaço de busca do XGBoost (objective_xgboost) permanece implementado no
#  arquivo, mesmo sem ser usado por CONFIGS_TUNING, para permitir reexecuções
#  rápidas caso a Fase 03 seja refeita e o XGBoost volte ao Top-3.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ETAPAS DO PROCESSO (resumo)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  N°  Etapa                        Estratégia                        Descrição
#  01  Carregamento                 read_parquet                      Lê TREINO_NORM e TESTE_NORM
#  02  Detecção automática          regex + tokens                    Localiza TARGET, KEY_* e REGIME_PRECO
#  03  Split anti-leakage           Filtro por AAAAMM + dedupe        Treino < mar/2026, teste = mar/abr/2026
#  04  Engenharia temporal          TemporalFE (24 derivadas)         Cíclicas, calendário, feriados — obrigatórias
#  05  Ranking de importância       LightGBM rápido (200 árvores)     Ordena candidatos extras por importância
#  06  Seleção por N_features       Obrigatórias + top-N do ranking   Reproduz o subconjunto usado na Fase 03
#  07  Tuning bayesiano             Optuna TPESampler (30 trials)     Um estudo independente por configuração
#  08  Validação interna            CV temporal expanding window      Objetivo = MAE médio das dobras
#  09  Treino final                 Melhores params + treino completo Reajusta o modelo com todo o histórico
#  10  Avaliação pontual            MAE/RMSE/MAPE/sMAPE/R²/NRMSE      Métricas completas em mar/abr 2026
#  11  Avaliação de picos           Spike_MAE / Spike_Bias            Erro em observações de PLD alto (> R$ 500)
#  12  Avaliação probabilística     Bootstrap de resíduos             Pinball, CRPS aproximado, cobertura 80%
#  13  Comparação com baseline      Δ MAE = tuning − Fase 03          Mede o ganho real da otimização
#  14  Persistência                 JSON de hiperparâmetros           Insumo direto para a Fase 05
#  15  Coleta de métricas           Varredura de arrays no globals()  Grava/atualiza metricas_recalculadas.csv
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONSIDERAÇÕES INICIAIS E OBSERVAÇÕES TÉCNICAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#  1. AMBIENTE DE EXECUÇÃO
#     O script foi desenvolvido e testado no Google Colab (Python 3.10+).
#     Instala automaticamente (via pip, silenciosamente) xgboost, lightgbm,
#     optuna, holidays e rich caso ainda não estejam presentes. Depende dos
#     arquivos TREINO_NORM.parquet e TESTE_NORM.parquet já normalizados por
#     etapas anteriores do pipeline.
#
#  2. O CONJUNTO DE TESTE NUNCA ENTRA NA OTIMIZAÇÃO (PONTO CRÍTICO)
#     A função objetivo de todos os estudos é cv_mae_temporal(), que particiona
#     APENAS o conjunto de treino. Março e Abril de 2026 são tocados uma única
#     vez por experimento, já com os hiperparâmetros congelados. Otimizar
#     diretamente o MAE de teste produziria números artificialmente bons e
#     inválidos para o TCC.
#
#  3. VALIDAÇÃO CRUZADA TEMPORAL (EXPANDING WINDOW)
#     cv_mae_temporal() não embaralha os dados: para cada dobra k, treina em
#     [0, fold_size·(k+1)) e valida no bloco imediatamente seguinte, sempre
#     avançando no tempo. Em séries de preço isso é obrigatório — um KFold
#     aleatório permitiria que o modelo "visse o futuro" ao prever o passado.
#     Por custo computacional, a CV usa apenas as N_CV_SAMPLES observações
#     mais recentes (15.000 ≈ 1,7 ano de dados horários), que são também as
#     mais representativas do regime atual do PLD.
#
#  4. TRATAMENTO DE FALHAS DENTRO DA CV
#     Combinações inviáveis de hiperparâmetros (por exemplo, restrições de
#     min_samples que esvaziam nós) fazem o fit levantar exceção. Nesses casos
#     a dobra é ignorada; se NENHUMA dobra concluir, a função devolve
#     float("inf"), o que faz o Optuna descartar naturalmente aquele trial.
#
#  5. LISTA EM VEZ DE DICIONÁRIO PARA AS CONFIGURAÇÕES
#     Como o RandomForest aparece duas vezes no Top-3, CONFIGS_TUNING é uma
#     lista e cada entrada recebe um label único ("RandomForest_50f",
#     "RandomForest_70f", "LightGBM_60f"). Esse label é a chave usada em
#     best_params_all, studies_all, nos títulos das tabelas e no JSON de
#     saída. Já OBJECTIVES continua indexado pelo TIPO de modelo, pois o
#     espaço de busca depende do algoritmo e não da quantidade de features.
#
#  6. CAPTURA EXPLÍCITA NO LAMBDA DO OPTUNA
#     Dentro do loop de trials, o lambda passado a study.optimize() recebe
#     X e y como argumentos padrão (X=X_tr, y=y_treino). Isso "congela" os
#     valores no momento da criação da função. Sem esse cuidado, o clássico
#     bug de closure em Python faria todos os lambdas enxergarem as variáveis
#     da ÚLTIMA iteração do loop de configurações.
#
#  7. MÉTRICAS COM FOCO EM PICOS E ASSIMETRIA
#     Além de MAE, RMSE, MAPE e R², são calculados sMAPE (evita explosão
#     quando o real é próximo de zero), NRMSE (RMSE normalizado pela média do
#     real) e, principalmente, Spike_MAE e Spike_Bias para observações com PLD
#     acima de SPIKE_ALTO_LIMIAR (R$ 500). Spike_Bias positivo indica que o
#     modelo SUBESTIMA os picos — o cenário mais arriscado para quem toma
#     decisão de exposição no mercado livre no Nordeste.
#
#  8. MÉTRICAS PROBABILÍSTICAS APROXIMADAS
#     Os modelos são pontuais (não nativamente probabilísticos), então a
#     incerteza é aproximada por bootstrap gaussiano dos resíduos de TREINO,
#     gerando quantis P10/P50/P90 e derivando Pinball Loss, CRPS aproximado e
#     cobertura do intervalo de 80%. Nesta fase são usadas 200 réplicas
#     (contra 150 da Fase 03), o que estabiliza um pouco os quantis sem custo
#     relevante. Uma cobertura muito abaixo de 80% indica intervalos
#     otimistas; muito acima, intervalos largos demais para serem úteis.
#
#  9. INTERPRETAÇÃO DO Δ MAE
#     A coluna "Δ MAE" das tabelas compara o MAE de teste pós-tuning com o
#     baseline da Fase 03 (mesma combinação modelo × n_features). Valores
#     negativos (verdes) indicam ganho real. É perfeitamente possível que
#     algum experimento piore: o tuning otimiza a CV interna, e um ganho ali
#     não se traduz automaticamente em ganho no período de teste — resultado
#     que, inclusive, merece discussão no texto do TCC.
#
# 10. COLETA AUTOMÁTICA DE MÉTRICAS (PARTE B)
#     Varre o namespace global por arrays numéricos com ao menos 10 elementos,
#     tentando casar uma variável de "verdade" (y_true, y_test, y_teste...)
#     com uma de "previsão" (y_pred_teste, y_pred_te, y_pred...). Detecta
#     ainda um possível caso de classificação binária, que não se aplica a
#     esta fase, puramente de regressão.
#
# 11. LIMITES CONHECIDOS
#     - N_TRIALS = 30 é um orçamento modesto para espaços de busca de 8 a 11
#       dimensões; serve para demonstrar o ganho da otimização, não para
#       esgotar o espaço. Aumentar para 100+ trials tende a melhorar
#       marginalmente, ao custo de tempo de execução.
#     - O MedianPruner está configurado mas tem efeito limitado, pois a função
#       objetivo não reporta valores intermediários (não há trial.report()
#       dentro da CV) — o pruning só atuaria com objetivos multi-step.
#     - As predições ficam dentro de run_fase_04(); se o coletor de métricas
#       (Parte B) não encontrar y_true/y_pred_teste no escopo global, basta
#       expô-los antes da célula de coleta, como foi feito na Fase 03.
#
# 12. REPRODUTIBILIDADE
#     RANDOM_STATE = 42 é usado nos modelos, no ranking de importância, no
#     sampler do Optuna (TPESampler(seed=...)) e no bootstrap de resíduos.
#     Com os mesmos dados de entrada, os hiperparâmetros vencedores e todas
#     as métricas são determinísticos entre execuções.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DEPENDÊNCIAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Biblioteca        Versão mínima  Finalidade
#  numpy             1.23           Operações numéricas e amostragem (bootstrap)
#  pandas            1.5            Manipulação de DataFrames e Parquet
#  scikit-learn      1.2            RandomForest e métricas de regressão
#  optuna            3.0            Otimização bayesiana dos hiperparâmetros
#  xgboost           —              Espaço de busca disponível (não usado no Top-3 atual)
#  lightgbm          —              Modelo tunado + ranking de importância
#  holidays          —              Feriados nacionais para engenharia temporal
#  rich              13.0           Tabelas, painéis e barras de progresso
#
#  Instalação automática (executada no início do script, se necessário):
#      pip install -q xgboost lightgbm optuna holidays rich
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
#              │   │           ├── TREINO_NORM.parquet      (entrada)
#              │   │           └── TESTE_NORM.parquet       (entrada)
#              │   └── fase_04_best_params.json             (saída)
#              └── codigos_modelagem/
#                  └── metricas_recalculadas.csv            (saída)
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  COMO EXECUTAR (Google Colab)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Célula 1 — Montar o Drive:
#      from google.colab import drive
#      drive.mount('/content/drive')
#
#  Célula 2 — Executar o script:
#      exec(open('fase04_tuning_hiperparametros_coleta_metricas.py').read())
#   ou simplesmente executar este arquivo como módulo principal.
#   As dependências (xgboost, lightgbm, optuna, holidays, rich) são instaladas
#   automaticamente na primeira execução, se necessário.
#
#  Tempo estimado: 3 experimentos × 30 trials, com CV de 3 dobras sobre 15.000
#  amostras. Em CPU do Colab, espere algo entre 20 e 60 minutos, dominado
#  pelos trials de RandomForest com n_estimators alto.
#
# ══════════════════════════════════════════════════════════════════════════════

import subprocess, sys
def _ensure(pkg, imp=None):
    """Importa o pacote; se falhar, instala silenciosamente via pip (Colab)."""
    try:
        __import__(imp or pkg)
    except ImportError:
        print(f"⏬ Instalando {pkg}...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

_ensure("xgboost"); _ensure("lightgbm")
_ensure("optuna"); _ensure("holidays"); _ensure("rich")

import warnings
warnings.filterwarnings("ignore")

import re
import time
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import optuna
# Silencia o log linha-a-linha do Optuna: o progresso é reportado pelo Rich.
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
# Tema visual do console (mesmas chaves das fases anteriores, para manter a
# identidade das saídas ao longo de todo o pipeline do TCC).
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
# CONSTANTES GLOBAIS DE CONTROLE
# ==============================================================================
BASE_DIR  = Path("/content/drive/MyDrive/TCC_PLD_Project/09-ESCRITA_TCC/PARTES_TCC/codigos/dados")
INPUT_DIR = BASE_DIR / "10_dados_normalizados"

CENARIO_DEFAULT   = "cenario_A_todos_anos"
STRATEGY_DEFAULT  = "HYBRID_AGGRESSIVE"
TARGET_CANONICAL  = "01_CCEE_NORDESTE_TARGET_PLD_HORA_NORDESTE"
SPIKE_ALTO_LIMIAR = 500.0    # R$/MWh acima do qual a observação conta como "pico"

# ── Datas de corte ─────────────────────────────────────────────────────────
TESTE_ANO          = 2026
TREINO_EXTRA_MESES = [1, 2]   # jan e fev/2026 → vão para TREINO
TESTE_MESES        = [3, 4]   # mar e abr/2026 → TESTE
# ───────────────────────────────────────────────────────────────────────────

RANDOM_STATE = 42   # semente única: modelos, ranking, sampler e bootstrap

# ── Parâmetros do tuning ───────────────────────────────────────────────────
N_TRIALS     = 30       # tentativas do Optuna por configuração
CV_FOLDS     = 3        # dobras da validação temporal interna
N_CV_SAMPLES = 15_000   # amostras mais recentes usadas na CV (custo × recência)

# ── Configurações levadas ao tuning (top-1, top-2 e top-3 da Fase 03) ───────
# Cada entrada é um experimento independente, com rótulo único (label).
# O label é usado como chave em dicionários, títulos e no JSON de saída — e é
# indispensável porque o RandomForest aparece DUAS vezes (50 e 70 features).
CONFIGS_TUNING = [
    {"model": "RandomForest", "n_features": 50, "baseline_mae": 46.524},
    {"model": "RandomForest", "n_features": 70, "baseline_mae": 48.568},
    {"model": "LightGBM",     "n_features": 60, "baseline_mae": 55.571},
]
for _c in CONFIGS_TUNING:
    _c["label"] = f"{_c['model']}_{_c['n_features']}f"
# ───────────────────────────────────────────────────────────────────────────

# ==============================================================================
# UTILITÁRIOS
# ==============================================================================
def safe_round(v, decimals=4):
    """Arredonda protegendo contra None, NaN e Inf (que quebram o json.dump)."""
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    return round(float(v), decimals)

def detectar_target(df):
    """Localiza a coluna-alvo: nome canônico ou, em fallback, por tokens."""
    if TARGET_CANONICAL in df.columns:
        return TARGET_CANONICAL
    for c in df.columns:
        cl = c.lower()
        if "nordeste" in cl and "target" in cl:
            return c
    raise KeyError("TARGET não encontrado")

def detectar_key_cols(df):
    """Mapeia KEY_ANO/MES/DIA/HORA, tolerando prefixos ou sufixos nos nomes."""
    mapping = {}
    for key in ["KEY_ANO", "KEY_MES", "KEY_DIA", "KEY_HORA"]:
        if key in df.columns:
            mapping[key] = key
        else:
            found = [c for c in df.columns if key.lower() in c.lower()]
            mapping[key] = found[0] if found else None
    return mapping

def detectar_regime_col(df):
    """Localiza REGIME_PRECO — variável derivada do próprio PLD, logo LEAKAGE."""
    cands = [c for c in df.columns if "REGIME_PRECO" in c.upper()]
    return cands[0] if cands else None

def load_parquet(path, label=""):
    """
    Lê o parquet e higieniza:
      - sanitiza caracteres que quebram o LightGBM em nomes de colunas;
      - troca ±inf por NaN e preenche tudo com 0.0 (dados já normalizados).
    """
    console.print(f"   [info]📂 {label}: {path.name}[/]")
    df = pd.read_parquet(path)
    df.columns = [re.sub(r'[":{},\[\]]', "_", str(c)) for c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mb = df.memory_usage(deep=True).sum() / 1024**2
    console.print(f"   [info]   Shape: {df.shape} | {mb:.1f} MB[/]")
    return df

def split_treino_teste(df_treino_raw, df_teste_raw, key_cols):
    """
    GARANTE:
        TREINO = tudo ATÉ fevereiro/2026  (histórico + jan/fev 2026)
        TESTE  = março + abril / 2026

    Robustez (mesma trava anti-leakage da Fase 03 corrigida):
      - Remove do parquet de TREINO qualquer linha do período de teste ou
        posterior (>= março/2026). Se o parquet de treino NÃO contiver 2026,
        nada é removido e o resultado é idêntico ao split antigo.
      - Deduplica jan/fev 2026 (parquet de treino x parquet de teste).
      - Faz checagem final de sanidade contra leakage.
    """
    k_ano  = key_cols.get("KEY_ANO")
    k_mes  = key_cols.get("KEY_MES")
    k_dia  = key_cols.get("KEY_DIA")
    k_hora = key_cols.get("KEY_HORA")

    # Chave numérica ano*100+mes para comparar períodos com segurança.
    # corte_treino = 202603 -> o treino só pode conter (ano*100+mes) < 202603
    corte_treino = TESTE_ANO * 100 + min(TESTE_MESES)   # 202603

    def _ano_mes(df):
        return df[k_ano].astype(int) * 100 + df[k_mes].astype(int)

    # 1) TESTE = mar + abr / 2026 (sempre vindo do parquet de teste)
    mask_teste = (
        (df_teste_raw[k_ano] == TESTE_ANO) &
        (df_teste_raw[k_mes].isin(TESTE_MESES))
    )
    df_teste_final = df_teste_raw[mask_teste].reset_index(drop=True)

    # 2) EXTRA = jan + fev / 2026 → migram do parquet de teste para o TREINO
    mask_extra = (
        (df_teste_raw[k_ano] == TESTE_ANO) &
        (df_teste_raw[k_mes].isin(TREINO_EXTRA_MESES))
    )
    df_extra = df_teste_raw[mask_extra]

    # 3) TREINO base: manter SOMENTE período < março/2026 (anti-leakage).
    #    Preserva todo o histórico e descarta mar/abr 2026 (ou meses
    #    posteriores) que por acaso estejam no parquet de treino.
    df_treino_base = df_treino_raw[_ano_mes(df_treino_raw) < corte_treino]
    n_leak = len(df_treino_raw) - len(df_treino_base)
    if n_leak > 0:
        console.print(
            f"   [warning]⚠  Removidas {n_leak:,} linhas >= mar/2026 do parquet "
            f"de TREINO (proteção contra leakage)[/]"
        )

    # 4) Concatena treino base + jan/fev 2026
    df_treino_final = pd.concat([df_treino_base, df_extra], ignore_index=True)

    # 5) Deduplica por chave temporal (jan/fev 2026 pode existir nos dois
    #    parquets). keep="last" prioriza a versão vinda do parquet de teste.
    chaves = [c for c in [k_ano, k_mes, k_dia, k_hora] if c]
    if chaves:
        antes = len(df_treino_final)
        df_treino_final = (
            df_treino_final.drop_duplicates(subset=chaves, keep="last")
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

    # 6) Validação explícita + checagem final de sanidade contra leakage
    treino_2026 = df_treino_final[df_treino_final[k_ano] == TESTE_ANO]
    ultimo_mes_treino = int(treino_2026[k_mes].max()) if len(treino_2026) else None
    meses_teste = sorted(
        df_teste_final[df_teste_final[k_ano] == TESTE_ANO][k_mes].unique().tolist()
    )
    console.print(
        f"   [muted]Treino → último mês 2026: {ultimo_mes_treino} "
        f"| Teste → meses 2026: {meses_teste}[/]"
    )

    leak = df_treino_final[
        (df_treino_final[k_ano] == TESTE_ANO) &
        (df_treino_final[k_mes].isin(TESTE_MESES))
    ]
    if len(leak) > 0:
        console.print(
            f"   [error]🚨 LEAKAGE: {len(leak):,} linhas de mar/abr 2026 no treino![/]"
        )
    else:
        console.print("   [success]🔒 Sem leakage: TREINO não contém mar/abr 2026[/]")

    return df_treino_final, df_teste_final

# ==============================================================================
# FEATURE ENGINEERING
# Idêntica à da Fase 03 — condição necessária para que a comparação
# baseline × tuning isole o efeito dos hiperparâmetros.
# ==============================================================================
class TemporalFE:
    """
    Gera as features temporais obrigatórias a partir das colunas-chave:
      • CYC_*  — codificação seno/cosseno (hora, dia da semana, dia do mês,
                 mês e dia do ano), preservando a continuidade circular
                 (23h e 0h ficam próximos, dezembro e janeiro também);
      • CAL_*  — flags de calendário (fim de semana, pico, madrugada, período
                 seco/chuvoso do NE, ano crítico, feriado, dia útil);
      • TEMP_* — trimestre, dia do ano, progresso do ano/mês e interações.
    """
    def __init__(self, use_holidays=True):
        self.use_holidays = use_holidays and HOLIDAYS_OK
        self.br_holidays  = (
            holidays.Brazil(years=range(2018, 2031)) if self.use_holidays else None
        )

    def _safe_date(self, a, m, d):
        """Constrói a data ou devolve None em combinações inválidas (31/02...)."""
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

        # Datas inválidas caem em 1900-01-01: não quebram o pipeline e ficam
        # facilmente identificáveis se aparecerem em volume relevante.
        datas_safe = [self._safe_date(a, m, d) or date(1900, 1, 1)
                      for a, m, d in zip(ano, mes, dia)]
        dia_semana = np.array([d.weekday()           for d in datas_safe], dtype=int)
        dia_do_ano = np.array([d.timetuple().tm_yday for d in datas_safe], dtype=int)

        # --- Codificação cíclica (evita a falsa descontinuidade 23h → 0h) ---
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

        # --- Flags de calendário e sazonalidade hidrológica do Nordeste ---
        df_out["CAL_is_fim_semana"]         = (dia_semana >= 5).astype(int)
        df_out["CAL_is_horario_pico"]       = ((hora >= 18) & (hora <= 21)).astype(int)
        df_out["CAL_is_madrugada"]          = ((hora >= 0)  & (hora <= 5)).astype(int)
        df_out["CAL_is_periodo_seco_NE"]    = np.isin(mes, [6,7,8,9,10,11]).astype(int)
        df_out["CAL_is_periodo_chuvoso_NE"] = np.isin(mes, [12,1,2,3]).astype(int)
        # Anos de crise hídrica/energética, marcados por PLD historicamente alto
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

        # --- Features temporais lineares e de progresso ---
        df_out["TEMP_trimestre"]       = ((mes - 1) // 3 + 1).astype(int)
        df_out["TEMP_dia_do_ano"]      = dia_do_ano
        df_out["TEMP_ano_normalizado"] = (ano - ano.min()).astype(int)
        df_out["TEMP_progresso_ano"]   = (dia_do_ano - 1) / 365.0
        df_out["TEMP_progresso_mes"]   = (dia - 1) / 31.0
        # Interação: pico de demanda em fim de semana tem comportamento próprio
        df_out["TEMP_weekend_pico"]    = (
            df_out["CAL_is_fim_semana"] * df_out["CAL_is_horario_pico"]
        )
        # Coluna auxiliar (prefixo "_" a exclui automaticamente das features)
        df_out["_DATA"] = pd.to_datetime(
            {"year": ano, "month": mes, "day": dia}, errors="coerce"
        ).dt.date
        return df_out

    def list_obrigatorias(self, key_cols):
        """Features que entram SEMPRE, qualquer que seja o N_features testado."""
        keys      = [v for v in key_cols.values() if v]
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


def rank_features_lgb(df_completo, candidates, target_col, random_state=RANDOM_STATE):
    """
    Ranqueia features por importância usando um LightGBM rápido.

    Recebe o df COMPLETO (com target) e a lista de candidatos.
    Usa apenas df[candidates] como X e df[target_col] como y.

    Calculado UMA única vez por execução e reaproveitado pelos 3 experimentos,
    exatamente como na Fase 03 — o que garante que o subconjunto de 50, 60 ou
    70 features seja o mesmo das duas fases.
    """
    ranker = lgb.LGBMRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        num_leaves=31, min_child_samples=20,
        random_state=random_state, n_jobs=-1, verbose=-1,
    )
    ranker.fit(df_completo[candidates].values, df_completo[target_col].values)
    return pd.DataFrame({
        "feature"   : candidates,
        "importance": ranker.feature_importances_,
    }).sort_values("importance", ascending=False)


def selecionar_top_n(obrigatorias, ranking_extras, n_total):
    """Garante as obrigatórias e completa com o topo do ranking até n_total."""
    if n_total <= len(obrigatorias):
        return list(obrigatorias)
    n_extras = n_total - len(obrigatorias)
    extras   = ranking_extras.head(n_extras)["feature"].tolist()
    return list(obrigatorias) + extras

# ==============================================================================
# MÉTRICAS
# ==============================================================================
def metricas_pontuais(y_true, y_pred, model_name=""):
    """Bateria completa de métricas de regressão + métricas de pico."""
    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    # Clipa o real em 1e-3 antes do MAPE: divisões por ~0 explodiriam a métrica
    y_safe = np.clip(y_true, 1e-3, None)
    mape  = mean_absolute_percentage_error(y_safe, y_pred) * 100
    r2    = r2_score(y_true, y_pred)
    me    = float(np.mean(y_true - y_pred))
    # sMAPE: simétrico, robusto quando y_true ~ 0 (comum no PLD em piso)
    smape = float(np.mean(
        2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    ) * 100)
    # NRMSE: RMSE normalizado pela média do real, comparável entre janelas
    nrmse = float(np.sqrt(mean_squared_error(y_true, y_pred))) / (
        np.mean(y_true) + 1e-8
    ) * 100
    # Métricas restritas aos picos (> R$ 500/MWh); exigem amostra mínima de 5
    spike_mask = y_true > SPIKE_ALTO_LIMIAR
    spike_mae  = (float(mean_absolute_error(y_true[spike_mask], y_pred[spike_mask]))
                  if spike_mask.sum() > 5 else np.nan)
    # Spike_Bias positivo = o modelo SUBESTIMA o pico (pior caso para decisão
    # de exposição no mercado livre)
    spike_bias = (float(np.mean(y_true[spike_mask] - y_pred[spike_mask]))
                  if spike_mask.sum() > 5 else np.nan)
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
    """Perda quantílica: penaliza assimetricamente sub e superestimação."""
    e = y_true - y_pred_q
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))

def coverage_score(y_true, lower, upper):
    """Fração de observações reais dentro do intervalo previsto."""
    return float(np.mean((y_true >= lower) & (y_true <= upper)))

def metricas_prob_aproximadas(y_true, y_pred, residuos_treino,
                               n_bootstrap=200, random_state=RANDOM_STATE):
    """
    Aproxima métricas probabilísticas para modelos pontuais.

    Estratégia: assume resíduo gaussiano com o desvio observado no TREINO e
    gera n_bootstrap réplicas ruidosas da predição, das quais se extraem os
    quantis P10/P50/P90. É uma aproximação (não captura heterocedasticidade),
    mas permite comparar incerteza entre modelos de forma consistente.
    """
    rng   = np.random.default_rng(random_state)
    n     = len(y_pred)
    sigma = np.std(residuos_treino)
    noise = rng.normal(0, sigma, size=(n_bootstrap, n))
    samps = y_pred[None, :] + noise
    p10   = np.quantile(samps, 0.10, axis=0)
    p50   = np.quantile(samps, 0.50, axis=0)
    p90   = np.quantile(samps, 0.90, axis=0)
    # CRPS aproximado pela média das 3 pinball losses (regra dos 2/3)
    crps  = (pinball_loss(y_true, p10, 0.10) +
             pinball_loss(y_true, p50, 0.50) +
             pinball_loss(y_true, p90, 0.90)) * 2.0 / 3.0
    return {
        "Pinball_Q10"    : safe_round(pinball_loss(y_true, p10, 0.10), 4),
        "Pinball_Q50"    : safe_round(pinball_loss(y_true, p50, 0.50), 4),
        "Pinball_Q90"    : safe_round(pinball_loss(y_true, p90, 0.90), 4),
        "CRPS_aprox"     : safe_round(crps, 4),
        "Coverage_80%"   : safe_round(coverage_score(y_true, p10, p90), 4),
        "Largura_P10_P90": safe_round(float(np.mean(p90 - p10)), 4),
    }

# ==============================================================================
# CV TEMPORAL INTERNO — função objetivo de TODOS os estudos do Optuna
# ==============================================================================
def cv_mae_temporal(model_factory, X, y, n_folds=CV_FOLDS,
                    n_samples=N_CV_SAMPLES, random_state=RANDOM_STATE):
    """
    CV temporal expanding window sobre os n_samples mais recentes.
    Preserva continuidade temporal — sem embaralhar, sem buracos.

    Layout das dobras (n_folds=3, fold_size = n // 4):
        fold 0: treino [0 .. 1F)   validação [1F .. 2F)
        fold 1: treino [0 .. 2F)   validação [2F .. 3F)
        fold 2: treino [0 .. 3F)   validação [3F .. 4F)

    O treino cresce e a validação está SEMPRE no futuro imediato — a mesma
    lógica do problema real (prever mar/abr a partir do passado).

    Trials que falham (hiperparâmetros inviáveis) são absorvidos: a dobra é
    descartada e, se nenhuma sobrar, devolve-se inf para o Optuna rejeitar.
    """
    n = len(X)
    # Usa apenas a cauda mais recente: reduz custo e prioriza o regime atual
    if n > n_samples:
        X = X[n - n_samples:]
        y = y[n - n_samples:]
        n = n_samples

    fold_size = n // (n_folds + 1)
    maes = []
    for k in range(n_folds):
        train_end = fold_size * (k + 1)
        val_start = train_end
        val_end   = val_start + fold_size
        if val_end > n:
            break
        X_tr, y_tr = X[:train_end], y[:train_end]
        X_va, y_va = X[val_start:val_end], y[val_start:val_end]
        # Dobras degeneradas não produzem estimativa confiável
        if len(X_tr) < 100 or len(X_va) < 10:
            continue
        try:
            m = model_factory()
            m.fit(X_tr, y_tr)
            maes.append(mean_absolute_error(y_va, m.predict(X_va)))
        except Exception:
            pass
    return float(np.mean(maes)) if maes else float("inf")

# ==============================================================================
# ESPAÇOS DE BUSCA (Optuna)
# Cada objective_* descreve o espaço de hiperparâmetros de um algoritmo e
# devolve o MAE de CV temporal — que o Optuna minimiza.
# ==============================================================================
def objective_randomforest(trial, X, y):
    """
    Espaço do RandomForest. Os eixos mais sensíveis aqui são max_depth (que
    controla diretamente o overfitting em série temporal) e max_features
    (que regula a descorrelação entre árvores).
    """
    params = dict(
        n_estimators      = trial.suggest_int("n_estimators",      100, 600, step=50),
        max_depth         = trial.suggest_int("max_depth",          5,  30),
        min_samples_split = trial.suggest_int("min_samples_split",  2,  20),
        min_samples_leaf  = trial.suggest_int("min_samples_leaf",   1,  10),
        max_features      = trial.suggest_categorical(
                                "max_features", ["sqrt", "log2", 0.3, 0.5, 0.8]
                            ),
        bootstrap         = trial.suggest_categorical("bootstrap", [True, False]),
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
    )
    return cv_mae_temporal(lambda: RandomForestRegressor(**params), X, y)


def objective_xgboost(trial, X, y):
    """
    Espaço do XGBoost. Mantido no arquivo mesmo fora do Top-3 atual, para
    permitir reexecução imediata caso a Fase 03 seja refeita.
    Escalas log em learning_rate, reg_alpha e reg_lambda: são parâmetros cujo
    efeito é multiplicativo, e amostrar linearmente desperdiçaria trials.
    """
    params = dict(
        n_estimators      = trial.suggest_int("n_estimators",       200, 800, step=50),
        max_depth         = trial.suggest_int("max_depth",           3,  10),
        learning_rate     = trial.suggest_float("learning_rate",     0.01, 0.20, log=True),
        subsample         = trial.suggest_float("subsample",         0.5,  1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree",  0.5,  1.0),
        colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.5,  1.0),
        reg_alpha         = trial.suggest_float("reg_alpha",         1e-4, 10.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda",        1e-4, 10.0, log=True),
        min_child_weight  = trial.suggest_int("min_child_weight",    1,   10),
        gamma             = trial.suggest_float("gamma",             0.0,  5.0),
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
        tree_method       = "hist",
        verbosity         = 0,
    )
    return cv_mae_temporal(lambda: xgb.XGBRegressor(**params), X, y)


def objective_lightgbm(trial, X, y):
    """
    Espaço do LightGBM. Aqui num_leaves é o principal controle de capacidade
    (o crescimento é leaf-wise, não level-wise), e min_child_samples atua como
    freio contra folhas construídas sobre poucas horas atípicas.
    """
    params = dict(
        n_estimators      = trial.suggest_int("n_estimators",      200, 800, step=50),
        max_depth         = trial.suggest_int("max_depth",          3,  10),
        learning_rate     = trial.suggest_float("learning_rate",    0.01, 0.20, log=True),
        num_leaves        = trial.suggest_int("num_leaves",         15, 127),
        subsample         = trial.suggest_float("subsample",        0.5,  1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5,  1.0),
        reg_alpha         = trial.suggest_float("reg_alpha",        1e-4, 10.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda",       1e-4, 10.0, log=True),
        min_child_samples = trial.suggest_int("min_child_samples",  5,   50),
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
        verbose           = -1,
    )
    return cv_mae_temporal(lambda: lgb.LGBMRegressor(**params), X, y)


# Objetivos indexados pelo TIPO de modelo (não pelo label): o espaço de busca
# depende do algoritmo, não da quantidade de features do experimento.
OBJECTIVES = {
    "RandomForest": objective_randomforest,
    "XGBoost"     : objective_xgboost,
    "LightGBM"    : objective_lightgbm,
}


def build_final_model(model_name, best_params):
    """Instancia o modelo definitivo com os hiperparâmetros vencedores."""
    if model_name == "RandomForest":
        return RandomForestRegressor(**best_params)
    elif model_name == "XGBoost":
        return xgb.XGBRegressor(**best_params)
    elif model_name == "LightGBM":
        return lgb.LGBMRegressor(**best_params)
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")

# ==============================================================================
# DISPLAYS (Rich) — apenas apresentação; nenhuma métrica é calculada aqui
# ==============================================================================
def print_header():
    """Painel de abertura com o resumo da configuração da fase."""
    labels_txt = " | ".join(
        f"{c['model']}({c['n_features']}f)" for c in CONFIGS_TUNING
    )
    console.print(Panel(
        Text.assemble(
            ("🧪 FASE 04 — TUNING DE HIPERPARÂMETROS (Optuna)\n", "bold green"),
            ("   Otimização bayesiana nas 3 melhores combinações da Fase 03\n", "muted"),
            (f"   {labels_txt}\n", "muted"),
            (f"   N_TRIALS={N_TRIALS} | CV_FOLDS={CV_FOLDS} | CV_SAMPLES={N_CV_SAMPLES:,}\n", "muted"),
            ("   Treino: histórico + jan/fev 2026 | Teste: mar/abr 2026\n", "muted"),
            (f"   Cenário: {CENARIO_DEFAULT} | Strategy: {STRATEGY_DEFAULT}\n", "muted"),
        ),
        title="[header_v2]  TCC PLD — EXPERIMENTO ITERATIVO  [/header_v2]",
        border_style="purple4", padding=(1, 4),
    ))


def print_tabela_comparativa(results):
    """
    Tabela-chave da fase: MAE antes (Fase 03) × depois (tuning), com Δ MAE.
    Δ negativo (verde) = o tuning melhorou aquela combinação.
    """
    t = Table(
        title="📊 COMPARATIVO FASE 03 (baseline) vs FASE 04 (tuning)",
        box=box.DOUBLE_EDGE, border_style="cyan",
        show_lines=True, min_width=130,
    )
    t.add_column("Modelo",       style="bold",    min_width=16)
    t.add_column("N feats",      justify="right", min_width=8)
    t.add_column("MAE baseline", justify="right", min_width=14)
    t.add_column("MAE tuning",   justify="right", min_width=14)
    t.add_column("Δ MAE",        justify="right", min_width=10)
    t.add_column("RMSE",         justify="right", min_width=10)
    t.add_column("R²",           justify="right", min_width=10)
    t.add_column("CRPS aprox",   justify="right", min_width=12)
    t.add_column("Spike MAE",    justify="right", min_width=12)
    t.add_column("Tempo tuning", justify="right", min_width=14)

    best_mae = min(r["MAE"] for r in results if r.get("MAE") is not None)

    for r in results:
        mae_v    = r.get("MAE")
        baseline = r.get("baseline_mae", 0)
        delta    = mae_v - baseline if mae_v is not None else None
        d_st     = "acerto" if (delta is not None and delta < 0) else "metric_bad"
        delta_str = (
            f"[{d_st}]{delta:+.3f}[/{d_st}]" if delta is not None else "—"
        )
        mae_st = "acerto" if mae_v == best_mae else "white"
        spike  = r.get("Spike_MAE")
        t.add_row(
            r["model"],
            str(r.get("n_features", "?")),
            f"{baseline:.3f}",
            f"[{mae_st}]{mae_v:.3f}[/{mae_st}]" if mae_v is not None else "—",
            delta_str,
            f"{r.get('RMSE', 0):.3f}",
            f"{r.get('R2', 0):.4f}",
            f"{r.get('CRPS_aprox', 0):.2f}",
            f"{spike:.2f}" if spike is not None and not np.isnan(spike) else "—",
            f"{r.get('tempo_tuning_s', 0):.0f}s",
        )
    console.print(t)


def print_tabela_prob(results_prob):
    """
    Métricas probabilísticas pós-tuning.
    Cobertura é destacada em verde quando fica a menos de 5 p.p. dos 80%
    nominais e em vermelho quando se afasta mais de 20 p.p. — nesse caso o
    intervalo não deve ser usado para decisão.
    """
    t = Table(
        title="📊 MÉTRICAS PROBABILÍSTICAS PÓS-TUNING",
        box=box.ROUNDED, border_style="magenta",
        show_lines=True, min_width=120,
    )
    t.add_column("Modelo",     style="bold",    min_width=16)
    t.add_column("N feats",    justify="right", min_width=8)
    t.add_column("Pin Q10",    justify="right", min_width=10)
    t.add_column("Pin Q50",    justify="right", min_width=10)
    t.add_column("Pin Q90",    justify="right", min_width=10)
    t.add_column("CRPS aprox", justify="right", min_width=12)
    t.add_column("Cov 80%",    justify="right", min_width=10)
    t.add_column("Largura",    justify="right", min_width=10)

    best_crps = min(r["CRPS_aprox"] for r in results_prob if r.get("CRPS_aprox"))

    for r in results_prob:
        crps_v  = r.get("CRPS_aprox")
        crps_st = "acerto" if crps_v == best_crps else "white"
        cov     = r.get("Coverage_80%", 0) or 0
        cov_str = f"{cov*100:.1f}%"
        if abs(cov - 0.80) < 0.05:
            cov_str = f"[acerto]{cov_str}[/acerto]"
        elif abs(cov - 0.80) > 0.20:
            cov_str = f"[metric_bad]{cov_str}[/metric_bad]"
        t.add_row(
            r["model"],
            str(r.get("n_features", "?")),
            f"{r.get('Pinball_Q10', 0):.2f}",
            f"{r.get('Pinball_Q50', 0):.2f}",
            f"{r.get('Pinball_Q90', 0):.2f}",
            f"[{crps_st}]{crps_v:.2f}[/{crps_st}]" if crps_v else "—",
            cov_str,
            f"{r.get('Largura_P10_P90', 0):.1f}",
        )
    console.print(t)


def print_best_params(label, best_params, study):
    """Hiperparâmetros vencedores de um experimento + MAE de CV atingido."""
    t = Table(
        title=f"🔧 MELHORES HIPERPARÂMETROS — {label}",
        box=box.ROUNDED, border_style="yellow",
        show_lines=False, min_width=60,
    )
    t.add_column("Parâmetro", style="bold cyan",  min_width=28)
    t.add_column("Valor",     style="bold white", min_width=20)
    for k, v in best_params.items():
        t.add_row(k, str(v))
    t.add_row("─" * 26, "─" * 18)
    t.add_row(
        "[muted]MAE CV (objetivo)[/muted]",
        f"[acerto]{study.best_value:.4f}[/acerto]",
    )
    t.add_row("[muted]Trials completados[/muted]", str(len(study.trials)))
    console.print(t)


def print_optuna_history(study, label, top_n=5):
    """
    Top-N trials do estudo. Útil no TCC para discutir a SENSIBILIDADE do
    modelo: se os 5 melhores trials têm MAE muito próximo com parâmetros
    bem diferentes, a superfície de otimização é plana e o ganho do tuning
    é pequeno; se há forte separação, os hiperparâmetros importam de fato.
    Exibe no máximo 5 colunas de parâmetros para não estourar a largura.
    """
    df_t = study.trials_dataframe().dropna(subset=["value"])
    df_t = df_t.sort_values("value").head(top_n)
    param_cols = [c for c in df_t.columns if c.startswith("params_")][:5]

    t = Table(
        title=f"📉 TOP {top_n} TRIALS — {label}",
        box=box.SIMPLE, border_style="dim",
        show_lines=False, min_width=50,
    )
    t.add_column("Trial",  justify="right", min_width=8)
    t.add_column("MAE CV", justify="right", min_width=12)
    for col in param_cols:
        t.add_column(col.replace("params_", ""), justify="right", min_width=10)

    for _, row in df_t.iterrows():
        cells = [str(int(row["number"])), f"{row['value']:.4f}"]
        for col in param_cols:
            cells.append(str(round(row[col], 4)) if pd.notna(row[col]) else "—")
        t.add_row(*cells)
    console.print(t)

# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================
def run_fase_04():
    print_header()

    # 1. Dados
    console.print("\n   [info]📂 Carregando dados...[/]")
    treino_path   = INPUT_DIR / CENARIO_DEFAULT / STRATEGY_DEFAULT / "TREINO_NORM.parquet"
    teste_path    = INPUT_DIR / CENARIO_DEFAULT / STRATEGY_DEFAULT / "TESTE_NORM.parquet"
    df_treino_raw = load_parquet(treino_path, "TREINO_NORM")
    df_teste_raw  = load_parquet(teste_path,  "TESTE_NORM")

    # 2. Detecção automática de colunas especiais
    target_col = detectar_target(df_treino_raw)
    key_cols   = detectar_key_cols(df_treino_raw)
    regime_col = detectar_regime_col(df_treino_raw)
    console.print(f"   [info]🎯 TARGET: {target_col}[/]")
    if regime_col:
        # REGIME_PRECO é derivado do próprio PLD → usar como feature é leakage
        console.print(f"   [info]🚨 REGIME_PRECO removido (leakage): {regime_col}[/]")

    # 3. Split: treino até fev/2026, teste = mar+abr/2026 (com trava anti-leakage)
    df_treino, df_teste = split_treino_teste(df_treino_raw, df_teste_raw, key_cols)

    # 4. Feature engineering temporal (aplicada aos dois conjuntos)
    console.print("\n   [info]🗓  Engenharia temporal...[/]")
    fe           = TemporalFE(use_holidays=True)
    df_treino    = fe.transform(df_treino, key_cols)
    df_teste     = fe.transform(df_teste,  key_cols)
    obrigatorias = [f for f in fe.list_obrigatorias(key_cols) if f in df_treino.columns]
    n_keys  = sum(1 for v in key_cols.values() if v and v in df_treino.columns)
    n_deriv = len(obrigatorias) - n_keys
    console.print(
        f"   [info]🔒 Obrigatórias: {len(obrigatorias)} "
        f"({n_keys} KEY + {n_deriv} derivadas)[/]"
    )

    # 5. Candidatos extras e ranking de importância (calculado uma única vez)
    console.print("   [info]📊 Ranqueando features por importância (LightGBM)...[/]")
    exclude = {target_col}
    if regime_col:
        exclude.add(regime_col)
    exclude.update([c for c in df_treino.columns if c.startswith("_")])  # auxiliares
    exclude.update(obrigatorias)                                        # já garantidas
    candidates_extras = [
        c for c in df_treino.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df_treino[c])
    ]
    ranking = rank_features_lgb(df_treino, candidates_extras, target_col)
    console.print(f"   [info]🎲 Candidatos extras ranqueados: {len(candidates_extras)}[/]")

    y_treino = df_treino[target_col].values.astype(np.float32)
    y_teste  = df_teste[target_col].values.astype(np.float32)

    # 6. Tuning Optuna por configuração
    #    Percorre uma LISTA (e não um dict) → permite RandomForest repetido
    results_pontuais = []
    results_prob     = []
    best_params_all  = {}
    studies_all      = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as prog:

        for cfg in CONFIGS_TUNING:
            model_name   = cfg["model"]
            n_feats      = cfg["n_features"]
            baseline_mae = cfg["baseline_mae"]
            label        = cfg["label"]

            console.print(
                f"\n   [highlight]━━━ {label} "
                f"| {n_feats} features "
                f"| baseline MAE={baseline_mae:.3f} ━━━[/]"
            )

            # Reconstrói exatamente o subconjunto de features da Fase 03
            features_subset = selecionar_top_n(obrigatorias, ranking, n_feats)
            n_real = len(features_subset)
            X_tr   = df_treino[features_subset].values.astype(np.float32)
            X_te   = df_teste[features_subset].values.astype(np.float32)

            objective_fn = OBJECTIVES[model_name]
            task = prog.add_task(
                f"Optuna {label} ({N_TRIALS} trials)", total=N_TRIALS
            )

            t0    = time.perf_counter()
            # Estudo independente por configuração; seed fixa → reprodutível
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=10, n_warmup_steps=2
                ),
            )

            # Loop trial a trial (n_trials=1 por chamada) apenas para poder
            # atualizar a barra de progresso do Rich a cada tentativa.
            for trial_num in range(N_TRIALS):
                # Captura explícita das variáveis no lambda para evitar closure bug:
                # sem X=X_tr, y=y_treino, todos os lambdas apontariam para o
                # último valor assumido pelas variáveis no loop externo.
                study.optimize(
                    lambda trial, X=X_tr, y=y_treino: objective_fn(trial, X, y),
                    n_trials=1, show_progress_bar=False,
                )
                prog.advance(task)
                if (trial_num + 1) % 10 == 0:
                    console.print(
                        f"      [muted]Trial {trial_num+1}/{N_TRIALS} "
                        f"| best MAE_CV={study.best_value:.4f}[/]"
                    )

            tempo_tuning = time.perf_counter() - t0
            best_params  = study.best_params

            console.print(
                f"   [success]✔  Optuna concluído: "
                f"MAE_CV={study.best_value:.4f} | {tempo_tuning:.0f}s[/]"
            )

            # Treino final: hiperparâmetros congelados + TODO o treino
            # (a CV usou só as 15k amostras mais recentes; aqui vai o histórico
            # inteiro, que é o modelo efetivamente avaliado em mar/abr 2026)
            console.print(f"   [info]🏋️  Treinando modelo final (full treino)...[/]")
            final_model = build_final_model(model_name, best_params)
            final_model.fit(X_tr, y_treino)
            y_pred_tr = final_model.predict(X_tr)
            y_pred_te = final_model.predict(X_te)

            # Métricas pontuais no período de teste (primeiro e único contato)
            m_pont = metricas_pontuais(y_teste, y_pred_te, model_name)
            m_pont["label"]          = label
            m_pont["n_features"]     = n_real
            m_pont["baseline_mae"]   = baseline_mae
            m_pont["tempo_tuning_s"] = round(tempo_tuning, 1)
            m_pont["CRPS_aprox"]     = None  # preenchido abaixo
            results_pontuais.append(m_pont)

            # Métricas probabilísticas: sigma vem dos resíduos de TREINO
            residuos = y_treino - y_pred_tr
            m_prob   = metricas_prob_aproximadas(y_teste, y_pred_te, residuos)
            m_prob["model"]      = model_name
            m_prob["label"]      = label
            m_prob["n_features"] = n_real
            results_prob.append(m_prob)

            # Injeta CRPS no pontual para a tabela comparativa
            m_pont["CRPS_aprox"] = m_prob["CRPS_aprox"]

            # Guardado por LABEL (não por modelo): o RF aparece 2x
            best_params_all[label] = best_params
            studies_all[label]     = study

            console.print(
                f"   [success]✔  TESTE: "
                f"MAE={m_pont['MAE']:.3f} | "
                f"R²={m_pont['R2']:.4f} | "
                f"CRPS={m_prob['CRPS_aprox']:.2f}[/]"
            )

            console.print()
            print_best_params(label, best_params, study)
            print_optuna_history(study, label, top_n=5)
            console.print()

    # 7. Relatórios finais consolidados
    console.print()
    print_tabela_comparativa(results_pontuais)
    console.print()
    print_tabela_prob(results_prob)

    # 8. Ranking final (menor MAE de teste vence)
    df_res = pd.DataFrame(results_pontuais).sort_values("MAE").reset_index(drop=True)
    best_r = df_res.iloc[0]

    t = Table(
        title="🏆 RANKING FINAL FASE 04",
        box=box.DOUBLE_EDGE, border_style="yellow",
        show_lines=True, min_width=90,
    )
    t.add_column("#",          justify="right", min_width=4)
    t.add_column("Modelo",     style="bold",    min_width=16)
    t.add_column("N feats",    justify="right", min_width=8)
    t.add_column("MAE",        justify="right", min_width=12)
    t.add_column("R²",         justify="right", min_width=10)
    t.add_column("Δ baseline", justify="right", min_width=14)

    for i, row in df_res.iterrows():
        pos      = i + 1
        baseline = row.get("baseline_mae", 0)
        delta    = row["MAE"] - baseline
        d_st     = "acerto" if delta < 0 else "metric_bad"
        m_st     = "bold green" if pos == 1 else "white"
        t.add_row(
            str(pos),
            f"[{m_st}]{row['model']}[/{m_st}]",
            str(int(row["n_features"])),
            f"[{m_st}]{row['MAE']:.3f}[/{m_st}]",
            f"{row['R2']:.4f}",
            f"[{d_st}]{delta:+.3f}[/{d_st}]",
        )
    console.print(t)

    # 9. Recomendação para a Fase 05
    best_baseline = best_r.get("baseline_mae", 0)
    console.print()
    console.print(Panel(
        Text.assemble(
            ("🎯 RECOMENDAÇÃO PARA FASE 05 (ENSEMBLE / STACKING):\n\n", "bold green"),
            ("   Melhor modelo tuned: ", "muted"),
            (f"{best_r['model']} ({int(best_r['n_features'])} features)\n", "bold"),
            ("   MAE após tuning: ", "muted"),
            (f"{best_r['MAE']:.3f}  ", "bold"),
            (f"(Δ={best_r['MAE'] - best_baseline:+.3f} vs Fase 03)\n", "muted"),
            ("   Próximos passos sugeridos:\n", "muted"),
            ("     1. Ensemble (Voting/Stacking) dos 3 modelos tuned\n", "bold"),
            ("     2. Análise de resíduos e sub-períodos (spike analysis)\n", "bold"),
            ("     3. Feature importance pós-tuning para interpretabilidade\n", "bold"),
            ("\n   Treino usado: histórico completo + jan/fev 2026\n", "muted"),
            ("   Teste avaliado: mar/abr 2026\n", "muted"),
        ),
        border_style="green", padding=(1, 4),
    ))

    # 10. Persistência dos hiperparâmetros (insumo direto da Fase 05)
    #     Chaveado pelo LABEL, pois o RandomForest aparece 2x — um dict por
    #     nome de modelo perderia silenciosamente uma das configurações.
    params_path = BASE_DIR / "fase_04_best_params.json"
    try:
        payload = {}
        for label, ps in best_params_all.items():
            # Converte tipos numpy → float nativo: np.int64 não é serializável
            payload[label] = {
                p: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for p, v in ps.items()
            }
        with open(params_path, "w") as f:
            json.dump(payload, f, indent=2)
        console.print(f"\n   [success]💾 Params salvos em: {params_path}[/]")
    except Exception as e:
        # Falha ao salvar não deve invalidar a execução inteira
        console.print(f"\n   [warning]⚠  Não foi possível salvar params: {e}[/]")

    return {
        "results_pontuais": results_pontuais,
        "results_prob"    : results_prob,
        "best_params_all" : best_params_all,
        "studies_all"     : studies_all,
        "best_label"      : best_r["label"],
        "best_model"      : best_r["model"],
        "best_n_features" : int(best_r["n_features"]),
        "best_MAE"        : best_r["MAE"],
    }

# ==============================================================================
# EXECUÇÃO
# ==============================================================================
fase_04_output = run_fase_04()


# ==============================================================================
# 🧾 COLETA_METRICAS_TCC_AUTO  (não editar — célula gerada automaticamente)
# ==============================================================================
# Coleta de métricas finais desta parte. Não altera o modelo; só lê o que está
# em memória e grava em metricas_recalculadas.csv.
import numpy as _np, pandas as _pd, os as _os
PARTE = 4
PASTA_SAIDA = r"/content/drive/MyDrive/TCC_PLD_Project/09-ESCRITA_TCC/PARTES_TCC/codigos/codigos_modelagem"

def _arr(v):
    """Converte a variável em vetor float 1-D; devolve None se não for possível."""
    try:
        a = _np.asarray(v, dtype=float).ravel()
        return a if a.size else None
    except Exception:
        return None

# Varre o namespace global atrás de arrays numéricos com ao menos 10 elementos
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
# Tenta casar, por ordem de preferência, um nome de "verdade" com um de "previsão"
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
        _nz = _np.abs(_a) > 1e-6   # protege o MAPE contra divisão por zero
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
    # Caso as predições tenham ficado presas dentro de run_fase_04(): basta
    # expor y_true / y_pred_teste no escopo global antes desta célula.
    print(f"[REG] Parte {PARTE}: não encontrei par real/predito de regressão. "
          f"Arrays: {[(k,v.size) for k,v in _cands.items()][:12]}")

# ---------- CLASSIFICAÇÃO (se houver score de probabilidade + verdade binária) ----------
# Não se aplica à Fase 04 (puramente regressão), mas o bloco é mantido para
# manter todas as células de coleta idênticas entre as partes do TCC.
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
# Reexecutar o script SUBSTITUI a linha da Parte 4 em vez de duplicá-la.
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

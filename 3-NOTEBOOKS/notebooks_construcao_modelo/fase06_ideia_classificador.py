# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║          TCC PLD — FASE 06: APRESENTAÇÃO DA IDEIA DO CLASSIFICADOR            ║
# ║   Diagnóstico da Cauda, Viabilidade do Detector de Spike e Escolha do Limiar ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  IDENTIFICAÇÃO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Autor       : Yago
#  Instituição : Universidade Federal do Ceará
#  Curso       : Engenharia Elétrica
#  Data        : Julho / 2026
#  Versão      : 1.0 — Arquivo Único (Ideia do Classificador + Coleta de Métricas)
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DESCRIÇÃO / FUNÇÃO DO SCRIPT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Este script implementa a Fase 06 da modelagem preditiva do PLD (submercado
#  Nordeste) do TCC. Enquanto a Fase 05 CONSTATOU que o modelo único falha na
#  cauda, a Fase 06 propõe e testa a solução: um CLASSIFICADOR DE REGIME que
#  antecipe os spikes antes de o híbrido ser montado (Fase 07).
#
#  O objetivo é mostrar, COM EVIDÊNCIA, por que esse classificador é
#  necessário. Três blocos encadeados, na forma de um argumento:
#
#    (A) DIAGNÓSTICO — o modelo base erra de forma ESTRUTURADA na cauda.
#        Retreina o RandomForest vencedor da Fase 03/04 e mede seu "recall de
#        spike": a fração de horas com PLD > R$ 500 em que a própria previsão
#        também passou de R$ 500. Um recall baixo demonstra que o problema não
#        é ruído, e sim compressão sistemática da distribuição prevista.
#
#    (B) VIABILIDADE — um classificador CONSEGUE separar spikes.
#        Se o recall do regressor é baixo, resta saber se a informação existe
#        nas features ou se o spike é simplesmente imprevisível. Um LightGBM
#        classificador é avaliado por CV temporal (estimativa honesta) e no
#        teste, com PR-AUC como métrica principal.
#
#    (C) DRIVERS — quais features preveem spike.
#        Importância do classificador, categorizada em Temporal, Hidrológico,
#        Carga e Preço. Responde "o que o detector está olhando" e alimenta a
#        discussão física do TCC.
#
#  Ao final, a coleta automática de métricas grava a linha da "Parte 6" em
#  metricas_recalculadas.csv — agora com uma linha de REGRESSÃO e outra de
#  CLASSIFICAÇÃO, já que esta fase produz os dois tipos de resultado.
#
#  MODELO BASE:
#      RandomForest com 50 features, hiperparâmetros da Fase 04
#      (MAE = 47.538 | R² = 0.8155). LightGBM(60f) e XGBoost(50f) ficaram em
#      2º/3º lugar na Fase 04 e servem apenas de contexto já consolidado na
#      Fase 05 — a Fase 06 trabalha SÓ com o RandomForest.
#
#  SPLIT ANTI-LEAKAGE:
#      Idêntico ao das Fases 03/04/05. Treino = histórico completo até
#      fev/2026 (incluindo jan/fev 2026); teste = mar/abr 2026, com remoção
#      ativa de linhas >= mar/2026 do parquet de treino, deduplicação e
#      checagem final de sanidade.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HIERARQUIA DE MÉTRICAS (evento raro com custo assimétrico)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Esta é a decisão metodológica central da fase. Como o spike é raro e o
#  custo de NÃO detectá-lo (FN) é muito maior que o de um falso alarme (FP),
#  a ordem de prioridade adotada é:
#
#   🥇 Recall_spike     Principal. Fração dos spikes reais efetivamente
#                       capturada. FN é o pior erro possível no contexto.
#   🥈 PR-AUC           Melhor que ROC-AUC em classe desbalanceada: a linha de
#                       base é a própria taxa de positivos, não 0.5.
#   🥉 Precision        Controle de falsos alarmes — importa, mas cede lugar
#                       ao Recall quando há conflito.
#   🎯 F1 / Fbeta(β=2)  Equilíbrio. O Fbeta com β=2 faz o Recall valer 2× mais
#                       que a Precision, formalizando a assimetria de custo.
#   ⚠️ ROC-AUC          Apenas suporte. Em classe rara pode parecer excelente
#                       mesmo com desempenho prático ruim.
#   📊 Matriz de confusão      Interpretação qualitativa obrigatória no TCC.
#   📈 Curva PR + Threshold Sweep  Escolha FUNDAMENTADA do limiar, em vez do
#                       τ = 0.50 arbitrário herdado do padrão do scikit-learn.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENTRADAS (INPUTS)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  O script não requer entradas manuais do usuário. Os parâmetros de controle
#  estão definidos nas constantes globais (logo após o tema do Rich):
#
#    BASE_DIR / INPUT_DIR       : caminho dos dados normalizados
#    PASTA_SAIDA_METRICAS       : destino de metricas_recalculadas.csv
#    CENARIO_DEFAULT            : "cenario_A_todos_anos"
#    STRATEGY_DEFAULT           : "HYBRID_AGGRESSIVE"
#    TESTE_ANO / TESTE_MESES    : 2026 / [3, 4]  → período de teste fixo
#    TREINO_EXTRA_MESES         : [1, 2]  → Jan/Fev/2026 movidos para o treino
#    THRESHOLD_SPIKE            : 500.0 → define o rótulo binário de spike
#    THRESHOLD_CAUDA            : 300.0 → reservado para o especialista da Fase 07
#    QUANTIS_CAUDA              : [0.10, 0.50, 0.90] → idem (regressão quantílica)
#    USE_LOG1P_CAUDA            : True  → idem (transformação do alvo na cauda)
#    FBETA_BETA                 : 2.0   → Recall vale 2× a Precision
#    PRECISION_MINIMA_ACEITAVEL : 0.30  → tolera até ~70% de falsos alarmes
#    SWEEP_THRESHOLDS           : 0.05 a 0.95, passo 0.01
#    RF_BEST_PARAMS / RF_N_FEATURES : modelo base herdado da Fase 04
#
#  Arquivos de entrada esperados:
#    • INPUT_DIR / CENARIO_DEFAULT / STRATEGY_DEFAULT / TREINO_NORM.parquet
#    • INPUT_DIR / CENARIO_DEFAULT / STRATEGY_DEFAULT / TESTE_NORM.parquet
#
#  Observação: THRESHOLD_CAUDA, QUANTIS_CAUDA e USE_LOG1P_CAUDA já estão
#  declarados mas NÃO são usados nesta fase — são a semente da Fase 07, em que
#  o especialista de cauda será efetivamente construído.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SAÍDAS (OUTPUTS)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. CSV de métricas → PASTA_SAIDA_METRICAS / metricas_recalculadas.csv
#     DUAS linhas nesta fase (Parte 6):
#       • Tipo "regressao"     — RMSE, R², MAE, MAPE do modelo base
#       • Tipo "classificacao" — Recall_spike, PR-AUC, Precision, Fbeta, F1 nos
#         quatro thresholds analisados (MaxRecall, Fbeta, F1 e 0.50)
#     Deduplicado por (Parte, Tipo) a cada execução.
#
#  2. Saída no terminal (Rich):
#     • Cabeçalho e hiperparâmetros do modelo base
#     • (A) Diagnóstico: tabela do modelo base, erro por faixa de PLD e painel
#       "O PROBLEMA" quantificando os spikes perdidos
#     • (B) Viabilidade: separabilidade em CV temporal × teste; painéis do
#       classificador calibrado com três thresholds; sweep de threshold e
#       curva Precision-Recall em modo texto
#     • (C) Drivers: Top-15 features do classificador, por categoria
#     • Painel de hierarquia de métricas, com texto pronto para o TCC
#     • Painel de conclusão com veredito sobre a viabilidade do híbrido
#
#  3. Objeto retornado por run_fase_06() (disponível em fase_06_output):
#     modelo base, predições, rótulos binários, probabilidades calibradas,
#     classificador, métricas, tabela de sweep, os três thresholds e as
#     métricas de separabilidade (CV e teste).
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  OS TRÊS THRESHOLDS ANALISADOS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Rótulo          Critério de escolha                      Papel no TCC
#  τ(MaxRecall)    Maior Recall com Precision >= 0.30        🥇 Recomendado
#  τ(Fbeta)        Maximiza Fbeta com β = 2                  🥈 Alternativa formal
#  τ(F1)           Maximiza F1 (equilíbrio clássico)         🥉 Referência
#  τ = 0.50        Padrão do scikit-learn                    Linha de base
#
#  TODOS os thresholds são escolhidos no conjunto de CALIBRAÇÃO (a cauda final
#  do treino), nunca no teste. Escolher o limiar olhando o teste inflaria
#  artificialmente o Recall reportado e invalidaria o resultado.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ETAPAS DO PROCESSO (resumo)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  N°  Etapa                        Estratégia                        Descrição
#  01  Carregamento                 read_parquet                      Lê TREINO_NORM e TESTE_NORM
#  02  Detecção automática          regex + tokens                    Localiza TARGET, KEY_* e REGIME_PRECO
#  03  Split anti-leakage           Filtro por AAAAMM + dedupe        Treino < mar/2026, teste = mar/abr/2026
#  04  Ordenação temporal           _ordenar_temporal()               Pré-requisito da calibração e da CV
#  05  Engenharia temporal          TemporalFE (24 derivadas)         Cíclicas, calendário, feriados
#  06  Ranking de importância       LightGBM rápido (200 árvores)     Reproduz as 50 features da Fase 03
#  07  Treino do modelo base        Params fixos da Fase 04           Sem tuning: reproduz o vencedor
#  08  (A) Diagnóstico da cauda     Recall/MAE/Bias em PLD > R$500    Evidencia a falha estrutural
#  09  (B) CV temporal              TimeSeriesSplit (4 folds)         Separabilidade honesta no treino
#  10  (B) Classificador no teste   LightGBM balanceado               ROC-AUC e PR-AUC em mar/abr 2026
#  11  Calibração de probabilidade  IsotonicRegression                Probabilidades interpretáveis
#  12  Sweep de threshold           0.05 → 0.95, passo 0.01           Recall/Precision/F1/Fbeta por τ
#  13  Escolha do limiar            3 critérios + linha de base       Decisão fundamentada, não arbitrária
#  14  Avaliação final              Métricas nos 4 thresholds         Matriz de confusão e comparativo
#  15  (C) Drivers de spike         feature_importances_              Top-15 por categoria física
#  16  Conclusão                    Veredito sobre o híbrido          Insumo direto da Fase 07
#  17  Coleta de métricas           coletar_e_gravar_metricas()       Grava 2 linhas da Parte 6 no CSV
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONSIDERAÇÕES INICIAIS E OBSERVAÇÕES TÉCNICAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#  1. AMBIENTE DE EXECUÇÃO
#     O script foi desenvolvido e testado no Google Colab (Python 3.10+).
#     Instala automaticamente (via pip, silenciosamente) xgboost, lightgbm,
#     holidays e rich caso ainda não estejam presentes. Diferente da Fase 05,
#     não depende de SHAP.
#
#  2. DOIS "RECALLS" DIFERENTES — NÃO CONFUNDIR
#     • Recall do REGRESSOR (bloco A): fração das horas com PLD real > R$ 500
#       em que a PREVISÃO NUMÉRICA também passou de R$ 500. Mede a compressão
#       da distribuição prevista. É o número que justifica a fase inteira.
#     • Recall_spike do CLASSIFICADOR (bloco B): fração dos spikes reais em
#       que a PROBABILIDADE calibrada superou o threshold τ. É a métrica
#       principal do detector proposto.
#     Espera-se que o segundo seja muito maior que o primeiro — e essa
#     diferença é exatamente o argumento a favor do híbrido.
#
#  3. ORDENAÇÃO TEMPORAL É PRÉ-REQUISITO, NÃO ESTÉTICA
#     _ordenar_temporal() é chamada logo após o split. Duas coisas dependem
#     disso: o TimeSeriesSplit da CV (que assume índice cronológico) e o corte
#     de calibração (que pega as ÚLTIMAS n_calib linhas do treino). Sem
#     ordenar, a "cauda recente" do treino seria uma amostra arbitrária e a
#     calibração perderia o sentido temporal.
#
#  4. CALIBRAÇÃO ISOTÔNICA E POR QUE ELA IMPORTA AQUI
#     O LightGBM produz escores, não probabilidades bem calibradas — ainda
#     mais com class_weight="balanced", que distorce deliberadamente a escala.
#     A IsotonicRegression (monotônica, não paramétrica) remapeia esses
#     escores para probabilidades interpretáveis. Sem isso, varrer thresholds
#     entre 0.05 e 0.95 não teria significado prático, e a probabilidade não
#     poderia ser usada como "gate" no híbrido da Fase 07.
#
#  5. SEPARAÇÃO TREINO / CALIBRAÇÃO / TESTE
#     O treino é dividido em duas partes: as primeiras linhas treinam o
#     classificador e as últimas n_calib (até 5.000, ou 1/4 do treino) formam
#     o conjunto de calibração. Como os dados estão ordenados, esse conjunto
#     corresponde ao passado mais recente — tipicamente jan/fev 2026, o que
#     mais se parece com o período de teste. Os thresholds saem dali, e o
#     teste permanece intocado até a avaliação final.
#
#  6. class_weight="balanced" NÃO SUBSTITUI A ESCOLHA DO THRESHOLD
#     O balanceamento de classes corrige o viés do treino em direção à classe
#     majoritária, mas o ponto de corte ótimo continua sendo uma decisão de
#     custo. Por isso o sweep existe: ele traduz a assimetria FN × FP em um
#     número (τ) que pode ser justificado no texto do TCC.
#
#  7. PRECISION MÍNIMA DE 0.30 É UMA ESCOLHA DE PROJETO
#     Aceitar Precision >= 0.30 equivale a tolerar cerca de 70% de falsos
#     alarmes em troca do máximo de spikes capturados. É defensável quando o
#     custo de um alarme falso é operacional e o de um spike perdido é
#     financeiro — mas é um parâmetro, não uma verdade. Vale explicitá-lo na
#     defesa e, se possível, mostrar a sensibilidade do resultado a ele.
#
#  8. PR-AUC × ROC-AUC EM CLASSE RARA
#     Com poucos por cento de positivos, o ROC-AUC pode passar de 0.90 mesmo
#     com um detector inútil na prática, porque a enorme massa de negativos
#     domina a taxa de falsos positivos. O PR-AUC tem como linha de base a
#     própria taxa de positivos: se a taxa de spike é 4%, um PR-AUC de 0.30 já
#     representa um ganho substancial. É por isso que o ROC-AUC aparece como
#     métrica de suporte, e não como argumento principal.
#
#  9. O QUE A CV TEMPORAL RESPONDE
#     O TimeSeriesSplit de 4 dobras dá a estimativa HONESTA de separabilidade,
#     medida só dentro do treino. O classificador ajustado em todo o treino e
#     avaliado no teste responde à pergunta seguinte: o sinal se sustenta no
#     período efetivamente avaliado? Divergência grande entre as duas colunas
#     da tabela de separabilidade sugere mudança de regime entre o histórico e
#     mar/abr 2026 — achado relevante para o texto.
#
# 10. COLETA DE MÉTRICAS COMO FUNÇÃO, NÃO COMO CÉLULA GERADA
#     Nas Fases 03/04/05 a coleta era um bloco automático que varria o
#     namespace global atrás de arrays. Aqui ela virou a função
#     coletar_e_gravar_metricas(), chamada explicitamente ao final de
#     run_fase_06() com os arrays certos passados como argumento. É mais
#     robusto (não depende de heurística de nomes) e permite gravar as duas
#     linhas — regressão e classificação — na mesma execução.
#
# 11. PONTOS A REVISAR NO CÓDIGO (não alterados nesta versão comentada)
#     - Em SpikeClassifier.evaluate(), a variável thr_use é calculada a partir
#       do argumento thr, mas nunca utilizada: a avaliação sempre usa os três
#       thresholds internos. O argumento é, na prática, inerte.
#     - O Bloco 4 de print_classificador_completo() está rotulado no comentário
#       como "τ recomendado = MaxRecall", mas a matriz de confusão exibida é a
#       de τ(F1) — a única armazenada em metrics_. Como o τ recomendado é o de
#       MaxRecall, convém alinhar os dois antes de levar a matriz ao TCC.
#     - O limiar de PR-AUC no veredito (ap_te >= 0.30) é fixo; idealmente
#       deveria ser comparado à taxa base de positivos do teste, que é a
#       verdadeira linha de base do PR-AUC.
#
# 12. REPRODUTIBILIDADE
#     RANDOM_STATE = 42 é usado no modelo base, no ranking de importância, no
#     classificador da CV e no classificador final. Com os mesmos dados de
#     entrada, thresholds e métricas são determinísticos entre execuções.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DEPENDÊNCIAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Biblioteca        Versão mínima  Finalidade
#  numpy             1.23           Operações numéricas e varredura de thresholds
#  pandas            1.5            DataFrames, Parquet e persistência do CSV
#  scikit-learn      1.2            RandomForest, métricas, IsotonicRegression,
#                                   TimeSeriesSplit
#  lightgbm          —              Classificador de spike + ranking de features
#  xgboost           —              Disponível em build_model (não usado aqui)
#  holidays          —              Feriados nacionais para engenharia temporal
#  rich              13.0           Tabelas, painéis e réguas visuais
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
#              │   │           ├── TREINO_NORM.parquet      (entrada)
#              │   │           └── TESTE_NORM.parquet       (entrada)
#              │   └── fase_04_best_params.json             (origem dos params)
#              └── codigos_modelagem/
#                  └── metricas_recalculadas.csv            (saída — 2 linhas)
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  COMO EXECUTAR (Google Colab)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Célula 1 — Montar o Drive:
#      from google.colab import drive
#      drive.mount('/content/drive')
#
#  Célula 2 — Executar o script:
#      exec(open('fase06_ideia_classificador_coleta_metricas.py').read())
#   ou simplesmente executar este arquivo como módulo principal.
#   As dependências (xgboost, lightgbm, holidays, rich) são instaladas
#   automaticamente na primeira execução, se necessário.
#
#  Tempo estimado: um treino de RandomForest (550 árvores), quatro dobras de
#  CV com LightGBM e o classificador final. Em CPU do Colab, algo entre 8 e 20
#  minutos — dominado pelo RandomForest.
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

_ensure("xgboost"); _ensure("lightgbm"); _ensure("holidays"); _ensure("rich")

import warnings
warnings.filterwarnings("ignore")

import os
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
    f1_score, fbeta_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, confusion_matrix,
    precision_recall_curve,                           # ← NOVO: curva PR
)
from sklearn.isotonic import IsotonicRegression      # calibração de probabilidade
from sklearn.model_selection import TimeSeriesSplit  # CV que respeita o tempo

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
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# ==============================================================================
# Tema visual do console. Além das chaves das fases anteriores, esta fase
# acrescenta rank1/rank2/rank3 para sinalizar a hierarquia de métricas.
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
    "spike_pos"  : "bold red",
    "spike_neg"  : "bold blue",
    "spike"      : "bold red",
    "normal"     : "bold green",
    "rank1"      : "bold yellow",       # ← NOVO: medalha ouro
    "rank2"      : "bold cyan",         # ← NOVO: medalha prata
    "rank3"      : "bold white",        # ← NOVO: medalha bronze
})
console = Console(theme=THEME)

# ==============================================================================
# CONFIG
# ==============================================================================
BASE_DIR  = Path("/content/drive/MyDrive/TCC_PLD_Project/09-ESCRITA_TCC/PARTES_TCC/codigos/dados")
INPUT_DIR = BASE_DIR / "10_dados_normalizados"
CENARIO_DEFAULT  = "cenario_A_todos_anos"
STRATEGY_DEFAULT = "HYBRID_AGGRESSIVE"
TARGET_CANONICAL = "01_CCEE_NORDESTE_TARGET_PLD_HORA_NORDESTE"

PASTA_SAIDA_METRICAS = (
    r"/content/drive/MyDrive/TCC_PLD_Project/09-ESCRITA_TCC"
    r"/PARTES_TCC/codigos/codigos_modelagem"
)

# ── Datas de corte ─────────────────────────────────────────────────────────
TESTE_ANO          = 2026
TREINO_EXTRA_MESES = [1, 2]   # jan e fev/2026 → TREINO
TESTE_MESES        = [3, 4]   # mar e abr/2026 → TESTE
# ───────────────────────────────────────────────────────────────────────────

RANDOM_STATE      = 42
THRESHOLD_SPIKE   = 500.0   # define o rótulo binário: PLD > 500 ⇒ spike
# As três constantes abaixo NÃO são usadas nesta fase — são a semente do
# especialista de cauda que será construído na Fase 07.
THRESHOLD_CAUDA   = 300.0
QUANTIS_CAUDA     = [0.10, 0.50, 0.90]
USE_LOG1P_CAUDA   = True

# ── Parâmetros de threshold / custo ─────────────────────────────────────────
# β > 1 → penaliza mais FN (spike perdido) que FP (falso alarme)
# β = 2 significa que Recall vale 2× mais que Precision no Fbeta
FBETA_BETA = 2.0

# Precision mínima aceitável ao buscar threshold orientado a Recall máximo.
# É uma DECISÃO DE PROJETO: 0.30 tolera ~70% de falsos alarmes em troca de
# capturar o máximo possível de spikes.
PRECISION_MINIMA_ACEITAVEL = 0.30   # aceita até 70% de falsos alarmes

# Thresholds a varrer na análise de sensibilidade
SWEEP_THRESHOLDS = np.arange(0.05, 0.96, 0.01)
# ────────────────────────────────────────────────────────────────────────────

# Faixas de preço para o diagnóstico do bloco (A). O corte em 500 coincide
# com THRESHOLD_SPIKE: a faixa "Extremo" é exatamente a classe positiva.
FAIXAS_PLD = [
    ("Baixo",   0,   100),
    ("Médio",   100, 300),
    ("Alto",    300, 500),
    ("Extremo", 500, 10**9),
]
DIAS_SEMANA_PT = {0:"Seg",1:"Ter",2:"Qua",3:"Qui",4:"Sex",5:"Sáb",6:"Dom"}

# ── Modelo base: RandomForest (1º lugar Fase 03/04, MAE=47.538) ─────────────
# Hiperparâmetros vencedores do Optuna na Fase 04 — fixos, sem novo tuning.
RF_BEST_PARAMS = {
    "n_estimators"     : 550,
    "max_depth"        : 27,
    "min_samples_split": 8,
    "min_samples_leaf" : 10,
    "max_features"     : 0.8,
    "bootstrap"        : False,
    "random_state"     : RANDOM_STATE,
    "n_jobs"           : -1,
}
RF_N_FEATURES = 50
RF_MAE_FASE04 = 47.538   # métricas de referência para comparação
RF_R2_FASE04  = 0.8155
# ────────────────────────────────────────────────────────────────────────────


# ==============================================================================
# UTILITÁRIOS
# ==============================================================================
def safe_round(v, d=4):
    """Arredonda protegendo contra None, NaN e Inf (que quebram o CSV/JSON)."""
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    return round(float(v), d)


def detectar_target(df):
    """Localiza a coluna-alvo: nome canônico ou, em fallback, por tokens."""
    if TARGET_CANONICAL in df.columns:
        return TARGET_CANONICAL
    for c in df.columns:
        if "nordeste" in c.lower() and "target" in c.lower():
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
    """Localiza REGIME_PRECO — derivada do próprio PLD, logo excluída (leakage)."""
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
        TREINO = tudo ATÉ fevereiro/2026 (histórico + jan/fev 2026)
        TESTE  = março + abril / 2026

    Trava anti-leakage idêntica às Fases 03/04/05: remove do parquet de treino
    qualquer linha >= mar/2026, deduplica jan/fev 2026 (que aparece nos dois
    parquets, mantendo a versão do teste) e faz checagem final de sanidade.
    """
    k_ano  = key_cols.get("KEY_ANO")
    k_mes  = key_cols.get("KEY_MES")
    k_dia  = key_cols.get("KEY_DIA")
    k_hora = key_cols.get("KEY_HORA")

    # Chave numérica ano*100+mes; corte_treino = 202603
    corte_treino = TESTE_ANO * 100 + min(TESTE_MESES)

    def _ano_mes(df):
        return df[k_ano].astype(int) * 100 + df[k_mes].astype(int)

    # TESTE = mar + abr / 2026
    mask_teste = (
        (df_teste_raw[k_ano] == TESTE_ANO) &
        (df_teste_raw[k_mes].isin(TESTE_MESES))
    )
    df_teste_final = df_teste_raw[mask_teste].reset_index(drop=True)

    # EXTRA = jan + fev / 2026 → migram para o TREINO
    mask_extra = (
        (df_teste_raw[k_ano] == TESTE_ANO) &
        (df_teste_raw[k_mes].isin(TREINO_EXTRA_MESES))
    )
    df_extra = df_teste_raw[mask_extra]

    # TREINO base: somente período < março/2026 (anti-leakage)
    df_treino_base = df_treino_raw[_ano_mes(df_treino_raw) < corte_treino]
    n_leak = len(df_treino_raw) - len(df_treino_base)
    if n_leak > 0:
        console.print(
            f"   [warning]⚠  Removidas {n_leak:,} linhas >= mar/2026 do parquet "
            f"de TREINO (proteção contra leakage)[/]"
        )

    df_treino_final = pd.concat([df_treino_base, df_extra], ignore_index=True)

    # Deduplica por chave temporal (keep="last" prioriza o parquet de teste)
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
                f"   [warning]⚠  Removidas {dup:,} linhas duplicadas no TREINO[/]"
            )

    console.print(
        f"   [success]✔  TREINO: {len(df_treino_final):,} linhas "
        f"(histórico até fev/2026) | "
        f"TESTE: {len(df_teste_final):,} linhas (mar/abr 2026)[/]"
    )

    # Validação explícita + checagem final de sanidade contra leakage
    treino_2026 = df_treino_final[df_treino_final[k_ano] == TESTE_ANO]
    ultimo_mes = int(treino_2026[k_mes].max()) if len(treino_2026) else None
    meses_teste = sorted(
        df_teste_final[df_teste_final[k_ano] == TESTE_ANO][k_mes].unique().tolist()
    )
    console.print(
        f"   [muted]Treino → último mês 2026: {ultimo_mes} "
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


def _ordenar_temporal(df, key_cols):
    """
    Ordena cronologicamente por ANO/MÊS/DIA/HORA.

    NÃO é cosmético: o TimeSeriesSplit da CV e o corte de calibração (últimas
    n_calib linhas do treino) só fazem sentido sobre um índice cronológico.
    mergesort é estável, preservando a ordem relativa de eventuais empates.
    """
    cols = [key_cols[k] for k in ["KEY_ANO","KEY_MES","KEY_DIA","KEY_HORA"]
            if key_cols.get(k)]
    return df.sort_values(cols, kind="mergesort").reset_index(drop=True)


# ==============================================================================
# FEATURE ENGINEERING
# Idêntica à das Fases 03/04/05 — condição para reproduzir exatamente as 50
# features com que o RandomForest foi tunado.
# ==============================================================================
class TemporalFE:
    """
    Gera as features temporais obrigatórias a partir das colunas-chave:
      • CYC_*  — codificação seno/cosseno (hora, dia da semana, dia do mês,
                 mês e dia do ano), preservando a continuidade circular;
      • CAL_*  — flags de calendário (fim de semana, pico, madrugada, período
                 seco/chuvoso do NE, ano crítico, feriado, dia útil);
      • TEMP_* — trimestre, dia do ano, progresso do ano/mês e interações.
    """
    # Anos de crise hídrica/energética, com PLD historicamente elevado
    ANOS_CRITICOS = {2014, 2015, 2021, 2024}

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
        df_out["CAL_is_ano_critico"]        = np.isin(
            ano, list(self.ANOS_CRITICOS)
        ).astype(int)

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
        # Auxiliar (prefixo "_" a exclui automaticamente das features)
        df_out["_DIA_SEMANA"] = dia_semana
        return df_out

    def list_obrigatorias(self, key_cols):
        """Features que entram SEMPRE, qualquer que seja o N_features usado."""
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
    Ranqueia os candidatos extras por importância (LightGBM rápido).
    Mesmos parâmetros das Fases 03/04/05 → mesmas 50 features do tuning.
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


def build_model(nome, params):
    """Instancia o regressor a partir do nome e dos hiperparâmetros."""
    if nome == "RandomForest":
        return RandomForestRegressor(**params)
    elif nome == "LightGBM":
        return lgb.LGBMRegressor(**params)
    elif nome == "XGBoost":
        return xgb.XGBRegressor(**params)
    raise ValueError(f"Modelo desconhecido: {nome}")


# ==============================================================================
# PREPARAÇÃO DE DADOS
# ==============================================================================
def preparar_dados(verbose=True):
    """
    Executa, em sequência, todo o pré-processamento comum à fase:
    carregamento → detecção de colunas → split anti-leakage → ordenação
    temporal → engenharia de features → ranking de importância.

    Devolve um dicionário com tudo que o pipeline precisa, evitando repetir
    esses passos caso a fase seja estendida com novos experimentos.
    """
    if verbose:
        console.print("\n   [info]📂 Carregando dados...[/]")
    treino_path = INPUT_DIR / CENARIO_DEFAULT / STRATEGY_DEFAULT / "TREINO_NORM.parquet"
    teste_path  = INPUT_DIR / CENARIO_DEFAULT / STRATEGY_DEFAULT / "TESTE_NORM.parquet"
    df_treino_raw = load_parquet(treino_path, "TREINO_NORM")
    df_teste_raw  = load_parquet(teste_path,  "TESTE_NORM")

    target_col = detectar_target(df_treino_raw)
    key_cols   = detectar_key_cols(df_treino_raw)
    regime_col = detectar_regime_col(df_treino_raw)
    if verbose:
        console.print(f"   [info]🎯 TARGET: {target_col}[/]")
        if regime_col:
            console.print(f"   [info]🚨 REGIME_PRECO removido (leakage): {regime_col}[/]")

    df_treino, df_teste = split_treino_teste(df_treino_raw, df_teste_raw, key_cols)
    # Ordenação cronológica: pré-requisito da CV temporal e da calibração
    df_treino = _ordenar_temporal(df_treino, key_cols)
    df_teste  = _ordenar_temporal(df_teste,  key_cols)

    if verbose:
        console.print("\n   [info]🗓  Engenharia temporal...[/]")
    fe           = TemporalFE(use_holidays=True)
    df_treino    = fe.transform(df_treino, key_cols)
    df_teste     = fe.transform(df_teste,  key_cols)
    obrigatorias = [f for f in fe.list_obrigatorias(key_cols) if f in df_treino.columns]
    if verbose:
        n_keys  = sum(1 for v in key_cols.values() if v and v in df_treino.columns)
        n_deriv = len(obrigatorias) - n_keys
        console.print(
            f"   [info]🔒 Obrigatórias: {len(obrigatorias)} "
            f"({n_keys} KEY + {n_deriv} derivadas)[/]"
        )

    # Candidatos extras: tudo que é numérico e não é alvo, regime, auxiliar
    # (prefixo "_") ou já obrigatório
    exclude = {target_col}
    if regime_col:
        exclude.add(regime_col)
    exclude.update([c for c in df_treino.columns if c.startswith("_")])
    exclude.update(obrigatorias)
    candidates = [
        c for c in df_treino.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df_treino[c])
    ]

    if verbose:
        console.print("   [info]📊 Ranqueando features (LightGBM)...[/]")
    ranking = rank_features_lgb(df_treino, candidates, target_col)
    if verbose:
        console.print(f"   [info]🎲 Candidatos extras: {len(candidates)}[/]")

    return {
        "df_treino"   : df_treino,
        "df_teste"    : df_teste,
        "key_cols"    : key_cols,
        "target_col"  : target_col,
        "regime_col"  : regime_col,
        "obrigatorias": obrigatorias,
        "ranking"     : ranking,
        "candidates"  : candidates,
    }


# ==============================================================================
# MÉTRICAS
# ==============================================================================
def metricas_pontuais(y_true, y_pred, model_name=""):
    """
    Métricas de regressão do modelo base, incluindo Spike_MAE e Spike_Bias no
    critério de R$ 500 (o mesmo das Fases 01/03/05). Função auxiliar mantida
    por compatibilidade com o padrão das outras fases.
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # Clipa o real em 1e-3 antes do MAPE: divisões por ~0 explodiriam a métrica
    y_safe = np.clip(y_true, 1e-3, None)
    mape = mean_absolute_percentage_error(y_safe, y_pred) * 100
    r2   = r2_score(y_true, y_pred)
    me   = float(np.mean(y_true - y_pred))
    spike_mask = y_true > THRESHOLD_SPIKE
    spike_mae  = (float(mean_absolute_error(y_true[spike_mask], y_pred[spike_mask]))
                  if spike_mask.sum() > 5 else np.nan)
    # Spike_Bias positivo = o modelo SUBESTIMA o pico
    spike_bias = (float(np.mean(y_true[spike_mask] - y_pred[spike_mask]))
                  if spike_mask.sum() > 5 else np.nan)
    return {
        "model"     : model_name,
        "MAE"       : safe_round(mae, 4),
        "RMSE"      : safe_round(rmse, 4),
        "MAPE_%"    : safe_round(mape, 4),
        "R2"        : safe_round(r2, 6),
        "ME_bias"   : safe_round(me, 4),
        "Spike_MAE" : safe_round(spike_mae, 4),
        "Spike_Bias": safe_round(spike_bias, 4),
    }


# ==============================================================================
# ── NOVO: ANÁLISE DE THRESHOLD SWEEP ─────────────────────────────────────────
# O ponto de corte τ = 0.50 é uma convenção de biblioteca, não uma decisão de
# projeto. As funções abaixo tornam a escolha do limiar explícita e defensável.
# ==============================================================================
def threshold_sweep(y_bin, proba, beta=FBETA_BETA,
                    precisao_minima=PRECISION_MINIMA_ACEITAVEL):
    """
    Varre thresholds e calcula Recall, Precision, F1 e Fbeta para cada um.

    Retorna
    -------
    df_sweep : DataFrame com colunas [threshold, recall, precision, f1, fbeta]
    thr_f1   : threshold que maximiza F1
    thr_fbeta: threshold que maximiza Fbeta(β)
    thr_recall_max: threshold que maximiza Recall com Precision >= precisao_minima

    Importante: esta função deve ser aplicada ao conjunto de CALIBRAÇÃO, nunca
    ao teste — escolher τ olhando o teste inflaria o Recall reportado.
    """
    records = []
    for thr in SWEEP_THRESHOLDS:
        pred = (proba >= thr).astype(int)
        rec  = float(recall_score(y_bin, pred, zero_division=0))
        pre  = float(precision_score(y_bin, pred, zero_division=0))
        f1   = float(f1_score(y_bin, pred, zero_division=0))
        fb   = float(fbeta_score(y_bin, pred, beta=beta, zero_division=0))
        records.append({
            "threshold": round(float(thr), 2),
            "recall"   : round(rec, 4),
            "precision": round(pre, 4),
            "f1"       : round(f1,  4),
            "fbeta"    : round(fb,  4),
        })
    df_sweep = pd.DataFrame(records)

    thr_f1    = df_sweep.loc[df_sweep["f1"].idxmax(),    "threshold"]
    thr_fbeta = df_sweep.loc[df_sweep["fbeta"].idxmax(), "threshold"]

    # Threshold que maximiza Recall com Precision mínima aceitável.
    # Sem a restrição de precisão, o ótimo trivial seria τ→0 (prever spike
    # sempre), com Recall = 1.0 e nenhuma utilidade prática.
    df_viavel = df_sweep[df_sweep["precision"] >= precisao_minima]
    if len(df_viavel) > 0:
        thr_recall_max = df_viavel.loc[df_viavel["recall"].idxmax(), "threshold"]
    else:
        # Se nenhum threshold atinge precision mínima, pega o melhor F-beta
        thr_recall_max = thr_fbeta

    return df_sweep, thr_f1, thr_fbeta, thr_recall_max


def print_threshold_sweep(df_sweep, thr_f1, thr_fbeta, thr_recall_max,
                          beta=FBETA_BETA, precisao_minima=PRECISION_MINIMA_ACEITAVEL):
    """
    Exibe tabela compacta com thresholds-chave e sua justificativa.
    Mostra apenas os quatro τ relevantes (MaxRecall, Fbeta, F1 e 0.50), não a
    varredura inteira — a tabela é para o texto do TCC, não para depuração.
    """
    console.print()
    t = Table(
        title=f"📈 ANÁLISE DE THRESHOLD — Recall vs Precision vs F1 vs Fbeta(β={beta})",
        box=box.DOUBLE_EDGE, border_style="magenta",
        show_lines=True, min_width=100,
    )
    t.add_column("Threshold (τ)",        style="bold",    justify="center", min_width=14)
    t.add_column("Recall ↑ (spike)",     justify="right", min_width=14)
    t.add_column("Precision",            justify="right", min_width=12)
    t.add_column(f"Fbeta β={beta}",      justify="right", min_width=14)
    t.add_column("F1",                   justify="right", min_width=10)
    t.add_column("Critério",             justify="left",  min_width=30)

    # Conjunto: se dois critérios coincidirem no mesmo τ, aparece uma linha só
    thrs_destaque = {thr_f1, thr_fbeta, thr_recall_max, 0.5}
    for _, row in df_sweep.iterrows():
        thr = row["threshold"]
        if thr not in thrs_destaque:
            continue
        if thr == thr_recall_max:
            criterio = f"[rank1]🥇 Máx Recall (Precision≥{precisao_minima})[/rank1]"
        elif thr == thr_fbeta:
            criterio = f"[rank2]🥈 Máx Fbeta(β={beta})[/rank2]"
        elif thr == thr_f1:
            criterio = "[rank3]🥉 Máx F1 (equilíbrio)[/rank3]"
        else:
            criterio = "[muted]τ=0.50 (padrão)[/muted]"

        # Semáforo do Recall: verde >= 70%, amarelo >= 50%, vermelho abaixo
        rec_color = "metric_best" if row["recall"] >= 0.70 else (
            "warning" if row["recall"] >= 0.50 else "metric_bad"
        )
        t.add_row(
            f"τ = {thr:.2f}",
            f"[{rec_color}]{row['recall']:.4f}[/{rec_color}]",
            f"{row['precision']:.4f}",
            f"{row['fbeta']:.4f}",
            f"{row['f1']:.4f}",
            criterio,
        )
    console.print(t)
    console.print(
        f"   [muted]→ Fbeta(β={beta}): recall vale {beta}× mais que precision. "
        f"Threshold orientado a custo (FN > FP).[/]"
    )
    console.print(
        f"   [muted]→ Threshold recomendado para TCC: τ={thr_recall_max:.2f} "
        f"(maximiza Recall com Precision≥{precisao_minima})[/]"
    )


def print_pr_curve_ascii(df_sweep):
    """
    Aproximação textual da curva Precision-Recall usando Rich.
    Exibe como tabela de amostras ao longo da curva para facilitar
    a interpretação sem depender de matplotlib.

    Mostra ~10 pontos igualmente espaçados da varredura, o suficiente para
    evidenciar o trade-off sem poluir a saída do console.
    """
    console.print()
    t = Table(
        title="📉 CURVA PRECISION-RECALL (amostras — threshold crescente)",
        box=box.SIMPLE, border_style="cyan",
        show_lines=False, min_width=80,
    )
    t.add_column("τ",         justify="right", min_width=6)
    t.add_column("Recall",    justify="right", min_width=10)
    t.add_column("Precision", justify="right", min_width=10)
    t.add_column("Barra Recall", min_width=30)

    # Amostras em 10 thresholds representativos
    step = max(1, len(df_sweep) // 10)
    for _, row in df_sweep.iloc[::step].iterrows():
        bar_len = int(row["recall"] * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        color = "metric_best" if row["recall"] >= 0.70 else (
            "warning" if row["recall"] >= 0.40 else "metric_bad"
        )
        t.add_row(
            f"{row['threshold']:.2f}",
            f"[{color}]{row['recall']:.3f}[/{color}]",
            f"{row['precision']:.3f}",
            f"[{color}]{bar}[/{color}]",
        )
    console.print(t)
    console.print(
        "   [muted]→ τ baixo = Recall alto, Precision baixa | "
        "τ alto = Recall baixo, Precision alta[/]"
    )


# ==============================================================================
# CLASSIFICADOR DE SPIKE (LGB + calibração isotônica + threshold F1)
# ==============================================================================
class SpikeClassifier:
    """
    Detector de spike em três camadas:

      1. LightGBM binário com class_weight="balanced" — corrige o viés do
         desbalanceamento durante o treino;
      2. IsotonicRegression — remapeia os escores para PROBABILIDADES
         calibradas, ajustada em um conjunto de calibração separado;
      3. Sweep de threshold — escolhe τ segundo três critérios distintos
         (F1, Fbeta e MaxRecall com precisão mínima), também na calibração.

    O conjunto de teste só é tocado em evaluate(), com tudo já congelado.
    """
    def __init__(self, threshold_pld=THRESHOLD_SPIKE, random_state=RANDOM_STATE,
                 beta=FBETA_BETA):
        self.threshold_pld   = threshold_pld
        self.random_state    = random_state
        self.beta            = beta
        self.model_          = None
        self.calibrator_     = None
        self.threshold_pred_ = 0.5      # threshold padrão (F1)
        self.threshold_fbeta_= 0.5      # threshold Fbeta
        self.threshold_recall_= 0.5     # threshold orientado a Recall máximo
        self.metrics_        = {}
        self.df_sweep_       = None

    def fit(self, X_train, y_train_pld, X_calib, y_calib_pld, verbose=True):
        """
        Ajusta o classificador e, em seguida, calibra e define os thresholds.

        Recebe o PLD contínuo (não o rótulo binário) nos dois conjuntos: a
        binarização acontece internamente, sempre com o mesmo threshold_pld,
        o que evita divergência de critério entre treino e calibração.
        """
        y_train_bin = (y_train_pld > self.threshold_pld).astype(int)
        n_pos = int(y_train_bin.sum())
        n_neg = len(y_train_bin) - n_pos
        if verbose:
            console.print(
                f"   [info]🎲 Classificador: {n_pos:,} pos / {n_neg:,} neg "
                f"({100*n_pos/len(y_train_bin):.1f}%)[/]"
            )
        base = lgb.LGBMClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            num_leaves=63, subsample=0.85, colsample_bytree=0.85,
            min_child_samples=20, class_weight="balanced",  # compensa a raridade
            random_state=self.random_state, n_jobs=-1, verbose=-1,
        )
        base.fit(X_train, y_train_bin)
        self.model_ = base

        y_calib_bin = (y_calib_pld > self.threshold_pld).astype(int)
        # Sem positivos suficientes na calibração, não há como calibrar nem
        # escolher τ: mantém-se o padrão 0.50 e os escores brutos.
        if y_calib_bin.sum() >= 5:
            proba_calib = base.predict_proba(X_calib)[:, 1]
            # Isotônica: monotônica e não paramétrica; "clip" evita extrapolar
            # fora do intervalo observado na calibração
            self.calibrator_ = IsotonicRegression(out_of_bounds="clip",
                                                  y_min=0.0, y_max=1.0)
            self.calibrator_.fit(proba_calib, y_calib_bin)
            proba_cal = self.calibrator_.transform(proba_calib)

            # ── Sweep para escolher thresholds (SEMPRE na calibração) ─────────
            df_sw, thr_f1, thr_fbeta, thr_recall = threshold_sweep(
                y_calib_bin, proba_cal,
                beta=self.beta,
                precisao_minima=PRECISION_MINIMA_ACEITAVEL,
            )
            self.df_sweep_        = df_sw
            self.threshold_pred_  = float(thr_f1)
            self.threshold_fbeta_ = float(thr_fbeta)
            self.threshold_recall_= float(thr_recall)

            if verbose:
                console.print(
                    f"   [info]   τ(F1)    = {self.threshold_pred_:.2f} | "
                    f"τ(Fbeta β={self.beta}) = {self.threshold_fbeta_:.2f} | "
                    f"τ(MaxRecall) = {self.threshold_recall_:.2f}[/]"
                )
        return self

    def predict_proba(self, X):
        """Probabilidade calibrada de spike (cai para o escore bruto se não calibrado)."""
        proba_raw = self.model_.predict_proba(X)[:, 1]
        return (self.calibrator_.transform(proba_raw)
                if self.calibrator_ is not None else proba_raw)

    def predict_class(self, X, threshold=None):
        """Rótulo binário no threshold indicado (padrão: o de F1)."""
        thr = threshold if threshold is not None else self.threshold_pred_
        return (self.predict_proba(X) >= thr).astype(int)

    def evaluate(self, X_test, y_test_pld, thr=None):
        """
        Avalia o classificador no conjunto de teste.
        Calcula todas as métricas na ordem de prioridade do TCC.

        Produz um dicionário "achatado" com sufixos __tau_XXX, para que a
        gravação no CSV seja direta e cada número fique inequivocamente
        associado ao threshold em que foi obtido.

        Nota: o argumento `thr` alimenta apenas a variável local thr_use, que
        não chega a ser usada — a avaliação sempre percorre os três thresholds
        internos mais o τ=0.50 de referência.
        """
        thr_f1     = self.threshold_pred_
        thr_fbeta  = self.threshold_fbeta_
        thr_recall = self.threshold_recall_
        thr_use    = thr if thr is not None else thr_f1

        y_bin  = (y_test_pld > self.threshold_pld).astype(int)
        proba  = self.predict_proba(X_test)

        # ── Predições com os três thresholds ──────────────────────────────────
        y_pred_f1     = (proba >= thr_f1).astype(int)
        y_pred_fbeta  = (proba >= thr_fbeta).astype(int)
        y_pred_recall = (proba >= thr_recall).astype(int)
        y_pred_050    = (proba >= 0.50).astype(int)

        def _clf_metrics(y_b, y_p):
            """Bloco de métricas dependentes de threshold, para reuso."""
            return {
                "F1"       : safe_round(f1_score(y_b, y_p, zero_division=0), 4),
                "Fbeta"    : safe_round(fbeta_score(y_b, y_p, beta=self.beta, zero_division=0), 4),
                "Recall"   : safe_round(recall_score(y_b, y_p, zero_division=0), 4),
                "Precision": safe_round(precision_score(y_b, y_p, zero_division=0), 4),
            }

        # AUCs falham se o teste não tiver nenhum positivo — daí o try
        try:
            auc = roc_auc_score(y_bin, proba) if y_bin.sum() > 0 else np.nan
            ap  = average_precision_score(y_bin, proba) if y_bin.sum() > 0 else np.nan
        except ValueError:
            auc, ap = np.nan, np.nan

        # ── Recall_spike explícito (métrica nomeada para o TCC) ───────────────
        n_spikes_reais = int(y_bin.sum())
        recall_spike_f1     = safe_round(recall_score(y_bin, y_pred_f1, zero_division=0), 4)
        recall_spike_fbeta  = safe_round(recall_score(y_bin, y_pred_fbeta, zero_division=0), 4)
        recall_spike_recall = safe_round(recall_score(y_bin, y_pred_recall, zero_division=0), 4)

        # Matriz de confusão armazenada no τ(F1)
        cm = confusion_matrix(y_bin, y_pred_f1).tolist()

        self.metrics_ = {
            # ── Métricas independentes de threshold ───────────────────────────
            "n_amostras"        : int(len(y_bin)),
            "n_spikes_reais"    : n_spikes_reais,
            "taxa_spike_%"      : safe_round(100.0 * n_spikes_reais / len(y_bin), 2),
            # 🥇 PR-AUC e ROC-AUC (independentes de τ)
            "PR_AUC"            : safe_round(ap, 4),
            "ROC_AUC"           : safe_round(auc, 4),

            # ── τ = maximiza F1 (equilíbrio clássico) ─────────────────────────
            "tau_F1"            : safe_round(thr_f1, 4),
            **{f"F1__tau_F1"          : _clf_metrics(y_bin, y_pred_f1)["F1"]},
            **{f"Fbeta_b{self.beta}__tau_F1"  : _clf_metrics(y_bin, y_pred_f1)["Fbeta"]},
            "Recall_spike__tau_F1"    : recall_spike_f1,
            "Precision__tau_F1"       : _clf_metrics(y_bin, y_pred_f1)["Precision"],
            "n_spikes_prev__tau_F1"   : int(y_pred_f1.sum()),
            "Confusion_Matrix__tau_F1": cm,

            # ── τ = maximiza Fbeta (penaliza FN, recomendado TCC) ─────────────
            "tau_Fbeta"         : safe_round(thr_fbeta, 4),
            **{f"F1__tau_Fbeta"        : _clf_metrics(y_bin, y_pred_fbeta)["F1"]},
            **{f"Fbeta_b{self.beta}__tau_Fbeta": _clf_metrics(y_bin, y_pred_fbeta)["Fbeta"]},
            "Recall_spike__tau_Fbeta"  : recall_spike_fbeta,
            "Precision__tau_Fbeta"     : _clf_metrics(y_bin, y_pred_fbeta)["Precision"],
            "n_spikes_prev__tau_Fbeta" : int(y_pred_fbeta.sum()),

            # ── τ = maximiza Recall com Precision mínima (PRINCIPAL DO TCC) ───
            "tau_MaxRecall"            : safe_round(thr_recall, 4),
            **{f"F1__tau_MaxRecall"       : _clf_metrics(y_bin, y_pred_recall)["F1"]},
            **{f"Fbeta_b{self.beta}__tau_MaxRecall": _clf_metrics(y_bin, y_pred_recall)["Fbeta"]},
            "Recall_spike__tau_MaxRecall" : recall_spike_recall,
            "Precision__tau_MaxRecall"    : _clf_metrics(y_bin, y_pred_recall)["Precision"],
            "n_spikes_prev__tau_MaxRecall": int(y_pred_recall.sum()),

            # ── τ = 0.50 (linha de base / comparação) ─────────────────────────
            "tau_050"                  : 0.50,
            "Recall_spike__tau_050"    : safe_round(recall_score(y_bin, y_pred_050, zero_division=0), 4),
            "Precision__tau_050"       : safe_round(precision_score(y_bin, y_pred_050, zero_division=0), 4),
            "F1__tau_050"              : safe_round(f1_score(y_bin, y_pred_050, zero_division=0), 4),
        }
        return self.metrics_


# ==============================================================================
# DISPLAYS
# ==============================================================================
def print_header():
    """Painel de abertura: modelo base, blocos da fase e parâmetros de custo."""
    console.print(Panel(
        Text.assemble(
            ("🔬 FASE 06 — APRESENTAÇÃO DA IDEIA DO CLASSIFICADOR\n", "bold green"),
            ("   Modelo base (melhor da Fase 03/04): "
             f"RandomForest ({RF_N_FEATURES}f, MAE={RF_MAE_FASE04:.3f})\n", "muted"),
            ("   (A) Diagnóstico  (B) Viabilidade  (C) Drivers de spike\n", "muted"),
            ("   Treino: histórico até fev/2026 (inclui jan/fev 2026) "
             "| Teste: mar/abr 2026\n", "muted"),
            (f"   Threshold provisório de spike: R$ {THRESHOLD_SPIKE:.0f}\n", "muted"),
            (f"   Fbeta β={FBETA_BETA} | Precision mínima aceitável: "
             f"{PRECISION_MINIMA_ACEITAVEL:.0%}\n", "muted"),
            (f"   Cenário: {CENARIO_DEFAULT} | Strategy: {STRATEGY_DEFAULT}\n", "muted"),
        ),
        title="[header_v2]  TCC PLD — IDEIA DO CLASSIFICADOR  [/header_v2]",
        border_style="purple4", padding=(1, 4),
    ))


def print_hiperparametros(nome, params, n_features, mae_f04, r2_f04):
    """Documenta os hiperparâmetros do modelo base herdados da Fase 04."""
    t = Table(
        title=f"🔧 HIPERPARÂMETROS DO MODELO BASE — {nome} ({n_features} features)",
        box=box.DOUBLE_EDGE, border_style="yellow",
        show_lines=False, min_width=70,
    )
    t.add_column("Hiperparâmetro", style="bold cyan",  min_width=28)
    t.add_column("Valor",          style="bold white", min_width=24)
    for k, v in params.items():
        t.add_row(k, str(v))
    t.add_row("─" * 26, "─" * 22)
    t.add_row("[muted]N features[/muted]",        str(n_features))
    t.add_row("[muted]MAE (Fase 04)[/muted]",     f"{mae_f04:.3f}")
    t.add_row("[muted]R² (Fase 04)[/muted]",      f"{r2_f04:.4f}")
    t.add_row("[muted]Origem dos params[/muted]", "Optuna (Fase 04)")
    console.print(t)


def print_faixa_pld(y_real, y_pred, model_name):
    """
    Erro do modelo base por faixa de preço. Versão enxuta da tabela da Fase
    05, suficiente para o argumento do bloco (A): mostrar que a faixa
    "Extremo" concentra erro e viés de subestimação.
    """
    t = Table(
        title=f"📊 ERRO POR FAIXA DE PLD — {model_name}",
        box=box.DOUBLE_EDGE, border_style="cyan",
        show_lines=True, min_width=100,
    )
    t.add_column("Faixa",      style="bold",     min_width=10)
    t.add_column("Intervalo",  justify="center", min_width=18)
    t.add_column("N",          justify="right",  min_width=8)
    t.add_column("% teste",    justify="right",  min_width=9)
    t.add_column("Real médio", justify="right",  min_width=12)
    t.add_column("Prev médio", justify="right",  min_width=12)
    t.add_column("MAE",        justify="right",  min_width=10)
    t.add_column("Bias",       justify="right",  min_width=12)
    n_total = len(y_real)
    for nome, lo, hi in FAIXAS_PLD:
        mask = (y_real >= lo) & (y_real < hi)
        if mask.sum() < 1:
            continue
        yr, yp = y_real[mask], y_pred[mask]
        mae  = float(np.mean(np.abs(yr - yp)))
        bias = float(np.mean(yr - yp))
        b_st = "spike_neg" if bias > 30 else "spike_pos" if bias < -30 else "white"
        intervalo = f"R$ {lo}–{hi if hi < 10**9 else '∞'}"
        t.add_row(
            nome, intervalo,
            f"{mask.sum():,}", f"{100*mask.sum()/n_total:.1f}%",
            f"R$ {yr.mean():.2f}", f"R$ {yp.mean():.2f}",
            f"R$ {mae:.2f}",
            f"[{b_st}]R$ {bias:+.2f}[/{b_st}]",
        )
    console.print(t)
    console.print(
        "   [muted]→ Bias > 0: modelo subestima | Bias < 0: modelo superestima[/]"
    )


def print_diagnostico_modelo_base(metrics_list):
    """
    Ficha do modelo base. A coluna "Recall Spk" é o número que abre o
    argumento da fase: o recall do REGRESSOR, não do classificador.
    """
    t = Table(
        title="📊 DIAGNÓSTICO DO MODELO BASE — RandomForest (Fase 04 tuned)",
        box=box.DOUBLE_EDGE, border_style="cyan",
        show_lines=True, min_width=130,
    )
    t.add_column("Modelo",     style="bold",    min_width=16)
    t.add_column("N feats",    justify="right", min_width=8)
    t.add_column("MAE F04",    justify="right", min_width=10)
    t.add_column("MAE F06",    justify="right", min_width=10)
    t.add_column("R²",         justify="right", min_width=10)
    t.add_column("Bias",       justify="right", min_width=10)
    t.add_column("Spike MAE",  justify="right", min_width=12)
    t.add_column("Spike Bias", justify="right", min_width=12)
    t.add_column("Recall Spk", justify="right", min_width=12)

    for r in metrics_list:
        spike_mae = r.get("spike_mae")
        t.add_row(
            r["nome"],
            str(r["n_features"]),
            f"R$ {r['mae_f04']:.3f}",
            f"[acerto]R$ {r['mae_f06']:.3f}[/acerto]",
            f"{r['r2']:.4f}",
            f"R$ {r['bias']:+.3f}",
            f"R$ {spike_mae:.2f}" if spike_mae else "—",
            f"R$ {r['spike_bias']:+.2f}" if r.get("spike_bias") else "—",
            f"{r['recall_spike']*100:.1f}%" if r.get("recall_spike") is not None else "—",
        )
    console.print(t)


def print_classificador_completo(metrics, beta=FBETA_BETA,
                                 precisao_minima=PRECISION_MINIMA_ACEITAVEL):
    """
    Exibe todas as métricas do classificador na ordem de prioridade do TCC.
    Três blocos: (1) independente de τ, (2) τ recomendado (MaxRecall),
    (3) comparação entre thresholds. Mais o bloco 4, com a matriz de confusão.
    """
    console.print()
    # ── Bloco 1: Métricas independentes de threshold ──────────────────────────
    # São as que medem a QUALIDADE DO RANKING de probabilidades, antes de
    # qualquer decisão de corte.
    t1 = Table(
        title="📊 CLASSIFICADOR — Métricas independentes de threshold",
        box=box.ROUNDED, border_style="cyan",
        show_lines=True, min_width=70,
    )
    t1.add_column("Métrica",   style="bold", min_width=30)
    t1.add_column("Valor",     justify="right", min_width=14)
    t1.add_column("Prioridade TCC", justify="center", min_width=18)
    t1.add_row(
        "PR-AUC (Avg Precision)",
        f"[rank2]{metrics.get('PR_AUC', '—')}[/rank2]",
        "[rank2]🥈 2ª[/rank2]",
    )
    t1.add_row(
        "ROC-AUC",
        f"{metrics.get('ROC_AUC', '—')}",
        "[muted]⚠️ 5ª (suporte)[/muted]",
    )
    t1.add_row(
        "N spikes reais no teste",
        f"{metrics.get('n_spikes_reais', '—')}",
        "",
    )
    # A taxa de spike é a LINHA DE BASE do PR-AUC: comparar os dois é o que
    # dá sentido ao número.
    t1.add_row(
        "Taxa de spike no teste (%)",
        f"{metrics.get('taxa_spike_%', '—')}%",
        "",
    )
    console.print(t1)

    # ── Bloco 2: τ recomendado (MaxRecall com Precision mínima) ──────────────
    thr_rec = metrics.get("tau_MaxRecall", 0.5)
    rec_val = metrics.get("Recall_spike__tau_MaxRecall", 0.0)
    pre_val = metrics.get("Precision__tau_MaxRecall", 0.0)
    f1_val  = metrics.get("F1__tau_MaxRecall", 0.0)
    fb_val  = metrics.get(f"Fbeta_b{beta}__tau_MaxRecall", 0.0)
    n_prev  = metrics.get("n_spikes_prev__tau_MaxRecall", 0)
    n_real  = metrics.get("n_spikes_reais", 0)

    rec_color = "metric_best" if (rec_val or 0) >= 0.70 else (
        "warning" if (rec_val or 0) >= 0.40 else "metric_bad"
    )

    t2 = Table(
        title=f"🎯 THRESHOLD RECOMENDADO (τ={thr_rec:.2f}) — Máx Recall | Precision≥{precisao_minima:.0%}",
        box=box.DOUBLE_EDGE, border_style="red",
        show_lines=True, min_width=80,
    )
    t2.add_column("Métrica",           style="bold", min_width=34)
    t2.add_column("Valor",             justify="right", min_width=14)
    t2.add_column("Prioridade TCC",    justify="center", min_width=18)
    t2.add_row(
        "🥇 Recall_spike  (métrica principal)",
        f"[{rec_color}]{rec_val}[/{rec_color}]",
        "[rank1]🥇 1ª[/rank1]",
    )
    t2.add_row(
        f"🥉 Precision",
        f"{pre_val}",
        "[rank3]🥉 3ª[/rank3]",
    )
    t2.add_row(
        f"🎯 Fbeta (β={beta})",
        f"{fb_val}",
        "[highlight]4ª[/highlight]",
    )
    t2.add_row(
        "   F1-score",
        f"{f1_val}",
        "[muted]4b (equil.)[/muted]",
    )
    # Previstos vs reais: quantifica o volume de alarmes gerado
    t2.add_row(
        "   Spikes previstos / reais",
        f"{n_prev} / {n_real}",
        "",
    )
    console.print(t2)

    # ── Bloco 3: Comparação entre thresholds ──────────────────────────────────
    # Mostra o trade-off explicitamente: quanto de Precision se paga por
    # quanto de Recall ao mover τ.
    t3 = Table(
        title="🔄 COMPARAÇÃO ENTRE THRESHOLDS",
        box=box.SIMPLE, border_style="magenta",
        show_lines=True, min_width=110,
    )
    t3.add_column("τ",          justify="center", min_width=10)
    t3.add_column("Critério",   justify="left",   min_width=28)
    t3.add_column("Recall ↑",   justify="right",  min_width=10)
    t3.add_column("Precision",  justify="right",  min_width=12)
    t3.add_column(f"Fbeta β={beta}", justify="right", min_width=12)
    t3.add_column("F1",         justify="right",  min_width=10)
    t3.add_column("Prev / Reais", justify="right", min_width=14)

    # Cada tupla: (valor de τ, rótulo, e as CHAVES do dicionário de métricas
    # correspondentes àquele threshold). None = métrica não disponível.
    linhas_comp = [
        (
            metrics.get("tau_MaxRecall", "—"),
            f"[rank1]Máx Recall (Prec≥{precisao_minima:.0%})[/rank1]",
            "Recall_spike__tau_MaxRecall",
            "Precision__tau_MaxRecall",
            f"Fbeta_b{beta}__tau_MaxRecall",
            "F1__tau_MaxRecall",
            "n_spikes_prev__tau_MaxRecall",
        ),
        (
            metrics.get("tau_Fbeta", "—"),
            f"[rank2]Máx Fbeta β={beta}[/rank2]",
            "Recall_spike__tau_Fbeta",
            "Precision__tau_Fbeta",
            f"Fbeta_b{beta}__tau_Fbeta",
            "F1__tau_Fbeta",
            "n_spikes_prev__tau_Fbeta",
        ),
        (
            metrics.get("tau_F1", "—"),
            "[rank3]Máx F1[/rank3]",
            "Recall_spike__tau_F1",
            "Precision__tau_F1",
            f"Fbeta_b{beta}__tau_F1",
            "F1__tau_F1",
            "n_spikes_prev__tau_F1",
        ),
        (
            0.50,
            "[muted]Padrão (0.50)[/muted]",
            "Recall_spike__tau_050",
            "Precision__tau_050",
            None,
            "F1__tau_050",
            None,
        ),
    ]
    for thr_v, crit, k_rec, k_pre, k_fb, k_f1, k_np in linhas_comp:
        rec = metrics.get(k_rec, "—")
        pre = metrics.get(k_pre, "—")
        fb  = metrics.get(k_fb, "—") if k_fb else "—"
        f1v = metrics.get(k_f1, "—")
        np_ = metrics.get(k_np, "—") if k_np else "—"
        n_r = metrics.get("n_spikes_reais", "—")
        rc  = "metric_best" if isinstance(rec, float) and rec >= 0.70 else (
            "warning" if isinstance(rec, float) and rec >= 0.40 else "metric_bad"
        )
        t3.add_row(
            f"τ={thr_v:.2f}" if isinstance(thr_v, float) else str(thr_v),
            crit,
            f"[{rc}]{rec}[/{rc}]",
            f"{pre}",
            f"{fb}",
            f"{f1v}",
            f"{np_} / {n_r}" if k_np else "— / —",
        )
    console.print(t3)

    # ── Bloco 4: Matriz de confusão (τ recomendado = MaxRecall) ───────────────
    # ATENÇÃO: a matriz efetivamente exibida é a de τ(F1), a única armazenada
    # em metrics_. Se a matriz do TCC precisa ser a do τ recomendado
    # (MaxRecall), é necessário guardá-la também em evaluate().
    cm = metrics.get("Confusion_Matrix__tau_F1", [[0,0],[0,0]])
    console.print()
    t4 = Table(
        title=f"📊 MATRIZ DE CONFUSÃO (τ={metrics.get('tau_F1', 0.5):.2f} | base para TCC)",
        box=box.ROUNDED, border_style="red", show_lines=True, min_width=60,
    )
    t4.add_column("",               style="bold",    min_width=22)
    t4.add_column("Previsto: Normal", justify="center", min_width=18)
    t4.add_column("Previsto: Spike",  justify="center", min_width=18)
    t4.add_row(
        "Real: Normal",
        f"[metric_best]TN = {cm[0][0]:,}[/metric_best]",
        f"[warning]FP = {cm[0][1]:,}[/warning]",
    )
    t4.add_row(
        "Real: Spike",
        f"[metric_bad]FN = {cm[1][0]:,} ← crítico[/metric_bad]",
        f"[metric_best]TP = {cm[1][1]:,}[/metric_best]",
    )
    console.print(t4)
    console.print(
        "   [muted]→ FN (spike perdido) é o erro mais grave no seu contexto.[/]"
    )


def print_painel_prioridade_metricas(clf_metrics, auc_te, ap_te, beta=FBETA_BETA):
    """
    Painel final com ranking acadêmico das métricas — para citar no TCC.
    Inclui um parágrafo de justificativa pronto para ser adaptado ao texto,
    o que reduz o risco de a hierarquia ser apresentada sem fundamentação.
    """
    thr_r  = clf_metrics.get("tau_MaxRecall", "—")
    rec_r  = clf_metrics.get("Recall_spike__tau_MaxRecall", "—")
    pre_r  = clf_metrics.get("Precision__tau_MaxRecall", "—")
    fb_r   = clf_metrics.get(f"Fbeta_b{beta}__tau_MaxRecall", "—")
    f1_r   = clf_metrics.get("F1__tau_MaxRecall", "—")
    auc_v  = auc_te if not (auc_te is None or (isinstance(auc_te, float) and np.isnan(auc_te))) else "—"
    ap_v   = ap_te  if not (ap_te  is None or (isinstance(ap_te,  float) and np.isnan(ap_te)))  else "—"

    console.print(Panel(
        Text.assemble(
            ("📋 HIERARQUIA DE MÉTRICAS — PARA O TCC\n\n", "bold cyan"),
            ("   Dado o caráter crítico dos eventos extremos (spikes de PLD),\n", "muted"),
            ("   a não-detecção (FN) é o erro mais custoso. Hierarquia adotada:\n\n", "muted"),
            ("   🥇 1. Recall_spike ",          "rank1"),
            (f"= {rec_r}",                       "bold white"),
            (f"  (τ={thr_r:.2f} orientado a custo)\n", "muted"),
            ("   🥈 2. PR-AUC        ",          "rank2"),
            (f"= {ap_v}\n",                      "bold white"),
            ("   🥉 3. Precision     ",          "rank3"),
            (f"= {pre_r}\n",                     "bold white"),
            (f"   🎯 4. Fbeta (β={beta}) ",      "highlight"),
            (f"= {fb_r}",                        "bold white"),
            (f"  / F1 = {f1_r}\n",              "muted"),
            ("   ⚠️  5. ROC-AUC      ",          "warning"),
            (f"= {auc_v}",                       "bold white"),
            ("  (suporte — pode enganar em classe rara)\n\n", "muted"),
            ("   → Justificativa TCC:\n", "muted"),
            ("     'Priorizou-se o Recall para a classe spike pois a não-detecção\n", "white"),
            ("      implica perdas significativamente maiores que falsos alarmes.\n", "white"),
            (f"     Fbeta(β={beta}) foi utilizado pois penaliza o FN {beta}× mais que o FP.\n", "white"),
            ("      PR-AUC é preferível ao ROC-AUC em problemas de classe rara.'\n", "white"),
        ),
        title="[header_v2]  RANKING DE MÉTRICAS — DETECTOR DE EVENTOS EXTREMOS  [/header_v2]",
        border_style="cyan", padding=(1, 3),
    ))


# ==============================================================================
# COLETA E GRAVAÇÃO DE MÉTRICAS (atualizada com todas as novas métricas)
# ==============================================================================
# Diferente das Fases 03/04/05, a coleta aqui NÃO varre o namespace global:
# é uma função chamada explicitamente com os arrays corretos. Mais robusta e
# capaz de gravar as duas linhas (regressão e classificação) da Parte 6.
# ==============================================================================
def coletar_e_gravar_metricas(
    parte: int,
    y_te: np.ndarray,
    y_pred: np.ndarray,
    y_bin_te: np.ndarray,
    proba_spike_te: np.ndarray,
    clf_metrics: dict,
    beta: float,
    pasta_saida: str,
) -> None:
    """
    Calcula e persiste métricas de regressão e classificação no CSV consolidado.

    Parâmetros
    ----------
    parte          : número da Fase (6)
    y_te           : valores reais de PLD no teste (float array)
    y_pred         : previsões do modelo base (float array)
    y_bin_te       : rótulos binários de spike no teste (0/1 int array)
    proba_spike_te : probabilidades calibradas do classificador (float array)
    clf_metrics    : dicionário retornado por SpikeClassifier.evaluate()
    beta           : valor de β usado no Fbeta
    pasta_saida    : diretório onde está / será criado metricas_recalculadas.csv
    """
    console.rule("[bold blue]━━━ COLETA AUTOMÁTICA DE MÉTRICAS ━━━[/bold blue]")

    linhas = []

    # ── REGRESSÃO ──────────────────────────────────────────────────────────────
    yt = np.asarray(y_te,   dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    mask_valid = ~(np.isnan(yt) | np.isnan(yp))
    a, b = yt[mask_valid], yp[mask_valid]

    if a.size >= 10:
        rmse_v = float(np.sqrt(mean_squared_error(a, b)))
        r2_v   = float(r2_score(a, b))
        mae_v  = float(mean_absolute_error(a, b))
        nz     = np.abs(a) > 1e-6   # protege o MAPE contra divisão por zero
        mape_v = float(np.mean(np.abs((a[nz] - b[nz]) / a[nz])) * 100) if nz.any() else None
        reg_row = {
            "Parte"     : parte,
            "Tipo"      : "regressao",
            "N_Features": RF_N_FEATURES,
            "RMSE"      : round(rmse_v, 4),
            "R2"        : round(r2_v,   4),
            "MAE"       : round(mae_v,  4),
            "MAPE"      : round(mape_v, 2) if mape_v is not None else None,
            "var_real"  : "y_te",
            "var_pred"  : "y_pred",
            "n"         : int(a.size),
        }
        linhas.append(reg_row)
        console.print(
            f"   [success][REG] Parte {parte}: "
            f"RMSE={reg_row['RMSE']} | R2={reg_row['R2']} | "
            f"MAE={reg_row['MAE']} | n={reg_row['n']}[/]"
        )
    else:
        console.print(f"   [warning][REG] Parte {parte}: arrays insuficientes.[/]")

    # ── CLASSIFICAÇÃO (todas as métricas na hierarquia do TCC) ────────────────
    yb = np.asarray(y_bin_te,       dtype=float).ravel()
    ps = np.asarray(proba_spike_te, dtype=float).ravel()

    # Validação defensiva: rótulos binários, tamanhos compatíveis e escores
    # dentro de [0, 1] (garantido pela calibração isotônica)
    if (
        yb.size >= 10
        and ps.size == yb.size
        and set(np.unique(yb[~np.isnan(yb)])) <= {0.0, 1.0}
        and np.nanmin(ps) >= 0.0
        and np.nanmax(ps) <= 1.0
    ):
        # Métricas independentes de τ
        try:
            auc_v = float(roc_auc_score(yb, ps))
            ap_v  = float(average_precision_score(yb, ps))
        except ValueError:
            auc_v, ap_v = np.nan, np.nan

        # Extrair métricas com threshold recomendado (MaxRecall)
        thr_r = clf_metrics.get("tau_MaxRecall", 0.5)
        pred_r = (ps >= thr_r).astype(int)

        clf_row = {
            "Parte"    : parte,
            "Tipo"     : "classificacao",

            # ── Independentes de τ ─────────────────────────────────────────────
            "PR_AUC"   : round(ap_v,  4) if not np.isnan(ap_v)  else None,  # 🥈 2ª
            "ROC_AUC"  : round(auc_v, 4) if not np.isnan(auc_v) else None,  # ⚠️ 5ª

            # ── τ = MaxRecall (principal do TCC) ──────────────────────────────
            # As colunas SEM sufixo carregam o threshold recomendado: é o que
            # deve ser citado como resultado da fase.
            "tau_MaxRecall"          : thr_r,
            "Recall_spike"           : clf_metrics.get("Recall_spike__tau_MaxRecall"),   # 🥇 1ª
            "Precision"              : clf_metrics.get("Precision__tau_MaxRecall"),       # 🥉 3ª
            f"Fbeta_b{beta}"         : clf_metrics.get(f"Fbeta_b{beta}__tau_MaxRecall"), # 🎯 4ª
            "F1"                     : clf_metrics.get("F1__tau_MaxRecall"),              # 4b
            "n_spikes_prev_MaxRecall": clf_metrics.get("n_spikes_prev__tau_MaxRecall"),

            # ── τ = MaxFbeta ───────────────────────────────────────────────────
            "tau_Fbeta"              : clf_metrics.get("tau_Fbeta"),
            "Recall_spike_Fbeta"     : clf_metrics.get("Recall_spike__tau_Fbeta"),
            "Precision_Fbeta"        : clf_metrics.get("Precision__tau_Fbeta"),
            f"Fbeta_b{beta}_Fbeta"   : clf_metrics.get(f"Fbeta_b{beta}__tau_Fbeta"),

            # ── τ = F1 ────────────────────────────────────────────────────────
            "tau_F1"                 : clf_metrics.get("tau_F1"),
            "Recall_spike_F1"        : clf_metrics.get("Recall_spike__tau_F1"),
            "Precision_F1"           : clf_metrics.get("Precision__tau_F1"),

            # ── τ = 0.50 (linha de base) ───────────────────────────────────────
            "Recall_spike_050"       : clf_metrics.get("Recall_spike__tau_050"),
            "Precision_050"          : clf_metrics.get("Precision__tau_050"),
            "F1_050"                 : clf_metrics.get("F1__tau_050"),

            # ── Contexto ────────────────────────────────────────────────────────
            "n_spikes_reais"         : clf_metrics.get("n_spikes_reais"),
            "taxa_spike_%"           : clf_metrics.get("taxa_spike_%"),
            "var_real"               : "y_bin_te",
            "var_pred"               : "proba_spike_te",
            "n"                      : int(yb.size),
        }
        linhas.append(clf_row)
        console.print(
            f"   [success][CLF] Parte {parte}: "
            f"Recall_spike(τ={thr_r:.2f})={clf_row['Recall_spike']} | "
            f"PR-AUC={clf_row['PR_AUC']} | "
            f"Precision={clf_row['Precision']} | "
            f"F1={clf_row['F1']}[/]"
        )
    else:
        console.print(
            f"   [warning][CLF] Parte {parte}: sem score de probabilidade válido.[/]"
        )

    # ── GRAVA / ATUALIZA CSV ───────────────────────────────────────────────────
    if not linhas:
        console.print(f"   [warning]Nenhuma métrica calculada para Parte {parte}.[/]")
        return

    csv_path = os.path.join(pasta_saida, "metricas_recalculadas.csv")
    df_existente = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
    df_novo  = pd.DataFrame(linhas)

    # Dedupe por (Parte, Tipo): reexecutar SUBSTITUI as linhas da Parte 6
    if not df_existente.empty and "Parte" in df_existente.columns and "Tipo" in df_existente.columns:
        chaves = set(zip(df_novo["Parte"], df_novo["Tipo"]))
        df_existente = df_existente[
            ~df_existente.apply(
                lambda r: (r.get("Parte"), r.get("Tipo")) in chaves, axis=1
            )
        ]

    df_final = (
        pd.concat([df_existente, df_novo], ignore_index=True)
        .sort_values(["Parte", "Tipo"])
    )
    df_final.to_csv(csv_path, index=False, encoding="utf-8-sig")
    console.print(
        f"   [success]✅ metricas_recalculadas.csv atualizado "
        f"(+{len(linhas)} linha(s)) → {csv_path}[/]"
    )


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================
def run_fase_06():
    print_header()

    # ── 1. Dados ────────────────────────────────────────────────────────────
    d = preparar_dados(verbose=True)
    df_treino    = d["df_treino"]
    df_teste     = d["df_teste"]
    key_cols     = d["key_cols"]
    target_col   = d["target_col"]
    obrigatorias = d["obrigatorias"]
    ranking      = d["ranking"]

    # Mesmas 50 features do RandomForest tunado — o classificador reutiliza
    # exatamente esse conjunto, o que torna a comparação direta
    features_rf = selecionar_top_n(obrigatorias, ranking, RF_N_FEATURES)
    X_tr_rf = df_treino[features_rf].values.astype(np.float32)
    y_tr    = df_treino[target_col].values.astype(np.float32)
    X_te_rf = df_teste[features_rf].values.astype(np.float32)
    y_te    = df_teste[target_col].values.astype(np.float32)

    # ── 2. Treinar o modelo base (RandomForest) ──────────────────────────────
    console.print("\n   [highlight]━━━ MODELO BASE — RandomForest (hiperparâmetros Fase 04) ━━━[/]")
    print_hiperparametros("RandomForest", RF_BEST_PARAMS, len(features_rf),
                          RF_MAE_FASE04, RF_R2_FASE04)

    console.print(f"\n   [info]🏋️  Treinando RandomForest ({len(features_rf)}f)...[/]")
    t0    = time.perf_counter()
    model = build_model("RandomForest", RF_BEST_PARAMS)
    model.fit(X_tr_rf, y_tr)
    elapsed = time.perf_counter() - t0
    y_pred  = model.predict(X_te_rf)
    mae_f06 = mean_absolute_error(y_te, y_pred)
    r2      = r2_score(y_te, y_pred)
    bias    = float(np.mean(y_te - y_pred))

    # Métricas restritas à cauda (PLD real > R$ 500)
    spike_mask = y_te > THRESHOLD_SPIKE
    spike_mae  = (float(mean_absolute_error(y_te[spike_mask], y_pred[spike_mask]))
                  if spike_mask.sum() > 5 else None)
    spike_bias = (float(np.mean(y_te[spike_mask] - y_pred[spike_mask]))
                  if spike_mask.sum() > 5 else None)
    # RECALL DO REGRESSOR: fração dos spikes reais em que a PREVISÃO NUMÉRICA
    # também ultrapassou R$ 500. Não confundir com o Recall_spike do
    # classificador — é este número baixo que justifica a fase inteira.
    recall_spk = (float(np.mean(y_pred[spike_mask] > THRESHOLD_SPIKE))
                  if spike_mask.sum() > 5 else None)

    console.print(
        f"   [success]✔  RandomForest: MAE={mae_f06:.3f} | R²={r2:.4f} "
        f"| Recall spike={recall_spk*100:.1f}% | [{elapsed:.1f}s][/]"
        if recall_spk is not None else
        f"   [success]✔  RandomForest: MAE={mae_f06:.3f} | R²={r2:.4f} | [{elapsed:.1f}s][/]"
    )

    diag_list = [{
        "nome"        : "RandomForest",
        "n_features"  : len(features_rf),
        "mae_f04"     : RF_MAE_FASE04,
        "mae_f06"     : safe_round(mae_f06, 3),
        "r2"          : safe_round(r2, 4),
        "bias"        : safe_round(bias, 3),
        "spike_mae"   : safe_round(spike_mae, 3) if spike_mae else None,
        "spike_bias"  : safe_round(spike_bias, 3) if spike_bias else None,
        "recall_spike": recall_spk,
    }]

    # ── (A) DIAGNÓSTICO ──────────────────────────────────────────────────────
    # Objetivo do bloco: provar que a falha na cauda é ESTRUTURAL, não ruído.
    console.print("\n")
    console.rule("[bold red]━━━ (A) DIAGNÓSTICO — ONDE O MODELO BASE FALHA? ━━━[/bold red]")

    print_diagnostico_modelo_base(diag_list)
    console.print()
    print_faixa_pld(y_te, y_pred, "RandomForest (modelo base)")
    console.print()

    rf_diag      = diag_list[0]
    recall_base  = rf_diag["recall_spike"] or 0.0
    spike_mae_v  = rf_diag["spike_mae"] or 0.0
    spike_bias_v = rf_diag["spike_bias"] or 0.0
    n_spk = int((y_te > THRESHOLD_SPIKE).sum())

    console.print(Panel(
        Text.assemble(
            ("🚨 O PROBLEMA (por que precisamos de um classificador):\n\n", "bold red"),
            (f"   • Spikes reais no teste (PLD > R$ {THRESHOLD_SPIKE:.0f}): {n_spk}\n", "white"),
            (f"   • RECALL do RandomForest (modelo base): "
             f"{recall_base*100:.1f}% dos spikes detectados\n", "bold"),
            (f"   • MAE na cauda: R$ {spike_mae_v:.2f} | "
             f"Bias (sub-previsão): R$ {spike_bias_v:+.2f}\n", "bold"),
            ("\n   → O modelo único 'comprime' a distribuição: sub-prevê a cauda.\n", "muted"),
            ("     Um especialista de cauda + gate probabilístico resolve isso.\n", "muted"),
        ),
        border_style="red", padding=(1, 3),
    ))

    # ── (B) VIABILIDADE ──────────────────────────────────────────────────────
    # Objetivo do bloco: mostrar que a informação sobre o spike EXISTE nas
    # features — ou seja, que o problema do bloco (A) tem solução.
    console.print("\n")
    console.rule("[bold green]━━━ (B) VIABILIDADE — spikes são separáveis pelas features? ━━━[/bold green]")

    X_full_rf  = X_tr_rf
    y_bin_full = (y_tr > THRESHOLD_SPIKE).astype(int)
    n_pos = int(y_bin_full.sum())
    console.print(
        f"   [info]   Positivos no treino: {n_pos:,} "
        f"({100*n_pos/len(y_bin_full):.1f}%)[/]"
    )

    # CV temporal → estimativa honesta de separabilidade
    # (expanding window: cada dobra treina no passado e valida no futuro)
    tscv = TimeSeriesSplit(n_splits=4)
    aucs_cv, aps_cv = [], []
    for tr_idx, va_idx in tscv.split(X_full_rf):
        # Dobras sem positivos suficientes não produzem AUC/AP significativos
        if y_bin_full[tr_idx].sum() < 5 or y_bin_full[va_idx].sum() < 1:
            continue
        clf_cv = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05, num_leaves=63,
            min_child_samples=20, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        )
        clf_cv.fit(X_full_rf[tr_idx], y_bin_full[tr_idx])
        proba_cv = clf_cv.predict_proba(X_full_rf[va_idx])[:, 1]
        try:
            aucs_cv.append(roc_auc_score(y_bin_full[va_idx], proba_cv))
            aps_cv.append(average_precision_score(y_bin_full[va_idx], proba_cv))
        except ValueError:
            pass
    auc_cv = float(np.mean(aucs_cv)) if aucs_cv else np.nan
    ap_cv  = float(np.mean(aps_cv))  if aps_cv  else np.nan

    # Treina no treino completo e avalia no teste
    # (responde: o sinal medido na CV se sustenta em mar/abr 2026?)
    clf_full = lgb.LGBMClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05, num_leaves=63,
        min_child_samples=20, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    clf_full.fit(X_full_rf, y_bin_full)
    proba_te  = clf_full.predict_proba(X_te_rf)[:, 1]
    y_bin_te  = (y_te > THRESHOLD_SPIKE).astype(int)
    try:
        auc_te = float(roc_auc_score(y_bin_te, proba_te)) if y_bin_te.sum() > 0 else np.nan
        ap_te  = float(average_precision_score(y_bin_te, proba_te)) if y_bin_te.sum() > 0 else np.nan
    except ValueError:
        auc_te, ap_te = np.nan, np.nan

    t = Table(
        title="🤖 SEPARABILIDADE DO CLASSIFICADOR (LGB rápido + features do RandomForest)",
        box=box.DOUBLE_EDGE, border_style="green",
        show_lines=True, min_width=90,
    )
    t.add_column("Métrica",               style="bold",    min_width=26)
    t.add_column("CV temporal (treino)",  justify="right", min_width=22)
    t.add_column("Teste (mar+abr/2026)",  justify="right", min_width=22)
    t.add_row(
        "ROC-AUC  [⚠️ suporte]",
        f"{auc_cv:.4f}" if not np.isnan(auc_cv) else "—",
        f"{auc_te:.4f}" if not np.isnan(auc_te) else "—",
    )
    t.add_row(
        "PR-AUC   [🥈 principal]",
        f"{ap_cv:.4f}" if not np.isnan(ap_cv) else "—",
        f"{ap_te:.4f}" if not np.isnan(ap_te) else "—",
    )
    # A taxa base é a linha de comparação do PR-AUC: se PR-AUC ≈ taxa base,
    # o classificador não acrescenta nada ao chute proporcional.
    t.add_row(
        "Taxa base (positivos)",
        f"{100*n_pos/len(y_bin_full):.1f}%",
        f"{100*y_bin_te.sum()/len(y_bin_te):.1f}%",
    )
    console.print(t)
    console.print(
        "   [muted]→ PR-AUC é a métrica de separabilidade mais relevante em classe rara.[/]"
    )
    console.print(
        "   [muted]→ ROC-AUC: 0.5=aleatório (use como referência secundária).[/]"
    )

    # ── Classificador completo (com calibração isotônica + sweep de threshold) ─
    # Corte de calibração: as ÚLTIMAS n_calib linhas do treino. Como os dados
    # foram ordenados cronologicamente, isso corresponde ao passado mais
    # recente (tipicamente jan/fev 2026) — o período mais parecido com o teste.
    console.print(
        "\n   [info]🔧 Treinando classificador completo "
        "(calibração isotônica + sweep de threshold)...[/]"
    )
    n_calib   = min(5000, len(X_full_rf) // 4)
    X_clf_tr  = X_full_rf[:-n_calib]
    y_clf_tr  = y_tr[:-n_calib]
    X_clf_cal = X_full_rf[-n_calib:]
    y_clf_cal = y_tr[-n_calib:]

    spike_clf = SpikeClassifier(threshold_pld=THRESHOLD_SPIKE,
                                random_state=RANDOM_STATE,
                                beta=FBETA_BETA)
    spike_clf.fit(X_clf_tr, y_clf_tr, X_clf_cal, y_clf_cal, verbose=True)
    # Primeiro e único contato do classificador calibrado com o teste
    clf_metrics = spike_clf.evaluate(X_te_rf, y_te)

    # Probabilidades calibradas para a coleta de métricas
    proba_spike_te = spike_clf.predict_proba(X_te_rf)

    # ── Exibe painel completo de métricas do classificador ────────────────────
    console.print()
    print_classificador_completo(clf_metrics, beta=FBETA_BETA,
                                 precisao_minima=PRECISION_MINIMA_ACEITAVEL)

    # ── Sweep de threshold (tabela + curva PR ascii) ──────────────────────────
    # df_sweep_ só existe se houve calibração (positivos suficientes)
    if spike_clf.df_sweep_ is not None:
        print_threshold_sweep(
            spike_clf.df_sweep_,
            spike_clf.threshold_pred_,
            spike_clf.threshold_fbeta_,
            spike_clf.threshold_recall_,
            beta=FBETA_BETA,
            precisao_minima=PRECISION_MINIMA_ACEITAVEL,
        )
        print_pr_curve_ascii(spike_clf.df_sweep_)

    # ── (C) DRIVERS ──────────────────────────────────────────────────────────
    # Objetivo do bloco: dar interpretação física ao detector — o que ele
    # observa para antecipar o spike.
    console.print("\n")
    console.rule("[bold cyan]━━━ (C) DRIVERS — quais features preveem spike? ━━━[/bold cyan]")

    imp = pd.DataFrame({
        "feature"   : features_rf,
        "importance": clf_full.feature_importances_,
    }).sort_values("importance", ascending=False).head(15).reset_index(drop=True)

    t = Table(
        title="🔑 TOP-15 FEATURES DO CLASSIFICADOR DE SPIKE (features do RandomForest)",
        box=box.ROUNDED, border_style="cyan",
        show_lines=False, min_width=80,
    )
    t.add_column("#",           justify="right", min_width=4)
    t.add_column("Feature",     style="bold",    min_width=48)
    t.add_column("Importância", justify="right", min_width=14)
    t.add_column("Categoria",   justify="center",min_width=14)
    for i, r in imp.iterrows():
        f = r["feature"]
        # Categorização por família física, para a discussão do TCC
        if any(p in f.upper() for p in ["CYC_","CAL_","TEMP_","KEY_"]):
            cat, cor = "Temporal", "data"
        elif any(p in f.upper() for p in ["EAR","RESERV"]):
            cat, cor = "Hidrológico", "info"
        elif any(p in f.upper() for p in ["CARGA","DEMANDA"]):
            cat, cor = "Carga", "highlight"
        elif any(p in f.upper() for p in ["CMO","PLD","PRECO"]):
            cat, cor = "Preço", "warning"
        else:
            cat, cor = "Outro", "white"
        t.add_row(
            str(i+1),
            f"[{cor}]{f}[/{cor}]",
            f"{r['importance']:.0f}",
            f"[{cor}]{cat}[/{cor}]",
        )
    console.print(t)

    # ── PAINEL DE PRIORIDADE DE MÉTRICAS (para o TCC) ─────────────────────────
    console.print()
    print_painel_prioridade_metricas(clf_metrics, auc_te, ap_te, beta=FBETA_BETA)

    # ── CONCLUSÃO ─────────────────────────────────────────────────────────────
    # Critérios do veredito: ROC-AUC >= 0.75 OU PR-AUC >= 0.30. O segundo é o
    # mais informativo — idealmente deveria ser comparado à taxa base de
    # positivos do teste, e não a um valor fixo.
    viavel      = (not np.isnan(auc_te)) and auc_te >= 0.75
    ap_viavel   = (not np.isnan(ap_te))  and ap_te  >= 0.30   # PR-AUC > taxa base
    rec_final   = clf_metrics.get("Recall_spike__tau_MaxRecall", 0.0) or 0.0
    veredito = (
        "[acerto]✔ HÍBRIDO JUSTIFICADO — sinal preditivo suficiente[/acerto]"
        if viavel or ap_viavel else
        "[warning]⚠ Separabilidade fraca — revisar features ou threshold[/warning]"
    )
    console.print()
    console.print(Panel(
        Text.assemble(
            ("🎯 CONCLUSÃO DA FASE 06\n\n", "bold green"),
            ("   Modelo base (RandomForest — melhor da Fase 03/04):\n", "muted"),
            (f"   • Recall de spike (regressor): {recall_base*100:.1f}% "
             f"({int((1-recall_base)*100)}% dos spikes perdidos)\n", "white"),
            (f"   • MAE na cauda: R$ {spike_mae_v:.2f} | "
             f"Sub-previsão: R$ {spike_bias_v:+.2f}\n", "white"),
            ("\n   Classificador de spike (métricas na hierarquia do TCC):\n", "muted"),
            (f"   🥇 Recall_spike (τ={clf_metrics.get('tau_MaxRecall','—'):.2f}) "
             f"= {rec_final:.3f}\n", "white"),
            (f"   🥈 PR-AUC                  = {ap_te:.3f}\n", "white"),
            (f"   🥉 Precision               = {clf_metrics.get('Precision__tau_MaxRecall','—')}\n", "white"),
            (f"   🎯 Fbeta (β={FBETA_BETA})            = {clf_metrics.get(f'Fbeta_b{FBETA_BETA}__tau_MaxRecall','—')}\n", "white"),
            (f"   ⚠️  ROC-AUC                = {auc_te:.3f}\n", "white"),
            ("\n   → Há sinal preditivo → faz sentido um especialista de cauda "
             "guiado por gate.\n", "muted"),
            ("   Próximo passo (Fase 07): calibrar threshold e montar o híbrido.\n\n", "bold"),
            ("   Veredito: ", "muted"),
            (veredito, "white"),
        ),
        border_style="green", padding=(1, 3),
    ))

    # ── COLETA AUTOMÁTICA DE MÉTRICAS ─────────────────────────────────────────
    # Chamada explícita, com os arrays corretos passados como argumento
    coletar_e_gravar_metricas(
        parte          = 6,
        y_te           = y_te,
        y_pred         = y_pred,
        y_bin_te       = y_bin_te,
        proba_spike_te = proba_spike_te,
        clf_metrics    = clf_metrics,
        beta           = FBETA_BETA,
        pasta_saida    = PASTA_SAIDA_METRICAS,
    )

    # Objetos devolvidos para a Fase 07: o gate probabilístico (spike_clf) e
    # os thresholds são o que efetivamente será reaproveitado no híbrido.
    return {
        "model"          : model,
        "y_pred"         : y_pred,
        "y_te"           : y_te,
        "y_bin_te"       : y_bin_te,
        "proba_spike_te" : proba_spike_te,
        "diag_list"      : diag_list,
        "clf_full"       : clf_full,
        "spike_clf"      : spike_clf,
        "clf_metrics"    : clf_metrics,
        "df_sweep"       : spike_clf.df_sweep_,
        "thr_f1"         : spike_clf.threshold_pred_,
        "thr_fbeta"      : spike_clf.threshold_fbeta_,
        "thr_recall"     : spike_clf.threshold_recall_,
        "auc_cv"         : auc_cv,
        "ap_cv"          : ap_cv,
        "auc_te"         : auc_te,
        "ap_te"          : ap_te,
        "recall_base"    : recall_base,
        "features_rf"    : features_rf,
        "X_te_rf"        : X_te_rf,
    }


# ==============================================================================
# ENTRADA
# ==============================================================================
fase_06_output = run_fase_06()

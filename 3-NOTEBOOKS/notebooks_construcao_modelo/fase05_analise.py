# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                TCC PLD — FASE 05: ANÁLISE PROFUNDA DE ERROS                   ║
# ║   Resíduos por Faixa, Spikes, Sazonalidade, SHAP e Diagnóstico do Modelo     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  IDENTIFICAÇÃO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Autor       : Yago
#  Instituição : Universidade Federal do Ceará
#  Curso       : Engenharia Elétrica
#  Data        : Julho / 2026
#  Versão      : 1.0 — Arquivo Único (Análise de Erros + Coleta de Métricas)
#                >>> VERSÃO COM MÉTRICAS COMPLETAS (sMAPE, NRMSE, R² por grupo,
#                    Spike_MAE/Spike_Bias no mesmo critério R$500 das Fases
#                    01/03) <<<
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DESCRIÇÃO / FUNÇÃO DO SCRIPT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Este script implementa a Fase 05 da modelagem preditiva do PLD (submercado
#  Nordeste) do TCC, unificando em um único arquivo Colab-friendly:
#
#    (A) ANÁLISE PROFUNDA DE ERROS — as Fases 01 a 04 responderam "QUAL modelo
#        erra menos". Esta fase responde "ONDE, QUANDO e POR QUE ele erra".
#        Um único modelo é retreinado (o vencedor da Fase 03/04) e suas
#        predições sobre mar/abr 2026 são dissecadas por faixa de preço, tipo
#        de spike, mês, hora, dia da semana, período do dia, estação do
#        Nordeste e regime de preço, além de explicadas via SHAP.
#
#    (B) COLETA AUTOMÁTICA DE MÉTRICAS — ao final da execução, varre as
#        variáveis em memória, tenta casar um par (verdade, predição) e
#        grava/atualiza a linha correspondente à "Parte 5" no arquivo
#        metricas_recalculadas.csv, com deduplicação por (Parte, Tipo).
#
#  MODELO ANALISADO (herdado das Fases 03/04):
#      RandomForest com 50 features
#      Fase 04 (pós-tuning Optuna): MAE = 47.538 | R² = 0.8155
#      Hiperparâmetros: n_estimators=550, max_depth=27, min_samples_split=8,
#                       min_samples_leaf=10, max_features=0.8, bootstrap=False
#
#      Os hiperparâmetros são FIXOS aqui — vieram do JSON produzido pela Fase
#      04 e estão escritos literalmente em MODELOS_FASE05. Nenhuma otimização
#      é refeita nesta fase; o objetivo é diagnosticar, não melhorar.
#
#  MODELOS_FASE05 É UMA LISTA (com um único elemento):
#      A estrutura de lista foi mantida propositalmente para permitir, sem
#      alterar o pipeline, reintroduzir os demais modelos tunados da Fase 04
#      (RandomForest 70f, LightGBM 60f) e comparar seus perfis de erro lado a
#      lado. Basta acrescentar dicionários à lista.
#
#  DOIS CRITÉRIOS DE "SPIKE" EM USO (PROPOSITALMENTE DIFERENTES):
#      1) Z_SPIKE (estatístico): |z-score| > 2 sobre a distribuição do PLD real
#         do período de teste. Detecta outliers relativos ao próprio período,
#         tanto positivos (SPIKE_POS) quanto negativos (SPIKE_NEG). É o
#         critério de analise_spikes() e da coluna "spike_tipo".
#      2) SPIKE_ALTO_LIMIAR = R$ 500 (financeiro): o mesmo critério absoluto
#         usado nas Fases 01/03/06 para Spike_MAE/Spike_Bias. É calculado
#         explicitamente aqui (coluna "is_spike_R500" e função
#         calcular_spike_mae_bias_r500) para que os números sejam DIRETAMENTE
#         comparáveis entre todas as fases do TCC.
#      Os dois convivem porque respondem a perguntas distintas: o z-score
#      pergunta "esta hora foi atípica para o período?"; o limiar de R$500
#      pergunta "esta hora representa risco financeiro relevante?".
#
#  SPLIT ANTI-LEAKAGE:
#      Idêntico ao das Fases 03 e 04. Treino = histórico completo até
#      fev/2026 (incluindo jan/fev 2026); teste = mar/abr 2026. A função
#      split_treino_teste() remove ativamente do parquet de treino qualquer
#      linha >= março/2026, deduplica períodos presentes nos dois parquets e
#      faz checagem final de sanidade, emitindo alerta explícito em caso de
#      vazamento.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENTRADAS (INPUTS)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  O script não requer entradas manuais do usuário. Os parâmetros de controle
#  estão definidos nas constantes globais (logo após o tema do Rich):
#
#    BASE_DIR / INPUT_DIR    : caminho dos dados normalizados
#    OUTPUT_DIR              : "12_resultados_fase05" — onde a documentação dos
#                              hiperparâmetros e métricas é salva
#    CENARIO_DEFAULT         : "cenario_A_todos_anos"
#    STRATEGY_DEFAULT        : "HYBRID_AGGRESSIVE"
#    TESTE_ANO / TESTE_MESES : 2026 / [3, 4]  → período de teste fixo (Mar/Abr)
#    TREINO_EXTRA_MESES      : [1, 2]  → Jan/Fev/2026 movidos para o treino
#    MODELOS_FASE05          : modelo(s) analisado(s) + hiperparâmetros da Fase 04
#    Z_SPIKE                 : 2.0    → limiar do critério estatístico de spike
#    SPIKE_ALTO_LIMIAR       : 500.0  → limiar do critério financeiro de spike
#    N_MIN_R2_GRUPO          : 10     → n mínimo para calcular R² por subgrupo
#    FAIXAS_PLD              : Baixo (0–100), Médio (100–300), Alto (300–500),
#                              Extremo (> 500)
#
#  Arquivos de entrada esperados:
#    • INPUT_DIR / CENARIO_DEFAULT / STRATEGY_DEFAULT / TREINO_NORM.parquet
#    • INPUT_DIR / CENARIO_DEFAULT / STRATEGY_DEFAULT / TESTE_NORM.parquet
#      → dados já normalizados, contendo a coluna-alvo (PLD Nordeste), as
#        colunas-chave temporais (KEY_ANO, KEY_MES, KEY_DIA, KEY_HORA) e,
#        opcionalmente, a coluna REGIME_PRECO.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SAÍDAS (OUTPUTS)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. Documentação dos hiperparâmetros → OUTPUT_DIR /
#     • hiperparametros_fase05.json  — params + métricas das Fases 04 e 05
#     • hiperparametros_fase05.md    — versão legível, pronta para o texto do TCC
#     (se OUTPUT_DIR não estiver acessível, o script cai para /content e avisa)
#
#  2. CSV de métricas → PASTA_SAIDA / metricas_recalculadas.csv
#     Uma linha por (Parte, Tipo) — aqui Parte=5 — com RMSE, R², MAE e MAPE.
#     Deduplicado automaticamente a cada execução.
#
#  3. Saída no terminal (Rich) — o principal produto desta fase:
#     • Cabeçalho e tabela de hiperparâmetros considerados
#     • Resumo geral dos resíduos (bias, percentis do erro absoluto, extremos)
#     • Erro por FAIXA DE PLD (com R², Bias, MAPE e sMAPE por faixa)
#     • Análise de spikes por z-score, com "recall de spike"
#     • Spike_MAE / Spike_Bias no critério R$ 500 (comparável às outras fases)
#     • Erro por mês, hora, dia da semana, período do dia e estação do NE
#     • Erro por REGIME_PRECO (quando a coluna existe)
#     • Importância global SHAP (Top 20) e SHAP separado NORMAL × SPIKE_POS
#     • Top 15 piores previsões, com data, hora e classificação de spike
#     • Painel de diagnóstico com insights automáticos
#     • Tabela comparativa final e recomendação para a Fase 06
#
#  4. Objeto retornado por run_fase_05() (disponível em fase_05_output):
#     todos_residuos (DataFrame de resíduos por modelo), resultados_comparativo,
#     melhor_modelo e melhor_mae.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DIMENSÕES DE ANÁLISE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Dimensão            Recorte                         O que revela
#  Faixa de PLD        Baixo/Médio/Alto/Extremo        Se o erro cresce com o preço
#  Spike (z-score)     SPIKE_POS/NORMAL/SPIKE_NEG      Se o modelo "enxerga" o pico
#  Spike (R$ 500)      PLD > R$ 500                    Erro no que é caro de errar
#  Mês                 Mar / Abr 2026                  Deriva dentro do teste
#  Hora                00h–23h                         Curva de carga e ponta
#  Dia da semana       Seg–Dom                         Efeito de dia útil × fim de semana
#  Período do dia      Madrugada/Manhã/Tarde/Noite     Agrupamento operacional
#  Estação NE          Chuvoso/Transição/Seco          Sazonalidade hidrológica
#  REGIME_PRECO        R0, R1, ...                     Comportamento por regime
#  SHAP                Global e por tipo de spike      Quais variáveis pesam onde
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ETAPAS DO PROCESSO (resumo)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  N°  Etapa                        Estratégia                       Descrição
#  01  Carregamento                 read_parquet                     Lê TREINO_NORM e TESTE_NORM
#  02  Detecção automática          regex + tokens                   Localiza TARGET, KEY_* e REGIME_PRECO
#  03  Split anti-leakage           Filtro por AAAAMM + dedupe       Treino < mar/2026, teste = mar/abr/2026
#  04  Engenharia temporal          TemporalFE (24 derivadas)        Cíclicas, calendário, feriados
#  05  Ranking de importância       LightGBM rápido (200 árvores)    Reproduz a seleção de features da Fase 03
#  06  Treino do modelo final       Params fixos da Fase 04          Sem tuning: só reproduz o vencedor
#  07  Construção dos resíduos      construir_df_residuos()          Erro, erro relativo, faixa, z-score, estação
#  08  Resumo geral                 Percentis do erro absoluto       Enxerga cauda pesada e outliers
#  09  Erro por faixa de PLD        Agrupamento por faixa            Onde o erro se concentra
#  10  Análise de spikes            z-score + recall de spike        Capacidade de antecipar picos
#  11  Spike_MAE / Bias (R$500)     Critério das Fases 01/03/06      Comparabilidade entre fases
#  12  Análise temporal             Mês, hora, dia, período, estação Padrões sistemáticos de erro
#  13  Análise por regime           Agrupamento por REGIME_PRECO     Erro condicional ao regime
#  14  Interpretabilidade           SHAP TreeExplainer (2.000 obs)   Importância global e por spike
#  15  Piores casos                 Top 15 por erro absoluto         Casos concretos para o texto
#  16  Diagnóstico automático       Regras sobre bias/recall/MAE     Insights prontos para discussão
#  17  Documentação                 JSON + Markdown                  Hiperparâmetros e métricas
#  18  Coleta de métricas           Varredura de arrays no globals() Grava/atualiza metricas_recalculadas.csv
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONSIDERAÇÕES INICIAIS E OBSERVAÇÕES TÉCNICAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#  1. AMBIENTE DE EXECUÇÃO
#     O script foi desenvolvido e testado no Google Colab (Python 3.10+).
#     Instala automaticamente (via pip, silenciosamente) xgboost, lightgbm,
#     shap, holidays e rich caso ainda não estejam presentes. Se o SHAP não
#     estiver disponível, a análise de interpretabilidade é pulada com aviso e
#     o restante do pipeline segue normalmente.
#
#  2. POR QUE O MODELO É RETREINADO AQUI
#     A Fase 04 não persistiu o objeto do modelo, apenas os hiperparâmetros.
#     Como o pré-processamento, o split e a seleção de features são
#     determinísticos (RANDOM_STATE = 42), retreinar reproduz o mesmo modelo.
#     Pequenas diferenças de MAE entre as colunas "MAE F04" e "MAE F05" das
#     tabelas indicam alguma fonte de não determinismo residual (versão de
#     biblioteca, paralelismo) e merecem verificação antes de irem ao texto.
#
#  3. REGIME_PRECO: EXCLUÍDO COMO FEATURE, USADO COMO LENTE
#     A coluna é removida do conjunto de candidatos (seria leakage: deriva do
#     próprio PLD). Ainda assim, ela é copiada para o DataFrame de resíduos e
#     usada em analise_regime() apenas para AGRUPAR os erros a posteriori.
#     Agrupar não é prever: o modelo nunca viu essa informação.
#
#  4. R² NÃO É CALCULADO EM GRUPOS PEQUENOS
#     calcular_metricas_grupo() só devolve R² quando o subgrupo tem pelo menos
#     N_MIN_R2_GRUPO observações. Em amostras muito pequenas o R² é
#     instável e pode até ficar fortemente negativo por acaso, o que geraria
#     leitura equivocada nas tabelas por faixa.
#
#  5. INTERPRETAÇÃO DO BIAS
#     Bias = média(real − previsto). Positivo significa que o modelo SUBESTIMA
#     o preço; negativo, que SUPERESTIMA. Nas tabelas por faixa, valores acima
#     de |R$ 30| são destacados em cor. Subestimar preço alto é o erro mais
#     custoso para quem decide exposição no mercado livre — daí a atenção
#     especial ao Bias na faixa "Extremo" e ao Spike_Bias.
#
#  6. RECALL DE SPIKE
#     Percentual dos spikes REAIS cuja PREVISÃO também caiu em zona de spike
#     (mesmo limiar de z-score, aplicado à série prevista). É a métrica mais
#     severa desta fase: modelos de árvore treinados com perda quadrática
#     tendem a suavizar extremos, e um recall baixo mostra, de forma direta,
#     que o modelo não serve como alarme de pico — ainda que seu MAE global
#     seja bom. Esse achado é o que motiva o "especialista de spike" sugerido
#     para a Fase 06.
#
#  7. AMOSTRAGEM DO SHAP
#     TreeExplainer é exato para modelos de árvore, mas custoso. A análise usa
#     uma amostra de até 2.000 observações do teste, sorteada com semente fixa
#     (RandomState(42)) — reprodutível entre execuções. O mesmo vetor de
#     índices é reaproveitado para recortar o DataFrame de resíduos e produzir
#     o SHAP separado por tipo de spike, garantindo alinhamento linha a linha.
#     A quebra NORMAL × SPIKE_POS é o ponto mais interessante para o TCC:
#     mostra se o modelo muda de "lógica" ao prever um pico ou se continua
#     pesando as mesmas variáveis de sempre.
#
#  8. MÉTRICAS COMPLETAS EM TODOS OS RECORTES
#     Além de MAE, RMSE e Bias, cada grupo traz MAPE, sMAPE (robusto a valores
#     próximos de zero) e NRMSE (RMSE normalizado pela média do grupo). Sem o
#     NRMSE, comparar o MAE da faixa "Baixo" com o da faixa "Extremo" seria
#     enganoso: erro de R$ 40 sobre R$ 60 e sobre R$ 700 têm significados
#     completamente diferentes.
#
#  9. DIAGNÓSTICO AUTOMÁTICO É HEURÍSTICO
#     diagnostico_final() aplica regras de limiar fixas (bias > R$ 10, MAE de
#     extremos > 3× o MAE geral, recall < 20% / < 50%, pior mês > 1,5× o
#     melhor). São critérios pragmáticos para gerar leitura rápida, não testes
#     estatísticos — no texto do TCC devem ser apresentados como observações,
#     e não como significância comprovada.
#
# 10. COLETA AUTOMÁTICA DE MÉTRICAS (PARTE B)
#     Varre o namespace global por arrays numéricos com ao menos 10 elementos,
#     tentando casar uma variável de "verdade" (y_true, y_test, y_teste...)
#     com uma de "previsão" (y_pred_teste, y_pred_te, y_pred...). Detecta
#     ainda um possível caso de classificação binária, que não se aplica a
#     esta fase, puramente de regressão.
#
# 11. LIMITES CONHECIDOS
#     - A análise é de um único modelo. Perfis de erro de LightGBM e do
#       RandomForest de 70 features não são comparados aqui, embora a
#       estrutura de lista permita fazê-lo sem mudanças no pipeline.
#     - O período de teste tem apenas dois meses; a análise "por mês" tem,
#       portanto, dois pontos, e a de "estação NE" fica concentrada no período
#       chuvoso. São recortes ilustrativos, não conclusões sazonais.
#     - As predições ficam dentro de run_fase_05(); se o coletor de métricas
#       (Parte B) não encontrar y_true/y_pred_teste no escopo global, basta
#       expô-los antes da célula de coleta, como foi feito na Fase 03.
#
# 12. REPRODUTIBILIDADE
#     RANDOM_STATE = 42 é usado no modelo, no ranking de importância e na
#     amostragem do SHAP. Com os mesmos dados de entrada, todas as tabelas
#     desta fase são determinísticas entre execuções.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DEPENDÊNCIAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Biblioteca        Versão mínima  Finalidade
#  numpy             1.23           Operações numéricas e estatísticas
#  pandas            1.5            DataFrames, Parquet e agrupamentos
#  scikit-learn      1.2            RandomForest e métricas de regressão
#  shap              0.41           Interpretabilidade (TreeExplainer)
#  lightgbm          —              Ranking de importância das features
#  xgboost           —              Disponível em build_model (não usado no modelo atual)
#  holidays          —              Feriados nacionais para engenharia temporal
#  rich              13.0           Tabelas, painéis e regras visuais
#
#  Instalação automática (executada no início do script, se necessário):
#      pip install -q xgboost lightgbm shap holidays rich
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
#              │   ├── fase_04_best_params.json             (origem dos params)
#              │   └── 12_resultados_fase05/                (saída)
#              │       ├── hiperparametros_fase05.json
#              │       └── hiperparametros_fase05.md
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
#      exec(open('fase05_analise_profunda_erros_coleta_metricas.py').read())
#   ou simplesmente executar este arquivo como módulo principal.
#   As dependências (xgboost, lightgbm, shap, holidays, rich) são instaladas
#   automaticamente na primeira execução, se necessário.
#
#  Tempo estimado: um único treino de RandomForest (550 árvores, max_depth=27)
#  sobre o histórico completo, mais o cálculo do SHAP. Em CPU do Colab, algo
#  entre 5 e 15 minutos — dominado pelo treino e pelo TreeExplainer.
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
_ensure("shap"); _ensure("holidays"); _ensure("rich")

import warnings
warnings.filterwarnings("ignore")

import re
import time
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
import lightgbm as lgb

# SHAP e holidays são opcionais: a ausência degrada a análise, não a quebra
try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

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
from rich.rule import Rule

# ==============================================================================
# Tema visual do console. Além das chaves herdadas das fases anteriores, esta
# fase acrescenta spike_pos/spike_neg para sinalizar direção do erro extremo.
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
})
console = Console(theme=THEME)

# ==============================================================================
# CONSTANTES GLOBAIS DE CONTROLE
# ==============================================================================
BASE_DIR   = Path("/content/drive/MyDrive/TCC_PLD_Project/09-ESCRITA_TCC/PARTES_TCC/codigos/dados")
INPUT_DIR  = BASE_DIR / "10_dados_normalizados"
OUTPUT_DIR = BASE_DIR / "12_resultados_fase05"   # documentação dos hiperparâmetros

CENARIO_DEFAULT  = "cenario_A_todos_anos"
STRATEGY_DEFAULT = "HYBRID_AGGRESSIVE"
TARGET_CANONICAL = "01_CCEE_NORDESTE_TARGET_PLD_HORA_NORDESTE"

# ── Datas de corte ─────────────────────────────────────────────────────────
TESTE_ANO          = 2026
TREINO_EXTRA_MESES = [1, 2]   # jan e fev/2026 → TREINO
TESTE_MESES        = [3, 4]   # mar e abr/2026 → TESTE
# ───────────────────────────────────────────────────────────────────────────

RANDOM_STATE      = 42
Z_SPIKE           = 2.0     # critério ESTATÍSTICO de spike (|z-score| > 2)
SPIKE_ALTO_LIMIAR = 500.0   # critério FINANCEIRO (mesmo das Fases 01/03/06)
N_MIN_R2_GRUPO    = 10      # abaixo disso, R² por grupo não é calculado (instável)

# Faixas de preço usadas para segmentar o erro. O corte em R$ 500 coincide
# com SPIKE_ALTO_LIMIAR: a faixa "Extremo" é exatamente o conjunto de spikes
# no critério financeiro.
FAIXAS_PLD = [
    ("Baixo",   0,   100),
    ("Médio",   100, 300),
    ("Alto",    300, 500),
    ("Extremo", 500, 999999),
]
DIAS_SEMANA_PT = {0:"Seg",1:"Ter",2:"Qua",3:"Qui",4:"Sex",5:"Sáb",6:"Dom"}
MESES_PT = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",
            7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}

# ── MELHOR modelo da Fase 03/04 (único avaliado na Fase 05) ─────────────────
# RandomForest com 50 features — hiperparâmetros reais vencedores do Optuna.
# Estrutura em LISTA para permitir reintroduzir os demais modelos tunados
# (RandomForest 70f, LightGBM 60f) sem alterar o pipeline.
MODELOS_FASE05 = [
    {
        "nome"      : "RandomForest",
        "n_features": 50,
        "mae_fase04": 47.538,   # métricas de referência, para comparação
        "r2_fase04" : 0.8155,
        "params"    : {
            "n_estimators"     : 550,
            "max_depth"        : 27,
            "min_samples_split": 8,
            "min_samples_leaf" : 10,
            "max_features"     : 0.8,
            "bootstrap"        : False,   # árvores treinam sobre a amostra completa
            "random_state"     : RANDOM_STATE,
            "n_jobs"           : -1,
        },
    },
]
# ────────────────────────────────────────────────────────────────────────────

# ==============================================================================
# UTILITÁRIOS
# ==============================================================================
def safe_round(v, d=4):
    """Arredonda protegendo contra None, NaN e Inf (que quebram o json.dump)."""
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
    """
    Localiza REGIME_PRECO. Como feature seria leakage (deriva do próprio PLD),
    mas a coluna é preservada para AGRUPAR os erros na análise a posteriori.
    """
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

    Trava anti-leakage idêntica às fases 03/04 corrigidas:
      - Mantém no TREINO apenas (ano*100+mes) < 202603, descartando qualquer
        linha >= mar/2026 que esteja no parquet de treino. Se não houver 2026
        no treino, nada é removido (resultado idêntico ao split antigo).
      - Deduplica jan/fev 2026 (parquet de treino x parquet de teste).
      - Checagem final de sanidade contra leakage.
    """
    k_ano  = key_cols.get("KEY_ANO")
    k_mes  = key_cols.get("KEY_MES")
    k_dia  = key_cols.get("KEY_DIA")
    k_hora = key_cols.get("KEY_HORA")

    # Chave numérica ano*100+mes para comparar períodos com segurança
    corte_treino = TESTE_ANO * 100 + min(TESTE_MESES)   # 202603

    def _ano_mes(df):
        return df[k_ano].astype(int) * 100 + df[k_mes].astype(int)

    # TESTE = mar + abr / 2026 (sempre vindo do parquet de teste)
    mask_teste = (
        (df_teste_raw[k_ano] == TESTE_ANO) &
        (df_teste_raw[k_mes].isin(TESTE_MESES))
    )
    df_teste_final = df_teste_raw[mask_teste].reset_index(drop=True)

    # EXTRA = jan + fev / 2026 → migram do parquet de teste para o TREINO
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
                f"   [warning]⚠  Removidas {dup:,} linhas duplicadas no TREINO "
                f"(chave {chaves})[/]"
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

# ==============================================================================
# FEATURE ENGINEERING
# Idêntica à das Fases 03/04 — condição necessária para reproduzir exatamente
# o mesmo conjunto de 50 features com que o modelo foi tunado.
# ==============================================================================
class TemporalFE:
    """
    Gera as features temporais obrigatórias a partir das colunas-chave:
      • CYC_*  — codificação seno/cosseno (hora, dia da semana, dia do mês,
                 mês e dia do ano), preservando a continuidade circular;
      • CAL_*  — flags de calendário (fim de semana, pico, madrugada, período
                 seco/chuvoso do NE, ano crítico, feriado, dia útil);
      • TEMP_* — trimestre, dia do ano, progresso do ano/mês e interações.
    Acrescenta ainda as auxiliares _DATA e _DIA_SEMANA (prefixo "_" as exclui
    automaticamente do conjunto de features).
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
        # Auxiliares para as análises (não entram como features)
        df_out["_DATA"]       = pd.to_datetime(
            {"year": ano, "month": mes, "day": dia}, errors="coerce"
        ).dt.date
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
    Recebe df COMPLETO (com target). X = candidates, y = target_col.

    Mesmos parâmetros das Fases 03/04: é o que garante que as 50 features
    selecionadas aqui sejam exatamente as usadas no tuning.
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
    """Instancia o modelo a partir do nome e dos hiperparâmetros da Fase 04."""
    if nome == "RandomForest":
        return RandomForestRegressor(**params)
    elif nome == "LightGBM":
        return lgb.LGBMRegressor(**params)
    elif nome == "XGBoost":
        return xgb.XGBRegressor(**params)
    raise ValueError(f"Modelo desconhecido: {nome}")

# ==============================================================================
# MÉTRICAS — versão completa (MAE, RMSE, Bias, MAPE, sMAPE, NRMSE, R² por grupo)
# ==============================================================================
def calcular_metricas_grupo(y_true, y_pred, n_min=5):
    """
    Bateria de métricas para um SUBGRUPO de observações (faixa, hora, mês...).

    Devolve None se o grupo for pequeno demais (n < n_min) — assim as funções
    de exibição simplesmente pulam a linha em vez de mostrar números frágeis.
    """
    if len(y_true) < n_min:
        return None
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    # Bias positivo = modelo SUBESTIMA; negativo = SUPERESTIMA
    bias = float(np.mean(y_true - y_pred))
    # Clipa o denominador em 1e-3: PLD no piso zeraria a divisão
    mape = float(np.mean(
        np.abs(y_true - y_pred) / np.clip(np.abs(y_true), 1e-3, None)
    )) * 100

    # sMAPE: simétrico, evita explosão quando y_true ~ 0
    smape = float(np.mean(
        2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )) * 100

    # NRMSE: RMSE normalizado pela média do real (compara grupos de magnitudes diferentes)
    nrmse = float(rmse / (np.mean(y_true) + 1e-8) * 100)

    # R² por grupo: só calculado se houver observações suficientes (senão é ruído)
    r2 = safe_round(float(r2_score(y_true, y_pred)), 4) if len(y_true) >= N_MIN_R2_GRUPO else None

    return {
        "n"          : len(y_true),
        "MAE"        : safe_round(mae, 3),
        "RMSE"       : safe_round(rmse, 3),
        "Bias"       : safe_round(bias, 3),
        "MAPE_%"     : safe_round(mape, 2),
        "sMAPE_%"    : safe_round(smape, 2),
        "NRMSE_%"    : safe_round(nrmse, 2),
        "R2"         : r2,
        "y_mean"     : safe_round(float(np.mean(y_true)), 2),
        "y_pred_mean": safe_round(float(np.mean(y_pred)), 2),
    }


def construir_df_residuos(df_teste, y_teste, y_pred, key_cols, regime_col=None):
    """
    Monta o DataFrame que sustenta TODA a análise desta fase: uma linha por
    hora do período de teste, com real, previsto, erro e todos os rótulos de
    agrupamento (faixa de PLD, tipo de spike, estação, período do dia...).
    """
    k_ano, k_mes  = key_cols["KEY_ANO"], key_cols["KEY_MES"]
    k_dia, k_hora = key_cols["KEY_DIA"], key_cols["KEY_HORA"]

    datas = pd.to_datetime({
        "year" : df_teste[k_ano].astype(int),
        "month": df_teste[k_mes].astype(int),
        "day"  : df_teste[k_dia].astype(int),
    }, errors="coerce")

    df = pd.DataFrame({
        "data"      : datas.dt.strftime("%Y-%m-%d"),
        "ano"       : df_teste[k_ano].astype(int).values,
        "mes"       : df_teste[k_mes].astype(int).values,
        "dia"       : df_teste[k_dia].astype(int).values,
        "hora"      : df_teste[k_hora].astype(int).values,
        "dia_semana": datas.dt.weekday,
        "real"      : y_teste,
        "previsto"  : y_pred,
    })
    # Convenção do erro: real − previsto (positivo = subestimação)
    df["erro"]       = df["real"] - df["previsto"]
    df["abs_erro"]   = df["erro"].abs()
    df["erro_rel_%"] = (df["erro"] / np.clip(df["real"].abs(), 1e-3, None)) * 100

    # --- Rótulo 1: faixa de preço (o corte em 500 = critério financeiro) ---
    def faixa(v):
        for nome, lo, hi in FAIXAS_PLD:
            if lo <= v < hi:
                return nome
        return FAIXAS_PLD[-1][0]
    df["faixa_PLD"] = df["real"].apply(faixa)

    # --- Rótulo 2: spike ESTATÍSTICO (z-score sobre o próprio período) ---
    media, std = df["real"].mean(), df["real"].std()
    z = (df["real"] - media) / std if std > 0 else 0
    df["z_score"]    = z
    df["spike_tipo"] = np.where(z > Z_SPIKE, "SPIKE_POS",
                         np.where(z < -Z_SPIKE, "SPIKE_NEG", "NORMAL"))

    # Flag adicional com o critério FINANCEIRO (R$500), consistente com Fases 01/03/06
    df["is_spike_R500"] = df["real"] > SPIKE_ALTO_LIMIAR

    # --- Rótulo 3: estação hidrológica do Nordeste ---
    def estacao_ne(m):
        if m in [12,1,2,3]: return "Chuvoso"
        if m in [6,7,8,9,10,11]: return "Seco"
        return "Transição"
    df["estacao_NE"] = df["mes"].apply(estacao_ne)

    # --- Rótulo 4: período operacional do dia ---
    def periodo(h):
        if 0 <= h <= 5:   return "Madrugada"
        if 6 <= h <= 11:  return "Manhã"
        if 12 <= h <= 17: return "Tarde"
        return "Noite/Pico"
    df["periodo_dia"] = df["hora"].apply(periodo)

    # REGIME_PRECO entra apenas como rótulo de agrupamento — nunca como feature
    if regime_col and regime_col in df_teste.columns:
        df["regime_preco"] = df_teste[regime_col].astype(int).values
    return df


def calcular_spike_mae_bias_r500(df_res):
    """
    Spike_MAE / Spike_Bias usando o MESMO critério (PLD > R$500) das Fases
    01/03/06 — mantido separado da análise de spikes por z-score (que serve
    a um propósito estatístico diferente), para permitir comparação direta
    dos números entre todas as fases do TCC.

    Exige mais de 5 observações; abaixo disso devolve (None, None, n).
    """
    sub = df_res[df_res["is_spike_R500"]]
    if len(sub) <= 5:
        return None, None, int(len(sub))
    spike_mae  = float(mean_absolute_error(sub["real"].values, sub["previsto"].values))
    spike_bias = float(np.mean(sub["real"].values - sub["previsto"].values))
    return safe_round(spike_mae, 3), safe_round(spike_bias, 3), int(len(sub))

# ==============================================================================
# DISPLAYS
# Cada função abaixo consome o df_res e imprime UMA dimensão de análise.
# Nenhuma delas altera o modelo ou os dados.
# ==============================================================================
def print_header():
    """Painel de abertura com o modelo analisado e suas métricas da Fase 04."""
    cfg = MODELOS_FASE05[0]
    console.print(Panel(
        Text.assemble(
            ("🔬 FASE 05 — ANÁLISE PROFUNDA DE ERROS\n", "bold green"),
            ("   Melhor modelo da Fase 03/04: "
             f"{cfg['nome']} ({cfg['n_features']} features) "
             f"| MAE Fase04={cfg['mae_fase04']:.3f} | R²={cfg['r2_fase04']:.4f}\n", "muted"),
            ("   Treino: histórico até fev/2026 (inclui jan/fev 2026) "
             "| Teste: mar/abr 2026\n", "muted"),
            (f"   Cenário: {CENARIO_DEFAULT} | Strategy: {STRATEGY_DEFAULT}\n", "muted"),
        ),
        title="[header_v2]  TCC PLD — ANÁLISE DE ERROS  [/header_v2]",
        border_style="purple4", padding=(1, 4),
    ))


def print_hiperparametros(cfg):
    """Exibe os hiperparâmetros considerados, para documentação/anotação."""
    t = Table(
        title=(f"🔧 HIPERPARÂMETROS CONSIDERADOS — "
               f"{cfg['nome']} ({cfg['n_features']} features)"),
        box=box.DOUBLE_EDGE, border_style="yellow",
        show_lines=False, min_width=70,
    )
    t.add_column("Hiperparâmetro", style="bold cyan",  min_width=28)
    t.add_column("Valor",          style="bold white", min_width=24)
    for k, v in cfg["params"].items():
        t.add_row(k, str(v))
    t.add_row("─" * 26, "─" * 22)
    t.add_row("[muted]N features[/muted]",        str(cfg["n_features"]))
    t.add_row("[muted]MAE (Fase 04)[/muted]",     f"{cfg['mae_fase04']:.3f}")
    t.add_row("[muted]R² (Fase 04)[/muted]",      f"{cfg['r2_fase04']:.4f}")
    t.add_row("[muted]Origem dos params[/muted]", "Optuna (Fase 04)")
    console.print(t)


def salvar_hiperparametros(resultados_full, output_dir=OUTPUT_DIR):
    """
    Salva os hiperparâmetros + métricas (Fase 04 e Fase 05) para documentação:
      - hiperparametros_fase05.json
      - hiperparametros_fase05.md

    Se o diretório do Drive não estiver acessível (Drive não montado, por
    exemplo), cai para /content e avisa, em vez de abortar a execução.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        destino = output_dir
    except Exception:
        destino = Path("/content")
        console.print(f"   [warning]⚠  Sem acesso a {output_dir}; salvando em {destino}[/]")

    # JSON — converte tipos numpy para nativos (np.int64 não é serializável)
    payload = {}
    for r in resultados_full:
        payload[r["nome"]] = {
            "n_features"     : r["n_features"],
            "hiperparametros": {
                k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in r["params"].items()
            },
            "mae_fase04"      : r["mae_f04"],
            "r2_fase04"       : r["r2_f04"],
            "mae_fase05"      : r["mae_f05"],
            "r2_fase05"       : r["r2"],
            "sMAPE_%_fase05"  : r.get("smape"),
            "NRMSE_%_fase05"  : r.get("nrmse"),
            "spike_mae_R500"  : r.get("spike_mae_r500"),
            "spike_bias_R500" : r.get("spike_bias_r500"),
        }
    p_json = destino / "hiperparametros_fase05.json"
    with open(p_json, "w") as f:
        json.dump(payload, f, indent=2)

    # Markdown legível — pronto para colar na seção de metodologia do TCC
    md = ["# Hiperparâmetros — Fase 05", ""]
    for r in resultados_full:
        md += [
            f"## {r['nome']} ({r['n_features']} features)",
            "",
            f"- MAE Fase 04: {r['mae_f04']:.3f} | R² Fase 04: {r['r2_f04']:.4f}",
            f"- MAE Fase 05: {r['mae_f05']:.3f} | R² Fase 05: {r['r2']:.4f}",
            f"- sMAPE Fase 05: {r.get('smape')} % | NRMSE Fase 05: {r.get('nrmse')} %",
            f"- Spike_MAE (R$500): {r.get('spike_mae_r500')} | "
            f"Spike_Bias (R$500): {r.get('spike_bias_r500')}",
            "",
            "| Hiperparâmetro | Valor |",
            "|---|---|",
        ]
        for k, v in r["params"].items():
            md.append(f"| `{k}` | {v} |")
        md.append("")
    p_md = destino / "hiperparametros_fase05.md"
    p_md.write_text("\n".join(md), encoding="utf-8")

    console.print("\n   [success]💾 Hiperparâmetros salvos para documentação:[/]")
    console.print(f"   [success]   • {p_json}[/]")
    console.print(f"   [success]   • {p_md}[/]")


def print_resumo_residuos(df_res, model_name):
    """
    Panorama global do erro. Os percentis (P75, P90, P95) são a parte mais
    informativa: se P95 for muito maior que a mediana, o erro tem cauda
    pesada — poucas horas concentram a maior parte do prejuízo.
    """
    t = Table(
        title=f"📋 RESUMO GERAL DOS RESÍDUOS — {model_name}",
        box=box.ROUNDED, border_style="cyan",
        show_lines=False, min_width=70,
    )
    t.add_column("Métrica", style="bold", min_width=30)
    t.add_column("Valor",   justify="right", min_width=20)
    erro = df_res["erro"].values
    abse = df_res["abs_erro"].values
    real = df_res["real"].values
    prev = df_res["previsto"].values

    rmse_geral  = float(np.sqrt(np.mean(erro**2)))
    smape_geral = float(np.mean(2*abse/(np.abs(real)+np.abs(prev)+1e-8))*100)
    nrmse_geral = float(rmse_geral/(np.mean(real)+1e-8)*100)

    rows = [
        ("N observações teste",     f"{len(df_res):,}"),
        ("Erro médio (bias)",        f"R$ {erro.mean():+.3f}"),
        ("Erro mediano",             f"R$ {float(np.median(erro)):+.3f}"),
        ("Desvio padrão do erro",    f"R$ {erro.std():.3f}"),
        ("MAE",                      f"R$ {abse.mean():.3f}"),
        ("RMSE",                     f"R$ {rmse_geral:.3f}"),
        ("sMAPE",                    f"{smape_geral:.2f}%"),
        ("NRMSE",                    f"{nrmse_geral:.2f}%"),
        ("Erro absoluto mediano",    f"R$ {float(np.median(abse)):.3f}"),
        ("Erro abs. P75",            f"R$ {float(np.percentile(abse, 75)):.3f}"),
        ("Erro abs. P90",            f"R$ {float(np.percentile(abse, 90)):.3f}"),
        ("Erro abs. P95",            f"R$ {float(np.percentile(abse, 95)):.3f}"),
        ("Erro abs. máximo",         f"R$ {abse.max():.3f}"),
        ("─"*28,                     "─"*18),
        ("PLD real médio",           f"R$ {df_res['real'].mean():.3f}"),
        ("PLD real mediano",         f"R$ {df_res['real'].median():.3f}"),
        ("PLD real máximo",          f"R$ {df_res['real'].max():.3f}"),
        ("PLD real mínimo",          f"R$ {df_res['real'].min():.3f}"),
        ("PLD previsto médio",       f"R$ {df_res['previsto'].mean():.3f}"),
    ]
    for k, v in rows:
        t.add_row(k, v)
    console.print(t)


def analise_por_faixa_pld(df_res, model_name):
    """
    Erro segmentado por faixa de preço. É a tabela mais importante da fase:
    revela se o MAE global é bom apenas porque a maioria das horas está na
    faixa "Baixo", enquanto a faixa "Extremo" (a que importa financeiramente)
    concentra erro desproporcional.
    """
    console.rule(f"[highlight]🔬 FAIXA DE PLD — {model_name}[/highlight]")
    t = Table(
        title=f"📊 MAE / RMSE / R² / Bias / N por FAIXA DE PLD — {model_name}",
        box=box.DOUBLE_EDGE, border_style="cyan",
        show_lines=True, min_width=130,
    )
    t.add_column("Faixa",      style="bold",    min_width=10)
    t.add_column("Intervalo",  justify="center",min_width=16)
    t.add_column("N",          justify="right", min_width=8)
    t.add_column("% teste",    justify="right", min_width=9)
    t.add_column("Real médio", justify="right", min_width=12)
    t.add_column("Prev médio", justify="right", min_width=12)
    t.add_column("MAE",        justify="right", min_width=10)
    t.add_column("RMSE",       justify="right", min_width=10)
    t.add_column("R²",         justify="right", min_width=9)
    t.add_column("Bias",       justify="right", min_width=10)
    t.add_column("MAPE %",     justify="right", min_width=9)
    t.add_column("sMAPE %",    justify="right", min_width=9)
    n_total = len(df_res)
    for nome, lo, hi in FAIXAS_PLD:
        sub = df_res[df_res["faixa_PLD"] == nome]
        if len(sub) == 0:
            continue
        m = calcular_metricas_grupo(sub["real"].values, sub["previsto"].values)
        if m is None:
            continue
        intervalo = f"R$ {lo}–{hi if hi < 999999 else '∞'}"
        # Cor do bias: vermelho quando superestima forte, azul quando subestima forte
        b_st = "spike_neg" if m["Bias"] > 30 else "spike_pos" if m["Bias"] < -30 else "white"
        r2_str = f"{m['R2']:.3f}" if m["R2"] is not None else "—"
        t.add_row(
            nome, intervalo,
            f"{m['n']:,}", f"{100*m['n']/n_total:.1f}%",
            f"R$ {m['y_mean']:.2f}", f"R$ {m['y_pred_mean']:.2f}",
            f"R$ {m['MAE']:.2f}", f"R$ {m['RMSE']:.2f}",
            r2_str,
            f"[{b_st}]R$ {m['Bias']:+.2f}[/{b_st}]",
            f"{m['MAPE_%']:.1f}%", f"{m['sMAPE_%']:.1f}%",
        )
    console.print(t)
    console.print(
        "   [muted]R² não é exibido para grupos pequenos (n < "
        f"{N_MIN_R2_GRUPO}) por ser instável em amostras reduzidas.[/]"
    )


def analise_spikes(df_res, model_name):
    """
    Análise dupla dos picos:
      (1) por z-score — inclui o RECALL DE SPIKE, que mede se a previsão
          também caiu em zona de pico quando o real foi pico;
      (2) por R$ 500 — Spike_MAE/Spike_Bias no critério das Fases 01/03/06.

    Devolve (spike_mae_r500, spike_bias_r500, n_spike_r500) para alimentar a
    tabela comparativa final.
    """
    console.rule(f"[highlight]🔬 SPIKES (z > {Z_SPIKE}) — {model_name}[/highlight]")
    media = df_res["real"].mean()
    std   = df_res["real"].std()
    console.print(
        f"   [muted]Média={media:.2f} | Std={std:.2f} | "
        f"Limiar spike (z-score): PLD > {media + Z_SPIKE*std:.2f} ou < {media - Z_SPIKE*std:.2f}[/]"
    )
    console.print(
        f"   [muted]⚠ Este critério (z-score) é ESTATÍSTICO e difere do critério "
        f"financeiro (PLD > R$ {SPIKE_ALTO_LIMIAR:.0f}) usado em Spike_MAE/Spike_Bias "
        f"abaixo e nas Fases 01/03/06.[/]"
    )
    t = Table(
        title=f"🚨 ANÁLISE DE SPIKES (z-score) — {model_name}",
        box=box.DOUBLE_EDGE, border_style="red",
        show_lines=True, min_width=100,
    )
    t.add_column("Tipo",       style="bold",    min_width=14)
    t.add_column("N",          justify="right", min_width=8)
    t.add_column("% teste",    justify="right", min_width=9)
    t.add_column("Real médio", justify="right", min_width=12)
    t.add_column("Prev médio", justify="right", min_width=12)
    t.add_column("MAE",        justify="right", min_width=10)
    t.add_column("Bias",       justify="right", min_width=12)
    t.add_column("Recall ★",   justify="right", min_width=14)
    n_total = len(df_res)
    for tipo in ["SPIKE_POS", "NORMAL", "SPIKE_NEG"]:
        sub = df_res[df_res["spike_tipo"] == tipo]
        if len(sub) == 0:
            continue
        m = calcular_metricas_grupo(sub["real"].values, sub["previsto"].values)
        if m is None:
            continue
        if tipo in ["SPIKE_POS", "SPIKE_NEG"]:
            # Recall: aplica o MESMO limiar de z-score à série PREVISTA,
            # usando média/desvio da série REAL como referência comum
            z_pred = (sub["previsto"] - media) / std
            acertou = ((z_pred > Z_SPIKE).sum() if tipo == "SPIKE_POS"
                       else (z_pred < -Z_SPIKE).sum())
            recall_str = f"{100*acertou/len(sub):.1f}% ({acertou}/{len(sub)})"
        else:
            recall_str = "—"
        cor = ("spike_pos" if tipo == "SPIKE_POS"
               else "spike_neg" if tipo == "SPIKE_NEG" else "white")
        t.add_row(
            f"[{cor}]{tipo}[/{cor}]",
            f"{m['n']:,}", f"{100*m['n']/n_total:.1f}%",
            f"R$ {m['y_mean']:.2f}", f"R$ {m['y_pred_mean']:.2f}",
            f"R$ {m['MAE']:.2f}", f"R$ {m['Bias']:+.2f}",
            recall_str,
        )
    console.print(t)
    console.print("   [muted]★ Recall: % spikes reais cuja PREVISÃO também caiu em zona de spike[/]")

    # ---------- Spike_MAE / Spike_Bias no critério R$500 (Fases 01/03/06) ----------
    # Números diretamente comparáveis com as tabelas das outras fases do TCC.
    spike_mae_r500, spike_bias_r500, n_spike_r500 = calcular_spike_mae_bias_r500(df_res)
    t2 = Table(
        title=f"💰 Spike_MAE / Spike_Bias (critério R$ {SPIKE_ALTO_LIMIAR:.0f}) — {model_name}",
        box=box.ROUNDED, border_style="magenta", show_lines=False, min_width=70,
    )
    t2.add_column("Métrica", style="bold", min_width=20)
    t2.add_column("Valor", justify="right", min_width=20)
    t2.add_row("N spikes (PLD > R$500)", f"{n_spike_r500}")
    t2.add_row("Spike_MAE", f"R$ {spike_mae_r500:.2f}" if spike_mae_r500 is not None else "—")
    t2.add_row(
        "Spike_Bias",
        f"R$ {spike_bias_r500:+.2f}" if spike_bias_r500 is not None else "—",
    )
    console.print(t2)
    return spike_mae_r500, spike_bias_r500, n_spike_r500


def analise_temporal(df_res, model_name):
    """
    Cinco recortes temporais em sequência: mês, hora, dia da semana, período
    do dia e estação do NE. As barras "█" são normalizadas pelo maior MAE de
    cada recorte — servem para leitura visual rápida, não como escala absoluta.
    """
    console.rule(f"[highlight]🔬 TEMPORAL — {model_name}[/highlight]")

    # Por mês (no teste atual, só Mar e Abr/2026)
    t = Table(title=f"📅 ERRO POR MÊS — {model_name}",
              box=box.ROUNDED, border_style="cyan",
              show_lines=False, min_width=80)
    t.add_column("Mês",       style="bold",    min_width=8)
    t.add_column("N",         justify="right", min_width=8)
    t.add_column("Real méd.", justify="right", min_width=12)
    t.add_column("MAE",       justify="right", min_width=10)
    t.add_column("Bias",      justify="right", min_width=12)
    t.add_column("Barra MAE", justify="left",  min_width=20)
    max_mae = df_res.groupby("mes")["abs_erro"].mean().max() or 1
    for mes_num in sorted(df_res["mes"].unique()):
        sub = df_res[df_res["mes"] == mes_num]
        m = calcular_metricas_grupo(sub["real"].values, sub["previsto"].values)
        if m is None: continue
        bar = "█" * int(20 * m["MAE"] / max_mae)
        t.add_row(MESES_PT.get(mes_num, str(mes_num)),
                  f"{m['n']:,}", f"R$ {m['y_mean']:.2f}",
                  f"R$ {m['MAE']:.2f}", f"R$ {m['Bias']:+.2f}", bar)
    console.print(t)

    # Por hora — revela se o erro se concentra na ponta (18h–21h)
    t = Table(title=f"🕐 ERRO POR HORA — {model_name}",
              box=box.ROUNDED, border_style="cyan",
              show_lines=False, min_width=80)
    t.add_column("Hora",      style="bold",    min_width=6)
    t.add_column("N",         justify="right", min_width=8)
    t.add_column("Real méd.", justify="right", min_width=12)
    t.add_column("MAE",       justify="right", min_width=10)
    t.add_column("Bias",      justify="right", min_width=12)
    t.add_column("Barra MAE", justify="left",  min_width=20)
    max_mae_h = df_res.groupby("hora")["abs_erro"].mean().max() or 1
    for h in sorted(df_res["hora"].unique()):
        sub = df_res[df_res["hora"] == h]
        m = calcular_metricas_grupo(sub["real"].values, sub["previsto"].values)
        if m is None: continue
        bar = "█" * int(20 * m["MAE"] / max_mae_h)
        t.add_row(f"{int(h):02d}h", f"{m['n']:,}", f"R$ {m['y_mean']:.2f}",
                  f"R$ {m['MAE']:.2f}", f"R$ {m['Bias']:+.2f}", bar)
    console.print(t)

    # Por dia da semana — testa se as features de calendário deram conta
    t = Table(title=f"📅 ERRO POR DIA DA SEMANA — {model_name}",
              box=box.ROUNDED, border_style="cyan",
              show_lines=False, min_width=60)
    t.add_column("Dia",  style="bold",    min_width=8)
    t.add_column("N",    justify="right", min_width=8)
    t.add_column("MAE",  justify="right", min_width=10)
    t.add_column("Bias", justify="right", min_width=12)
    for d in sorted(df_res["dia_semana"].unique()):
        sub = df_res[df_res["dia_semana"] == d]
        m = calcular_metricas_grupo(sub["real"].values, sub["previsto"].values)
        if m is None: continue
        t.add_row(DIAS_SEMANA_PT.get(d, str(d)), f"{m['n']:,}",
                  f"R$ {m['MAE']:.2f}", f"R$ {m['Bias']:+.2f}")
    console.print(t)

    # Por período do dia — agrupamento operacional das 24 horas
    t = Table(title=f"⏰ ERRO POR PERÍODO DO DIA — {model_name}",
              box=box.ROUNDED, border_style="cyan",
              show_lines=False, min_width=70)
    t.add_column("Período",   style="bold",    min_width=14)
    t.add_column("N",         justify="right", min_width=8)
    t.add_column("Real méd.", justify="right", min_width=12)
    t.add_column("MAE",       justify="right", min_width=10)
    t.add_column("Bias",      justify="right", min_width=12)
    for p in ["Madrugada", "Manhã", "Tarde", "Noite/Pico"]:
        sub = df_res[df_res["periodo_dia"] == p]
        if len(sub) == 0: continue
        m = calcular_metricas_grupo(sub["real"].values, sub["previsto"].values)
        if m is None: continue
        t.add_row(p, f"{m['n']:,}", f"R$ {m['y_mean']:.2f}",
                  f"R$ {m['MAE']:.2f}", f"R$ {m['Bias']:+.2f}")
    console.print(t)

    # Por estação NE — com teste de mar/abr, tende a cair quase todo em "Chuvoso"
    t = Table(title=f"🌧️  ERRO POR ESTAÇÃO NE — {model_name}",
              box=box.ROUNDED, border_style="cyan",
              show_lines=False, min_width=70)
    t.add_column("Estação",   style="bold",    min_width=12)
    t.add_column("N",         justify="right", min_width=8)
    t.add_column("Real méd.", justify="right", min_width=12)
    t.add_column("MAE",       justify="right", min_width=10)
    t.add_column("Bias",      justify="right", min_width=12)
    for e in ["Chuvoso", "Transição", "Seco"]:
        sub = df_res[df_res["estacao_NE"] == e]
        if len(sub) == 0: continue
        m = calcular_metricas_grupo(sub["real"].values, sub["previsto"].values)
        if m is None: continue
        t.add_row(e, f"{m['n']:,}", f"R$ {m['y_mean']:.2f}",
                  f"R$ {m['MAE']:.2f}", f"R$ {m['Bias']:+.2f}")
    console.print(t)


def analise_regime(df_res, model_name):
    """
    Erro por REGIME_PRECO. Lembrete: a coluna foi EXCLUÍDA das features (é
    leakage); aqui ela serve só como lente de leitura a posteriori. Se a
    coluna não existir no parquet, a análise é silenciosamente pulada.
    """
    if "regime_preco" not in df_res.columns:
        return
    console.rule(f"[highlight]🔬 REGIME_PRECO — {model_name}[/highlight]")
    t = Table(
        title=f"📊 ERRO POR REGIME_PRECO — {model_name}",
        box=box.DOUBLE_EDGE, border_style="magenta",
        show_lines=True, min_width=90,
    )
    t.add_column("Regime",     style="bold",    min_width=10)
    t.add_column("N",          justify="right", min_width=8)
    t.add_column("% teste",    justify="right", min_width=9)
    t.add_column("Real médio", justify="right", min_width=12)
    t.add_column("Prev médio", justify="right", min_width=12)
    t.add_column("MAE",        justify="right", min_width=10)
    t.add_column("Bias",       justify="right", min_width=12)
    n_total = len(df_res)
    for r in sorted(df_res["regime_preco"].unique()):
        sub = df_res[df_res["regime_preco"] == r]
        m = calcular_metricas_grupo(sub["real"].values, sub["previsto"].values)
        if m is None: continue
        t.add_row(f"R{int(r)}", f"{m['n']:,}",
                  f"{100*m['n']/n_total:.1f}%",
                  f"R$ {m['y_mean']:.2f}", f"R$ {m['y_pred_mean']:.2f}",
                  f"R$ {m['MAE']:.2f}", f"R$ {m['Bias']:+.2f}")
    console.print(t)


def analise_shap(model, X_test, feature_names, df_res, model_name, top_n=20):
    """
    Interpretabilidade via SHAP (TreeExplainer, exato para modelos de árvore).

    Duas saídas:
      1) Importância GLOBAL (|SHAP| médio) das top_n features, colorida por
         família (temporais, reservatório/EAR, carga, preço/CMO);
      2) Top 10 features separadamente em horas NORMAIS e em SPIKE_POS — o
         recorte mais interessante do TCC, pois mostra se o modelo muda de
         "lógica" ao prever um pico.

    Amostra até 2.000 observações com semente fixa; o mesmo vetor de índices
    recorta df_res, o que mantém o alinhamento linha a linha entre SHAP e
    classificação de spike.
    """
    if not SHAP_OK:
        console.print("   [warning]⚠  SHAP não disponível[/]")
        return None
    console.rule(f"[highlight]🔬 SHAP — {model_name}[/highlight]")
    console.print("   [info]🔍 Calculando SHAP values (~30s)...[/]")
    n_sample = min(2000, len(X_test))
    idx = np.random.RandomState(RANDOM_STATE).choice(len(X_test), n_sample, replace=False)
    X_s = X_test[idx]
    try:
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X_s)
    except Exception as e:
        # Falha do SHAP (memória, versão incompatível) não derruba o pipeline
        console.print(f"   [error]✘  SHAP falhou: {e}[/]")
        return None

    # Importância global = média do valor absoluto do SHAP por feature
    mean_abs = np.abs(shap_vals).mean(axis=0)
    df_imp = pd.DataFrame({
        "feature"      : feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).head(top_n)

    t = Table(
        title=f"🌍 IMPORTÂNCIA GLOBAL TOP {top_n} — {model_name}",
        box=box.DOUBLE_EDGE, border_style="cyan",
        show_lines=False, min_width=80,
    )
    t.add_column("#",            justify="right", min_width=4)
    t.add_column("Feature",      style="bold",    min_width=35)
    t.add_column("|SHAP| médio", justify="right", min_width=14)
    t.add_column("Barra",        justify="left",  min_width=20)
    max_imp = df_imp["mean_abs_shap"].max()
    for rank, (_, row) in enumerate(df_imp.iterrows(), 1):
        f = row["feature"]
        # Cor por família de variável, para leitura rápida da tabela
        if any(p in f.upper() for p in ["CYC_","CAL_","TEMP_","KEY_"]):
            cor = "data"        # temporais/calendário
        elif any(p in f.upper() for p in ["EAR","RESERV"]):
            cor = "info"        # reservatórios / energia armazenada
        elif any(p in f.upper() for p in ["CARGA","DEMANDA"]):
            cor = "highlight"   # carga e demanda
        elif any(p in f.upper() for p in ["CMO","PLD","PRECO"]):
            cor = "warning"     # preço / custo marginal
        else:
            cor = "white"
        bar = "█" * int(20 * row["mean_abs_shap"] / max_imp)
        t.add_row(str(rank), f"[{cor}]{f}[/{cor}]",
                  f"{row['mean_abs_shap']:.3f}", bar)
    console.print(t)

    # SHAP por regime de spike: mesma amostra, recortada por tipo
    spike_idx = df_res.iloc[idx].reset_index(drop=True)
    for tipo in ["NORMAL", "SPIKE_POS"]:
        mask = (spike_idx["spike_tipo"] == tipo).values
        if mask.sum() < 10:   # amostra pequena demais para ser informativa
            continue
        mean_abs_sub = np.abs(shap_vals[mask]).mean(axis=0)
        df_sub = pd.DataFrame({
            "feature"      : feature_names,
            "mean_abs_shap": mean_abs_sub,
        }).sort_values("mean_abs_shap", ascending=False).head(10)
        cor_t = "spike_pos" if tipo == "SPIKE_POS" else "success"
        ts = Table(
            title=f"[{cor_t}]TOP 10 FEATURES — {tipo} (n={mask.sum()}) — {model_name}[/{cor_t}]",
            box=box.ROUNDED,
            border_style="magenta" if tipo == "SPIKE_POS" else "green",
            show_lines=False, min_width=60,
        )
        ts.add_column("#",            justify="right", min_width=4)
        ts.add_column("Feature",      style="bold",    min_width=35)
        ts.add_column("|SHAP| médio", justify="right", min_width=14)
        for rank, (_, row) in enumerate(df_sub.iterrows(), 1):
            ts.add_row(str(rank), row["feature"], f"{row['mean_abs_shap']:.3f}")
        console.print(ts)
    return df_imp


def piores_casos(df_res, model_name, top_n=15):
    """
    Top-N horas com maior erro absoluto, com data e hora identificadas.
    Serve para citar casos concretos no texto do TCC e para investigar se os
    piores erros coincidem com eventos conhecidos do sistema elétrico.
    """
    console.rule(f"[highlight]🔬 TOP-{top_n} PIORES PREVISÕES — {model_name}[/highlight]")
    df_pior = df_res.nlargest(top_n, "abs_erro").reset_index(drop=True)
    t = Table(
        title=f"❌ {top_n} MAIORES ERROS ABSOLUTOS — {model_name}",
        box=box.DOUBLE_EDGE, border_style="red",
        show_lines=False, min_width=110,
    )
    t.add_column("#",        justify="right",  min_width=4)
    t.add_column("Data",     style="data",     min_width=12)
    t.add_column("Hora",     justify="right",  min_width=5)
    t.add_column("Real",     justify="right",  min_width=10)
    t.add_column("Previsto", justify="right",  min_width=10)
    t.add_column("Erro",     justify="right",  min_width=10)
    t.add_column("Erro %",   justify="right",  min_width=10)
    t.add_column("Faixa",    justify="center", min_width=10)
    t.add_column("Spike",    justify="center", min_width=12)
    for i, row in df_pior.iterrows():
        s_st = ("spike_pos" if row["spike_tipo"] == "SPIKE_POS"
                else "spike_neg" if row["spike_tipo"] == "SPIKE_NEG" else "muted")
        t.add_row(
            str(i+1), row["data"], f"{int(row['hora']):02d}h",
            f"R$ {row['real']:.2f}", f"R$ {row['previsto']:.2f}",
            f"R$ {row['erro']:+.2f}", f"{row['erro_rel_%']:+.1f}%",
            row["faixa_PLD"],
            f"[{s_st}]{row['spike_tipo']}[/{s_st}]",
        )
    console.print(t)


# ==============================================================================
# TABELA COMPARATIVA FINAL DOS MODELOS
# ==============================================================================
def tabela_comparativa_modelos(resultados_modelos):
    """
    Consolida tudo em uma linha por modelo. Com um único modelo na lista, é
    principalmente uma ficha-resumo — mas a estrutura já suporta comparação
    caso outros modelos tunados sejam acrescentados a MODELOS_FASE05.
    """
    console.rule("[bold yellow]📊 RESUMO FINAL DO(S) MODELO(S)[/bold yellow]")
    t = Table(
        title="🏆 RESULTADO FASE 05",
        box=box.DOUBLE_EDGE, border_style="yellow",
        show_lines=True, min_width=150,
    )
    t.add_column("Modelo",         style="bold",    min_width=16)
    t.add_column("N feats",        justify="right", min_width=8)
    t.add_column("MAE F04",        justify="right", min_width=10)
    t.add_column("MAE F05",        justify="right", min_width=10)
    t.add_column("RMSE F05",       justify="right", min_width=10)
    t.add_column("R²",             justify="right", min_width=10)
    t.add_column("sMAPE %",        justify="right", min_width=10)
    t.add_column("NRMSE %",        justify="right", min_width=10)
    t.add_column("Bias geral",     justify="right", min_width=12)
    t.add_column("Spike_MAE(R500)",justify="right", min_width=14)
    t.add_column("Spike_Bias(R500)",justify="right",min_width=15)
    t.add_column("Recall spike(z)",justify="right", min_width=15)

    best_mae = min(r["mae_f05"] for r in resultados_modelos)

    for r in resultados_modelos:
        mae_st = "acerto" if r["mae_f05"] == best_mae else "white"
        spike_mae_r500  = r.get("spike_mae_r500")
        spike_bias_r500 = r.get("spike_bias_r500")
        t.add_row(
            r["nome"],
            str(r["n_features"]),
            f"R$ {r['mae_f04']:.3f}",
            f"[{mae_st}]R$ {r['mae_f05']:.3f}[/{mae_st}]",
            f"R$ {r['rmse']:.3f}" if r.get("rmse") is not None else "—",
            f"{r['r2']:.4f}",
            f"{r['smape']:.2f}%" if r.get("smape") is not None else "—",
            f"{r['nrmse']:.2f}%" if r.get("nrmse") is not None else "—",
            f"R$ {r['bias']:+.3f}",
            f"R$ {spike_mae_r500:.2f}" if spike_mae_r500 is not None else "—",
            f"R$ {spike_bias_r500:+.2f}" if spike_bias_r500 is not None else "—",
            f"{r['recall_spike']:.1f}%" if r.get("recall_spike") is not None else "—",
        )
    console.print(t)


def diagnostico_final(df_res, model_name):
    """
    Converte os números em leitura textual, aplicando regras de limiar:
      • |bias geral| > R$ 10           → viés sistemático relevante
      • MAE da faixa Extremo > 3× geral → falha específica em preços altos
      • recall de spike < 20% / < 50%   → crítico / moderado
      • pior mês > 1,5× melhor mês      → heterogeneidade temporal

    São heurísticas pragmáticas, não testes estatísticos — no TCC devem ser
    apresentadas como observações. Devolve o recall de spike para a tabela
    comparativa final.
    """
    console.rule(f"[bold yellow]🎯 DIAGNÓSTICO — {model_name}[/bold yellow]")
    mae_geral  = df_res["abs_erro"].mean()
    bias_geral = df_res["erro"].mean()
    mae_faixa  = df_res.groupby("faixa_PLD")["abs_erro"].mean()
    bias_faixa = df_res.groupby("faixa_PLD")["erro"].mean()
    spike_pos  = df_res[df_res["spike_tipo"] == "SPIKE_POS"]
    media, std = df_res["real"].mean(), df_res["real"].std()
    recall_spike = None
    if len(spike_pos) > 0:
        z_pred = (spike_pos["previsto"] - media) / std
        recall_spike = float((z_pred > Z_SPIKE).mean() * 100)
    mae_mes  = df_res.groupby("mes")["abs_erro"].mean()
    pior_mes = mae_mes.idxmax() if len(mae_mes) > 0 else None

    insights = []
    # --- Regra 1: viés sistemático global ---
    if abs(bias_geral) > 10:
        dir_ = "SUBESTIMA" if bias_geral > 0 else "SUPERESTIMA"
        insights.append(("✘ Viés geral", "bold red",
            f"Modelo {dir_} em média R$ {abs(bias_geral):.2f}"))
    else:
        insights.append(("✔ Bias geral baixo", "bold green",
            f"Bias = R$ {bias_geral:+.2f} (próximo de zero)"))

    # --- Regra 2: desempenho na faixa de preço extremo ---
    if "Extremo" in mae_faixa.index:
        mae_ext  = mae_faixa["Extremo"]
        bias_ext = bias_faixa["Extremo"]
        if mae_ext > 3 * mae_geral:
            insights.append(("✘ Falha em extremos", "bold red",
                f"MAE PLD>R$500 = R$ {mae_ext:.2f} ({mae_ext/mae_geral:.1f}× MAE geral)"))
            insights.append(("✘ Bias em extremos", "bold red",
                f"Bias PLD>R$500 = R$ {bias_ext:+.2f}"))

    # --- Regra 3: capacidade de antecipar picos ---
    if recall_spike is not None:
        if recall_spike < 20:
            insights.append(("✘ Recall spike crítico", "bold red",
                f"Apenas {recall_spike:.1f}% dos spikes são detectados"))
        elif recall_spike < 50:
            insights.append(("⚠ Recall spike moderado", "bold yellow",
                f"{recall_spike:.1f}% dos spikes detectados"))
        else:
            insights.append(("✔ Recall spike OK", "bold green",
                f"{recall_spike:.1f}% dos spikes detectados"))

    # --- Regra 4: estabilidade do erro ao longo do tempo ---
    if pior_mes:
        mae_pior   = mae_mes[pior_mes]
        mae_melhor = mae_mes.min()
        if mae_pior > 1.5 * mae_melhor:
            insights.append(("⚠ Heterogeneidade temporal", "bold yellow",
                f"Pior mês ({MESES_PT[pior_mes]}) tem MAE "
                f"{mae_pior/mae_melhor:.1f}× pior que melhor"))

    t = Table(
        title=f"🔍 INSIGHTS — {model_name}",
        box=box.DOUBLE_EDGE, border_style="yellow",
        show_lines=True, min_width=100,
    )
    t.add_column("Status",    style="bold", min_width=24)
    t.add_column("Descrição", min_width=70)
    for status, cor, desc in insights:
        t.add_row(f"[{cor}]{status}[/{cor}]", desc)
    console.print(t)
    return recall_spike

# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================
def run_fase_05():
    print_header()

    # 1. Dados
    console.print("\n   [info]📂 Carregando dados...[/]")
    treino_path   = INPUT_DIR / CENARIO_DEFAULT / STRATEGY_DEFAULT / "TREINO_NORM.parquet"
    teste_path    = INPUT_DIR / CENARIO_DEFAULT / STRATEGY_DEFAULT / "TESTE_NORM.parquet"
    df_treino_raw = load_parquet(treino_path, "TREINO_NORM")
    df_teste_raw  = load_parquet(teste_path,  "TESTE_NORM")

    # Detecção automática de colunas especiais
    target_col = detectar_target(df_treino_raw)
    key_cols   = detectar_key_cols(df_treino_raw)
    regime_col = detectar_regime_col(df_treino_raw)
    console.print(f"   [info]🎯 TARGET: {target_col}[/]")
    if regime_col:
        # Removida das FEATURES; reaproveitada apenas como agrupador na análise
        console.print(f"   [info]🚨 REGIME_PRECO removido (leakage): {regime_col}[/]")

    # 2. Split: treino até fev/2026, teste = mar+abr/2026 (com trava anti-leakage)
    df_treino, df_teste = split_treino_teste(df_treino_raw, df_teste_raw, key_cols)

    # 3. Feature engineering temporal (aplicada aos dois conjuntos)
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

    # 4. Ranking de features (df COMPLETO com target)
    #    Reproduz o mesmo ranking da Fase 03 → mesmas 50 features do tuning
    console.print("   [info]📊 Ranqueando features (LightGBM)...[/]")
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
    console.print(f"   [info]🎲 Candidatos extras: {len(candidates_extras)}[/]")

    y_treino = df_treino[target_col].values.astype(np.float32)
    y_teste  = df_teste[target_col].values.astype(np.float32)

    # 5. Loop pelo(s) modelo(s) — aqui só o melhor: RandomForest 50
    resultados_comparativo = []
    todos_residuos         = {}

    for cfg in MODELOS_FASE05:
        nome       = cfg["nome"]
        n_feats    = cfg["n_features"]
        mae_f04    = cfg["mae_fase04"]
        r2_f04     = cfg["r2_fase04"]
        params     = cfg["params"]

        console.print(f"\n")
        console.rule(
            f"[bold magenta]▶▶▶  {nome}  |  {n_feats} features  "
            f"|  MAE Fase04={mae_f04:.3f}  ◀◀◀[/bold magenta]"
        )

        # ← Apresenta os hiperparâmetros considerados (para anotar)
        print_hiperparametros(cfg)

        features = selecionar_top_n(obrigatorias, ranking, n_feats)
        console.print(f"   [info]✔  {len(features)} features selecionadas[/]")

        X_tr = df_treino[features].values.astype(np.float32)
        X_te = df_teste[features].values.astype(np.float32)

        # Treino — hiperparâmetros FIXOS da Fase 04, sem nenhuma otimização
        console.print(f"   [info]🏋️  Treinando {nome}...[/]")
        t0    = time.perf_counter()
        model = build_model(nome, params)
        model.fit(X_tr, y_treino)
        elapsed  = time.perf_counter() - t0
        y_pred   = model.predict(X_te)
        mae_f05  = mean_absolute_error(y_teste, y_pred)
        rmse_f05 = float(np.sqrt(mean_squared_error(y_teste, y_pred)))
        r2       = r2_score(y_teste, y_pred)
        smape_f05 = float(np.mean(
            2*np.abs(y_teste-y_pred)/(np.abs(y_teste)+np.abs(y_pred)+1e-8)
        )*100)
        nrmse_f05 = float(rmse_f05/(np.mean(y_teste)+1e-8)*100)
        # MAE aqui deve reproduzir o da Fase 04; divergência = investigar
        console.print(
            f"   [success]✔  Treinado em {elapsed:.1f}s | "
            f"MAE={mae_f05:.3f} | RMSE={rmse_f05:.3f} | R²={r2:.4f} | "
            f"sMAPE={smape_f05:.2f}% | NRMSE={nrmse_f05:.2f}%[/]"
        )

        # Resíduos — base de TODAS as análises seguintes
        df_res = construir_df_residuos(df_teste, y_teste, y_pred, key_cols, regime_col)
        todos_residuos[nome] = df_res

        # Análises (ordem: do panorama geral ao caso individual)
        console.print()
        print_resumo_residuos(df_res, nome)
        console.print()
        analise_por_faixa_pld(df_res, nome)
        console.print()
        spike_mae_r500, spike_bias_r500, n_spike_r500 = analise_spikes(df_res, nome)
        console.print()
        analise_temporal(df_res, nome)
        console.print()
        analise_regime(df_res, nome)
        console.print()
        analise_shap(model, X_te, features, df_res, nome, top_n=20)
        console.print()
        piores_casos(df_res, nome, top_n=15)
        console.print()
        recall_spike = diagnostico_final(df_res, nome)

        # Métricas para comparativo
        bias = float(df_res["erro"].mean())
        resultados_comparativo.append({
            "nome"           : nome,
            "n_features"     : len(features),
            "params"         : params,
            "mae_f04"        : mae_f04,
            "r2_f04"         : r2_f04,
            "mae_f05"        : safe_round(mae_f05, 3),
            "rmse"           : safe_round(rmse_f05, 3),
            "r2"             : safe_round(r2, 4),
            "smape"          : safe_round(smape_f05, 2),
            "nrmse"          : safe_round(nrmse_f05, 2),
            "bias"           : safe_round(bias, 3),
            "spike_mae_r500" : spike_mae_r500,
            "spike_bias_r500": spike_bias_r500,
            "n_spike_r500"   : n_spike_r500,
            "recall_spike"   : recall_spike,
        })

    # 6. Tabela comparativa final
    console.print()
    tabela_comparativa_modelos(resultados_comparativo)

    # 7. Salvar hiperparâmetros + métricas (documentação)
    salvar_hiperparametros(resultados_comparativo)

    # 8. Recomendação Fase 06
    #    As sugestões 3 e 4 respondem diretamente aos achados desta fase:
    #    recall de spike baixo → especialista de spike; erro assimétrico em
    #    preços altos → regressão quantílica com intervalos.
    melhor = min(resultados_comparativo, key=lambda x: x["mae_f05"])
    console.print()
    console.print(Panel(
        Text.assemble(
            ("🎯 RECOMENDAÇÃO PARA FASE 06 (ENSEMBLE / STACKING):\n\n", "bold green"),
            ("   Modelo analisado na Fase 05: ", "muted"),
            (f"{melhor['nome']} ({melhor['n_features']} features) "
             f"| MAE={melhor['mae_f05']:.3f}\n", "bold"),
            ("   Próximos passos sugeridos:\n", "muted"),
            ("     1. Ensemble (Voting/Stacking) com os demais modelos tuned\n", "bold"),
            ("     2. Weighted average por R² ou MAE inverso\n", "bold"),
            ("     3. Especialista de spike (classificador + regressor condicional)\n", "bold"),
            ("     4. Quantile Regression (Q10/Q50/Q90) para intervalos de confiança\n", "bold"),
            ("\n   Treino usado: histórico até fev/2026 (inclui jan/fev 2026)\n", "muted"),
            ("   Teste avaliado: mar/abr 2026\n", "muted"),
        ),
        border_style="green", padding=(1, 4),
    ))

    return {
        "todos_residuos"        : todos_residuos,
        "resultados_comparativo": resultados_comparativo,
        "melhor_modelo"         : melhor["nome"],
        "melhor_mae"            : melhor["mae_f05"],
    }

# ==============================================================================
# EXECUÇÃO
# ==============================================================================
fase_05_output = run_fase_05()


# ==============================================================================
# 🧾 COLETA_METRICAS_TCC_AUTO  (não editar — célula gerada automaticamente)
# ==============================================================================
# Coleta de métricas finais desta parte. Não altera o modelo; só lê o que está
# em memória e grava em metricas_recalculadas.csv.
import numpy as _np, pandas as _pd, os as _os
PARTE = 5
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
    # Caso as predições tenham ficado presas dentro de run_fase_05(): basta
    # expor y_true / y_pred_teste no escopo global antes desta célula.
    print(f"[REG] Parte {PARTE}: não encontrei par real/predito de regressão. "
          f"Arrays: {[(k,v.size) for k,v in _cands.items()][:12]}")

# ---------- CLASSIFICAÇÃO (se houver score de probabilidade + verdade binária) ----------
# Não se aplica à Fase 05 (puramente regressão), mas o bloco é mantido para
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
# Reexecutar o script SUBSTITUI a linha da Parte 5 em vez de duplicá-la.
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

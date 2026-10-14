# Checkpoint 5 — Statistical Computing e Machine Learning

**FIAP · Tecnólogo em Inteligência Artificial · 2º Semestre**
Parte 1: Análise Exploratória, Descritiva e Probabilística do dataset **WINES** (Python)

Parte 2: Agrupamento com **K-means**, padronização das 13 medições químicas,
seleção de k por Elbow e silhueta e interpretação dos perfis dos grupos.

## Integrantes

| Nome completo | RM |
|---|---|
| Caio Ryan Velten Alves | 572913 |
| Carlos Alberto Oliveira Nascimento | 565694 |
| Davi França Aranda | 572751 |
| João Vitor Fernandes Gomes | 573882 |
| Murilo de Oliveira Moreira | 573534 |

## Conteúdo do repositório

| Arquivo | Descrição |
|---|---|
| `CP5_Statistical_Computing_Wine.ipynb` | Notebook com todas as células **executadas** e código **comentado** com interpretações |
| `CP5_Machine_Learning_Wine.ipynb` | Notebook de ML com pré-processamento, K-means, Elbow, silhueta e perfis químicos |
| `preparar_ml.py` | Gera e executa novamente o notebook de ML |
| `CP5_Integrantes.pdf` | Identificação dos cinco integrantes, com nomes completos e RMs |
| `wine.csv` | Dataset WINES (versão GitHub gist — 178 vinhos, 13 medições químicas + cultivar) |
| `requirements.txt` | Bibliotecas necessárias |

## Variáveis analisadas

- **Alcohol** — teor alcoólico (% vol.)
- **Malic.acid** — ácido málico (g/L), indicador de acidez

## Itens atendidos

| Item | Onde está no notebook |
|---|---|
| **a)** Tabela de Distribuição de Frequências (Sturges, fi, fri, Fi, Fri, xi) + Histograma, Polígono de Frequências e Ogiva | Seção 3 |
| **b)** Tendência Central (média, mediana, moda bruta e de Czuber), Dispersão (amplitude, variância, desvio-padrão, CV, DMA, IQR) e Separatrizes (quartis, decis, cercas de Tukey, boxplot) | Seção 4 |
| **c)** Distribuições de Probabilidade (Normal para o álcool, Log-Normal para o ácido málico, com testes de aderência e QQ-plot) + cálculos probabilísticos (teórico × empírico, Binomial, TCL, probabilidade condicional) | Seção 5 |

## Como executar

```bash
pip install -r requirements.txt
jupyter notebook CP5_Statistical_Computing_Wine.ipynb
```

Para ML, abra `CP5_Machine_Learning_Wine.ipynb` no Jupyter e execute as células
em ordem, a partir da pasta do repositório. Para gerar e executar automaticamente:

```bash
python preparar_ml.py
```

A coluna `Wine` não participa do ajuste nem da seleção de k. A PCA é usada
somente na visualização; o agrupamento usa as 13 medições padronizadas.

## Origem dos arquivos

A análise estatística e o dataset foram mantidos a partir do [repositório do grupo](https://github.com/IgaoPrograma/cp5-wine-statistical-computing).
Este repositório reúne essa parte, o notebook de Machine Learning e o PDF de integrantes.

## Entrega

O enunciado pede PDF com nomes completos e RMs dos integrantes e link do
repositório com código comentado (notebook com células executadas, `.py` ou `.r`).
Os prazos das disciplinas são distintos e devem ser conferidos nas atribuições
do Teams. Conforme orientação atual da disciplina, ML não exige apresentação oral.

O notebook também funciona com a versão do dataset do Kaggle (`harrywang/wine-dataset-for-clustering`): os nomes das colunas são padronizados automaticamente na leitura. A análise por cultivar (seções 4.5 e 5.3) só é exibida quando o arquivo contém a coluna `Wine`.

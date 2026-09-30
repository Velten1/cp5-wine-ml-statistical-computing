# Checkpoint 5 — Statistical Computing with R & Python

**FIAP · Tecnólogo em Inteligência Artificial · 2º Semestre**
Parte 1: Análise Exploratória, Descritiva e Probabilística do dataset **WINES** (Python)

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

O notebook também funciona com a versão do dataset do Kaggle (`harrywang/wine-dataset-for-clustering`): os nomes das colunas são padronizados automaticamente na leitura. A análise por cultivar (seções 4.5 e 5.3) só é exibida quando o arquivo contém a coluna `Wine`.

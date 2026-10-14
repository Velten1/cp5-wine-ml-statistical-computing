# Checkpoint 5 — Análise de vinhos

**FIAP · Tecnólogo em Inteligência Artificial · 2º semestre**

**Statistical Computing with R & Python · Machine Learning & Modelling**

Analisamos o dataset WINES por meio de estatística descritiva, modelos
probabilísticos e agrupamento com K-means. O conjunto contém 178 vinhos,
13 medições químicas e a identificação de três cultivares.

## Integrantes

| Nome completo | RM |
|---|---|
| Caio Ryan Velten Alves | 572913 |
| Carlos Alberto Oliveira Nascimento | 565694 |
| Davi França Aranda | 572751 |
| João Vitor Fernandes Gomes | 573882 |
| Murilo de Oliveira Moreira | 573534 |

## Análise estatística

Estudamos o teor alcoólico (`Alcohol`) e a concentração de ácido málico
(`Malic.acid`). Para cada variável, calculamos distribuições de frequência,
medidas de tendência central, dispersão e quartis, com histogramas, ogivas e
boxplots. Comparamos modelos probabilísticos e suas estimativas com as
frequências observadas na amostra.

O teor alcoólico apresenta média de aproximadamente 13,00% e baixa dispersão
relativa. O ácido málico apresenta assimetria à direita e maior dispersão.
A Normal e a Log-Normal são utilizadas como aproximações para os eventos
analisados, com suas limitações discutidas no notebook.

## Agrupamento com K-means

Excluímos a coluna `Wine` do ajuste e padronizamos as 13 medições químicas.
Comparamos a inércia pelo método Elbow e a silhueta média para selecionar
o número de grupos. A solução com três grupos apresentou silhueta de 0,2849.

| Grupo | Vinhos | Teor alcoólico médio (% vol.) | Ácido málico médio* |
|---|---:|---:|---:|
| 1 | 65 | 12,25 | 1,90 |
| 2 | 51 | 13,13 | 3,31 |
| 3 | 62 | 13,68 | 2,00 |

O grupo 3 apresenta maior teor alcoólico médio, enquanto o grupo 2 tem maior
concentração média de ácido málico. A PCA é utilizada somente para visualizar
os grupos; o K-means utiliza as 13 medições padronizadas. Os identificadores
correspondem à execução com `random_state=42`.

\* Valores na escala do dataset. A fonte não documenta a unidade do ácido
málico; essa variável não representa isoladamente acidez total ou pH.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| [CP5_Statistical_Computing_Wine.ipynb](CP5_Statistical_Computing_Wine.ipynb) | Análises exploratória, descritiva e probabilística |
| [CP5_Machine_Learning_Wine.ipynb](CP5_Machine_Learning_Wine.ipynb) | Pré-processamento, K-means, seleção de k e interpretação dos grupos |
| [CP5_Integrantes.pdf](CP5_Integrantes.pdf) | Identificação dos integrantes |
| `wine.csv` | Dataset utilizado nas análises |
| `requirements.txt` | Dependências do ambiente Python |
| `preparar_ml.py` | Geração e execução do notebook de agrupamento |

## Execução

Na pasta do projeto:

```bash
pip install -r requirements.txt
jupyter notebook
```

Os notebooks contêm as saídas das células executadas. A reprodução das análises
utiliza o arquivo `wine.csv` na mesma pasta dos notebooks.

O notebook de agrupamento também pode ser gerado e executado com:

```bash
python preparar_ml.py
```

## Referências

- [Dataset WINES](https://gist.github.com/tijptjik/9408623)
- [Análise estatística do grupo](https://github.com/IgaoPrograma/cp5-wine-statistical-computing)
- [K-means — scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [Silhouette score — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)

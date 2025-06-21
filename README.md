# Análise de Redes de Colaboração em Títulos de Streaming

Este projeto realiza a análise de redes de colaboração entre atores e diretores com base em um dataset de produções de plataformas de streaming (Netflix, Amazon Prime, Disney+). A análise é feita por meio de construção de grafos e aplicação de métricas clássicas da teoria dos grafos, utilizando a biblioteca `igraph` para garantir alto desempenho mesmo em redes com dezenas de milhares de vértices.

## 📁 Estrutura do Projeto

```
├── main_rapido.py # Script principal otimizado com igraph
├── netflix_amazon_disney_titles.csv # Dataset de entrada
├── README.md # Este arquivo
```


## 🚀 Funcionalidades

O script realiza as seguintes análises:

1. **Construção de grafos**:
   - Grafo direcionado: de atores para diretores.
   - Grafo não-direcionado: entre atores que atuaram juntos.

2. **Estatísticas de rede**:
   - Quantidade de vértices e arestas.
   - Número de componentes conexas e fortemente conexas.

3. **Algoritmos de grafos**:
   - Árvore Geradora Mínima (MST).
   - Centralidade de Grau.
   - Centralidade de Intermediação (com `cutoff` para performance).
   - Centralidade de Proximidade.

## ⚙️ Requisitos

- Python 3.8 ou superior
- `python-igraph`

### Instalação de dependências:

```
pip install python-igraph
```

Lógica do Código
O script utiliza csv.DictReader para ler o arquivo.

Cria dois grafos usando igraph, um com relações direcionadas (atores → diretores) e outro com co-atores.

As análises são feitas com os métodos nativos de igraph, que são otimizados em C para máxima performance.

A centralidade de intermediação pode retornar zero se o vértice não estiver em caminhos curtos no grafo. Isso é esperado com cutoff=3.

Em redes muito grandes, recomenda-se manter o uso de cutoff para evitar tempos de execução excessivos.

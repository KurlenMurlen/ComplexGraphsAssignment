import csv
import os
from igraph import Graph as IGraph

# limpa e padroniza nomes
def clean_name(name):
    if not name or not isinstance(name, str):
        return None
    return name.strip().upper()

# constroi grafos com igraph
def build_graph_igraph_from_csv(filename, directed=False):
    edges = []
    name_to_index = {}
    index = 0

    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if not row.get('director') or not row.get('cast'):
                continue

            directors = [clean_name(d) for d in row['director'].split(',') if clean_name(d)]
            cast = [clean_name(c) for c in row['cast'].split(',') if clean_name(c)]

            for name in directors + cast:
                if name not in name_to_index:
                    name_to_index[name] = index
                    index += 1

            if directed:
                for director in directors:
                    for actor in cast:
                        edges.append((name_to_index[actor], name_to_index[director]))
            else:
                for i in range(len(cast)):
                    for j in range(i + 1, len(cast)):
                        edges.append((name_to_index[cast[i]], name_to_index[cast[j]]))

    g = IGraph(edges=edges, directed=directed)
    index_to_name = [None] * len(name_to_index)
    for name, idx in name_to_index.items():
        index_to_name[idx] = name
    g.vs["name"] = index_to_name
    return g

# main principal
def main():
    csv_filename = "netflix_amazon_disney_titles.csv"
    sample_actor = "BOB ODENKIRK"

    print("arquivo existe?", os.path.exists(csv_filename))
    print("construindo grafos com igraph...")

    g_dir = build_graph_igraph_from_csv(csv_filename, directed=True)
    g_undir = build_graph_igraph_from_csv(csv_filename, directed=False)





    print("\n=== atividade 1 ===")
    print(f"grafo direcionado - vertices: {g_dir.vcount()}, arestas: {g_dir.ecount()}")
    print(f"grafo nao-direcionado - vertices: {g_undir.vcount()}, arestas: {g_undir.ecount()}")




    print("\n=== atividade 2 ===")
    print(f"componentes fortemente conexas (grafo direcionado): {len(g_dir.components(mode='STRONG'))}")
    print(f"componentes conexas (grafo nao-direcionado): {len(g_undir.components())}")




    print("\n=== atividade 3 ===")
    mst = g_undir.spanning_tree()
    print(f"arvore geradora minima - total de arestas: {len(mst.es)}")




    print("\n=== atividade 4 ===")
    if sample_actor in g_undir.vs["name"]:
        idx = g_undir.vs.find(name=sample_actor).index
        grau = g_undir.degree(idx)
        max_grau = g_undir.vcount() - 1
        centralidade_grau = grau / max_grau if max_grau > 0 else 0
        print(f"centralidade de grau para {sample_actor}: {centralidade_grau:.8f}")
    else:
        print(f"{sample_actor} nao encontrado no grafo")




    print("\n=== atividade 5 ===")
    if sample_actor in g_undir.vs["name"]:
        idx = g_undir.vs.find(name=sample_actor).index

        try:
            # calcula apenas caminhos ate 3 arestas de distancia
            intermediacao = g_undir.betweenness(vertices=[idx], directed=False, cutoff=3)[0]
            n = g_undir.vcount()
            if n > 2:
                intermediacao /= ((n - 1) * (n - 2) / 2)
            print(f"centralidade de intermediacao (estimada com cutoff=3) para {sample_actor}: {intermediacao:.8f}") #tive que colocar 8f professor, aqui ficou curto mesmo
        except Exception as e:
            print(f"erro ao calcular intermediacao: {e}")
    else:
        print(f"{sample_actor} nao encontrado no grafo")




    print("\n=== atividade 6 ===")
    if sample_actor in g_undir.vs["name"]:
        idx = g_undir.vs.find(name=sample_actor).index
        proximidade = g_undir.closeness(idx, normalized=True)
        print(f"centralidade de proximidade para {sample_actor}: {proximidade:.4f}")
    else:
        print(f"{sample_actor} nao encontrado no grafo")




if __name__ == "__main__":
    main()

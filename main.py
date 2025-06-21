import csv
from collections import defaultdict, deque
import heapq
import math
import os

class Graph:
    def __init__(self, directed=False):
        self.adjacency_list = defaultdict(dict)
        self.directed = directed
        self.vertices = set()
    
    def add_edge(self, u, v, weight=1):
        # adiciona uma aresta ao grafo
        self.vertices.add(u)
        self.vertices.add(v)
        
        if self.directed:
            self.adjacency_list[u][v] = self.adjacency_list[u].get(v, 0) + weight
        else:
            self.adjacency_list[u][v] = self.adjacency_list[u].get(v, 0) + weight
            self.adjacency_list[v][u] = self.adjacency_list[v].get(u, 0) + weight
    
    def get_vertices(self):
        # retorna a lista de vertices do grafo
        return list(self.vertices)
    
    def get_edges(self):
        # retorna a lista de arestas do grafo
        edges = []
        for u in self.adjacency_list:
            for v, weight in self.adjacency_list[u].items():
                if self.directed or (not self.directed and (v, u, weight) not in edges):
                    edges.append((u, v, weight))
        return edges
    
    def get_neighbors(self, vertex):
        # retorna os vizinhos de um vertice
        return self.adjacency_list.get(vertex, {})
    
    def __str__(self):
        # representacao em string do grafo
        result = []
        for vertex in self.adjacency_list:
            connections = []
            for neighbor, weight in self.adjacency_list[vertex].items():
                connections.append(f"{neighbor}({weight})")
            result.append(f"{vertex} -> {', '.join(connections)}")
        return "\n".join(result)

def clean_name(name):
    # padroniza os nomes para maiusculas e sem espacos
    if not name or not isinstance(name, str):
        return None
    return name.strip().upper()

def build_graphs_from_csv(filename):
    # constroi os dois grafos a partir do arquivo csv
    directed_graph = Graph(directed=True)
    undirected_graph = Graph(directed=False)
    
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            if not row.get('director') or not row.get('cast'):
                continue
            directors = [clean_name(d) for d in row['director'].split(',') if clean_name(d)]
            cast = [clean_name(actor) for actor in row['cast'].split(',') if clean_name(actor)]
            
            for director in directors:
                for actor in cast:
                    directed_graph.add_edge(actor, director)
            
            for i in range(len(cast)):
                for j in range(i + 1, len(cast)):
                    undirected_graph.add_edge(cast[i], cast[j])
    
    return directed_graph, undirected_graph

def count_components(graph):
    # conta componentes conexas
    if graph.directed:
        return count_strongly_connected_components(graph)
    else:
        return count_connected_components(graph)

def count_connected_components(graph):
    # conta componentes conexas em grafo nao direcionado
    visited = set()
    count = 0
    
    for vertex in graph.get_vertices():
        if vertex not in visited:
            count += 1
            queue = deque([vertex])
            visited.add(vertex)
            while queue:
                current = queue.popleft()
                for neighbor in graph.get_neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
    
    return count

def count_strongly_connected_components(graph):
    # algoritmo de kosaraju para grafos direcionados
    visited = set()
    order = []
    
    def dfs_first_pass(vertex):
        stack = [(vertex, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                order.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for neighbor in graph.get_neighbors(node):
                if neighbor not in visited:
                    stack.append((neighbor, False))
    
    for vertex in graph.get_vertices():
        if vertex not in visited:
            dfs_first_pass(vertex)
    
    transposed = Graph(directed=True)
    for u in graph.adjacency_list:
        for v in graph.adjacency_list[u]:
            transposed.add_edge(v, u)
    
    visited = set()
    count = 0
    
    def dfs_second_pass(vertex):
        stack = [vertex]
        visited.add(vertex)
        while stack:
            node = stack.pop()
            for neighbor in transposed.get_neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
    
    for vertex in reversed(order):
        if vertex not in visited:
            count += 1
            dfs_second_pass(vertex)
    
    return count

def minimum_spanning_tree(graph, start_vertex):
    # calcula a arvore geradora minima
    if graph.directed:
        raise ValueError("mst so pode ser calculada em grafos nao direcionados")
    
    if start_vertex not in graph.vertices:
        return None, 0
    
    mst = Graph(directed=False)
    visited = set([start_vertex])
    edges = []
    total_cost = 0
    
    for neighbor, weight in graph.get_neighbors(start_vertex).items():
        heapq.heappush(edges, (weight, start_vertex, neighbor))
    
    while edges and len(visited) < len(graph.vertices):
        weight, u, v = heapq.heappop(edges)
        if v not in visited:
            visited.add(v)
            mst.add_edge(u, v, weight)
            total_cost += weight
            for neighbor, w in graph.get_neighbors(v).items():
                if neighbor not in visited:
                    heapq.heappush(edges, (w, v, neighbor))
    
    return mst, total_cost

def degree_centrality(graph, vertex):
    # calcula a centralidade de grau
    if vertex not in graph.vertices:
        return 0.0
    
    degree = len(graph.get_neighbors(vertex))
    max_possible_degree = len(graph.vertices) - 1
    
    if max_possible_degree == 0:
        return 0.0
    
    return degree / max_possible_degree

def betweenness_centrality(graph, vertex):
    # calcula a centralidade de intermediacao
    if vertex not in graph.vertices:
        return 0.0
    
    betweenness = 0.0
    vertices = graph.get_vertices()
    n = len(vertices)
    
    for s in vertices:
        if s == vertex:
            continue
        pred = {v: [] for v in vertices}
        dist = {v: -1 for v in vertices}
        sigma = {v: 0 for v in vertices}
        dist[s] = 0
        sigma[s] = 1
        queue = deque([s])
        stack = []
        
        while queue:
            v = queue.popleft()
            stack.append(v)
            for neighbor in graph.get_neighbors(v):
                if dist[neighbor] < 0:
                    queue.append(neighbor)
                    dist[neighbor] = dist[v] + 1
                if dist[neighbor] == dist[v] + 1:
                    sigma[neighbor] += sigma[v]
                    pred[neighbor].append(v)
        
        delta = {v: 0 for v in vertices}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                betweenness += delta[w]
    
    normalization = (n - 1) * (n - 2) if graph.directed else (n - 1) * (n - 2) / 2
    if normalization == 0:
        return 0.0
    
    return betweenness / normalization

def closeness_centrality(graph, vertex):
    # calcula a centralidade de proximidade
    if vertex not in graph.vertices:
        return 0.0
    
    distances = {v: -1 for v in graph.vertices}
    distances[vertex] = 0
    queue = deque([vertex])
    
    while queue:
        current = queue.popleft()
        for neighbor in graph.get_neighbors(current):
            if distances[neighbor] == -1:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    
    reachable_distances = [d for d in distances.values() if d > 0]
    if not reachable_distances:
        return 0.0
    
    sum_distances = sum(reachable_distances)
    n = len(reachable_distances)
    
    closeness = n / sum_distances
    normalized_closeness = closeness * (n / (len(graph.vertices) - 1))
    
    return normalized_closeness

def main():
    print("diretorio atual:", os.getcwd())
    print("arquivos no diretorio:", os.listdir('.'))
    print("arquivo existe?", os.path.exists('netflix_amazon_disney_titles.csv'))
    print("construindo grafos...")
    directed_graph, undirected_graph = build_graphs_from_csv('T:/ComplexGraphsAssignment/netflix_amazon_disney_titles.csv')

    
    print("\n=== atividade 1 ===")
    print(f"grafo direcionado - vertices: {len(directed_graph.vertices)}, arestas: {len(directed_graph.get_edges())}")
    print(f"grafo nao-direcionado - vertices: {len(undirected_graph.vertices)}, arestas: {len(undirected_graph.get_edges())}")
    
    print("\n=== atividade 2 ===")
    print(f"componentes fortemente conexas (grafo direcionado): {count_components(directed_graph)}")
    print(f"componentes conexas (grafo nao-direcionado): {count_components(undirected_graph)}")
    
    print("\n=== atividade 3 ===")
    sample_actor = "BOB ODENKIRK"
    mst, mst_cost = minimum_spanning_tree(undirected_graph, sample_actor)
    print(f"arvore geradora minima para {sample_actor} - custo total: {mst_cost}")
    
    print("\n=== atividade 4 ===")
    print(f"centralidade de grau para {sample_actor}: {degree_centrality(undirected_graph, sample_actor):.4f}")
    
    print("\n=== atividade 5 ===")
    print(f"centralidade de intermediacao para {sample_actor}: {betweenness_centrality(undirected_graph, sample_actor):.4f}")
    
    print("\n=== atividade 6 ===")
    print(f"centralidade de proximidade para {sample_actor}: {closeness_centrality(undirected_graph, sample_actor):.4f}")

if __name__ == "__main__":
    main()

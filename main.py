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
        """Adiciona uma aresta ao grafo"""
        self.vertices.add(u)
        self.vertices.add(v)
        
        if self.directed:
            self.adjacency_list[u][v] = self.adjacency_list[u].get(v, 0) + weight
        else:
            self.adjacency_list[u][v] = self.adjacency_list[u].get(v, 0) + weight
            self.adjacency_list[v][u] = self.adjacency_list[v].get(u, 0) + weight
    
    def get_vertices(self):
        """Retorna a lista de vértices do grafo"""
        return list(self.vertices)
    
    def get_edges(self):
        """Retorna a lista de arestas do grafo"""
        edges = []
        for u in self.adjacency_list:
            for v, weight in self.adjacency_list[u].items():
                if self.directed or (not self.directed and (v, u, weight) not in edges):
                    edges.append((u, v, weight))
        return edges
    
    def get_neighbors(self, vertex):
        """Retorna os vizinhos de um vértice"""
        return self.adjacency_list.get(vertex, {})
    
    def __str__(self):
        """Representação em string do grafo"""
        result = []
        for vertex in self.adjacency_list:
            connections = []
            for neighbor, weight in self.adjacency_list[vertex].items():
                connections.append(f"{neighbor}({weight})")
            result.append(f"{vertex} -> {', '.join(connections)}")
        return "\n".join(result)

def clean_name(name):
    """Padroniza os nomes: maiúsculas e sem espaços extras"""
    if not name or not isinstance(name, str):
        return None
    return name.strip().upper()

def build_graphs_from_csv(filename):
    """Constrói os dois grafos a partir do arquivo CSV"""
    directed_graph = Graph(directed=True)
    undirected_graph = Graph(directed=False)
    
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            # Ignora linhas com diretor ou elenco vazios
            if not row.get('director') or not row.get('cast'):
                continue
                
            # Limpa e processa os diretores
            directors = [clean_name(d) for d in row['director'].split(',') if clean_name(d)]
            
            # Limpa e processa o elenco
            cast = [clean_name(actor) for actor in row['cast'].split(',') if clean_name(actor)]
            
            # Adiciona arestas para o grafo direcionado (atores -> diretores)
            for director in directors:
                for actor in cast:
                    directed_graph.add_edge(actor, director)
            
            # Adiciona arestas para o grafo não-direcionado (atores <-> atores)
            for i in range(len(cast)):
                for j in range(i + 1, len(cast)):
                    undirected_graph.add_edge(cast[i], cast[j])
    
    return directed_graph, undirected_graph

def count_components(graph):
    """Conta componentes conexas (não-direcionado) ou fortemente conexas (direcionado)"""
    if graph.directed:
        return count_strongly_connected_components(graph)
    else:
        return count_connected_components(graph)

def count_connected_components(graph):
    """Conta componentes conexas em um grafo não-direcionado"""
    visited = set()
    count = 0
    
    for vertex in graph.get_vertices():
        if vertex not in visited:
            count += 1
            # BFS para visitar todos os vértices conectados
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
    """Conta componentes fortemente conexas em um grafo direcionado usando o algoritmo de Kosaraju"""
    visited = set()
    order = []
    
    # Primeira passada (ordem de término)
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
    
    # Grafo transposto
    transposed = Graph(directed=True)
    for u in graph.adjacency_list:
        for v in graph.adjacency_list[u]:
            transposed.add_edge(v, u)
    
    # Segunda passada no grafo transposto
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
    """Retorna a Árvore Geradora Mínima (Prim) e seu custo total"""
    if graph.directed:
        raise ValueError("MST só pode ser calculada em grafos não-direcionados")
    
    if start_vertex not in graph.vertices:
        return None, 0
    
    mst = Graph(directed=False)
    visited = set([start_vertex])
    edges = []
    total_cost = 0
    
    # Inicializa a heap com as arestas do vértice inicial
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
    """Calcula a centralidade de grau de um vértice"""
    if vertex not in graph.vertices:
        return 0.0
    
    degree = len(graph.get_neighbors(vertex))
    max_possible_degree = len(graph.vertices) - 1
    
    if max_possible_degree == 0:
        return 0.0
    
    return degree / max_possible_degree

def betweenness_centrality(graph, vertex):
    """Calcula a centralidade de intermediação de um vértice"""
    if vertex not in graph.vertices:
        return 0.0
    
    betweenness = 0.0
    vertices = graph.get_vertices()
    n = len(vertices)
    
    for s in vertices:
        if s == vertex:
            continue
        # BFS para encontrar todos os caminhos mais curtos de s para outros vértices
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
    
    # Normalização
    if graph.directed:
        normalization = (n - 1) * (n - 2)
    else:
        normalization = (n - 1) * (n - 2) / 2
    
    if normalization == 0:
        return 0.0
    
    return betweenness / normalization

def closeness_centrality(graph, vertex):
    """Calcula a centralidade de proximidade de um vértice"""
    if vertex not in graph.vertices:
        return 0.0
    
    # BFS para calcular as distâncias mais curtas
    distances = {v: -1 for v in graph.vertices}
    distances[vertex] = 0
    queue = deque([vertex])
    
    while queue:
        current = queue.popleft()
        for neighbor in graph.get_neighbors(current):
            if distances[neighbor] == -1:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    
    # Remove vértices inalcançáveis
    reachable_distances = [d for d in distances.values() if d > 0]
    if not reachable_distances:
        return 0.0
    
    sum_distances = sum(reachable_distances)
    n = len(reachable_distances)
    
    # Centralidade de proximidade (harmônica para lidar com grafos não conexos)
    closeness = n / sum_distances
    
    # Normalização
    normalized_closeness = closeness *(n / (len(graph.vertices) - 1))
    
    return normalized_closeness










def main():
    print("Arquivo existe?", os.path.exists('netflix_amazon_disney_tittles.csv'))
    # Construir os grafos
    print("Construindo grafos...")
    directed_graph, undirected_graph = build_graphs_from_csv('../netflix_amazon_disney_tittles.csv')
    
    # Atividade 1: Construção dos grafos
    print("\n=== Atividade 1 ===")
    print(f"Grafo direcionado - Vértices: {len(directed_graph.vertices)}, Arestas: {len(directed_graph.get_edges())}")
    print(f"Grafo não-direcionado - Vértices: {len(undirected_graph.vertices)}, Arestas: {len(undirected_graph.get_edges())}")
    
    # Atividade 2: Contagem de componentes
    print("\n=== Atividade 2 ===")
    print(f"Componentes fortemente conexas (grafo direcionado): {count_components(directed_graph)}")
    print(f"Componentes conexas (grafo não-direcionado): {count_components(undirected_graph)}")
    
    # Atividade 3: Árvore Geradora Mínima
    print("\n=== Atividade 3 ===")
    sample_actor = "BOB ODENKIRK"
    mst, mst_cost = minimum_spanning_tree(undirected_graph, sample_actor)
    print(f"Árvore Geradora Mínima para {sample_actor} - Custo total: {mst_cost}")
    
    # Atividade 4: Centralidade de Grau
    print("\n=== Atividade 4 ===")
    print(f"Centralidade de Grau para {sample_actor}: {degree_centrality(undirected_graph, sample_actor):.4f}")
    
    # Atividade 5: Centralidade de Intermediação
    print("\n=== Atividade 5 ===")
    print(f"Centralidade de Intermediação para {sample_actor}: {betweenness_centrality(undirected_graph, sample_actor):.4f}")
    
    # Atividade 6: Centralidade de Proximidade
    print("\n=== Atividade 6 ===")
    print(f"Centralidade de Proximidade para {sample_actor}: {closeness_centrality(undirected_graph, sample_actor):.4f}")

    if __name__ == "__main__":
     main()
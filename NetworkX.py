import networkx as nx
import matplotlib.pyplot as plt
from heapq import heappush, heappop

G=nx.Graph()
G.add_edges_from([(1,2),(1,3)])
print(G.number_of_edges())

print(list(G.nodes))
print(list(G.edges))
print(list(G.adj[1]))
print(G.degree[1])

print(G[1]) #Lo mismo que G.adj[1]
print(G[1][2])
print(G.edges[1, 2])

G.add_edge(1,3)
G[1][3]['color'] = 'azul'
G.edges[1,2]['color'] = 'rojo'

'''G=nx.path_graph(4)
nx.write_edgelist(G, "test.edgelist")
G=nx.path_graph(4)
fh=open("test.edgelist",'wb')
nx.write_edgelist(G, fh)
nx.write_edgelist(G, "test.edgelist.gz")
nx.write_edgelist(G, "test.edgelist.gz", data=False)'''
print('\n')

fh = open("Grafo2-myciel3", 'rb')
H = nx.read_edgelist(fh,nodetype=int)
print('nodos: ')
print(list(H))
print('aristas: ')
print(list(H.edges))
print('adyacencias: ')
print(H.adj)
print('grados: ')
print(H.degree)
fh.close()
print('\ngreedy: ')
#d={}
d = nx.coloring.greedy_color(H, strategy=nx.coloring.strategy_independent_set)
print(d)
l=H.nodes
#print('Grafo: ')
#print(list(H))

color_map = []
#for node in d:
i = 1
for ele,con in l.items():
    #print(str(node) + ',' + str(color))
    for node, color in d.items():
        if node == ele:
            break
    if color == 0:
        color_map.append('blue')
    elif color == 1:
        color_map.append('green')
    elif color == 2:
        color_map.append('red')
    else: color_map.append('yellow')

nx.draw(H,node_color = color_map,with_labels = True)
plt.show()

print('\nGrafo aleatorio: ')
R=nx.fast_gnp_random_graph(27,0.1)
print(list(R))
print('adyacencias: ')
print(H.adj)

nx.draw(R,with_labels=True)
plt.draw()
plt.show()

'''#heap
heap=[]
for nodo in H:
    heappush(heap, nodo)
print('Heap: ')
while heap:
    print(heappop(heap))'''

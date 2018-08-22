import networkx as nx
import time

path='C:/Users/jessi/PycharmProjects/GrafosRand/GrafosAleatorios/'



'''Lee el grafo desde un archivo'''
def Lee_Grafo(nombre):
    return nx.read_edgelist(nombre)


archivosmall = open("k-small","w")
archivo2small = open("tiempo-small","w")
archivoind = open("k-independent","w")
archivo2ind = open("tiempo-independent","w")

for i in range(10):
    #print('Grafo#' + str(i+1))
    start_time = time.time()
    carpeta='Grafos20P0.7'
    nombre= '/G0.7_'+ str(i+1)
    Nombre_benchmark = carpeta + nombre
    G = Lee_Grafo(path + Nombre_benchmark)
    d = nx.coloring.greedy_color(G, strategy='independent_set')# smallest_last independent_set
    archivo2ind.write(str(time.time() - start_time) + '\n')
    r = nx.coloring.greedy_color(G, strategy='smallest_last')  # smallest_last independent_set
    archivo2small.write(str(time.time() - start_time) + '\n')
    #print(d)
    #print(str(max(d.values())+1))
    archivoind.write(str(max(d.values())+1) + '\n')    #Más 1 porque empieza en cero
    archivosmall.write(str(max(r.values()) + 1) + '\n')  # Más 1 porque empieza en cero

archivoind.close()
archivo2ind.close()
archivosmall.close()
archivo2small.close()
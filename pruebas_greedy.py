import networkx as nx
import time

path='C:/Users/jessi/PycharmProjects/GrafosRand/GrafosAleatorios/'



'''Lee el grafo desde un archivo'''
def Lee_Grafo(nombre):
    return nx.read_edgelist(nombre)


archivo = open("k","w")
archivo2 = open("tiempo","w")
for i in range(10):
    #print('Grafo#' + str(i+1))
    start_time = time.time()
    carpeta='Grafos100P0.7'
    nombre= '/G100_0.7_'+ str(i+1)
    Nombre_benchmark = carpeta + nombre
    G = Lee_Grafo(path + Nombre_benchmark)
    d = nx.coloring.greedy_color(G, strategy='independent_set')# smallest_last independent_set
    #print(d)
    archivo.write(str(max(d.values())) + '\n')
    archivo2.write(str(time.time() - start_time) + '\n')
archivo.close()
archivo2.close()
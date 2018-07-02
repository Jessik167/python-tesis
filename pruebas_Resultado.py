import Busquedas_locales
import networkx as nx


path='C:/Users/jessi/PycharmProjects/GrafosRand/GrafosAleatorios/'

numColores = 10
numNodosAle = 40
probabilidad = 0.5
grafo = 3


'''Lee el grafo desde un archivo'''
def Lee_Grafo(nombre):
    return nx.read_edgelist(nombre, nodetype=int)

carpeta='Grafos' + str(numNodosAle) + 'P' + str(probabilidad)
nombre= '/G' + str(numNodosAle) + '_' + str(probabilidad) +'_' + str(grafo)
Nombre_benchmark = carpeta + nombre
#Nombre_benchmark = 'myciel3'

G = Lee_Grafo(path + Nombre_benchmark)
#G = Lee_Grafo(Nombre_benchmark)

individuo = {0: {2: 2, 16: 16, 21: 21, 24: 24, 32: 32}, 1: {28: 28, 11: 11, 0: 0, 12: 12, 27: 27, 26: 26}, 2: {33: 33, 37: 37, 23: 23, 5: 5}, 3: {22: 22, 7: 7, 25: 25, 31: 31, 13: 13, 9: 9}, 4: {34: 34, 10: 10, 17: 17, 36: 36}, 5: {29: 29, 30: 30, 35: 35, 19: 19}, 6: {38: 38, 18: 18, 14: 14, 4: 4}, 7: {39: 39, 3: 3, 1: 1}, 8: {20: 20, 6: 6, 8: 8}, 9: {15: 15}}
print("Número de aristas monocromáticas: " + str(Busquedas_locales.numeroAristasMono(G,individuo,numColores)))
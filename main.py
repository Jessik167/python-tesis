import bolsas
import Cruza
import Tabu
import random
import copy
import numpy as np
import networkx as nx
import time


path='C:/Users/jessi/PycharmProjects/GrafosRand/GrafosAleatorios/'
#Nombre_benchmark = "2-Fullins_3" #'inithx.i.1' #"fpsol2.i.1" #'flat1000_60_0'  #"myciel3" #"myciel4" #'2-Fullins_3' #"DSJC250-5" #

tam_poblacion = 20
numGeneraciones = 20
numColores = 7
numNodosAle = 40
probabilidad = 0.4
grafo = 2





'''Lee el grafo desde un archivo'''
def Lee_Grafo(nombre):
    return nx.read_edgelist(nombre)



'''Inicializa la población con estrategia greedy, y luego realiza una búsqueda local'''
def InicializacionPoblacion(poblacion, probabilidades, G):# Crea instancias de colorado del grafo con estrategia greedy de tamaño de la población
    for ind in range(tam_poblacion):    #Lo realiza hasta el tamaño de la población
        poblacion.append(bolsas.crea_individuo(G, numColores))  #crea un individuo
        probabilidades.append(bolsas.cuenta_nodos(poblacion[ind], numColores, numNodos))    # cuenta los nodos de cada bolsa del individuo

    for ind in range(tam_poblacion):
        poblacion[ind] = Tabu.Busqueda_Tabu(G, poblacion[ind], probabilidades[ind], numColores) # Genera la búsqueda local en la población inicial
    probabilidades = [] #Limpia la lista de probabilidad



def CruzaIndividuos(poblacion,probabilidades,numNodos, padres):
    individuos = [] #Lista que contendrá a los dos individuos a elegir
    nuevos_indiv = []  #Lista que contendrá al nuevo individuo

    padres.append(random.randint(0, tam_poblacion - 1)) #Elige el índice a elegir del padre 1
    padres.append(random.randint(0, tam_poblacion - 1)) #Elige el índice a elegir del padre 2

    individuos.append(copy.deepcopy(poblacion[padres[0]]))   # toma un individuo al azar de la población y lo copia en individuos
    individuos.append(copy.deepcopy(poblacion[padres[1]]))   # toma un individuo al azar de la población y lo copia en individuos

    nuevos_indiv.append(Cruza.GPX(individuos, numColores))   # Cruza a los padres y forma un nuevo individuo
    probabilidades.append(bolsas.cuenta_nodos(nuevos_indiv[0], numColores, numNodos)) # cuenta los nodos de cada bolsa del individuo

    return nuevos_indiv



def BusquedaLocal(nuevo_individuo, probabilidad, G, bestmono,best_indv):
    aristas_mono = Tabu.numeroAristasMono(G, nuevo_individuo, numColores) #Calcula el número de aristas monocromáticas del individuo

    if aristas_mono != 0:   #Si el número de aristas monocromáticas es mayor que cero, realiza la búsqueda local
       nuevo_individuo = Tabu.Busqueda_Tabu(G, nuevo_individuo, probabilidad, numColores)  #regresa al individuo después de realzar la búsqueda local
       aristas_mono = Tabu.numeroAristasMono(G, nuevo_individuo, numColores)  #Calcula el número de aristas monocromáticas del grafo

    if aristas_mono < bestmono: #Guarda al individuo que obtuvo el menor número de aristas monocromáticas
        bestmono= aristas_mono
        best_indv = nuevo_individuo

    return best_indv,bestmono



def ActualizaPoblacion(pobla,nuevoindv,indiceP,G):
    aristas_monoP1 = Tabu.numeroAristasMono(G, pobla[indiceP[0]], numColores)  # Calcula el número de aristas monocromáticas del individuo
    aristas_monoP2 = Tabu.numeroAristasMono(G, pobla[indiceP[1]], numColores)  # Calcula el número de aristas monocromáticas del individuo
    aristas_monoH = Tabu.numeroAristasMono(G, nuevoindv, numColores)

    if aristas_monoP1 < aristas_monoP2: #Si el padre 2 es peor que el padre 1 entonces...
        poblacion[indiceP[1]]= nuevoindv    #El hijo reemplaza al padre 2
    else:                                   #Si no...
        poblacion[indiceP[0]] = nuevoindv   #El hijo reemplaza al padre 1



##   Algoritmo Genético   ##
#archivo = open("Resultados","w")
#for i in range(10):
start_time = time.time()
#    print('\n***Grafo#' + str(i + 1))
carpeta='Grafos' + str(numNodosAle) + 'P' + str(probabilidad)
nombre= '/G' + str(numNodosAle) + '_' + str(probabilidad) +'_' + str(grafo)#+ str(i+1)
Nombre_benchmark = carpeta + nombre

ind = 0
best = np.inf
bestAnt = np.inf
best_ind = []
poblacion = []
probabilidades=[]

#G = bolsas.crea_grafo(Nombre_benchmark) #Crea el grafo apartir de un archivo de texto
    #G=bolsas.crea_grafo_aleatorio(numNodosAle, probabilidad)   #Crea un grafo a partir de un número de nodos y una probabilidad
G = Lee_Grafo(path + Nombre_benchmark)
numNodos = len(G)

InicializacionPoblacion(poblacion, probabilidades, G)
for gen in range(numGeneraciones):
    ind_padres = []
    nuevos_individuos = CruzaIndividuos(poblacion, probabilidades, numNodos, ind_padres)
    best_ind , best = BusquedaLocal(nuevos_individuos[0], probabilidades[gen], G, best, best_ind)   #Realiza la búsqueda local por cada uno de los individuos de la población
    ActualizaPoblacion(poblacion,nuevos_individuos[0],ind_padres,G)

    print('mejor número de aristas monocromáticas por generación: ' + str(best))
    if best == bestAnt and (gen != 1 or gen != 2):
        best_col = gen
        break
    else:
        bestAnt = best

print('\n\nmejor número de aristas monocromáticas: ')
print(str(best))
print('mejor individuo: ')
print(best_ind)
print("%s seconds" % (time.time() - start_time))
#bolsas.colorea_grafo(G,poblacion[best_col])
#    archivo.write(str(best) + '\t' + str(time.time() - start_time) + '\t' + str(best_ind) + '\n')
#archivo.close()


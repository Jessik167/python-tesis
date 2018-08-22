from numba import cuda, vectorize
from numba.cuda.random import create_xoroshiro128p_states
import numpy as np
import networkx as nx
import scipy as sp
import bolsas
import Cruza
import Busquedas_locales
import random
import copy
import math
import time


path='C:/Users/jessi/PycharmProjects/GrafosRand/GrafosAleatorios/'
Nombre_benchmark = '2-Fullins_3'#"myciel3"#"inithx.i.1" #'inithx.i.1' #"fpsol2.i.1" #'flat1000_60_0'  #"myciel3" #"myciel4" #"DSJC250-5" #

tam_poblacion = 20
numGeneraciones = 20
numColores = 4
numNodosAle = 100
probabilidad = 0.7
grafo = 10



'''Lee el grafo desde un archivo'''
def Lee_Grafo(nombre):
    return nx.read_edgelist(nombre, nodetype=int)



'''Inicializa la población con estrategia greedy, y luego realiza una búsqueda local'''
def InicializacionPoblacion(poblacion, probabilidades, G,numNodos):# Crea instancias de colorado del grafo con estrategia greedy de tamaño de la población
    for ind in range(tam_poblacion):    #Lo realiza hasta el tamaño de la población
        if ind == 0:
            poblacion.append(TomaGreedy(G))
        else:
            poblacion.append(bolsas.crea_individuo(G, numColores))  #crea un individuo
        probabilidades.append(bolsas.cuenta_nodos(poblacion[ind], numColores, numNodos))    # cuenta los nodos de cada bolsa del individuo




def TomaGreedy(G):
    d = nx.coloring.greedy_color(G, strategy='smallest_last')   #Utiliza estrategia greedy
    Bolsas_colores = {k: {} for k in range(numColores)}
    for valores in d.items():   #Recorre los nodos
        nodo = valores[0]
        if valores[1] < numColores: #Pregunta si el color del greedy está dentro del rango de colores
            Bolsas_colores[valores[1]][nodo] = nodo #Si si, ingresa el nodo en la bolsa que eligió el greedy
        else:
            Bolsas_colores[random.randint(0, numColores - 1)][nodo] = nodo  #Si no está, ingresa el nodo en una bolsa aleatoria
    return Bolsas_colores




def CruzaIndividuos(poblacion,numNodos, padres,probabilidades):
    individuos = [] #Lista que contendrá a los dos individuos a elegir
    nuevos_indiv = {}  #Lista que contendrá al nuevo individuo
    indices = ()

    indices += (random.randint(0, tam_poblacion - 1),)  #Elige el índice a elegir del padre 1
    while True:
        r = random.randint(0, tam_poblacion - 1)
        if r != indices[0]:
            indices += (r,) #Elige el índice a elegir del padre 2
            break
    padres.append(indices)

    individuos.append(copy.deepcopy(poblacion[padres[-1][0]]))   # toma el individuo al azar de la población (padre 1)
    individuos.append(copy.deepcopy(poblacion[padres[-1][1]]))   # toma el individuo al azar de la población (padre 2)

    nuevos_indiv = Cruza.GPX(individuos, numColores)   # Cruza a los padres y forma un nuevo individuo
    probabilidades.append(bolsas.cuenta_nodos(nuevos_indiv, numColores, numNodos)) # cuenta los nodos de cada bolsa del individuo

    return nuevos_indiv



'''Manda a llamar a la búsqueda local (metrópolis ó Escalando la colina)'''
def BusquedaLocal(nuevo_individuo, probabilidad, Aristmono, M):
     #nuevo_individuo, Arist = Busquedas_locales.Busqueda_Escalando(M, nuevo_individuo, probabilidad,
     #                                                       Aristmono, numColores)  # regresa al individuo después de realzar la búsqueda local
     nuevo_individuo, Arist = Busquedas_locales.Busqueda_Metropolis(M, nuevo_individuo, probabilidad,
                                                             Aristmono,numColores)  # regresa al individuo después de realzar la búsqueda local
     return Arist



'''Manda a llamar a la búsqueda local en CUDA(metrópolis ó Escalando la colina)'''
@cuda.jit#(device=cuda, debug=True)
def BusquedaLocal_CUDA(nuevo_individuos, probabilidades,Aristmono,MatrizAdjacencia, H,numNodos,tam_pobla,rng_states):
    bx = cuda.blockIdx.x
    thx = cuda.threadIdx.x
    id = bx * H + thx
    if id < tam_pobla:
        Busquedas_locales.Busqueda_MetropolisCUDA(MatrizAdjacencia, nuevo_individuos[id], probabilidades[id],
                                                                      Aristmono[id],numColores,numNodos,rng_states,id)  # regresa al individuo después de realzar la búsqueda local
        cuda.syncthreads()


'''Reemplaza al peor de los padres elegidos con el hijo, compara el número de aristas monocromáticas de los individuos implicados'''
def ActualizaPoblacion(poblacion, hijo, AristasMono, AristasMonohijo,probabilidades, probab_hijo, indiceP):
    if AristasMono[indiceP[0]] < AristasMono[indiceP[1]]: #Si el padre 2 tiene mayor número de aristas mono que el padre 1 entonces...
        poblacion[indiceP[1]]= hijo    #El hijo reemplaza al padre 2
        AristasMono[indiceP[1]]= AristasMonohijo    #actualiza las aristas mono del padre reemplazado con las del hijo
        probabilidades[indiceP[1]]= probab_hijo
    else:                                   #Si no...
        poblacion[indiceP[0]] = hijo   #El hijo reemplaza al padre 1
        AristasMono[indiceP[0]] = AristasMonohijo   #actualiza las aristas mono del padre reemplazado con las del hijo
        probabilidades[indiceP[0]] = probab_hijo


'''Compara los resultados'''
def sonIguales(resultados,tam):
    primero = resultados[0]  #Toma el primer resultado
    i = 1

    for i in range(tam):    #Lo compara con todos los demas
        if primero == resultados[i]:    #Si es igual continua
            continue
        else:
            return False    #Si no, termina
    return True #Si termina el ciclo es que todos fueron iguales


def convierte_individuonumpy(individuo,probabilidades,Amono,indi_np,prob_np,AmonoNp,tam,bolsas):
    for i in range(tam):
        AmonoNp[i] = Amono[i]
        for j in range(bolsas):
            prob_np[i][j]= probabilidades[i][j]
            for r in individuo[i][j]:
                indi_np[i][j][r-1] = 1


##   Algoritmo Genético   ##
def main():
    #archivo = open("Resultados","w")
    #for i in range(10):
    start_time = time.time()

    #print('\n***Grafo#' + str(i + 1))
    #carpeta='Grafos' + str(numNodosAle) + 'P' + str(probabilidad)
    #nombre= '/G' + str(numNodosAle) + '_' + str(probabilidad) +'_' + str(grafo)#+ str(i+1)
    #nombre = '/G' + str(probabilidad) + '_' + str(i + 1)  # + str(grafo)
    #Nombre_benchmark = carpeta + nombre

    ind = 0
    termina = False
    NumGenIgual = 3
    best = np.inf
    bestAnt = []
    best_ind = []
    poblacion = []
    probabilidades=[]
    AMonocromaticas = []
    nuevos_individuos = []
    AMonoNuevo = []
    probabilidadeshijo = []
    ind_padres = []

    G = bolsas.crea_grafo(Nombre_benchmark) #Crea el grafo apartir de un archivo de texto
    #G=bolsas.crea_grafo_aleatorio(numNodosAle, probabilidad)   #Crea un grafo a partir de un número de nodos y una probabilidad
    #G = Lee_Grafo(path + Nombre_benchmark)
    Matriz_Adyacencia = nx.to_numpy_matrix(G,nodelist=sorted(G.nodes()),dtype=int)
    #print(Matriz_Adyacencia)
    numNodos = len(G)

    InicializacionPoblacion(poblacion, probabilidades, G,numNodos)

    for ind in range(tam_poblacion):    #Realiza la búsqueda local en la población inicial
        AMonocromaticas.append(Busquedas_locales.numeroAristasMono(G, poblacion[ind], numColores))  # calcula el número de aristas monocromáticas del individuo

        if AMonocromaticas[ind] < best:  # Guarda al individuo que obtuvo el menor número de aristas monocromáticas
            best = copy.deepcopy(AMonocromaticas[ind])
            best_ind = copy.deepcopy(poblacion[ind])

        AMonocromaticas[ind] = BusquedaLocal(poblacion[ind],probabilidades[ind],AMonocromaticas[ind],Matriz_Adyacencia)

        if AMonocromaticas[ind] == 0:
            best = 0
            best_ind = poblacion[ind]
            termina= True
            break
        else:
            if AMonocromaticas[ind] < best:  # Guarda al individuo que obtuvo el menor número de aristas monocromáticas
                best = copy.deepcopy(AMonocromaticas[ind])
                best_ind = copy.deepcopy(poblacion[ind])


    if not termina: #Continua si arriba NO encontró un individuo con aristas monocromáticas igual a cero
        mitad_poblacion = int(tam_poblacion / 2)
        for gen in range(numGeneraciones):
            #individuos de arreglos numpy
            nuevos_individuosNp = np.zeros((mitad_poblacion, numColores, numNodos), dtype= np.int64)
            proba_hijoNp = np.zeros((mitad_poblacion,numColores), dtype=np.float64)
            AMonoNuevoNp = np.zeros((mitad_poblacion), dtype=np.int64)
            bloques= 10
            hilos= 21
            rng_states = create_xoroshiro128p_states( (bloques*hilos)* bloques, seed=1)

            nuevos_individuos = []
            AMonoNuevo = []
            ind_padres = []
            probabilidadeshijo = []

            for p in range(mitad_poblacion):
                nuevos_individuos.append(CruzaIndividuos(poblacion, numNodos, ind_padres, probabilidadeshijo))
                AMonoNuevo.append(Busquedas_locales.numeroAristasMono(G, nuevos_individuos[p], numColores))  # Calcula el número de aristas monocromáticas del individuo nuevo

                if AMonoNuevo[p] < best:  # Guarda si alguno de los hijos obtuvo menor número de aristas monocromáticas
                    best = copy.deepcopy(AMonoNuevo[p])
                    best_ind = copy.deepcopy(nuevos_individuos[p])
                if AMonoNuevo[p] == 0:
                    break

            convierte_individuonumpy(nuevos_individuos, probabilidadeshijo,AMonoNuevo, nuevos_individuosNp, proba_hijoNp,AMonoNuevoNp,
                                     mitad_poblacion, numColores)
            BusquedaLocal_CUDA[bloques,hilos](nuevos_individuosNp,proba_hijoNp,AMonoNuevoNp, Matriz_Adyacencia,hilos,numNodos,mitad_poblacion,rng_states)   #realiza la búsqueda local en CUDA
            print("Sale")
            for p in range(mitad_poblacion):
                #AMonoNuevo[p] = BusquedaLocal(nuevos_individuos[p],probabilidadeshijo[p],AMonoNuevo[p], G)   #realiza la búsqueda local con el número de aristas monocromáticas del nuevo individuo
                ActualizaPoblacion(poblacion,nuevos_individuos[p],AMonocromaticas,AMonoNuevo[p], probabilidades, probabilidadeshijo[p], ind_padres[p]) #reemplaza al hijo con el peor individuo

                if AMonoNuevo[p] < best:  # Guarda si alguno de los hijos obtuvo menor número de aristas monocromáticas
                    best = copy.deepcopy(AMonoNuevo[p])
                    best_ind = copy.deepcopy(nuevos_individuos[p])
                if AMonoNuevo[p] == 0:
                    break

            if AMonoNuevo[p] == 0:
                break

            #print('Número de aristas monocromáticas del hijo por generación: ' + str(AMonoNuevo))
            bestAnt.append(AMonoNuevo)
            if (gen+1) % NumGenIgual == 0:
                if sonIguales(bestAnt,NumGenIgual):
                    bestAnt = []
                    break
                bestAnt = []

    print('\n\nmejor número de aristas monocromáticas: ')
    print(str(best))
    print('mejor individuo: ')
    print(best_ind)
    print("%s seconds" % (time.time() - start_time))
    #bolsas.Muestra_Grafo(G)
    #bolsas.colorea_grafo(G,best_ind)
    #if best == 0:
    #    archivo.write('Grafo#' + str(i + 1) + '\t' + str(numColores) + '\t' + str(time.time() - start_time) + '\n')
    #archivo.close()

if __name__ == '__main__':
    main()
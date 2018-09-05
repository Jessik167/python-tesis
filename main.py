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
Nombre_benchmark = "inithx.i.1"#'2-Fullins_3'#"myciel3" #'inithx.i.1' #"fpsol2.i.1" #'flat1000_60_0'  #"myciel3" #"myciel4" #"DSJC250-5" #

#Datos del Algoritmo genético
tam_poblacion = 32
numGeneraciones = 20
numColores = 2

#Datos del grafo aleatorio
numNodosAle = 100
probabilidad = 0.7
grafo = 1



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



'''Toma la solución greedy smallest last y la convierte a un individuo en diccionario'''
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



'''Recibe dos padres, los cruza con el algoritmo GPX, retorna un nuevo individuo (hijo)'''
def CruzaPadres(individuos,probabilidades,AMonoNuevo,numNodos,G):
    nuevo_indiv = {}  # Lista que contendrá al nuevo individuo
    nuevo_indiv = Cruza.GPX(individuos, numColores)  # Cruza a los padres y forma un nuevo individuo
    probabilidades.append(bolsas.cuenta_nodos(nuevo_indiv, numColores, numNodos))  # cuenta los nodos de cada bolsa del individuo

    # Calcula el número de aristas monocromáticas del individuo nuevo
    AMonoNuevo.append(Busquedas_locales.numeroAristasMono(G, nuevo_indiv,numColores))

    return nuevo_indiv



'''Toma aleatoriamente dos de los indices dentro del total de la población, que serán los padres que se cruzarán'''
def EligePadres(poblacion,numNodos, padres,probabilidades):
    individuos = [] #Lista que contendrá a los dos individuos a elegir
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

    return individuos




'''Manda a llamar a la búsqueda local (metrópolis ó Escalando la colina)'''
def BusquedaLocal(nuevo_individuo, probabilidad, Aristmono, M):
     #nuevo_individuo, Arist = Busquedas_locales.Busqueda_Escalando(M, nuevo_individuo, probabilidad,Aristmono, numColores)  # regresa al individuo después de realzar la búsqueda local
     nuevo_individuo, Arist = Busquedas_locales.Busqueda_Metropolis(M, nuevo_individuo, probabilidad, Aristmono,numColores)  # regresa al individuo después de realzar la búsqueda local
     return Arist



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


'''Convierte el hijo en un diccionario de colores'''
def convert_hijo(Arr,numNodos):
    Bolsas_colores = {k: {} for k in range(numColores)}
    for i in range(numColores):
        for j in range(numNodos):
            if Arr[i][j] == 1:
                Bolsas_colores[i][j] = j
    return Bolsas_colores



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



'''Actualiza el mejor individuo encontrado con el menor número de aristas monocromáticas (NAM)'''
def ActualizaMejor(AMonocromaticas,poblacion,best,best_ind,ind,termina):
    if AMonocromaticas[ind] < best:  # Guarda al individuo que obtuvo el menor número de aristas monocromáticas
        best[0] = AMonocromaticas[ind]
        best_ind[0] = poblacion[ind]
        #return AMonocromaticas[ind], poblacion[ind]
    if AMonocromaticas[ind]==0:
        termina = True



'''Realiza una búsqueda local en la población inicial, calcula el número de aristas monocromáticas y actualiza el mejor'''
def Busqueda_Local(G,AMonocromaticas,poblacion,probabilidades,best,best_ind,termina):
    for ind in range(tam_poblacion):  # Realiza la búsqueda local en la población inicial
        AMonocromaticas.append(Busquedas_locales.numeroAristasMono(G, poblacion[ind],
                                                                   numColores))  # calcula el número de aristas monocromáticas del individuo

        ActualizaMejor(AMonocromaticas,poblacion,best,best_ind,ind,termina)

        AMonocromaticas[ind] = BusquedaLocal(poblacion[ind], probabilidades[ind], AMonocromaticas[ind], G)

        if AMonocromaticas[ind] == 0:
            best = 0
            best_ind = poblacion[ind]
            return 1
        else:
            ActualizaMejor(AMonocromaticas, poblacion, best, best_ind, ind, termina)
            if termina == True:
                break
    return 0


'''Si tres generaciones no encuentra nada mejor, el progama termina'''
def Termina_Generacion(bestAnt,AMonoNuevo,gen,NumGenIgual):
    bestAnt.append(AMonoNuevo)
    if (gen + 1) % NumGenIgual == 0:
        if sonIguales(bestAnt, NumGenIgual):
            bestAnt = []
            return 0
        bestAnt = []



'''Función encargada de imprimir al mejor individuo, el mejor número de aristas monocromáticas, y el tiempo total del programa'''
def imprimeResultados(best,best_ind,start_time,archivo,archivo2):
    print('\n\nmejor número de aristas monocromáticas: ')
    print(str(best))
    print('mejor individuo: ')
    print(best_ind)
    print("%s seconds" % (time.time() - start_time))
    # bolsas.Muestra_Grafo(G)
    # bolsas.colorea_grafo(G,best_ind)
    # if best == 0:
    archivo.write(str(best[0])+ '\n')
    archivo2.write(str(time.time() - start_time)+ '\n')
    # archivo.close()


#***************************************  Algoritmo Genético   *********************************************
def main():
    # =================================== Definicion de Grafos y lectura de grafos aleatorios ==================
    archivo = open("NAM","w")
    archivo2 = open("Tiempo", "w")
    #for i in range(10):

    #print('\n***Grafo#' + str(i + 1))
    #carpeta='Grafos' + str(numNodosAle) + 'P' + str(probabilidad)
    #nombre= '/G' + str(numNodosAle) + '_' + str(probabilidad) +'_' + str(grafo)#+ str(i+1)
    #nombre = '/G' + str(probabilidad) + '_' + str(i + 1)#+ str(grafo)
    #Nombre_benchmark = carpeta + nombre

    G = bolsas.crea_grafo(Nombre_benchmark) #Crea el grafo apartir de un archivo de texto
    # G=bolsas.crea_grafo_aleatorio(numNodosAle, probabilidad)   #Crea un grafo a partir de un número de nodos y una probabilidad
    #G = Lee_Grafo(path + Nombre_benchmark)
    # print(Matriz_Adyacencia)
    numNodos = max(G.nodes)

    # ==========================================================================================================
    # =================================== Definicion de variables y estructuras de datos ========================
    ind = 0
    termina = False
    NumGenIgual = 3
    best = np.array([np.inf])
    bestAnt = []
    best_ind = [0]
    poblacion = []
    probabilidades=[]
    AMonocromaticas = []
    nuevos_individuos = []
    AMonoNuevo = []
    probabilidadeshijo = []
    ind_padres = []
    #Estructuras para los nuevos individuos
    nuevos_individuos = []
    AMonoNuevo = []
    ind_padres = []
    probabilidadeshijo = []
    mitad_poblacion = int(tam_poblacion / 2)
    # ==========================================================================================================
    # =================================== Inicia Algoritmo genético =============================================
    start_time = time.time()
    InicializacionPoblacion(poblacion, probabilidades, G,numNodos)
    if Busqueda_Local(G, AMonocromaticas, poblacion, probabilidades, best, best_ind, termina):
        termina = True
    if not termina: #Continua si arriba NO encontró un individuo con aristas monocromáticas igual a cero
        for gen in range(numGeneraciones):
            for p in range(mitad_poblacion):
                Padres = EligePadres(poblacion, numNodos, ind_padres, probabilidadeshijo)
                nuevos_individuos.append(CruzaPadres(Padres, probabilidadeshijo, AMonoNuevo, numNodos, G))
                ActualizaMejor(AMonocromaticas, poblacion, best, best_ind, ind, termina)
            for j in range(mitad_poblacion):
                AMonoNuevo[j] = BusquedaLocal(nuevos_individuos[j], probabilidadeshijo[j],AMonoNuevo[j], G)   #realiza la búsqueda local con el número de aristas monocromáticas del nuevo individuo
                ActualizaPoblacion(poblacion, nuevos_individuos[j], AMonocromaticas, AMonocromaticas[j], probabilidades, probabilidadeshijo[j], ind_padres[j]) #reemplaza al hijo con el peor individuo
                ActualizaMejor(AMonocromaticas, poblacion, best, best_ind, ind, termina)
            if Termina_Generacion(bestAnt, AMonoNuevo, gen, NumGenIgual) == 0:
                break
    imprimeResultados(best, best_ind, start_time,archivo,archivo2)
    # =================================== Termina Algoritmo genético =============================================
    print("Terminó")

if __name__ == '__main__':
    main()
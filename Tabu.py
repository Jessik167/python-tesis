import networkx as nx
import random

A = 10
alfa = 0.6
iteraciones = 20
busqueda_vecindario = 100


'''Mueve un nodo al azar a otra bolsa (se mueve en el vecindario)'''
def vecino (G,individuo, probabilidades, numColores):
    #best = numeroAristasMono(G,individuo,numColores)
    for i in range(busqueda_vecindario):
        while True:
            bolsaProbabilistica = bolsaAleatoriaProbabilidad(probabilidades, numColores) #Elige una bolsa con probabilidad a su número de nodos
            if individuo[bolsaProbabilistica]:  #verifica que la bolsa que eligió no esté vacía
                break
        dic = dict(individuo[bolsaProbabilistica])   #Toma el diccionario con los nodos de la bolsa elegida
        nodo = random.choice(list(dic.keys()))   #Elige un nodo al azar de la bolsa
        ListaAdj = G[nodo]
        monoAct = num_mono_bolsa(dic, ListaAdj) #calcula el número de aristas monocromáticas del nodo en la bolsa elegida
        BolsaNueva = BolsaAleatoria(bolsaProbabilistica, numColores)
        monopost = num_mono_bolsa(individuo[BolsaNueva], ListaAdj)
        if monopost != 0 and monopost < monoAct:
            del dic[nodo]  # Elimina el nodo de la bolsa
            individuo[BolsaNueva][nodo] = nodo  # Inserta el nodo en otra bolsa al azar



'''Elige una bolsa aleatoria que no sea la que ya eligió probabilísticamente'''
def BolsaAleatoria(bolsa,numColores):
    while True:
        rand = random.randint(0, numColores - 1)    #Elige una bolsa aleatoria
        if rand != bolsa:   #si la bolsa es diferente a la elegida
            return rand #Regresa el indice de la bolsa



'''Elige una bolsa con probabilidad basada en el número de nodos contenidos en la bolsa'''
def bolsaAleatoriaProbabilidad (probabilidades, numColores):
    r = random.uniform(0, 1) #selecciona un número al azar del cero al uno
    l = 0
    for i in range(numColores): #recorre hasta el número de bolsas
        if (r >= l and r < l + probabilidades[i]):  #si cae entre l y la probabilidad de la bolsa i
            return i    #retorna el indice i
        else:
            l = l + probabilidades[i]    #si no a la variable l le suma probabilidad de la bolsa i


'''Calcula el número de aristas monocromáticas de la bolsa'''
def num_mono_bolsa (bolsa, adyacentes):
    num_mono = 0
    for i in bolsa:
        if i in adyacentes:
            num_mono = num_mono + 1
    return num_mono



'''Calcula el número monocromático del individuo'''
def numeroAristasMono(G, individuo, numColores):
    num_mono = 0
    for i in range(numColores):
        nodos_bolsa = list(individuo[i])
        for j in range(len(nodos_bolsa)):
            #adj = G[nodos_bolsa[j]]
            for r in range(len(nodos_bolsa)):
                if G.has_edge(nodos_bolsa[j], nodos_bolsa[r]): #or G.has_edge(nodos_bolsa[r], nodos_bolsa[j]):
                    num_mono = num_mono + 1
    return num_mono



'''Realiza la búsqueda local en el individuo'''
def Busqueda_Tabu(G, individuo, probabilidades, numColores):
    #Lista_tabu = []
    #num_conflictos = 0
    #tabu_ternure = random.randint(0, A-1) + alfa * num_conflictos
    #Lista_tabu.append(individuo)
    #for i in range(iteraciones):
    vecino(G, individuo, probabilidades, numColores)
    #numeroAristasMono(G,individuo,numColores)
    return individuo


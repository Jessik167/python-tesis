import main
import time

numNodosAle = 20
probabilidad = 0.5
numColores = 7
grafo = 4

path='C:/Users/jessi/PycharmProjects/GrafosRand/GrafosAleatorios/'



#archivo = open("Resultados","w")
#for i in range(10):
start_time = time.time()
#print('Grafo#' + str(i + 1))
carpeta='Grafos' + str(numNodosAle) + 'P' + str(probabilidad)
nombre= '/G' + str(probabilidad) +'_' + str(grafo)#+ str(i+1)
Nombre_benchmark = carpeta + nombre
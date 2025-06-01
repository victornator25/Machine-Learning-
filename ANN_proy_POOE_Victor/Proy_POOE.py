import numpy as np
#como se va usar varias veces conviene ponerle un alias
from numpy.random import uniform as u
from numpy.random import normal  as n 

import matplotlib.pyplot as plt
import matplotlib as mpl

"""
trataré de explicar cada paso para que se a entendible
y no se me olvide qué hice y para que sirven algunos métodos
"""

# Definimos los valores de nuestra neurona de entrada
#con solo dos neuronas de entrada
#con una distribucion uniforme
y_in = u(low=-1, high=1, size=2)
 

class Neural_Net:
    """
    por convencion escribimos los nombres en inglés
    Neural net -- red neuronal
    layers --capas
    neurons -- neuronas
    """
 
    def __init__(self, n_layers=20, n_neurons=50):
        """
        __init__ para inicilizar parametros y constructor
        y se definen las capas y neuronas por capa
        
        """
        """
        se inicializa un vector de pesos de entrada 
        de tamaño de neuronas de entrada por el numero de neuronas por capa
        """
        self.w_in_u = u(low=-1, high=1, size=(2, n_neurons)) 
        # los bias solo se ocupa un vector del tamaño del n_neurons
        self.b_in_u = u(low=-1, high=1, size= n_neurons)

        """
        Definimos pesos y bias en las capas internas, conectando capas ocultas
        del tamaño de las capas ocultas por cuantas conecciones y numero de salidas
        este es uno de los pasos mas raros porque para que se hagan las conexiones
        con nodos de las capas ocultas debe haber una matriz 
        del numero de capas x el numero de neuronas por capa y cada neurona está conectada 
        con todas de la siguiente capa, por eso el tamaño de los pesos es así
        
        """
        self.w_hidden_u = u(low=-4, high=4, size=(n_layers, n_neurons, n_neurons))
        # el bias para las capas ocultas solo son una matrix del tamaño del numero de capas 
        #por en numero de neuronas, estas no estan conectadas con todas las neuronas
        self.b_hidden_u = u(low=-1, high=1, size=(n_layers, n_neurons))

        # Definimos pesos y bias en las capa de salida
        #para este caso solo una neurona de salida para el Feed Forward
        self.w_out_u = u(low=-1, high=1, size=(n_neurons,1)) 
        self.b_out_u = u(low=-1, high=1, size= 1)
        
        
        #######___Ahora con distribucion normal se inicializan_
        #### los mismos tamaños para cada nivel      
        self.w_in_n = n(loc=0.0, scale=0.5, size=(2, n_neurons)) 
        self.b_in_n = n(loc=0.0, scale=0.5, size= n_neurons)
 
        self.w_hidden_n = n(loc=0.0, scale=0.9, size=(n_layers, n_neurons, n_neurons))
        self.b_hidden_n = n(loc=0.0, scale=0.5, size=(n_layers, n_neurons))

        self.w_out_n = n(loc=0.0, scale=0.5, size=(n_neurons,1)) 
        self.b_out_n = n(loc=0.0, scale=0.5, size= 1)

    @staticmethod
    
     
    """
    
    con @staticmethod se resuelve un problema que le envie por correo
    ya funciona con ambas funciones de activacion
    lo que hace es que anula un primer argumento que tendría que recibir la funcion 
    y ahora corresponden bien los argumentos que recibe
    
    """
    
    
    def sigmoid_fun(y_in, w, b):

        z = np.dot(y_in, w) + b
        #cada entrada de b se le suma a la matriz producto
        #de los pesos de la capa de entrada por pesos de la capa siguiente 
        s =  1. / (1. + np.exp(-z))
        #funcion de activación
        
        return s
    
    @staticmethod
    
    def tanh_fun(y_in, w, b):
        # funcion de activacion tangencial hiperbolica
        a = np.dot(y_in, w) + b
        
       
        k =  (2. /(1.+ np.exp(-2*a))) + 1
       
        return k
  
   
  
   """
   se hacen los feed Forwards
   este es el que define la arquitectura
   es una arquitectura relativamente sencilla
   """

    def FF_sigmoid_u(self, y_in):
        #se pasa la funcion de activacion por las neuronas
        #de estrada con los respectivos pesos y bias
        
        y = self.sigmoid_fun(y_in, self.w_in_u, self.b_in_u) 
        #se hacen dos iteraciones para pasar por cada capa
        for i in range(self.w_hidden_u.shape[0]):  
            # el metodo shape para por cada neurona úmero de capas ocultas
            #shape devuelve una tupla para cada i al rango del num de capas
            #a continuacion se hace una iteracion
            #pasando la funcion por cada nodo
            y = self.sigmoid_fun(y, self.w_hidden_u[i], self.b_hidden_u[i]) 
        output = self.sigmoid_fun(y, self.w_out_u, self.b_out_u)
        return output
        #esto es la salida de un red neuronal solo un valor 
        #para esta arquitectura solo hay un valor de retorno 
        #de toda las capas de la red
        
        """ 
        Lo mismo con las demas dunciones y distribuciones
        respectivamente
        """
    def FF_sigmoid_n(self, y_in):
    
        y = self.sigmoid_fun(y_in, self.w_in_n, self.b_in_n) 
        for i in range(self.w_hidden_n.shape[0]): 
            y = self.sigmoid_fun(y, self.w_hidden_n[i], self.b_hidden_n[i]) 
        output = self.sigmoid_fun(y, self.w_out_n, self.b_out_u)
        
        return output
    

    
    def FF_tanh_u(self, y_in):
    
        y = self.tanh_fun(y_in, self.w_in_u, self.b_in_u)       
        for i in range(self.w_hidden_u.shape[0]): 
            y = self.tanh_fun(y, self.w_hidden_u[i], self.b_hidden_u[i]) 
        output = self.tanh_fun(y, self.w_out_u, self.b_out_u)
       
        return output
    
    
    def FF_tanh_n(self, y_in):
    
        y = self.tanh_fun(y_in, self.w_in_n, self.b_in_n) 
        for i in range(self.w_hidden_n.shape[0]):  
            y = self.tanh_fun(y, self.w_hidden_n[i], self.b_hidden_n[i]) 
        output = self.tanh_fun(y, self.w_out_n, self.b_out_u)
        return output
    
    """ FIN DE LOS fEEDORWARDS """
    
    def visualizar(self, grid_size=50, colormap='seismic'):
        """Función para visualizar el mapeo de la red neuronal en un 
        una rejilla bidimensional.
        
        grid_size : int
            El tamaño a utlizar para crear rejilla. La rejilla se crea de 
            tamaño (grid_size, grid_size) se definió de 400
        colormap : str
            El mapa de color a utilizar
            utilicé el seismic de la documentacion de colormaps
            ya que en este se pueden diferenciar claramente los valores
            negativos azules
            muy cercanos a cero o cero son de tono blanco
            rojo son positivos
           
        """

        # se crea una rejilla
        x = np.linspace(-0.5, 0.5, grid_size)
        y = np.linspace(-0.5, 0.5, grid_size)
        xx, yy = np.meshgrid(x, y)

        # Para todas las coordenadas (x, y) en la rejilla
        # hacemos una única lista con los pares de puntos
        x_flat = xx.flatten()
        y_flat = yy.flatten()
        #vector de entradas con todas las rejillas x,y
        y_in = zip(x_flat, y_flat)
        #el zip hace que tome los valores iterados con cada y_in
        #y le asigna un pixel y se iteran de el tamaño de la rejilla
        #es decir cada pixel representa la salida de una red neuronal
        
        y_in = np.array(list(y_in))
        #los pasamos a una lista 

        # Hacemos feedforward con la red
        #y probamos con cada distribucion y funcion
        y_out_1 = self.FF_sigmoid_u(y_in)
        #y_out_1 = self.FF_sigmoid_n(y_in)
        #y_out_1 = self.FF_tanh_u(y_in)
        #y_out_1 = self.FF_tanh_n(y_in)
        

        # Redimensionamos a la rejilla
        #cambiar las dimensiones a una de n x n
        y_out_2d_1 = np.reshape(y_out_1, (grid_size, grid_size))
        
        cmap=colormap
        # Graficamos los resultados de la red
        plt.figure(figsize=(10, 10))
        plt.axes([0, 0, 1, 1])
        plt.imshow(y_out_2d_1,extent=[-0.5, 0.5, -0.5, 0.5],interpolation='bessel', cmap=cmap)
        #quitamos los ejes para que se vea limpia
        plt.axis(False)
        plt.show()
        return y_out_1
       
    
#definimos un objeto a
A=Neural_Net()
#Corremos la vusualizacion
A.visualizar()
#cada imagen es diferente con cada iteracion
#pero dependiendo la funcion y distribucion se puede 
#predecir mas o menos como va a estar la imagen
#eso se explica en el doc

#-------- Importamos las librerias ---------
import os   
import tensorflow as tf          #Libreria de inteligencia artificial
import cv2                       #Libreria de OpenCV
import matplotlib.pyplot as plt  #Libreria para observar las imagenes
import numpy as np               #Operaciones
from tensorflow.keras.callbacks import TensorBoard #Libreria para observar el fincionamiento de la red
from tensorflow.keras.preprocessing.image import ImageDataGenerator #libreria para modificar mis imagenes.

# ------------ Almacenamos la direccion de las imagenes ----------
entrenamiento = "C:/Users/rodri/Desktop/Proyecto IXALaB/Red Neuronal Convolucional/Dataset/Entrenamiento"
validacion = "C:/Users/rodri/Desktop/Proyecto IXALaB/Red Neuronal Convolucional/Dataset/Validacion"

listaTrain = os.listdir(entrenamiento)
listaTest = os.listdir(validacion)

#-------- Establecemos algunos parametros ----------
ancho, alto = 200, 200

#Listas Entrenamiento
etiquetas = []
fotos = []
datos_train = []
con = 0

#Listas Validacion
etiquetas2 = []
fotos2 = []
datos_vali = []
con2 = 0

#---------- Extraemos un una lista las fotos y entra las etiquetas -----------
#Entrenamiento
for namedir in listaTrain:
    nombre = entrenamiento + '/' + namedir #leemos las fotos

    for fileName in os.listdir(nombre): #Asiganmos las etiquetas a cada foto
        etiquetas.append(con) #Valor de la etiqueta (0 la primer etiqueta y 1 a la segunda)
        img = cv2.imread(nombre + '/' + fileName, 0) #Leemos la imagen
        img = cv2.resize(img, (ancho,alto), interpolation=cv2.INTER_CUBIC) #Redimensionamos las imagenes
        img = img.reshape(ancho, alto, 1)  #dejamos 1 solo canal
        datos_train.append([img, con])
        fotos.append(img) #Agregamos las imagenes en EDG
    
    con = con + 1

#Validacion
for namedir2 in listaTest:
    nombre2 = validacion + '/' + namedir2 #Leeremos las fotos

    for fileName2 in os.listdir(nombre2):   #Asignamos las etiquetas a cada foto
        etiquetas2.append(con2) #Valor de la etiqueta(asignamos 0 a la primer etiqueta y 1 a la segunda)
        img2 = cv2.imread(nombre2 + '/' + fileName2, 0) #Leemos la imagen
        img2 = cv2.resize(img2, (ancho, alto), interpolation=cv2.INTER_CUBIC) #redimensionamos las imagenes
        img2 = img2.reshape(ancho, alto, 1)   #Dejamos 1 solo canal
        datos_vali.append([img2, con2])
        fotos2.append(img2) #Agregamos las imagenes en EDG
    
    con2 = con2 + 1

# ----------- Normalizamos las imagenes (0 o 1) -------------
fotos = np.array(fotos).astype(float) / 255
print(fotos.shape)
fotos2 = np.array(fotos2).astype(float) / 255
print(fotos2.shape)
#Pasamos las listas a Array
etiquetas = np.array(etiquetas)
etiquetas2 = np.array(etiquetas2)

imgTrainGen = ImageDataGenerator(
    rotation_range = 50,     #Rotacion aleatoria de las imagenes
    width_shift_range = 0.3, #Mover imagen a los lados
    height_shift_range = 0.3,#Mover la imagen arriba y abajo
    shear_range = 15,        #Inclinamos la imagen
    zoom_range = [0.5, 1.5], #Hacemos zoom a la imagen
    vertical_flip = True,    #Flip verticales aleatorios
    horizontal_flip = True   #Flip horizontal aleatorio
)

imgTrainGen.fit(fotos)
plt.figure(figsize=(20,8))
for imagen, etiqueta in imgTrainGen.flow(fotos, etiquetas, batch_size=10, shuffle=False):
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(imagen[i], cmap='gray')
    plt.show()
    break

imgTrain = imgTrainGen.flow(fotos, etiquetas, batch_size = 32)

# --------- Estructura de la Red Neuronal Convolucional -------------
# Modelo con Capas Densas
ModeloDenso = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (200, 200, 1)), #Capa de entrada con 40mil neuronas
    tf.keras.layers.Dense(150, activation = 'relu'),      #Capa Densa con 150 neuronas
    tf.keras.layers.Dense(150, activation = 'relu'),      #Capa Densa con 150 neuronas
    tf.keras.layers.Dense(1, activation = 'sigmoid'),
])

#Modelo con Capas Convolucionales
ModeloCNN = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (200, 200, 1)), #Capa de entrada Convolucional 32, Kernel 3x3
    tf.keras.layers.MaxPooling2D(2,2),                      #Capa de Max Pooling
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'), #Capa Convolucional con 64 Kernel
    tf.keras.layers.MaxPooling2D(2,2),                      #Capa de Max Pooling
    tf.keras.layers.Conv2D(128, (3,3), activation = 'relu'),#Capa Convolucional con 128 Kernel
    tf.keras.layers.MaxPooling2D(2,2),                      #Capa de Max Pooling

    #Capas Densas de clasificacion
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation = 'relu'),       #Capa densa con 256 neuronas
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

#Modelo con Capas Convolucionales y Drop Out
ModeloCNN2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (200, 200, 1)), #Capa de entrada Convolucional 32, Kernel 3x3
    tf.keras.layers.MaxPooling2D(2,2),                      #Capa de Max Pooling
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'), #Capa Convolucional con 64 Kernel
    tf.keras.layers.MaxPooling2D(2,2),                      #Capa de Max Pooling
    tf.keras.layers.Conv2D(128, (3,3), activation = 'relu'),#Capa Convolucional con 128 Kernel
    tf.keras.layers.MaxPooling2D(2,2),                      #Capa de Max Pooling

    #Capas Densas de clasificacion
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation = 'relu'),       #Capa densa con 256 neuronas
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

#---------------- Compilamos los modelos: agregamos el optimizador y la funcion de perdida --------------
ModeloDenso.compile(optimizer = 'adam',
                    loss = 'binary_crossentropy',
                    metrics = ['accuracy'])

ModeloCNN.compile(optimizer = 'adam',
                    loss = 'binary_crossentropy',
                    metrics = ['accuracy'])

ModeloCNN2.compile(optimizer = 'adam',
                    loss = 'binary_crossentropy',
                    metrics = ['accuracy'])


#-------------- Observaremos y Entrenamos las redes -------------------

# Entrenamos modelo Denso
BoardDenso = TensorBoard(log_dir = "C:/Users/rodri/Desktop/Proyecto IXALaB/Red Neuronal Convolucional")
ModeloDenso.fit(imgTrain, batch_size=32, validation_data=(fotos2,etiquetas2),
                epochs=100, callbacks=[BoardDenso], steps_per_epoch=int(np.ceil(len(fotos) / float(32))),
                validation_steps= int(np.ceil(len(fotos2) / float(32))))
# Guardamos el modelo
ModeloDenso.save('ClasificadorDenso.h5')
ModeloDenso.save_weights('pesosDenso.h5')
print('Terminamos Modelo Denso')

# Entrenamos CNN sin DO
BoardCNN = TensorBoard(log_dir = "C:/Users/rodri/Desktop/Proyecto IXALaB/Red Neuronal Convolucional")
ModeloCNN.fit(imgTrain, batch_size=32, validation_data=(fotos2,etiquetas2),
                epochs=100, callbacks=[BoardCNN], steps_per_epoch=int(np.ceil(len(fotos) / float(32))),
                validation_steps= int(np.ceil(len(fotos2) / float(32))))
# Guardamos el modelo
ModeloCNN.save('ClasificadorCNN.h5')
ModeloCNN.save_weights('pesosCNN.h5')
print('Terminamos Modelo CNN 1')

# Entrenamos CNN con DO
BoardCNN2 = TensorBoard(log_dir = "C:/Users/rodri/Desktop/Proyecto IXALaB/Red Neuronal Convolucional")
ModeloCNN2.fit(imgTrain, batch_size=32, validation_data=(fotos2,etiquetas2),
                epochs=100, callbacks=[BoardCNN2], steps_per_epoch=int(np.ceil(len(fotos) / float(32))),
                validation_steps= int(np.ceil(len(fotos2) / float(32))))
# Guardamos el modelo
ModeloCNN2.save('ClasificadorCNN2.h5')
ModeloCNN2.save_weights('pesosCNN2.h5')
print('Terminamos Modelo CNN 2')

# para la cmd donde se abre el tensor board
# tensorboard --logdir="C:/Users/rodri/Desktop/Proyecto IXALaB/Red Neuronal Convolucional/Board"
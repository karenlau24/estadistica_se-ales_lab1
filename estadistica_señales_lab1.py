# -- coding: utf-8 --

import math

#LIBRERIA wfdb PARA UTILIZAR LAS SEÑALES DESCARGADAS DE PhysioNet

import wfdb

#IMPORTAR MATPLOTLIB PARA DESARROLLAR LOS GRAFICOS NECESARIOS 

import matplotlib.pyplot as plt 

#IMPORTAR SPICY PARA CÁLCULAR EL PDF 

from scipy.stats import norm

#LIBRERIA NUMPY PARA MANIPULACION NUMERICA 

import numpy as np

#SE DECLARAN LAS VARIABLES QUE SE UTILIZARÁN EN LOS CÁLCULOS 

suma = 0 
cont = 0
cuadrado = 0
varianza = 0
minimo = 0
maximo = 0
N_intervalos = 0
bins = 0
ruido = 0
N = 1024 


#EN ESTE CASO, SE UTILIZÓ UNA SEÑAL DE VOZ DE UNA PACIENTE FEMENINA DE 54 AÑOS
#LA CUAL TIENE UN DIÁGNOSTICO DE DISFONÍA HIPOCINÉTICA Y QUE LLEVA UN 
#ESTILO DE SALUDABLE

#LA FRECUENCIA DE MUESTREO FUE DE 8000HZ (8000 DATOS POR SEGUNDO)
#RESOLUCIÓN DE 32 BITS 
#FILTROS APLICADOS (NO SE ESPECIFICA CUÁLES)


#LA ANTERIOR INFORMACIÓN FUE TOMADA DE PHYSIONET, LINK: 
#https://physionet.org/content/voiced/1.0.0/

#CARGAR LA SEÑAL, SE REQUIEREN LOS ARCHIVOS .DAT Y .HEA

signal = wfdb.rdrecord('voice005')

#OBTENER LOS VALORES EN EJE Y DE LA SEÑAL 

valores = signal.p_signal


#--------------------------------CALCULO DE LA MEDIA ARITMETICA 

#SIN FUNCION
for valor in valores:
    suma += valor
    cont += 1
    
    
media = suma / cont  

#CON FUNCION 

media2 = np.mean(valores)


#FIGURA QUE VA A CONTENER AMBOS GRAFICOS
plt.figure(figsize=(12, 6))

#GRAFICAR LA MEDIA DE MANERA MANUAL
plt.subplot(2, 1, 1)
plt.plot(valores)
plt.xlabel('Muestra')
plt.ylabel('Amplitud')
plt.title('Voice signal 005 promedio manual')
plt.axhline(y=media, color='r', linestyle='--', label='Media')
plt.legend()

#GRAFICAR LA MEDIA CON NUMPY
plt.subplot(2, 1, 2)
plt.plot(valores)
plt.xlabel('Muestra')
plt.ylabel('Amplitud')
plt.title('Voice signal 005 promedio np')
plt.axhline(y=media2, color='g', linestyle='--', label='Media')
plt.legend()

plt.tight_layout()
plt.show()


#-------------------------------CALCULO DE LA DESVIACION ESTANDAR 


#SIN FUNCION
for valor in valores:
    resta = valor - media 
    cuadrado += (resta*resta)
    varianza = cuadrado / cont
    
desv = math.sqrt(varianza)

#CON FUNCION

desv2 = np.std(valores)

#FIGURA QUE VA A CONTENER AMBOS GRAFICOS
plt.figure(figsize=(12, 6))

#GRAFICAR LA MEDIA DE MANERA MANUAL
plt.subplot(2, 1, 1)
plt.plot(valores)
plt.xlabel('Muestra')
plt.ylabel('Amplitud')
plt.title('Voice signal 005 desv manual')
plt.axhline(y=desv, color='r', linestyle='--', label='Desv std')
plt.legend()

#GRAFICAR LA MEDIA CON NUMPY
plt.subplot(2, 1, 2)
plt.plot(valores)
plt.xlabel('Muestra')
plt.ylabel('Amplitud')
plt.title('Voice signal 005 desv np')
plt.axhline(y=desv2, color='g', linestyle='--', label='Desv std')
plt.legend()

plt.tight_layout()
plt.show()




#-------------------------------------------COEFICIENTE DE VARIACIÓN

#SIN FUNCION
coe = (desv / media) * 100
coe_float = float(coe)

#CON FUNCION

coe2 = (desv2 / media2) * 100
coe2_float = float(coe2)

#FIGURA QUE VA A CONTENER AMBOS GRAFICOS
plt.figure(figsize=(12, 6))

#GRAFICAR EL COEFICIENTE DE VARIACION DE MANERA MANUAL
plt.subplot(2, 1, 1)
plt.plot(valores)
plt.xlabel('Muestra')
plt.ylabel('Amplitud')
plt.title('Voice signal 005 coeficiente variacion')
plt.text(0.1, 0.9, f"CV: {coe_float:.2f}%",  
         transform=plt.gca().transAxes, fontsize=9, ha='right', va='top')
plt.legend()

#GRAFICAR EL COEFICIENTE DE VARIACION CON NUMPY
plt.subplot(2, 1, 2)
plt.plot(valores)
plt.xlabel('Muestra')
plt.ylabel('Amplitud')
plt.title('Voice signal 005 coeficiente variacion np')
plt.text(0.1, 0.9, f"CV: {coe2_float:.2f}%",  
         transform=plt.gca().transAxes, fontsize=9, ha='right', va='top')
plt.legend()

plt.tight_layout()
plt.show()


#------------------------------------------------HISTOGRAMA 

#SIN FUNCIONES 

#ENCONTRAR EL RANGO Y DEFINIR LOS INTERVALOS 
minimo = np.min(valores)
maximo = np.max(valores)
N_intervalos = 20
intervalos = (maximo - minimo) / N_intervalos

#SE CREA UNA LISTA PARA GUARDAR ALLÍ LAS FRECUENCIAS 
freq = [0] * N_intervalos

#CONTAR LA FRECUENCIA EN CADA INTERVALO 
for valor in valores:
    index = int((valor - minimo) // intervalos)
    if 0 <= index < N_intervalos:
        freq[index] += 1

#CREAR LOS LIMITES DE LOS INTERVALOS 
bins = np.linspace(minimo, maximo, N_intervalos+1)

#GRAFICAR EL HISTOGRAMA

#DEFINIR LOS COLORES DE LAS BARRAS DEL HISTOGRAMA
colores = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan'] * (N_intervalos // 10 + 1)

plt.bar(bins[:-1], freq, width=intervalos, align='edge', color=colores, edgecolor='black')
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Histograma manual')
plt.show()


# CON FUNCION

plt.figure(figsize=(12, 6))
plt.hist(valores, bins=N_intervalos, color='orange', edgecolor='black')
plt.xlabel('Valores')
plt.ylabel('Frecuencia')
plt.title('Histograma con funcion')
plt.show()


#---------------------------------FUNCION DE PROBABILIDAD 

#SIN FUNCIONES 

#CALCULAR LA PROBABILIDAD EN CADA INTERVALO 
probabilidad = freq / (cont * intervalos)

#CREAR EL LIMITE DE LOS INTERVALOS 
bins = np.linspace(minimo, maximo, N_intervalos+1)

# Graficar la función de probabilidad
plt.plot(bins[:-1], probabilidad, label='Función de Probabilidad')
plt.xlabel('Valores de la señal')
plt.ylabel('Probabilidad')
plt.title('Función de Probabilidad de la Señal')
plt.legend()
plt.show()

#CON FUNCIONES 


pdf = norm.pdf(valores, media, desv)

#GRAFICAR LA FUNCIÓN DE PROBABILIDAD HECHA CON LA FUNCION PDF 
plt.plot(valores, pdf)
plt.xlabel('Amplitud')
plt.ylabel('Probabilidad')
plt.title('Función de probabilidad de la señal')
plt.show()






#---------------------CONTAMINACION CON RUIDO, Y RELACION SEÑAL-RUIDO----------------

#---------------------------------------------------------GAUSSIANO

#GENERAR RUIDO GAUSSIANO

vectores = valores.flatten()
ruido_gaussiano = np.random.normal(0, 1, len(vectores))

#NORMALIZAR EL RUIDO 

voice_max = np.max(np.abs(vectores))
voice_min = np.min(np.abs(vectores))
ruido_max = np.max(np.abs(ruido_gaussiano))
ruido_normalizado = ruido_gaussiano / ruido_max * (voice_max - voice_min)

#FIGURA QUE VA A CONTENER AMBOS GRAFICOS
plt.figure(figsize=(12, 6))

#GRAFICAR EL COEFICIENTE DE VARIACION DE MANERA MANUAL
plt.subplot(2, 1, 1)
plt.plot(ruido_gaussiano)
plt.xlabel('Muestra')
plt.ylabel('Amplitud')
plt.title('Ruido Gaussiano')
plt.legend()

#GRAFICAR EL COEFICIENTE DE VARIACION CON NUMPY
plt.subplot(2, 1, 2)
plt.plot(ruido_normalizado)
plt.xlabel('Muestra')
plt.ylabel('Amplitud')
plt.title('Ruido Gaussiano normalizado')
plt.legend()

plt.tight_layout()
plt.show()


#SUMAR AMBAS SEÑALES 

senal_ruidosa = vectores + ruido_normalizado

#FIGURA QUE VA A CONTENER AMBOS GRAFICOS
plt.figure(figsize=(12, 6))

#GRAFICAR EL COEFICIENTE DE VARIACION DE MANERA MANUAL
plt.subplot(2, 1, 1)
plt.plot(vectores)
plt.xlabel('Muestra')
plt.ylabel('Amplitud')
plt.title('Señal original')
plt.legend()

#GRAFICAR EL COEFICIENTE DE VARIACION CON NUMPY
plt.subplot(2, 1, 2)
plt.plot(senal_ruidosa)
plt.xlabel('Muestra')
plt.ylabel('Amplitud')
plt.title('Señal contaminada')
plt.legend()

plt.tight_layout()
plt.show()

#----------------------------------------------------------- RUIDO IMPULSO
# GENERACION DE RUIDO DE IMPULSO
impulsos = int(0.0002*len(valores))# Número de impulsos, ajusta este valor según sea necesario
indices = np.random.choice(len(valores), impulsos, replace=False)
ruido_impulso = np.copy(vectores)
ruido_impulso[indices] = np.max(vectores) 

#NORMALIZAR EL RUIDO 
ruido_impulso_max = np.max(np.abs(ruido_impulso))
ruido_impulso_normalizado = ruido_impulso / ruido_impulso_max * (voice_max - voice_min)

#SUMAR LAS SEÑALES 
senal_ruidosa2 = vectores + ruido_impulso_normalizado

#FIGURA QUE VA A CONTENER AMBOS GRAFICOS
plt.figure(figsize=(12, 6))

#GRAFICAR EL COEFICIENTE DE VARIACION DE MANERA MANUAL
plt.subplot(2, 1, 1)
plt.plot(ruido_impulso)
plt.xlabel('Muestra')
plt.ylabel('Amplitud')
plt.title('Ruido Impulso')
plt.legend()

#GRAFICAR EL COEFICIENTE DE VARIACION CON NUMPY
plt.subplot(2, 1, 2)
plt.plot(senal_ruidosa2)
plt.xlabel('Muestra')
plt.ylabel('Amplitud')
plt.title('Señal contaminada con ruido Impulso')
plt.legend()

plt.tight_layout()
plt.show()

# -------------------------------------------------------- RUIDO ARTEFACTO

# Generar ruido de baja frecuencia combinado con ruido blanco
frecuencia_baja = 0.1  # Frecuencia de la onda cuadrada
t = np.arange(len(valores))
ruido_onda_cuadrada = np.sign(np.sin(2 * np.pi * frecuencia_baja * t))

# Generar ruido blanco
ruido_blanco = np.random.normal(0, 0.5, len(valores))

# Combinar ambos ruidos
ruido_artefacto = ruido_onda_cuadrada + ruido_blanco

# Normalizar el ruido artefacto para ajustarlo a la amplitud de la señal
voice_max = np.max(np.abs(valores))
voice_min = np.min(np.abs(valores))
ruido_max = np.max(np.abs(ruido_artefacto))
ruido_artefacto_normalizado = ruido_artefacto / ruido_max * (voice_max - voice_min)

# Sumar el ruido artefacto a la señal original
senal_ruidosa3 = vectores + ruido_artefacto_normalizado

# FIGURA QUE VA A CONTENER AMBOS GRAFICOS
plt.figure(figsize=(12, 6))

# GRAFICAR EL RUIDO ARTEFACTO NORMALIZADO
plt.subplot(2, 1, 1)
plt.plot(ruido_artefacto_normalizado)
plt.xlabel('Muestra')
plt.ylabel('Amplitud')
plt.title('Ruido Artefacto')
plt.legend()

# GRAFICAR LA SEÑAL CONTAMINADA CON RUIDO ARTEFACTO
plt.subplot(2, 1, 2)
plt.plot(senal_ruidosa3)
plt.xlabel('Muestra')
plt.ylabel('Amplitud')
plt.title('Señal contaminada con ruido Artefacto')
plt.legend()

plt.tight_layout()
plt.show()

#------------------------------------------------------CALCULO DE SNR 

#FUNCIÓN QUE VA A CALCULAR LA POTENCIA DE LAS SEÑALES, NECESARIAS PARA CALCULAR EL SNR
def pot(signal):
  
    return np.mean(signal**2)

#SE ENVÍAN LAS RESPECTIVAS SEÑALES A LA FUNCIÓN POT PARA EL CÁLCULO DE LAS POTENCIAS 
potSenal = pot(vectores)
potGaussN = pot(ruido_normalizado)
potImpulsoN = pot(ruido_impulso_normalizado)
potArtN = pot(ruido_artefacto_normalizado)
potGauss = pot(ruido_gaussiano)
potImpulso = pot(ruido_impulso)
potArt = pot(ruido_artefacto)


#SE IMPRIMEN LAS POTENCIAS DE CADA SEÑAL
print("Potencia de voice 005:", potSenal)
print("Potencia del ruido gaussiano normalizado:", potGaussN)
print("Potencia del ruido impulso normalizado:", potImpulsoN)
print("Potencia del ruido artefacto normalizado:", potArtN)
print("Potencia del ruido gaussiano:", potGauss)
print("Potencia del ruido impulso:", potImpulso)
print("Potencia del ruido artefacto:", potArt)

#SE CÁLCULA EL SNR ENTRE LA SEÑAL ORIGINAL Y EL CORRESPONDIENTE RUIDO 
snrGaussN = 10 * np.log10(potSenal / potGaussN)
snrImpulsoN = 10 * np.log10(potSenal / potImpulsoN)
snrArtN = 10 * np.log10(potSenal / potArtN)
snrGauss = 10 * np.log10(potSenal / potGauss)
snrImpulso = 10 * np.log10(potSenal / potImpulso)
snrArt = 10 * np.log10(potSenal / potArt)



print("SNR voice 005 / ruido gaussiano normalizado:", snrGaussN, "dB")
print("SNR voice 005 / ruido de impulso normalizado:", snrImpulsoN, "dB")
print("SNR voice 005 / ruido de artefacto normalizado:", snrArtN, "dB")
print("SNR voice 005 / ruido gaussiano:", snrGauss, "dB")
print("SNR voice 005 / ruido de impulso:", snrImpulso, "dB")
print("SNR voice 005 / ruido de artefacto:", snrArt, "dB")
















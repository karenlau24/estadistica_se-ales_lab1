## CONTENIDOS 

1. Información general
2. Tecnologías 
3. Instalación 
4. Análisis  
5. Conclusión 

### INFORMACIÓN GENERAL 

Este código realiza un análisis básico de una señal de voz obtenida de la base de datos PhysioNet.
El objetivo principal es caracterizar la señal y evaluar el impacto de diferentes tipos de ruido en su calidad.

### TECNOLOGÍAS 

-Python 3: Lenguaje de programación principal.
-NumPy: Biblioteca para realizar operaciones numéricas y trabajar con arreglos.
-SciPy: Biblioteca que proporciona herramientas para la computación científica y técnica.
-Matplotlib: Biblioteca para crear visualizaciones estáticas, animadas e interactivas en Python.
-wfdb: Biblioteca específica para leer y escribir archivos de señales fisiológicas en el formato WFDB, como los de PhysioNet.

### INSTALACIÓN 

-Clonar el repositorio:  git clone https://tu_repositorio.git
-Instalar las dependencias:  pip install numpy scipy matplotlib wfdb
-Ejecutar el código:  python estadistica_señales_lab1.py

### ANÁLISIS  

El código realiza el siguiente análisis:

-Cálculo de estadísticas: Se calculan la media, desviación estándar y coeficiente de variación de la señal.
-Visualización: Se generan gráficos de la forma de onda de la señal y su histograma.
-Contaminación de ruido: Se añaden diferentes tipos de ruido (gaussiano, impulso y artefacto) a la señal original.
-Cálculo de la relación señal-ruido (SNR): Se cuantifica la calidad de la señal en presencia de ruido.

### CONCLUSIÓN 

Este código proporciona una base sólida para el análisis de señales de voz. Los resultados obtenidos muestran la importancia de considerar el ruido al analizar señales biológicas. 
Futuras investigaciones pueden profundizar en el análisis de señales de voz y explorar aplicaciones más avanzadas.



 





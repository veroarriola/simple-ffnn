# simple-ffnn
**Proyecto PAPIME PE104425: Visualización animada en 3D de Redes Neuronales**

Código básico para entrenar una red ffnn y crear la animación en blender

## Instalación

* Instalar [Blender 4.4](https://www.blender.org/thanks/) descargando la versión en archivo comprimido y descomprimiendo. 
* Si se utiliza en algún lenguaje que no sea inglés, entrar a ```Editar->Preferencias->Interfaz->Idioma``` y desmarcar ```Traducir->Nuevos datos```, pues esto traduce los nombres de los nodos en la API.
* Clonar este repositorio dentro del directorio recién creado, al lado del ejecutable ```blender```.
* Instalar prerrequisitos desde la instalación local de ```python```:

```
sudo apt update
sudo apt upgrade
cd 4.4/python/
./bin/python3.11 -m pip install --upgrade pip
./bin/python3.11 -m pip install torch torchvision torchaudio
./bin/python3.11 -m pip install tensorboard
./bin/python3.11 -m pip install pytorch-ignite
./bin/python3.11 -m pip install seaborn seaborn_image
./bin/python3.11 -m pip install typeguard
./bin/python3.11 -m pip install pyyaml
```

## Uso

La producción de videos se realiza en dos etapas:
1. Entrenamiento de la red.
2. Creación de la animación en Blender.

### Entrenamiento

El código para el entrenamiento se encuentra en ```bffnn/mnistTorch.py``` y por defecto se ejecuta una rutina de
entrenamiento preconfigurada invocando:

```
./4.4/python/bin/python3.11 simple-ffnn/bffnn/mnistTorch.py
```

Si falla la descarga del conjunto de datos MNIST (porque frecuentemente es el caso),
se pueden descargar los archivos manualmente de [Github fgnt](https://github.com/fgnt/mnist/tree/master).
Colocarlos en el directorio ```simple-ffnn/nn-data/MNIST/raw``` y volver a ejecutar.

Este guión utiliza el archivo de configuración que se incluye como ejemplo en
```simple-ffnn/nn-saved/net_003/nn_config.yaml```.  Para entrenar una red con parámetros distintos
se puede modificar este archivo o, de preferencia, crear otro directorio dentro de ```nn-saved```
con su respectivo ```nn_config.yaml```.

Al ejecutar ```mnistTorch.py``` se guardarán en ```simple-ffnn/nn-saved/net_003/``` archivos de
```PyTorch``` con los valores de los pesos de la red al final de cada época de entrenamiento.
Estos archivos serán utilizados por el guión para ```Blender```, ```trainVideo.py```, para
determinar el estado de las conexiones de la red al animar su proceso de entrenamiento.

### Renderizado de video

Se incluye un guión de ejemplo, que produce una animación con cámara fija de una red FFNN.
El guión se puede editar fácilmente para ilustrar la evolución de los pesos
y su funcionamiento para diferentes imágenes de entrada.
Para probarlo utilizándolo como esta los pasos son los siguientes:

1. Ejecutar ```Blender```
2. En la sección ```Scripting``` abrir el archivo ```trainVideo.py``` y presionar ```Play```

Con estos pasos se deberá:
1. Crear la geometría de la red: neuronas, conexiones, sesgos y materiales para los objetos.
2. Cargar los datos del conjunto de entrenamiento (se puede reemplazar por el de prueba)
3. Cargar los valores de los primeros pesos entrenados y asignar los materiales a los pesos acordemente.
4. Renderizar un cuadro en formato ```png``` con los pesos y ejemplar cargados.

Se dejó comentada una línea para cambiar el fotograma activo.
El objetivo es que nuestros animadores puedan configurar cámaras, luces o cualquier otro elemento
que deseen usar en la animación en la forma tradicional y que con el código puedan desplazarse a
los fotogramas requeridos antes de renderizar cada cuadro.
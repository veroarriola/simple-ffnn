# simple-ffnn
**Proyecto PAPIME PE104425**

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

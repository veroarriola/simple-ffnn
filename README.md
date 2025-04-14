# simple-ffnn
**Proyecto PAPIME PE104425**

Código básico para entrenar una red ffnn y crear la animación en blender

## Instalación

* Instalar [Blender 4.4](https://www.blender.org/thanks/) descargando la versión en archivo comprimido y descomprimiendo.
* Clonar este repositorio dentro del directorio recién creado, al lado del ejecutable ```blender```.
* Instalar prerrequisitos desde la instalación local de ```python```:

```
cd 4.4/python/
./bin/python3.11 -m pip install --upgrade pip
./bin/python3.11 -m pip install torch torchvision torchaudio
./bin/python3.11 -m pip install pytorch-ignite
./bin/python3.11 -m pip install seaborn seaborn_image
```
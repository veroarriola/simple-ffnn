#print("In __init__", locals())
import importlib

# Forces Blender to reload modules everytime it executes the script.

if 'mnistTorch' in locals():
    print("Reloading", mnistTorch)
    importlib.reload(mnistTorch)
if 'blenderFNN' in locals():
    print("Reloading", blenderFNN)
    importlib.reload(blenderFNN)

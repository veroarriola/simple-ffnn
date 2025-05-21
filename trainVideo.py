import bpy
import os
import sys
from pathlib import Path

#
# Add network module to path
#
print("Current directory:", os.getcwd())
mnist_module_path = os.path.join(os.path.dirname(bpy.data.filepath), "simple-ffnn")
#mnist_module_path = os.path.dirname(bpy.data.filepath)
print("Adding to path:", mnist_module_path)
sys.path.insert(0, mnist_module_path)

#
# Loads the neural network visualization
#
import importlib
import bffnn
importlib.reload(bffnn)

from bffnn.mnistTorch import NetConfig, MNISTNet, MNISTDataSet
from bffnn.blenderFNN import MNISTFFNNViz

import numpy as np
import torch.nn as nn

#
# Load network and datasets
#
MODULE_DIR = mnist_module_path
DATA_DIR   = "nn-data"
MODEL_DIR  = "net_001"

net_config = NetConfig(MODULE_DIR, MODEL_DIR)
print(net_config)

net = MNISTNet(net_config['IMG_INPUT_SIZE'] * net_config['IMG_INPUT_SIZE'],
                   net_config['HIDDEN1_SIZE'],
                   net_config['HIDDEN2_SIZE'],
                   net_config['OUTPUT_SIZE'])

#
# Instantiate visualization class, it creates the geometry if necessary
#
nv = MNISTFFNNViz(net,
                  (net_config['IMG_INPUT_SIZE'], net_config['IMG_INPUT_SIZE']),
                  (net_config['HIDDEN1_SIZE'], 1),
                  (net_config['HIDDEN2_SIZE'], 1),
                  (net_config['OUTPUT_SIZE'], 1))
                  
FULL_GEOMETRY = True
if FULL_GEOMETRY:
    nv.add_bias()
    nv.add_weights()

    # Test setting colors
    nv.weight_0.set_uniform_color(128)
    nv.weight_1.set_uniform_color(50)
    nv.weight_2.set_uniform_color(200)
                  
LOAD_DATA = True
RENDER_FRAMES = False
RENDER_IMAGE_NAME = MODEL_DIR

if LOAD_DATA:
    # Load MNIST data
    data_set = MNISTDataSet(net_config.data_dir, net_config['BATCH_SIZE'])
    
    # Load weights as stored during training
    frame = 0
    for weights_file in net_config.files_of_weights():
        #print(weights_file)
        net.load(os.path.join(net_config.model_dir, weights_file))
        nv.update_params()
        nv.viz_number(data_set.trainset[0][0])
        if RENDER_FRAMES:
            # Use as needed:
            #bpy.context.scene.frame_current = 10
            bpy.context.scene.render.filepath = f"{RENDER_IMAGE_NAME}_{frame}.png"
            bpy.ops.render.render(write_still=True)
        break   #  Only one sample

#     
    

    
#     bpy.nv = nv
    
#     if SAVE_KEYPOINTS:        
#         FRAME_SKIP=5
#         n = 100
#         samples = np.random.choice(len(mnist_testset), n, replace=False)
#         #for i in range(n):
#         for i, index in enumerate(samples):
#             nv.viz_number(mnist_testset[index][0])
#             nv.save_activations_to_frame(i * FRAME_SKIP)
#             nv.save_activations_to_frame((i + 1) * FRAME_SKIP - 1)


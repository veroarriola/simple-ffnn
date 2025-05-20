import bpy
C = bpy.context
D = bpy.data

from mathutils import Vector
import numpy as np

import torch
import torch.nn as nn

LAYER0_Y_POS = -14
LAYER1_Y_POS = 14
LAYER2_Y_POS = 28
LAYER3_Y_POS = 42


INPUT_GRID_COLLECTION = "Number grid"
HIDDEN1_GRID_COLLECTION = "Hidden1 grid"
HIDDEN2_GRID_COLLECTION = "Hidden2 grid"
OUTPUT_GRID_COLLECTION = "Output grid"

W0_WEIGHTS_COLLECTION = "W_0"
W1_WEIGHTS_COLLECTION = "W_1"
W2_WEIGHTS_COLLECTION = "W_2"

B0_BIAS_COLLECTION = "B_0"
B1_BIAS_COLLECTION = "B_1"
B2_BIAS_COLLECTION = "B_2"

EMISSION_MATERIAL_ID = "Emi"
#DIFFUSE_GRAY_MATERIAL_ID = "ColorMat"

EMISSION_MATERIAL_BASE_NAME = "Emi"
EMISSION_MAX_VALUE = 100
DIFFUSE_GRAD_MATERIAL_BASE_NAME = "icefire"
FACTOR_BRILLO = 10


def recurLayerCollection(layerColl, collName):
    '''
    Return the layer collection to which the named collection belongs.
    https://blenderartists.org/t/how-do-i-set-a-collection-to-be-active/1445630
    '''
    found = None
    if (layerColl.name == collName):
        return layerColl
    for layer in layerColl.children:
        found = recurLayerCollection(layer, collName)
        if found:
            return found
        
def calc_rows(n, cols):
    '''
    Obtains the number of rows with cols columns needed to display
    n elements in a grid.
    '''
    rows = n // cols
    rows = rows if n % cols == 0 else rows + 1
    return rows

def create_obj_emission_material(obj, color, strength):
    '''
    color: (r, g, b, a)
    '''
    mat = D.materials.new(name=EMISSION_MATERIAL_ID)
    mat.use_nodes = True
    
    if mat.node_tree:
        mat.node_tree.links.clear()
        mat.node_tree.nodes.clear()

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    output = nodes.new(type='ShaderNodeOutputMaterial')
    
    shader = nodes.new(type='ShaderNodeEmission')
    nodes['Emission'].inputs['Color'].default_value = color
    nodes['Emission'].inputs['Strength'].default_value = strength
    
    links.new(shader.outputs[0], output.inputs[0])

    obj.data.materials.clear()
    obj.data.materials.append(mat)

def verify_emission_materials(color):
    '''
    Verifies that the gradient emission materials exist,
    otherwise it creates them.
    '''
    if not EMISSION_MATERIAL_BASE_NAME + str(EMISSION_MAX_VALUE//2) in D.materials:
        for i in range(EMISSION_MAX_VALUE):
            mat = D.materials.new(name=f"{EMISSION_MATERIAL_BASE_NAME}{i}")
            mat.use_nodes = True
            
            if mat.node_tree:
                mat.node_tree.links.clear()
                mat.node_tree.nodes.clear()

            nodes = mat.node_tree.nodes
            links = mat.node_tree.links

            output = nodes.new(type='ShaderNodeOutputMaterial')
            shader = nodes.new(type='ShaderNodeEmission')

            nodes['Emission'].inputs['Color'].default_value = color
            nodes['Emission'].inputs['Strength'].default_value = (5.0 * i / EMISSION_MAX_VALUE)
            
            links.new(shader.outputs[0], output.inputs[0])


# def create_diffuse_gray_material(obj, val):
#     '''
#     val: in [0,1.0]
#     '''
#     # Create a new material for the object
#     mat = D.materials.new(name=DIFFUSE_GRAY_MATERIAL_ID)
#     obj.data.materials.clear()
#     obj.data.materials.append(mat)

#     # Create a new color and set it on the material
#     color = (val, val, val, 1.0)
#     mat.diffuse_color = color

#     #    # Insert a keyframe for the material color every frame_num frames
#     #    frame = i * frame_num
#     #    mat.keyframe_insert(data_path="diffuse_color", frame=frame)
def verify_diffuse_materials():
    """
    Verifies that the gradient material exist, otherwise it
    creates them.
    """
    import seaborn as sns

    palette = sns.color_palette(DIFFUSE_GRAD_MATERIAL_BASE_NAME, as_cmap=True).reversed()

    for i, color in enumerate(palette.colors):
        mat_name = DIFFUSE_GRAD_MATERIAL_BASE_NAME + str(i)
        if not mat_name in D.materials:
            mat = D.materials.new(name=mat_name)
            color_a = (*color, 1.0)
            mat.diffuse_color = color_a
            print(mat, "created")

def create_curve_object(point_a, point_b, collection, material, thickness=0.01):
    '''
    Creates a line curve between points a and b
    and assignes the indicated material.
    Addapted from:
    https://github.com/DanieliusKr/neural-network-blender/blob/main/blender_script.py
    '''
    # Create a new curve
    curve_data = D.curves.new('Curve', 'CURVE')
    curve_data.dimensions = '3D'
    
    # Create a new spline
    spline = curve_data.splines.new('POLY')
    spline.points.add(1)
    spline.points[0].co = point_a
    spline.points[1].co = point_b
    
    # Set the thickness of the curve
    curve_data.bevel_depth = thickness
    
    # Create a new object and link it to the scene
    obj = D.objects.new('Curve', curve_data)
    collection.objects.link(obj)
    
    # Set the object to use the curve as its data
    obj.data = curve_data
    #create_diffuse_gray_material(obj, color)
    obj.data.materials.clear()
    obj.data.materials.append(material)
   
    return obj


class AbstractVisualizationComponent:
    '''
    Each network component places its visualization objects
    inside a collection.  This base class provides common functionality.
    '''
    def __init__(self, collection_name) -> None:
        '''
        Set the collection, if it does not exist calls for the overridable
        method _create_components(self, layer_collection).
        It also calls _init_material_array() to obtain references to
        component materials.
        '''
        self.collection_name = collection_name
        self.collection = None
        self.component_locations = None
        self._set_collection()
        self.material_array = self._init_material_array()

    def _set_collection(self):
        '''
        Returns the collection if it is already there, otherwise it
        creates it with _create_components(self, layer_collection).
        It also sets the component_locations
        '''
        if not self.collection_name in D.collections:
            self.collection = collection = D.collections.new(self.collection_name)
            C.scene.collection.children.link(collection)
            # TODO: Should layer_collection be an attribute?
            layer_collection = recurLayerCollection(C.view_layer.layer_collection, collection.name)
            self.component_locations = self._create_components(layer_collection)

            print(self.collection_name, "layer created")
        else:
            self.collection = collection = D.collections[self.collection_name]
            component_locations = []
            for c in self.collection.all_objects:
                c4 = c.location.to_4d()
                c4.w = 0
                component_locations.append(c4)
            self.component_locations = component_locations

            print(self.collection_name, "layer detected")
        return collection


class WeigthLayer(AbstractVisualizationComponent):
    '''
    Creates lines between positions of two layers.
    '''
    def __init__(self, input_locations, output_locations, collection_name) -> None:
        self.input_locations = input_locations
        self.output_locations = output_locations
        super().__init__(collection_name)

    def _create_components(self, layer_collection):
        '''
        Creates the lines to represent the weights
        '''
        collection = self.collection
        mat = D.materials[DIFFUSE_GRAD_MATERIAL_BASE_NAME + str(128)]
        for point_b in self.output_locations:
            for point_a in self.input_locations:
                create_curve_object(point_a, point_b, collection, mat)

    # TODO: Now materials must be reassigned, not modified when values change.  
    # def _init_material_array(self):
    #     '''
    #     Creates a local 2D array as direct access to the grids of
    #     materials of each curve within the collection of this component.
    #     First dimension corresponds to neurons at output layer,
    #     second dimension are neurons at input layer.
    #     '''
    #     num_input = len(self.input_locations)
    #     num_output = len(self.output_locations)
    #     grid_materials = np.empty((num_output, num_input), dtype=object)
    #     i = 0
    #     j = 0
    #     for curve in self.collection.all_objects:
    #         #print([mat for mat in curve.data.materials])
    #         #grid_materials[i,j] = curve.data.materials[DIFFUSE_GRAY_MATERIAL_ID].diffuse_color
    #         grid_materials[i,j] = curve.data.materials[0]
    #         j += 1
    #         if j == num_input:
    #             j = 0
    #             i += 1
    #     return grid_materials
    
    def set_uniform_color(self, value):
        '''
        Sets the same emission value to all materials in layer
        '''
        mat = D.materials[DIFFUSE_GRAD_MATERIAL_BASE_NAME + str(value)]
        for curve in self.collection.objects:
            curve.data.materials[0] = mat

    def viz_tensor(self, tensor, min_val, max_val):
        '''
        Visualizes a 2D view tensor representing the connections
        between two fully connected layers.
        First dimension corresponds to neurons at output layer,
        second dimension are neurons at input layer.
        min: minimum value for scale, must be less or equal to the smallest
        value in tensor
        max: maximum value for scale, must be greater than or equal to the
        biggest value in tensor
        '''
        curves = self.collection.objects
        k = 0
        val_range = max_val - min_val
        for row in tensor:
            for val in row:
                val = val.item()
                value = int((val - min_val) * 255 / val_range)
                #print(f"[{min_val} - {max_val}]", val, value)
                mat = D.materials[DIFFUSE_GRAD_MATERIAL_BASE_NAME + str(value)]
                curves[k].data.materials[0] = mat
                k += 1

    # def save_to_frame(self, frame_number, group):
    #     '''
    #     Adds keyframes to the curves material's
    #     '''
    #     success = False
    #     for i, cube in enumerate(self.collection.objects):
                # TODO: materials is a collection... there are no keyframe_insert there... how do I save this? Can not be done: instead assign and render.
    #         success = cube.data.materials.keyframe_insert(data_path='default_value', frame=frame_number, group=group)
    #     return success


class BiasLayer(AbstractVisualizationComponent):
    '''
    Creates a small box under each neuron to show the value of its
    bias.
    '''
    def __init__(self, neuron_locations, size, collection_name) -> None:
        self.neuron_locations = neuron_locations
        self.size = size
        super().__init__(collection_name)

    def _create_components(self, layer_collection):
        '''
        Creates the bias cubes below the neuron cubes.
        '''
        C.view_layer.active_layer_collection = layer_collection
        size = self.size

        offset = Vector((0, 0, 5 * size / 4)).to_4d()
        offset.w = 0
        material = D.materials["icefire128"]
        for location_vec in self.neuron_locations:
            cube_loc = location_vec-offset
            cube_loc = cube_loc.to_3d()
            print(location_vec, offset, cube_loc)
            bpy.ops.mesh.primitive_cube_add(location=cube_loc)
            cube = C.active_object
            cube.scale = (size, size, size / 4)
            cube.data.materials.clear()
            cube.data.materials.append(material)

    def viz_tensor(self, tensor, min_val, max_val):
        """
        Receives the 1D tensor of bias values for one layer.
        """
        cubes = self.collection.objects
        k = 0
        val_range = max_val - min_val
        for val in tensor:
            val = val.item()
            value = int((val - min_val) * 255 / val_range)
            mat = D.materials[DIFFUSE_GRAD_MATERIAL_BASE_NAME + str(value)]
            cubes[k].data.materials[0] = mat
            k += 1


class FullLayer2D(AbstractVisualizationComponent):
    '''
    Shows a fully connected layer neurons in 2D grid.
    '''
    def __init__(self, num_neurons, num_cols, collection_name, y_position, spacing=1, size=0.5) -> None:
        
        self.num_neurons = num_neurons
        self.num_cols = num_cols
        self.num_rows = calc_rows(num_neurons, num_cols)
        self.y_position = y_position
        self.spacing = spacing
        self.size = size

        super().__init__(collection_name)
    
    def _create_components(self, layer_collection):
        '''
        Positions cubes for a linear layer in 2D, so that the layer does node_tree
        look that wide.
        '''
        #print("Arg:", self.layer_collection, type(self.layer_collection))
        #print()
        C.view_layer.active_layer_collection = layer_collection

        cube_locations = []
        n = self.num_neurons
        n_x = self.num_cols
        n_z = self.num_rows
        y_position = self.y_position
        i = 0
        j = 0
        num = 0
        size = self.size
        size_space = self.spacing + size
        while num < n:
            x = j * size_space - (n_x - 1) * size_space / 2
            y = y_position
            z =  (n_z - 1) * size_space / 2 - i * size_space

            # Create a new cube
            bpy.ops.mesh.primitive_cube_add(location=(x, y, z))
            cube = C.active_object
            cube.scale = (size, size, size)

            cube_locations.append(Vector((x, y, z, 0)))
            #create_obj_emission_material(cube, (180, 150, 200, 0.9), 0.5)
            cube.data.materials.clear()
            cube.data.materials.append(D.materials[EMISSION_MATERIAL_BASE_NAME + str(EMISSION_MAX_VALUE//2)])
            num += 1
            j += 1
            if j == n_x:
                j = 0
                i += 1
        return cube_locations
    
    def _init_material_array(self):
        '''
        Creates a local 2D array as direct access to the grid
        materials of each cube within the given collection
        '''
        n_x = self.num_cols
        n_z = self.num_rows
        grid_materials = np.empty((n_z, n_x), dtype=object)
        i = 0
        j = 0
        for cube in self.collection.all_objects:
            grid_materials[i,j] = cube.data.materials[0].node_tree.nodes['Emission'].inputs['Strength']
            j += 1
            if j == n_x:
                j = 0
                i += 1
        return grid_materials
    
    def set_uniform_emission(self, value):
        '''
        Sets the same emission value to all materials in layer
        value: between 0 and 1.
        '''
        materials = np.reshape(self.material_array, -1)
        for i in range(len(materials)):
            materials[i].default_value = value
        for cube in self.collection.objects:
            cube.data.materials[0] = D.materials[EMISSION_MATERIAL_BASE_NAME + str(value)]
            #cube.data.materials[0].node_tree.nodes['Emission'].inputs['Strength'].default_value = value
    
    def viz_tensor(self, tensor):
        '''
        Visualizes a 1D view tensor with the same number of neurons.
        '''
        # materials = np.reshape(self.material_array, -1)
        # for i in range(len(materials)):
        #     materials[i].default_value = tensor[i] / FACTOR_BRILLO
        #for i, cube in enumerate(self.collection.objects):
        #    cube.data.materials[0].node_tree.nodes['Emission'].inputs['Strength'].default_value = tensor[i] / FACTOR_BRILLO
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        proportion = (EMISSION_MAX_VALUE - 1) / (max_val - min_val)
        for i, cube in enumerate(self.collection.objects):
            value = int((tensor[i] - min_val) * proportion)
            cube.data.materials[0] = D.materials[EMISSION_MATERIAL_BASE_NAME + str(value)]
            
    def save_to_frame(self, frame_number, group):
        '''
        Adds keyframes to the material's strength of all cubes in layer
        '''
        success = False
        for i, cube in enumerate(self.collection.objects):
            success = cube.data.materials[0].node_tree.nodes['Emission'].inputs['Strength'].keyframe_insert(data_path='default_value', frame=frame_number, group=group)
        return success


class MNISTFFNNViz:
    '''
    Network for visualization
    '''
    # Development inspired by:
    # https://www.youtube.com/watch?v=23k4okrrH7A&t=435s
    # Materials:
    # https://vividfax.github.io/2021/01/14/blender-materials.html
    def __init__(self, net,
                 input_grid_size, hidden1_grid_size,
                 hidden2_grid_size, output_grid_size):
        self.net = net
        input_size = net.fc1.in_features
        hidden1_size = net.fc1.out_features
        hidden2_size = net.fc2.out_features
        output_size = net.fco.out_features

        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size

        #C.scene.eevee.use_bloom = True  # Deprecated by Blender
        verify_emission_materials((180, 150, 200, 0.9))
        verify_diffuse_materials()

        # Layers
        self.input_layer = FullLayer2D(
            input_size,
            input_grid_size[0],
            INPUT_GRID_COLLECTION,
            LAYER0_Y_POS
        )

        self.hidden1_layer = FullLayer2D(
            hidden1_size,
            hidden1_grid_size[0],
            HIDDEN1_GRID_COLLECTION,
            LAYER1_Y_POS
        )

        self.hidden2_layer = FullLayer2D(
            hidden2_size,
            hidden2_grid_size[0],
            HIDDEN2_GRID_COLLECTION,
            LAYER2_Y_POS
        )
        
        self.output_layer = FullLayer2D(
            output_size,
            output_grid_size[0],
            OUTPUT_GRID_COLLECTION,
            LAYER3_Y_POS
        )

    def add_weights(self):
        """
        Weight edges
        """
        self.weight_0 = WeigthLayer(
            self.input_layer.component_locations,
            self.hidden1_layer.component_locations,
            W0_WEIGHTS_COLLECTION
        )
        self.weight_1 = WeigthLayer(
            self.hidden1_layer.component_locations,
            self.hidden2_layer.component_locations,
            W1_WEIGHTS_COLLECTION
        )
        self.weight_2 = WeigthLayer(
            self.hidden2_layer.component_locations,
            self.output_layer.component_locations,
            W2_WEIGHTS_COLLECTION
        )

    def add_bias(self):
        """
        Bias cubes
        """
        self.bias_0 = BiasLayer(
            self.hidden1_layer.component_locations,
            self.hidden1_layer.size,
            B0_BIAS_COLLECTION
        )
        self.bias_1 = BiasLayer(
            self.hidden2_layer.component_locations,
            self.hidden2_layer.size,
            B1_BIAS_COLLECTION
        )
        self.bias_2 = BiasLayer(
            self.output_layer.component_locations,
            self.output_layer.size,
            B2_BIAS_COLLECTION
        )

    def update_params(self):
        '''
        Must be called when the values of the network's weights have
        changed to update their visualization.
        '''
        b_0 = self.net.fc1.bias.data
        b_1 = self.net.fc2.bias.data
        b_2 = self.net.fco.bias.data

        w_0 = self.net.fc1.weight.data
        w_1 = self.net.fc2.weight.data
        w_2 = self.net.fco.weight.data
        
        #print("Mins", torch.min(w_0), torch.min(w_1), torch.min(w_2))
        #print("Maxs", torch.max(w_0), torch.max(w_1), torch.max(w_2))
        min_val = min(torch.min(b_0), torch.min(b_1), torch.min(b_2))
        max_val = max(torch.max(b_0), torch.max(b_1), torch.max(b_2))
        self.bias_0.viz_tensor(b_0, min_val, max_val)
        self.bias_1.viz_tensor(b_1, min_val, max_val)
        self.bias_2.viz_tensor(b_2, min_val, max_val)

        min_val = min(torch.min(w_0), torch.min(w_1), torch.min(w_2))
        max_val = max(torch.max(w_0), torch.max(w_1), torch.max(w_2))
        self.weight_0.viz_tensor(w_0, min_val, max_val)
        self.weight_1.viz_tensor(w_1, min_val, max_val)
        self.weight_2.viz_tensor(w_2, min_val, max_val)

    def viz_number(self, input_tensor):
        '''
        Updates input cubes to show the number in the tensor.
        '''
        net = self.net
        self.input_layer.viz_tensor(input_tensor.flatten())
        net.forward(input_tensor)
        self.hidden1_layer.viz_tensor(net.ac1.flatten())
        self.hidden2_layer.viz_tensor(net.ac2.flatten())
        self.output_layer.viz_tensor(net.aco.flatten())

    def save_activations_to_frame(self, frame_number):
        '''
        Save state of network neurons to frame.
        '''
        print("Inserting keyframes at ", frame_number)
        neurons_group = "Neurons"
        self.input_layer.save_to_frame(frame_number, neurons_group)
        self.hidden1_layer.save_to_frame(frame_number, neurons_group)
        self.hidden2_layer.save_to_frame(frame_number, neurons_group)
        success = self.output_layer.save_to_frame(frame_number, neurons_group)
        print(success)

    # def save_weights_to_frame(self, frame_number):
    #     params_group = "Params"
    #     self.weight_0.save_to_frame(frame_number, params_group)
    #     self.weight_1.save_to_frame(frame_number, params_group)
    #     success = self.weight_2.save_to_frame(frame_number, params_group)
    #     print(success)

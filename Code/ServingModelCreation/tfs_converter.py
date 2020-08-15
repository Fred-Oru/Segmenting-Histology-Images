import tensorflow as tf
import keras.backend as K
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.compat.v1 import saved_model
import os
import sys


class ExportConfig(object):
    # model name
    MODEL_NAME = "glomerulia_mrcnn"

    # path to the mask r-cnn source code
    MRCNN_DIR = os.path.abspath('../')

    # keras model path
    KERAS_MODEL_DIR = os.path.abspath('../logs/V2.3/')

    # keras model weights path
    KERAS_WEIGHTS_PATH = os.path.abspath('../logs/V2.3/mask_rcnn_glomerulus_0040.h5')

    # tf serving export dir
    EXPORT_DIR = os.path.abspath('../logs/TFS_models')

    # Version of the tf serving model
    VERSION_NUMBER = 1

    # Graph optimisation transforms
    TRANSFORMS = ["remove_nodes(op=Identity)", 
                 "merge_duplicate_nodes",
                 "strip_unused_nodes",
                 "fold_constants(ignore_errors=true)",
                 "fold_batch_norms",
                 # "quantize_nodes", 
                 # "quantize_weights"
                 ]

sys.path.append(ExportConfig.MRCNN_DIR)

from mrcnn.model import MaskRCNN
from glomerulus import GlomerulusInferenceConfig

# def get_model_config():
#     """Returns inference config
#     Ammend hyperparameters as necessary
#     """
#     class InferenceConfig(GlomerulusConfig):
#         GPU_COUNT = 1
#         IMAGES_PER_GPU = 1
#     return InferenceConfig()


def describe_graph(graph_def, show_nodes = False):
    """Displays the summary of the Keras/Tensorflow model graph.
    This is a diagnostic function.
    Graph is broken down by node types: Input, output, quantization, etc.
    graph_def: Tensorflow graph def of the model
    """
    print(f"Input Feature Nodes: {[node.name for node in graph_def.node if node.op == 'Placeholder']}")
    print(f"Unused Nodes: {[node.name for node in graph_def.node if 'unused' in node.name]}")
    print(f"Output Nodes: {[node.name for node in graph_def.node if ('predictions' in node.name or 'softmax' in node.name)]}")
    print(f"Quantization Node Count: {len([node.name for node in graph_def.node if 'quant' in node.name])}")
    print(f"Constant Count: {len([node for node in graph_def.node if node.op =='Const'])}")
    print(f"Variable Count: {len([node for node in graph_def.node if 'Variable' in node.op])}")
    print(f"Identity Count: {len([node for node in graph_def.node if node.op =='Identity'])}")
    print("", f"Total nodes: {len(graph_def.node)}", "")

    if show_nodes == True:
        for node in graph_def.node:
            print(f"Op:{node.op} - Name: {node.name}")


def get_model_size(export_dir, version, model_file = "saved_model.pb"):
    """Displays the size and variable count of the exported model.
    This is a diagnostic function.
    export_dir: Path to exported model
    version: Version number of the exported model
    model_file: Exported model file name
    """

    model_dir = os.path.join(export_dir, str(version))
    model_file_path = os.path.join(model_dir, model_file)
    print(model_file_path, '')
    pb_size = os.path.getsize(model_file_path)
    variables_size = 0

    if os.path.exists(
      os.path.join(model_dir,"variables/variables.data-00000-of-00001")):
        variables_size = os.path.getsize(os.path.join(
            model_dir,"variables/variables.data-00000-of-00001"))
        variables_size += os.path.getsize(os.path.join(
            model_dir,"variables/variables.index"))

    print(f"Model size: {round(pb_size / (1024.0),3)} KB")
    print(f"Variables size: {round( variables_size / (1024.0),3)} KB")
    print(f"Total Size: {round((pb_size + variables_size) / (1024.0),3)} KB")


def freeze_model(model, transforms = None, clear_devices = True):
    """Freezes the keras/tensorflow model into a frozen graph.
    model: Keras/Tensorflow model
    transforms: list of optional transforms to be applied ontu the graph
    clear_devices: Boolean flag to clear device data from graph nodes
    Returns:
    frozen_graph/optimized_graph: Frozen tensorflow graph
    """
    input_names = [input_tensor.op.name for input_tensor in model.inputs][:4]
    output_names = [out.op.name for out in model.outputs][:4]
    freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()))

    g = tf.compat.v1.get_default_graph()
    input_graph_def = g.as_graph_def()

    if clear_devices:
        for node in input_graph_def.node:
            node.device = ""

    frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
        master_session, input_graph_def, output_names, freeze_var_names)

    print("*" * 80)
    print("FROZEN GRAPH SUMMARY")
    describe_graph(frozen_graph)
    print("*" * 80)

    if transforms:
        optimized_graph = TransformGraph(frozen_graph, input_names, output_names, transforms)
        print("*" * 80)
        print("OPTIMIZED GRAPH SUMMARY")
        describe_graph(optimized_graph)
        print("*" * 80)
        return optimized_graph
    else:
        return frozen_graph


def export_saved_model(export_dir, version):
    """Exports the graph as a Tensorflow SavedModel format.
    export_dir: Path to exported model
    version: Version number of the exported model
    """
    export_dir = os.path.join(export_dir, str(version))
    builder = saved_model.builder.SavedModelBuilder(export_dir)
    signature = {}

    g = tf.compat.v1.get_default_graph()

    input_image = saved_model.build_tensor_info(g.get_tensor_by_name("input_image:0"))
    input_image_meta = saved_model.build_tensor_info(g.get_tensor_by_name("input_image_meta:0"))
    input_anchors = saved_model.build_tensor_info(g.get_tensor_by_name("input_anchors:0"))

    output_detection = saved_model.build_tensor_info(g.get_tensor_by_name("mrcnn_detection/Reshape_1:0"))
    output_mask = saved_model.build_tensor_info(g.get_tensor_by_name("mrcnn_mask/Reshape_1:0"))

    signature[saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
    saved_model.signature_def_utils.build_signature_def(
        inputs = {"input_image": input_image, "input_image_meta": input_image_meta, "input_anchors": input_anchors},
        outputs = {"mrcnn_detection/Reshape_1": output_detection, "mrcnn_mask/Reshape_1": output_mask},
        method_name = saved_model.signature_constants.PREDICT_METHOD_NAME)

    builder.add_meta_graph_and_variables(export_session,
        [saved_model.tag_constants.SERVING],
        signature_def_map = signature)
    builder.save()


if __name__ == '__main__':
    # Get model config
#     model_config = get_model_config()
    model_config = GlomerulusInferenceConfig()
    export_dir = os.path.join(ExportConfig.EXPORT_DIR, ExportConfig.MODEL_NAME)
    if not os.path.exists(export_dir):
        os.mkdir(export_dir)
        
    # Load maask rcnn keras model and the pretrained weights
    model = MaskRCNN(mode = "inference", model_dir = ExportConfig.KERAS_MODEL_DIR, config = model_config)
    model.load_weights(ExportConfig.KERAS_WEIGHTS_PATH, by_name = True)

    with K.get_session() as master_session:
        graph_def = freeze_model(model.keras_model, transforms = ExportConfig.TRANSFORMS)

        with tf.Session(graph = tf.Graph()) as export_session:
            tf.import_graph_def(graph_def, name = "")
            export_saved_model(export_dir, ExportConfig.VERSION_NUMBER)

    # Print the size of the tf-serving model
    print("*" * 80)
    get_model_size(export_dir, ExportConfig.VERSION_NUMBER)
    print("*" * 80)
    print("COMPLETED")

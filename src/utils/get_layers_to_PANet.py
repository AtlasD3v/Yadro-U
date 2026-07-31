import torch
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

shapes = {}
hooks = []

input_H = None
input_W = None

PARAMS = [8, 16, 32]

def get_shape_hook(module_name):
    def hook(module, input, output):
        # [shapes[module_name] = output.shape[-2:] for num in PARAMS if (input_h // num == output.shape[-2]) and (input_W // num == output.shape[-1])]
        if not isinstance(output, torch.Tensor):
            return   # SE-блок и подобные иногда возвращают не то, что ожидаем - подстрахуемся
        
        for num in PARAMS:
            if output.shape[-2] == (input_H // num) and output.shape[-1] == (input_W // num):
                shapes[output.shape[-2:]] = module_name
    return hook

def get_needed_layers(model: torch.nn.Module):
    global input_H, input_W

    dummy_input = torch.randn((1, 3, 640, 640))
    input_size = dummy_input.shape[-2:]
    input_H = input_size[0]
    input_W = input_size[1]



    
    for m_name, module in model.features.named_children():
       full_name = f"features.{m_name}"
       hook_handle = module.register_forward_hook(get_shape_hook(module_name= full_name))
       hooks.append(hook_handle)
    
    print("--- СТАРТУЕМ FORWARD PASS ---")
    output = model(dummy_input)
    print("--- КОНЕЦ FORWARD PASS ---")

    _ = [hook.remove() for hook in hooks]

    print(shapes)

def _get_graph_node_names(model: torch.nn.Module):
    train_nodes, eval_nodes = get_graph_node_names(model)
    print(eval_nodes)
    print()
    print(train_nodes)

weights = torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
model = torchvision.models.mobilenet_v3_large(weights = weights)

get_needed_layers(model)
# _get_graph_node_names(model)
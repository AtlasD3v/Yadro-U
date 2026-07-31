from collections import defaultdict
import torch

# Гиперпараметры
SIZE_H = 640
SIZE_W = 640
PARTS = 8  # Количество частей для разбиения


def start_dummy_model(model: torch.nn.Module, dummy_input: torch.Tensor, hooks_arr: list, stage_name: str):
    """Запускает forward pass для сбора информации с хуков и затем удаляет их."""
    print(f"--- СТАРТУЕМ FORWARD PASS: {stage_name} ---")
    model.eval()
    
    # Подготовка входа для Faster R-CNN (ожидает список тензоров на инференсе)
    input_data = [dummy_input[0]] if dummy_input.ndim == 4 else dummy_input
    
    with torch.inference_mode():
        _ = model(input_data)
        
    print(f"--- КОНЕЦ FORWARD PASS: {stage_name} ---\n")

    # Удаляем хуки после прогона
    for hook in hooks_arr:
        hook.remove()


# --- ХУКИ ---

def build_min_size_hook(state: dict):
    """Хук для поиска минимального разрешения карт признаков."""
    def hook(module, input, output):
        outputs = output.values() if isinstance(output, dict) else [output]
        for out in outputs:
            if isinstance(out, torch.Tensor) and out.ndim >= 2:
                state["min_h"] = min(state["min_h"], out.shape[-2])
                state["min_w"] = min(state["min_w"], out.shape[-1])
    return hook


def build_stride_threshold_hook(module_name: str, thr_h: list, thr_w: list, result_dict: dict):
    """Хук для Группировки слоев по диапазонам порогов (Режим 1)."""
    def hook(module, input, output):
        outputs = output.values() if isinstance(output, dict) else [output]
        for out in outputs:
            if isinstance(out, torch.Tensor) and out.ndim >= 2:
                h_out, w_out = out.shape[-2], out.shape[-1]
                
                for i in range(len(thr_h) - 1):
                    if (thr_h[i] <= h_out < thr_h[i + 1]) and (thr_w[i] <= w_out < thr_w[i + 1]):
                        key = ((thr_h[i], thr_w[i]), (thr_h[i + 1], thr_w[i + 1]))
                        if module_name not in result_dict[key]:
                            result_dict[key].append(module_name)
    return hook


def build_all_sizes_hook(sizes_set: set):
    """Хук для сбора всех уникальных квадратных размеров карт признаков (Режим 2)."""
    def hook(module, input, output):
        outputs = output.values() if isinstance(output, dict) else [output]
        for out in outputs:
            if isinstance(out, torch.Tensor) and out.ndim >= 2:
                h_out, w_out = out.shape[-2], out.shape[-1]
                if h_out == w_out:
                    sizes_set.add(h_out)
    return hook


def build_select_by_size_hook(module_name: str, target_sizes: set, result_dict: dict):
    """Хук для группировки слоев по точным размерам (Режим 2)."""
    def hook(module, input, output):
        outputs = output.values() if isinstance(output, dict) else [output]
        for out in outputs:
            if isinstance(out, torch.Tensor) and out.ndim >= 2:
                h_out = out.shape[-2]
                if h_out in target_sizes:
                    key = str(h_out)
                    if module_name not in result_dict[key]:
                        result_dict[key].append(module_name)
    return hook


# --- РЕЖИМЫ РАБОТЫ ---

def run_mode_1(model: torch.nn.Module, backbone_block: torch.nn.Module, backbone_name: str, dummy_input: torch.Tensor, h: int, w: int):
    layers_dict = defaultdict(list)
    state = {"min_h": h, "min_w": w}

    # Шаг 1: Определение минимального размера
    size_hooks = [
        b_module.register_forward_hook(build_min_size_hook(state))
        for _, b_module in backbone_block.named_children()
    ]
    start_dummy_model(model, dummy_input, size_hooks, "ОПРЕДЕЛЕНИЕ MIN_SIZE")

    min_h, min_w = state["min_h"], state["min_w"]

    # Шаг 2: Расчет порогов
    step_h = (h - min_h) // PARTS
    step_w = (w - min_w) // PARTS
    thresholds_h = [h - (step_h * x) for x in range(PARTS, -1, -1)]
    thresholds_w = [w - (step_w * x) for x in range(PARTS, -1, -1)]

    print(f"Пороги H: {thresholds_h}\nПороги W: {thresholds_w}")

    # Шаг 3: Группировка
    stride_hooks = [
        b_block.register_forward_hook(
            build_stride_threshold_hook(f"{backbone_name}.{b_name}", thresholds_h, thresholds_w, layers_dict)
        )
        for b_name, b_block in backbone_block.named_children()
    ]
    start_dummy_model(model, dummy_input, stride_hooks, "ГРУППИРОВКА СЛОЕВ (РЕЖИМ 1)")

    return layers_dict


def run_mode_2(model: torch.nn.Module, backbone_block: torch.nn.Module, backbone_name: str, dummy_input: torch.Tensor):
    layers_dict = defaultdict(list)
    found_sizes = set()

    # Шаг 1: Сбор всех уникальных размеров
    size_hooks = [
        b_module.register_forward_hook(build_all_sizes_hook(found_sizes))
        for _, b_module in backbone_block.named_children()
    ]
    start_dummy_model(model, dummy_input, size_hooks, "СБОР ВСЕХ РАЗМЕРОВ")

    # Шаг 2: Группировка по найденным размерам
    stride_hooks = [
        b_block.register_forward_hook(
            build_select_by_size_hook(f"{backbone_name}.{b_name}", found_sizes, layers_dict)
        )
        for b_name, b_block in backbone_block.named_children()
    ]
    start_dummy_model(model, dummy_input, stride_hooks, "ГРУППИРОВКА СЛОЕВ (РЕЖИМ 2)")

    return layers_dict


# --- ГЛАВНАЯ ФУНКЦИЯ ---

def main_func(model: torch.nn.Module, backbone_block: torch.nn.Module, backbone_name: str, mode_number: int):
    dummy_input = torch.randn((1, 3, SIZE_H, SIZE_W))
    h, w = dummy_input.shape[-2:]
    print(f"Размер входа: ({h}, {w})")

    if mode_number == 1:
        final_dict = run_mode_1(model, backbone_block, backbone_name, dummy_input, h, w)
    elif mode_number == 2:
        final_dict = run_mode_2(model, backbone_block, backbone_name, dummy_input)
    else:
        raise ValueError(f"Неизвестный режим: {mode_number}. Доступны только 1 и 2.")


    all_names = {name for name, _ in backbone_block.named_children()}
    covered_names = {name.split('.')[-1] for names_list in final_dict.values() for name in names_list}
    missing = all_names - covered_names
    assert not missing, f"Эти блоки не попали ни в одну группу: {missing}"

    model.train()
    return dict(final_dict)
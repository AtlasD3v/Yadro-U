import torch
from src.models.custom_layers_blocks import my_blocks

CHANNELS_OUT = 256

class Neck(torch.nn.Module):
    def __init__(self, in_channels_list, out_channels, num_extra_levels = 0): #in_channels_list - список глубин поступивших карт признаков
        super().__init__()
        self.in_ch_list = in_channels_list
        self.out_ch = out_channels
        self.num_extra_levels = num_extra_levels
        
        # 1. Приведение каналов к единому стандарту (1x1 свёртки)
        self.standartizate_blocks = torch.nn.ModuleList(
            [my_blocks.Standartizate_block(ch_in, self.out_ch) for ch_in in self.in_ch_list]
        )#слои, которые приводят все входящие тензоры карт признаков к единой глубине out_channels свёртками 1x1 


        # 2. Анти-артефактные свёртки для Top-Down пути (FPN)
        # Для 3-х входных слоев нам нужно 2 такие свёртки (для P4 и P3)
        self.anti_artefact_convs1 = torch.nn.ModuleList(
            [my_blocks.Anti_artefact_blocks(in_channels=self.out_ch, out_channels=self.out_ch) for _ in range(len(self.in_ch_list) - 1)]
        )

        # 3. Свёртки сжатия (Downsample 3x3, stride=2) для Bottom-Up пути (PANet)
        self.down_top_layer = torch.nn.ModuleList(
            [my_blocks.Down_top_layer(in_channels=self.out_ch, out_channels=self.out_ch) for _ in range(len(self.in_ch_list) - 1)]
        )
        
        # 4. Финальные свёртки после слияния на Bottom-Up пути
        self.anti_artefact_convs2 = torch.nn.ModuleList(
            [my_blocks.Anti_artefact_blocks(in_channels=self.out_ch, out_channels=self.out_ch) for _ in range(len(self.in_ch_list) - 1)]
        )

        #добавляем свёртки, которые будут сужать N5, N6, чтобы получилось N3, N4, N5, N6, N7 (в карте признаков N7 одна ячейка будет отвечать за 128 пикселей) - дабы захватить больший receptive field
        self.extra_features_extractor_level = None
        self.is_extra_levels = False

        if self.num_extra_levels > 0:
            self.is_extra_levels = True

            self.extra_features_extractor_level = torch.nn.ModuleList(
                [my_blocks.Extra_level_extractor_layer(in_channels=self.out_ch, out_channels=self.out_ch) for _ in range(self.num_extra_levels)]
            )

    def forward(self, inputs):
        # inputs — это список тензоров из бэкбона: [C3, C4, C5]
        # Размеры: C3(N x N), C4(N//2 x N//2), C5(N//4 x N//4)
        
        #Выравниваем каналы у всех тензоров до out_channels (256)
        # Теперь в проекциях строго 256 каналов

        feature_maps = [self.standartizate_blocks[i](inputs[i]) for i in range(len(inputs))] #приводим все карты признаков к одинаковой глубине

        #реализация top-down прохода: (так как карты признаков идут в порядке убывания, то начинаем с конца массива, так как нам надо начинать с самой маленькой карты признаков)

        for i in range(len(feature_maps) - 1, 0, -1):
            p = torch.nn.functional.interpolate(input=feature_maps[i], scale_factor=2, mode='nearest') #апсемлим
            fused = feature_maps[i - 1] + p #складываем более высокоуровневную карту признаков с интерполированной низкоуровневой картой
            feature_maps[i - 1] = self.anti_artefact_convs1[i - 1](fused) #применяем анти-артефактные свёртки к получившемуся результату после сложения
        
        #реализация down-top прохода
        for i in range(len(feature_maps) - 1):
            n_past = self.down_top_layer[i](feature_maps[i])
            n_fused = feature_maps[i + 1] + n_past
            feature_maps[i + 1] = self.anti_artefact_convs2[i](n_fused)
        
        if self.is_extra_levels:
            for i in range(self.num_extra_levels):
                extra_N = self.extra_features_extractor_level[i](feature_maps[-1])#каждый раз берётся последний элемент массива, то есть, когда массив features_map 
                #только приходит в этот цикл, последним значением является N5, мы применяем к нему сжатие, групповую нормализацию и SiLu.

                #В следующей итерации цикла на месте features_map[-1] будет лежать уже новый N_i, полученный в результате применения self.extra_features_extractor_level[i] к N_i-1
                feature_maps.append(extra_N) #добавляем extra_N в общий массив карт признаков, по которым будет делаться предикт (на нулевой итерации - это N6)

        return feature_maps

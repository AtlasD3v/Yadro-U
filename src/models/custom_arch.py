import torch
import torchvision
from torchvision import models as cv_models
from src.models.neck import PANet_neck as neck
from src.models.custom_layers_blocks import my_blocks

weights = cv_models.MobileNet_V3_Large_Weights.DEFAULT
model = cv_models.mobilenet_v3_large(weights = weights)



class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass



class AnchorFreeDetecionHead(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hyper_param_tower_len, num_classes):
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.h_p_t_l = hyper_param_tower_len
        self.num_classes = num_classes

        self.cls_tower = torch.nn.Sequential(
            *[
                my_blocks.Head_towers_blocks(in_channels=self.in_ch, out_channels=self.out_ch) if x == 0 else my_blocks.Head_towers_blocks(in_channels=self.out_ch, out_channels=self.out_ch) for x in range(self.h_p_t_l)
            ]
        )

        self.reg_tower = torch.nn.Sequential(
            *[
                my_blocks.Head_towers_blocks(in_channels=self.in_ch, out_channels=self.out_ch) if x == 0 else my_blocks.Head_towers_blocks(in_channels=self.out_ch, out_channels=self.out_ch) for x in range(self.h_p_t_l)
            ]
        ) 

        self.cls_pred = torch.nn.Conv2d(in_channels=self.out_ch, out_channels=self.num_classes, kernel_size=3, stride=1, padding=1)
        self.reg_pred = torch.nn.Conv2d(in_channels=self.out_ch, out_channels=4, kernel_size=3, stride=1, padding=1)


    def forward(self, input):
        cls = self.cls_pred(self.cls_tower(input))
        reg = self.reg_pred(self.reg_tower(input))

        return cls, reg
    

class MultiLevelHead(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, num_of_neck_levels): #num_of_neck_levels - количество возвращаемых карт признаков из "шеи"
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.hyper_param_tower_len = 4
        self.num_classes = num_classes
        self.num_of_neck_levels = num_of_neck_levels

        self.detection_head = AnchorFreeDetecionHead(
            in_channels=self.in_ch, 
            out_channels=self.out_ch, 
            hyper_param_tower_len=self.hyper_param_tower_len, 
            num_classes=self.num_classes
        )

        self.heads_scalers = torch.nn.ModuleList(
            [my_blocks.ScaleExp(1.0) for _ in range(self.num_of_neck_levels)]
        )

    def forward(self, inputs):
        cls_results, reg_results = [], []

        for input, scale in zip(inputs, self.heads_scalers):
            result_cls, result_reg = self.detection_head(input)

            total_reg_result = torch.nn.functional.relu(scale(result_reg)) #сначала компенсируем масштаб для каждого из уровней пирамиды своим scale'ом,
            #потом применяем к выходу регрессии relu (так как расстояния не могут быть отрицательными), 
            
            cls_results.append(result_cls)
            reg_results.append(total_reg_result)

        return cls_results, reg_results 
   










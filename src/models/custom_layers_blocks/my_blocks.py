import torch


class Depth_point_wise_convs(torch.nn.Module):
    def __init__(self, channels_in, channels_out, d_kernel_size = 3, d_stride = 1, d_padding = 1):
        super().__init__()
        self.in_ch = channels_in
        self.out_ch = channels_out

        self.d_kernel_size = d_kernel_size
        self.d_stride = d_stride
        self.d_padding = d_padding

        self.depthwise_conv = torch.nn.Conv2d(
            in_channels=self.in_ch,
            out_channels= self.in_ch, 
            kernel_size=self.d_kernel_size, 
            stride=self.d_stride,
            padding= self.d_padding, 
            groups=self.in_ch, 
            bias=False
        )
        self.pointwise_conv = torch.nn.Conv2d(
            in_channels=self.in_ch, 
            out_channels = self.out_ch, 
            kernel_size=1, 
            stride=1, 
            bias=False
        )

    def forward(self, input):
        return self.pointwise_conv(self.depthwise_conv(input))

class Standartizate_block(torch.nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.in_ch = channels_in
        self.out_ch = channels_out

        self.conv = torch.nn.Conv2d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=1, stride=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(num_features=self.out_ch)

    def forward(self, input):
        return self.bn(self.conv(input))
    
class Anti_artefact_blocks(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.conv = torch.nn.Conv2d(in_channels= self.in_ch, out_channels= self.out_ch, kernel_size=3, stride = 1, padding=1,bias=False)
        self.bn_anti_art = torch.nn.BatchNorm2d(num_features=self.out_ch)
        self.SiLu = torch.nn.SiLU(inplace=True)

    def forward(self, input):
        return self.SiLu(self.bn_anti_art(self.conv(input)))   
        
class Down_top_layer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_ch = in_channels
        self.out_ch = out_channels

        self.depthwise = Depth_point_wise_convs(
            channels_in = self.in_ch, channels_out = self.out_ch, d_kernel_size=3, d_stride = 2, d_padding=1
        )
    
    def forward(self, input):
        return self.depthwise(input)
    
class Extra_level_extractor_layer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_ch = in_channels
        self.out_ch = out_channels

        self.conv_to_extra = torch.nn.Conv2d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_to_extra = torch.nn.BatchNorm2d(num_features=self.out_ch)
        self.SiLu_extra = torch.nn.SiLU(inplace=True)
    
    def forward(self, input):
        return self.SiLu_extra(self.bn_to_extra(self.conv_to_extra(input)))


class Head_towers_blocks(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels

        self.conv_head = torch.nn.Conv2d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn_head = torch.nn.GroupNorm(num_groups=16, num_channels=self.out_ch)
        self.SiLu_head = torch.nn.SiLU(inplace=True)

    def forward(self, input):
        return self.SiLu_head(self.gn_head(self.conv_head(input)))

class ScaleExp(torch.nn.Module):
    def __init__(self, init_value = 1.0):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, input):
        return input * self.scale
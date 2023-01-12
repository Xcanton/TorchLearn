# Convolution Layers

|参数|参数类型|||
|:----:|:----:|:----|:----|
|`in_channels`|int|Number of channels in the input image|	输入图像通道数|
|`out_channels`|int|Number of channels produced by the convolution|卷积产生的通道数|
|`kernel_size`|(int or tuple)|Size of the convolving kernel|卷积核尺寸，可以设为1个int型数或者一个(int, int)型的元组。例如(2,3)是高2宽3卷积核|
|`stride`|(int or tuple, optional)|Stride of the convolution. Default: 1|卷积步长，默认为1。可以设为1个int型数或者一个(int, int)型的元组。|
|`padding`|(int or tuple, optional)|Zero-padding added to both sides of the input. Default: 0|填充操作，控制padding_mode的数目。|
|`padding_mode`|(string, optional)|'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'|padding模式，默认为Zero-padding 。|
|`dilation`|(int or tuple, optional)|Spacing between kernel elements. Default: 1|	扩张操作：控制kernel点（卷积核点）的间距，默认值:1。|
|`groups`|(int, optional)|Number of blocked connections from input channels to output channels. Default: 1|group参数的作用是控制分组卷积，默认不分组，为1组。|
|`bias`|(bool, optional)|If True, adds a learnable bias to the output. Default: True|为真，则在输出中添加一个可学习的偏差。默认：True。|

# Reference
[1] [https://pytorch.org/docs/1.7.0/generated/torch.nn.Conv1d.html?highlight=conv#torch.nn.Conv1d](https://pytorch.org/docs/1.7.0/generated/torch.nn.Conv1d.html?highlight=conv#torch.nn.Conv1d)
[2] [https://pytorch.org/docs/1.7.0/generated/torch.nn.Conv1d.html?highlight=conv#torch.nn.Conv2d](https://pytorch.org/docs/1.7.0/generated/torch.nn.Conv1d.html?highlight=conv#torch.nn.Conv2d)
[3] [https://pytorch.org/docs/1.7.0/generated/torch.nn.Conv1d.html?highlight=conv#torch.nn.Conv3d](https://pytorch.org/docs/1.7.0/generated/torch.nn.Conv1d.html?highlight=conv#torch.nn.Conv3d)
[4] [https://blog.csdn.net/qq_34243930/article/details/107231539](https://blog.csdn.net/qq_34243930/article/details/107231539)
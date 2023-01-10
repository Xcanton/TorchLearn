# Base Module Class in PyTorch

  &ensp;&ensp;&ensp;&ensp;[<b>torch.nn.Module</b>](https://pytorch.org/docs/1.7.0/_modules/torch/nn/modules/module.html#Module) is the base class for all layers, modules and neural networks. By using this, self created modules can be easier and lighter. Official example is as follow.

  ```
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
  ```
To implement a self-made module, There are <b>three parallel steps</b> that should be done:
- Inheritance base class torch.nn.Module
- Assign your submodule as regular attributes
- Implement <b>_\_\_init\_\__</b> function and <b>_forward_</b> founction

Corresponding to parts in official example, these three parallel steps represent the following code separately.

<b>Inheritance base class torch.nn.Module</b>
```
class Model(nn.Module):
```
<b>Assign your submodule as regular attributes</b>
```
super(Model, self).__init__()
```
<b>Implement <b>_\_\_init\_\__</b> function and <b>_forward_</b> founction</b>
```
def __init__(self):
def forward(self, x):
```

# Design your own layer


# Reference
[1] [https://pytorch.org/docs/1.7.0/generated/torch.nn.Module.html#torch.nn.Module](https://pytorch.org/docs/1.7.0/generated/torch.nn.Module.html#torch.nn.Module)
[2] [https://blog.csdn.net/qq_27825451/article/details/90705328?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-90705328-blog-109994420.pc_relevant_3mothn_strategy_recovery&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-90705328-blog-109994420.pc_relevant_3mothn_strategy_recovery&utm_relevant_index=1](https://blog.csdn.net/qq_27825451/article/details/90705328?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-90705328-blog-109994420.pc_relevant_3mothn_strategy_recovery&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-90705328-blog-109994420.pc_relevant_3mothn_strategy_recovery&utm_relevant_index=1)
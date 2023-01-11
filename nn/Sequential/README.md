# Sequential

  &ensp;&ensp;&ensp;&ensp;[<b>torch.nn.Sequential</b>](https://pytorch.org/docs/1.7.0/_modules/torch/nn/modules/container.html#Sequential) is a pipeline container, which connects layers sequentially. </br>  &ensp;&ensp;&ensp;&ensp;Acconding to the official examples, there are 2 ways to use [<b>torch.nn.Sequential</b>](https://pytorch.org/docs/1.7.0/_modules/torch/nn/modules/container.html#Sequential). However, if we take _Object-oriented Programming form_ into consideration, there are at least 3 ways to implement this container:
  - Sequential by Index
  - Sequential by Ordered Dict
  - Sequential by Object


<b>Sequential by Index</b>
```
# Example of using Sequential
model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )
```

  &ensp;&ensp;&ensp;&ensp;_Sequential by Index_ is able to implement the designed network, but it has its limits. When using this, layers cannot be named manually. The default name of each layer is its index, which starts from 0 and is in order of its depth.
```
Sequential(
    (0): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
    (1): ReLU()
    (2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
    (3): ReLU()
)
```

<b>Sequential by Ordered Dict</b>
```
from collections import OrderedDict

# Example of using Sequential with OrderedDict
model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ]))
```

  &ensp;&ensp;&ensp;&ensp;_Sequential by Ordered Dict_ takes the advantage of _OrderedDict_, which is a advanced data structure in default python package _collections_. By doing so, layers can be named whatever you want as well as retain original functions. </br>
```
Sequential(
    (conv1): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
    (relu1): ReLU()
    (conv2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
    (relu2): ReLU()
)
```
\
The above two are quit <b>_Pythonic_</b>  .

<b>Sequential by Object</b>
```
from collections import OrderedDict

# Example of using Sequential with Object
model = nn.Sequential()
model.add_module("conv1", nn.Conv2d(1,20,5))
model.add_module("relu1", nn.ReLU())
model.add_module("conv2", nn.Conv2d(20,64,5))
model.add_module("relu2", nn.ReLU())
```
  &ensp;&ensp;&ensp;&ensp;_Example of using Sequential with Object_ looks like _<b>Keras</b> style_. By using _add_module_ function, layers can be appended to the network module with its name. This function is inherited from [_torch.nn.Module_](https://github.com/Xcanton/TorchLearn/tree/master/nn/Module), which is used for layer construction. </br>

# Reference
[1] [https://pytorch.org/docs/1.7.0/generated/torch.nn.Sequential.html?highlight=sequential#torch.nn.Sequential](https://pytorch.org/docs/1.7.0/generated/torch.nn.Sequential.html?highlight=sequential#torch.nn.Sequential)
[2] [https://blog.csdn.net/qq_27825451/article/details/90550890](https://blog.csdn.net/qq_27825451/article/details/90550890)
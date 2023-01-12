# Special Tensor Form
    
  &ensp;&ensp;&ensp;&ensp;Some special forms of tensor can be created by official functions offerred by torch. These functions are in direct subfolders of <b>_[torch](https://pytorch.org/docs/1.7.0/torch.html#torch)_</b>. The only reason why I plant them here is that, it belongs to _<b>[tensor definision](https://github.com/Xcanton/TorchLearn/tree/master/Tensor)</b> due to their results_. </br>&ensp;&ensp;&ensp;&ensp;The most common form is a matrix fill with the same value. Two of them are the zero matrix and the one matrix. Torch offers official functions that creates or fill matrix with 0 or 1. The details are expanded as follow.

## ones
  &ensp;&ensp;&ensp;&ensp;<b>_[torch.ones](https://pytorch.org/docs/1.7.0/generated/torch.ones.html#torch-ones)_</b> is capable of creating a one matrix. It returns a tensor filled with the scalar value 1, with the shape defined by the variable argument size. 
  - Parameters
    - size 
    - out (optional)
    - dtype (optional)
    - layout (optional)
    - device (optional)
    - requires_grad (False: optional)

  &ensp;&ensp;&ensp;&ensp;From the parameter requirements we can see that, the size of tensor must be defined when using this function. Examples are shown as follow.
  ```
>>> torch.ones(2, 3)
tensor([[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]])

>>> torch.ones(5)
tensor([ 1.,  1.,  1.,  1.,  1.])
  ```

## ones_like
  &ensp;&ensp;&ensp;&ensp;Similar to the <b>_[torch.ones](https://pytorch.org/docs/1.7.0/generated/torch.ones.html#torch-ones)_</b> function, <b>_[torch.ones_like](https://pytorch.org/docs/1.7.0/generated/torch.ones_like.html#torch-ones-like)_</b> also returns a tensor filled with the scalar value 1, but there is a slight difference. The size of the tensor is obtained from a given tensor, which is not optional when using this function. 
  - Parameters
    - input (Tensor)  
    - dtype (optional)
    - layout (optional)
    - device (optional)
    - requires_grad (False: optional)
    - memory_format (optional)

  &ensp;&ensp;&ensp;&ensp;Example is shown as follow.
  ```
>>> input = torch.empty(2, 3)
>>> torch.ones_like(input)
tensor([[ 1.,  1.,  1.],
        [ 1.,  1.,  1.]])
  ```

## zeros
  &ensp;&ensp;&ensp;&ensp;<b>_[torch.zeros](https://pytorch.org/docs/1.7.0/generated/torch.zeros.html#torch-zeros)_</b> and <b>_[torch.ones](https://pytorch.org/docs/1.7.0/generated/torch.ones.html#torch-ones)_</b> are exactly the same, except for one difference. Instead of filling the matrix with 1, <b>_[torch.zeros](https://pytorch.org/docs/1.7.0/generated/torch.zeros.html#torch-zeros)_</b> returns a tensor filled with the scalar value 0, with the shape defined by the variable argument size. Examples are shown as follow.
  ```
>>> torch.zeros(2, 3)
tensor([[ 0.,  0.,  0.],
        [ 0.,  0.,  0.]])

>>> torch.zeros(5)
tensor([ 0.,  0.,  0.,  0.,  0.])
  ```

## zeros_like
  &ensp;&ensp;&ensp;&ensp;<b>_[torch.zeros_like](https://pytorch.org/docs/1.7.0/generated/torch.zeros_like.html#torch-zeros-like)_</b> and <b>_[torch.ones_like](https://pytorch.org/docs/1.7.0/generated/torch.ones_like.html#torch-ones-like)_</b> are exactly the same, except for one difference. Instead of filling the matrix with 1, <b>_[torch.zeros_like](https://pytorch.org/docs/1.7.0/generated/torch.zeros_like.html#torch-zeros-like)_</b> returns  a tensor filled with the scalar value 0, with the same size as input. Example is shown as follow.
  ```
>>> input = torch.empty(2, 3)
>>> torch.zeros_like(input)
tensor([[ 0.,  0.,  0.],
        [ 0.,  0.,  0.]])
  ```


# Reference
[1] [https://pytorch.org/docs/1.7.0/torch.html](https://pytorch.org/docs/1.7.0/torch.html)

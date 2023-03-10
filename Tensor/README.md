# Tensor
  &ensp;&ensp;&ensp;&ensp;<b>_[torch.Tensor](https://pytorch.org/docs/1.7.0/tensors.html#torch-tensor)_</b> is the basic data structure of torch. A <b>_[torch.Tensor](https://pytorch.org/docs/1.7.0/tensors.html#torch-tensor)_</b> is a multi-dimensional matrix containing elements of a single data type. </br>&ensp;&ensp;&ensp;&ensp;Tensor can be defined by sequence data from a python list or a numpy array. When data is acceptable, tensor definition can be done as follows,
  ```
>>> torch.tensor([[1., -1.], [1., -1.]])
tensor([[ 1.0000, -1.0000],
        [ 1.0000, -1.0000]])
>>> torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
tensor([[ 1,  2,  3],
        [ 4,  5,  6]])
  ```

  &ensp;&ensp;&ensp;&ensp;Fortunately, <b>_[torch.Tensor](https://pytorch.org/docs/1.7.0/tensors.html#torch-tensor)_</b> always copies the original data. Using it can avoid modification of the original data. If you want to avoid copy, use <b>_[torch.as_tensor](https://pytorch.org/docs/1.7.0/generated/torch.as_tensor.html#torch.as_tensor)_</b> instead.

  &ensp;&ensp;&ensp;&ensp;<b>_[Special forms of tensor](https://github.com/Xcanton/TorchLearn/tree/master/Tensor/Special%20Tensor%20Form)_</b> can be created directly by functions offered by torch. 
  
## _Modify options_
  &ensp;&ensp;&ensp;&ensp;After defining a tensor, some options modifying its value might be needed such as <b>_[random_](https://pytorch.org/docs/1.7.0/tensors.html#torch.Tensor.random_)_</b>, <b>_[normal_](https://pytorch.org/docs/1.7.0/tensors.html#torch.Tensor.normal_)_</b>, <b>_[uniform_](https://pytorch.org/docs/1.7.0/tensors.html#torch.Tensor.uniform_)_</b>, <b>_[new_full](https://pytorch.org/docs/1.7.0/tensors.html#torch.Tensor.new_full)_</b>, <b>_[new_empty](https://pytorch.org/docs/1.7.0/tensors.html#torch.Tensor.new_empty)_</b>, <b>_[new_ones](https://pytorch.org/docs/1.7.0/tensors.html#torch.Tensor.new_ones)_</b>, <b>_[new_zeros](https://pytorch.org/docs/1.7.0/tensors.html#torch.Tensor.new_zeros)_</b>, and <b>_[fill_diagonal_](https://pytorch.org/docs/1.7.0/tensors.html#torch.Tensor.fill_diagonal_)_</b>. Others can be found in <b>[the official api document](https://pytorch.org/docs/1.7.0/tensors.html#torch-tensor)<b>.

## view
  &ensp;&ensp;&ensp;&ensp;<b>_[torch.Tensor.view](https://pytorch.org/docs/1.7.0/tensors.html#torch.Tensor.view)_</b> is capable of reshaping tensors with compatible size. It is similar to <b>_tensorflow.reshape_</b>. </br>&ensp;&ensp;&ensp;&ensp;In dimension definision, -1 can be used to infer from other dimensions. Official examples are as follows,

  ```
>>> x = torch.randn(4, 4)
>>> x.size()
torch.Size([4, 4])
>>> y = x.view(16)
>>> y.size()
torch.Size([16])
>>> z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
>>> z.size()
torch.Size([2, 8])

>>> a = torch.randn(1, 2, 3, 4)
>>> a.size()
torch.Size([1, 2, 3, 4])
>>> b = a.transpose(1, 2)  # Swaps 2nd and 3rd dimension
>>> b.size()
torch.Size([1, 3, 2, 4])
>>> c = a.view(1, 3, 2, 4)  # Does not change tensor layout in memory
>>> c.size()
torch.Size([1, 3, 2, 4])
>>> torch.equal(b, c)
False
  ```

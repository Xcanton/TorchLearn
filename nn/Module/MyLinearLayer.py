import torch

class MyLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MyLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x):
        x = x + self.bias
        x = torch.matmul(x, self.weight)
        return x

if __name__ == "__main__":
    class MyTestNet(torch.nn.Module):
        def __init__(self):
            super(MyTestNet, self).__init__()
            self.mylayer = MyLayer(5,3)
        
        def forward(self, x):
            x = self.mylayer(x)
            return x
    model = MyTestNet()
    print(model)

import torch.nn as nn
class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.feature_dim = feature_dim
        self.Wax = nn.Linear(feature_dim, feature_dim)
        self.Va = nn.Linear(feature_dim, feature_dim, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, Xt):
        at = self.Wax(Xt)
        at = self.tanh(at)
        at = self.Va(at)
        pt = self.softmax(at)
        x_tilda = Xt * pt
        return x_tilda
    
""" 
    at = Va · tanh(Wax · xt + ba),
    pt = softmax(at),
    x_tilda = xt · pt,
    where:
    - xt is the input feature vector at timestep t.
    - at is the output feature vector at timestep
    - Wax is a linear layer with feature_dim input and feature_dim output.
    - Va is a linear layer with feature_dim input and feature_dim output, without bias.
    - tanh is the hyperbolic tangent function.
    - softmax is the softmax function applied along the last dimension.
    - x_tilda is the weighted sum of input features xt and attention weights pt.
"""
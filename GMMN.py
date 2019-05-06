import torch

class GMMN(torch.nn.Module):
    """
    The architecture is shown as Figure 1 (left) in the paper.
    in_dim : dimensions of inputs to the layer
    out_dim : dimensions of outputs of the layer
    """
    def __init__(self, in_dim, out_dim):
        super(GMMN, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, 64)
        self.fc2 = torch.nn.Linear(64, 256)
        self.fc3 = torch.nn.Linear(256, 256)
        self.fc4 = torch.nn.Linear(256, 784)
        self.fc5 = torch.nn.Linear(784, out_dim)

    """
    Forward propagation of the GMMN
    input:  Input batch of samples from the uniform
    """
    def forward(self, input):
        h1 = torch.relu(self.fc1(input))
        h2 = torch.relu(self.fc2(h1))
        h3 = torch.relu(self.fc3(h2))
        h4 = torch.relu(self.fc4(h3))
        x  = torch.sigmoid(self.fc5(h4))
        return x

import torch
import torch.nn.functional as F

class autoencoder(torch.nn.Module):
    """
        The architecture is shown as Figure 1 (right) in the paper.
        Either the encoder or the decoder consists of two fully connected layers.
        in_dim : dimensions of inputs to the layer,i.e., image size
        out_dim : dimensions of outputs of the layer, i.e., encoded size
    """
    def __init__(self, in_dim, out_dim):
        super(autoencoder,self).__init__()
        # Encoder: get the latent representation using the input
        self.en_fc1 = torch.nn.Linear(in_dim, 1024)
        self.en_fc2 = torch.nn.Linear(1024, out_dim)
        # Decoder: get the reconstruction using the latent representation
        self.de_fc1 = torch.nn.Linear(out_dim, 1024)
        self.de_fc2 = torch.nn.Linear(1024, in_dim)

    """
    Forward propagation of the autoencoder.
    drate : dropout rate
    """

    def forward(self, input, index = 3, drate = 0):
        if index == 3:
            # Encoder
            en1 = torch.sigmoid(F.dropout(self.en_fc1(input), p = drate))
            en2 = torch.sigmoid(F.dropout(self.en_fc2(en1), p = drate))
            # Decoder
            de1 = torch.sigmoid(self.de_fc1(en2))
            de2 = torch.sigmoid(self.de_fc2(de1))
            return en1, en2, de1, de2
        elif index == 2:
            # Encoder
            en2 = torch.sigmoid(F.dropout(self.en_fc2(input), p = drate))
            # Decoder
            de1 = torch.sigmoid(self.de_fc1(en2))
            de2 = torch.sigmoid(self.de_fc2(de1))
            return en2, de1, de2

        elif index == 1:
            # Encoder
            # Decoder
            de1 = torch.sigmoid(self.de_fc1(input))
            de2 = torch.sigmoid(self.de_fc2(de1))
            return de1, de2
        elif index == 0:
            # Encoder
            # Decoder
            de2 = torch.sigmoid(self.de_fc2(input))
            return de2

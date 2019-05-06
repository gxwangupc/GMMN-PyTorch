import torch
import argparse
from GMMN import GMMN
from Autoencoder import autoencoder
import os
#from Dataloader import *
from torch.autograd import Variable
from torchvision import datasets, transforms

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist', help='input dataset')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--noise_size', type=int, default=10, help='size of noise')
parser.add_argument('--image_size', type=int, default=784, help='size of the input images')
parser.add_argument('--encoded_size', type=int, default=32, help='encoded size')
parser.add_argument('--nepoch_ae', type=int, default=500, help='number of epochs for training the autoencoder')
parser.add_argument('--nepoch_gmmn', type=int, default=500, help='number of epochs for training the gmmn')
parser.add_argument('--models', default='./models', help='path to save models')
parser.add_argument('--save_ae', default='./models/autoencoder.pth', help='path to save autoencoder.pth')
parser.add_argument('--save_gmmn', default='./models/gmmn.pth', help='path to save gmmn.pth')
parser.add_argument('--nrows', type=int, default=10, help='rows for visualizion')
parser.add_argument('--ncols', type=int, default=10, help='columns for visualizion')
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=0, help='number of GPUs to use')
parser.add_argument("--visualize", default="gmmn", choices=["autoencoder", "gmmn"], help='select one for visualization' )
args = parser.parse_args()

"""
Path to load data and save models, respectively.
"""
if not os.path.exists(args.dataroot):
    os.mkdir(args.dataroot)

if not os.path.exists(args.models):
    os.mkdir(args.models)
"""
Use GPU if available and assign a GPU.
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(args.ngpu)
"""
Load MNIST training images resorting to two packages datasets & transforms from torchvision.
"""
trans = transforms.Compose([transforms.ToTensor()])
trainx = datasets.MNIST(root=args.dataroot, train=True, transform=trans, download=True)
train_loader = torch.utils.data.DataLoader(
        dataset=trainx,
        batch_size=args.batch_size,
        shuffle=True)
"""
trainx  = loadMNIST()
#trainx = loadLFW()
"""

"""
Train the autoencoder.
Adam is employed to optimize the network.
"""
ae_net = autoencoder(args.image_size, args.encoded_size).to(device)
ae_optim = torch.optim.Adam(ae_net.parameters())

def log(input):
    return torch.log(input + 1e-8)

'''
Load saved autoencoder model if available.
'''
if os.path.exists(args.save_ae):
    ae_net.load_state_dict(torch.load(args.save_ae))
    print("Loaded the saved autoencoder model...")
else:
    # a trick for speeding up.
    torch.backends.cudnn.benchmark = True
    for ep in range(args.nepoch_ae):
        avg_loss = 0
        for idx, (img, _) in enumerate(train_loader):
            img = img.view(img.size()[0], -1)
            img = Variable(img).to(device)
            """
            Greedy layer-wise pretraining of the auto-encoder. I have not understood this scheme, and the following module 
            would be a false implementation.
            """
            '''
            Get the hidden representation to this layer by forward propagating on the previously trained layers.
            '''
            rep0, rep1, rep2, rep3 = ae_net(img, index = 3)
            '''
            Reconstruct using these hidden representations.
            '''
           # rec0 = ae_net(rep0, index = 0)
           # rec1, _ = ae_net(rep1, index = 1)
           # rec2, _, _ = ae_net(rep2, index = 2)
            '''
            Reconstraction errors: cross entropy losses the hidden representations and the correponding reconstructions.
            '''
           # loss0 = - torch.sum(img * log(rec0) + (1 - img) * log(1 - rec0))
           # loss1 = - torch.sum(rep0 * log(rec1) + (1 - rep0) * log(1 - rec1))
           # loss2 = - torch.sum(rep1 * log(rec2) + (1 - rep1) * log(1 - rec2))

            """
            Fine-tune the auto-encoder.
            """
            loss3 = - torch.sum(img * log(rep3) + (1 - img) * log(1 - rep3))
            #loss = torch.sum((img - decoded) ** 2)
            ae_optim.zero_grad()
           # loss0.backward(retain_graph = True)
           # loss1.backward(retain_graph = True)
           # loss2.backward(retain_graph = True)
            loss3.backward()
            ae_optim.step()
           # avg_loss += loss0.item() + loss1.item() + loss2.item() + loss3.item()
            avg_loss += loss3.item()
        avg_loss /= (idx + 1)

        print("Autoencoder Training: Epoch - [%2d] completed, average loss - [%.4f]" %(ep + 1, avg_loss))
    '''
    Save the autoencoder.pth.
    '''
    torch.save(ae_net.state_dict(), args.save_ae)

print("The autoencoder has been successfully trained.")

"""
Train the GMMN.
"""
gmmn_net = GMMN(args.noise_size, args.encoded_size).to(device)
gmmn_optimizer = torch.optim.Adam(gmmn_net.parameters(), lr=0.001)
'''
Load saved GMMN model if available.
'''
if os.path.exists(args.save_gmmn):
    gmmn_net.load_state_dict(torch.load(args.save_gmmn))
    print("Loaded the previously saved GMMN model...")
else:
    """
    Scale column for the MMD measure, as described in section 2 in the paper.
    M: Number of samples taken from dataset in one pass.
    N :  Number of samples to be generated in one pass.
    """
    def get_scale_matrix(M, N):
        # first 'N' entries have '1/N', next 'M' entries have '-1/M'
        s1 = (torch.ones((N, 1)) * 1.0 / N).to(device)
        s2 = (torch.ones((M, 1)) * -1.0 / M).to(device)
        return torch.cat((s1, s2), 0)
    """
    Calculates cost of the network, which is square root of the mixture of 'K' RBF kernels.
    x      :       Batch from the dataset.
    samples: Samples from the uniform distribution.
    sigma  :   Bandwidth parameters for the 'K' kernels.
    """
    def train_one_step(x, samples, sigma=[1]):
        samples = Variable(samples).to(device)
        # generate codes from the uniform samples
        gen_samples = gmmn_net(samples)
        X = torch.cat((gen_samples, x), 0)
        # dot product between all combinations of rows in 'X'
        XX = torch.matmul(X, X.t())
        # dot product of rows with themselves
        X2 = torch.sum(X * X, 1, keepdim=True)
        # exponent entries of the RBF kernel (without the sigma) for each
        # combination of the rows in 'X'
        # -0.5 * (x^Tx - 2*x^Ty + y^Ty)
        exp = XX - 0.5 * X2 - 0.5 * X2.t()

        # scaling constants for each of the rows in 'X', i.e., batch_size
        M = gen_samples.size()[0]
        N = x.size()[0]
        s = get_scale_matrix(M, N)
        # scaling factors of each of the kernel values, corresponding to the exp values
        S = torch.matmul(s, s.t())

        loss = 0
        # for each bandwidth parameter, compute the MMD value and add them all.
        for bw in sigma:
            # kernel values for each combination of the rows in 'X'.
            kernel_val = torch.exp(exp / bw)
            loss += torch.sum(S * kernel_val)

        loss = torch.sqrt(loss)

        gmmn_optimizer.zero_grad()
        loss.backward()
        gmmn_optimizer.step()
        return loss

    # training loop
    # a trick for speeding up.
    torch.backends.cudnn.benchmark = True
    for ep in range(args.nepoch_gmmn):
        avg_loss = 0
        for idx, (img, _) in enumerate(train_loader):
            img = img.view(img.size()[0], -1)
            with torch.no_grad():
                img = Variable(img).to(device)
                _, encoded, _, _ = ae_net(img)

            # uniform random noise between [-1, 1]
            random_noise = torch.rand((args.batch_size, args.noise_size)) * 2 - 1
            loss = train_one_step(encoded, random_noise)
            avg_loss += loss.item()

        avg_loss /= (idx + 1)
        print("GMMN Training: Epoch - [%3d] completed, average loss - [%.4f]" %(ep+1, avg_loss))
    '''
    Save the gmmn.pth.
    '''
    torch.save(gmmn_net.state_dict(), args.save_gmmn)

print("The GMMN has been successfully trained.")

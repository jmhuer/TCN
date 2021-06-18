
from TCN.tcn import TemporalBlock, TemporalConvNet
import torch
from torch import nn
from TCN.tcn import TemporalConvNet
import torch.nn.functional as F

import torch
import torch.optim as optim
torch.manual_seed(42)

from kwta import Sparsify1D_kactive
from synthetic_data import create_synthetic_data


# class TCN(nn.Module):
#     def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
#         super(TCN, self).__init__()
#         self.encoder = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
#         kernel_size, padding, stride, dilation = self.encoder.network[-1].conv1.kernel_size, self.encoder.network[-1].conv1.padding, self.encoder.network[-1].conv1.stride, self.encoder.network[-1].conv1.dilation
#         print(kernel_size, padding, stride )
#         in_channels, out_channels = self.encoder.network[-1].conv1.in_channels, self.encoder.network[-1].conv1.out_channels
#         print("in", in_channels, out_channels)
#         self.decoder = torch.nn.ConvTranspose1d(in_channels=num_channels[-1], out_channels=1, kernel_size=5, padding=0, dilation=1, stride=1)
#
#     def forward(self, x):
#         # x needs to have dimension (N, C, L) in order to be passed into CNN
#         output = self.encoder(x.transpose(1, 2))
#         print("~~~~~~~~out size ", output.size())
#         output = self.decoder(output).double().transpose(1, 2)
#         return output





synth_data = create_synthetic_data(size = 5000)

class autoencoder(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, wta_k):
        super(autoencoder, self).__init__()
        self.wta = Sparsify1D_kactive(k = wta_k)
        self.feature = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        # self.encoder = torch.nn.Conv1d(in_channels=5, out_channels=10, kernel_size=5, padding=0, bias=False, stride=5)
        # self.decoder = torch.nn.ConvTranspose1d(in_channels=10, out_channels=1, kernel_size=5, padding=0, bias=False, stride=5)
        # self.encoder.weight.data.fill_(0.3)
        # self.code = None
    # def get_kernels(self):
    #     return self.decoder.weight.data[:,0,:]
    # def feature_map(self, x):
    #     code = self.wta(self.encoder(x))
    #     return code
    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        output = self.feature(x).double()
        # print("~~~~~~~~feature size ", output.size())
        # self.code = self.wta(self.encoder(output))
        # # print("~~~~~~~~code size ", code.size())
        # output = self.decoder(self.code ).double()
        return output




device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using device: ", device)

model = autoencoder(input_size=1, output_size=400, num_channels=[1, 15, 25], kernel_size=5, dropout=0.2, wta_k = 5).to(device)
inputs = torch.tensor(synth_data[:,None,:]).float().to(device)
print("Input size: ", inputs.size())
out = model(inputs)
print("Output size: ", out.size(), "\n")


loss_fn = torch.nn.L1Loss().to(device)
optimizer = optim.SGD(model.parameters(), lr=.05, weight_decay = 0.00001, momentum=0.05) ##this has weight decay just like you implemented
epochs = 3000
history = {"loss": []}
for i in range(epochs):
  optimizer.zero_grad()
  output = model(inputs)

  #decaying WTA
  if i % 500 == 0 and i != 0:
      model.wta.k = max(1, model.wta.k - 1)
      print("model.wta.k: ", model.wta.k)

  loss = loss_fn(output, inputs)
  loss.backward()
  optimizer.step()
  history["loss"].append(float(loss))
  if i % 1 == 0:
      print("Epoch : {} \t Loss : {} \t ".format(i, round(float(loss),7)))
      # print("\nneg encoder ", float((model.encoder.weight.ravel() < 0).sum(dim=0)))














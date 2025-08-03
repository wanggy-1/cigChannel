# 3D-UNet model.
# x: 128x128 resolution for 32 frames.
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation,)

def conv_block_3d_last(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),)

def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        activation,)

def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

def conv_block_2_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation,)

class UNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(UNet, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.2, inplace=True)
        
        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = max_pooling_3d()
 
        # Bridge
        self.bridge = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)

        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.up_1 = conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.up_2 = conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
        self.up_3 = conv_block_2_3d(self.num_filters * 3, self.num_filters * 1, activation)
        
        # Output
        self.out = conv_block_3d_last(self.num_filters, out_dim)
        
    
    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x) # -> [1, 4, 128, 128, 128]
        pool_1 = self.pool_1(down_1) # -> [1, 4, 64, 64, 64]
        
        down_2 = self.down_2(pool_1) # -> [1, 8, 64, 64, 64]
        pool_2 = self.pool_2(down_2) # -> [1, 8, 32, 32, 32]
        
        down_3 = self.down_3(pool_2) # -> [1, 16, 32, 32, 32]
        pool_3 = self.pool_3(down_3) # -> [1, 16, 16, 16, 16]
        
        # Bridge
        bridge = self.bridge(pool_3) # -> [1, 128, 4, 4, 4]

        # Up sampling
        trans_1 = self.trans_1(bridge) # -> [1, 128, 8, 8, 8]
        concat_1 = torch.cat([trans_1, down_3], dim=1) # -> [1, 192, 8, 8, 8]
        up_1 = self.up_1(concat_1) # -> [1, 64, 8, 8, 8]
        
        trans_2 = self.trans_2(up_1) # -> [1, 64, 16, 16, 16]
        concat_2 = torch.cat([trans_2, down_2], dim=1) # -> [1, 96, 16, 16, 16]
        up_2 = self.up_2(concat_2) # -> [1, 32, 16, 16, 16]
        
        trans_3 = self.trans_3(up_2) # -> [1, 32, 32, 32, 32]
        concat_3 = torch.cat([trans_3, down_1], dim=1) # -> [1, 48, 32, 32, 32]
        up_3 = self.up_3(concat_3) # -> [1, 16, 32, 32, 32]
        
        # Output
        out = self.out(up_3) # -> [1, 3, 128, 128, 128]
        out = F.softmax(out, dim=1)
        # out = self.last_relu(out)
        return out

if __name__ == "__main__":
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  image_size = 128
  x = torch.Tensor(1, 1, image_size, image_size, image_size)
  x.to(device)
  print("x size: {}".format(x.size()))
  
  model = UNet(in_dim=1, out_dim=1, num_filters=4)
  
  out = model(x)
  print("out size: {}".format(out.size()))
import torch.nn as nn
import torch
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction


class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        
        # 512 = points sampled in farthest point sampling. Points number after FPS.
        # 0.2 = search radius in local region. After FPS, the SA need to group points. Points within the radius range will be regarded as one group.
        # 32 = how many points in each local region
        # [64,64,128] = output size for MLP on each point
        # 3 = 3-dim coordinates
        
        dropout_val = 0.5
        
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=256, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa1_1 = PointNetSetAbstraction(npoint=64, radius=0.4, nsample=256, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)

        self.sa2 = PointNetSetAbstraction(npoint=512, radius=0.4, nsample=512, in_channel=in_channel, mlp=[64, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel= in_channel, mlp=[64, 128, 256, 512], group_all=True)
        self.sa4 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        # fc1 input:1024
        self.fc1 = nn.Linear(1536, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(dropout_val)
        # fc2 input:512
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(dropout_val)
        # fc3 input:256
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape# xyz:[B, 6, 1024]
        if self.normal_channel:
            norm = xyz[:, 3:, :]#[B, 3, 1024]
            xyz = xyz[:, :3, :]#xyz:[B, 3, 1024]
        else:
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)#l1_xyz:[B, 3, 512], l1_points:[B, 128, 512]
        l1_1xyz, l1_1points = self.sa1_1(l1_xyz, l1_points)#l1_1xyz:[B, 3, 64], l1_points:[B, 256, 64]
        l2_xyz, l2_points = self.sa2(xyz, norm)
      
        l3_xyz, l3_points = self.sa3(xyz, norm)                                                  
        l12_xyz = torch.cat([l1_1xyz, l2_xyz], 2)#[B, 3, 576] 
        l12_points = torch.cat([l1_1points, l2_points], 2)#[B, 256, 576]
        l4_xyz, l4_points = self.sa4(l12_xyz, l12_points)#l4_xyz:[B, 3, 1], l4_points:[B, 1024, 1]
        l34_points = torch.cat([l3_points, l4_points], 1)#[B, 1536, 1]

        x = l34_points.view(B, 1536)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)
        return total_loss

if __name__ == '__main__':
    model = get_model(13)
    xyz = torch.rand(6, 9, 2048)
    x, l3_points = (model(xyz))
    print(x.shape)
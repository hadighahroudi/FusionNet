import torch
from torch import nn
import torchvision

class VCNNI(nn.Module):
    def __init__(self, input_shape, n_outputs = 4):
        super(VCNNI, self).__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels= 30, out_channels= 64, kernel_size= (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= (2, 2))
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels= 64, out_channels= 64, kernel_size= (3, 3)),
            nn.ReLU()
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels= 64, out_channels= 64, kernel_size= (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= (2, 2)),
            nn.Dropout(0.5)
        )

        dummy = torch.rand(input_shape)
        dummy_features = self.conv_block_3(self.conv_block_2(self.conv_block_1(dummy)))
        flattened_size = len(dummy_features.view(-1))

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = flattened_size, out_features= 2048),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(in_features= 2048, out_features= n_outputs)

    def forward(self, x): # x shape -> (batch, depth, height, width)
        features = self.conv_block_3(self.conv_block_2(self.conv_block_1(x)))
        return self.fc2(self.fc1(features))
        
    def forward_multi_view(self, x): # x shape -> (views, depth, height, width)
        features = self.conv_block_3(self.conv_block_2(self.conv_block_1(x)))
        features = self.fc1(features) # x shape -> (views, features)
        pooled_features = features.max(dim = 0).values # x shape -> (features)
        return self.fc2(pooled_features.unsqueeze(0)) 


class InceptionBlock1(nn.Module):
    def __init__(self):
        super(InceptionBlock1, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels = 30, out_channels = 20, kernel_size = 1, stride = 1)
        self.conv_3x3 = nn.Conv2d(in_channels = 30, out_channels = 20, kernel_size = 3, stride = 1, padding = 'same')
        self.conv_5x5 = nn.Conv2d(in_channels = 30, out_channels = 20, kernel_size = 5, stride = 1, padding = 'same')

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out1 = self.conv_1x1(x)
        out2 = self.conv_3x3(x)
        out3 = self.conv_5x5(x)
        out = torch.cat([out1, out2, out3], dim = 1)
        out = self.relu(out)
        out = self.dropout(out)
        return out

class InceptionBlock2(nn.Module):
    def __init__(self):
        super(InceptionBlock2, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels = 60, out_channels = 30, kernel_size = 1, stride = 1)
        self.conv_3x3 = nn.Conv2d(in_channels = 60, out_channels = 30, kernel_size = 3, stride = 1, padding = 'same')

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        out1 = self.conv_1x1(x)
        out2 = self.conv_3x3(x)
        out = torch.cat([out1, out2], dim = 1)
        out = self.relu(out)
        out = self.dropout(out)
        return out
    
class VCNNII(nn.Module):
    def __init__(self, input_shape, n_outputs = 4):
        super(VCNNII, self).__init__()

        self.inception_block_1 = InceptionBlock1()
        self.inception_block_2 = InceptionBlock2()

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels = 60, out_channels = 30, kernel_size = 3, stride = 1, padding = 'same'),
            nn.ReLU(),
            nn.Dropout(0.5)
            )

        dummy = torch.rand(input_shape)
        if dummy.ndim == 3: # No batch dim
            dummy = dummy.unsqueeze(0)
        dummy_features = self.conv_block_3(self.inception_block_2(self.inception_block_1(dummy)))
        flattened_size = len(dummy_features.view(-1))

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = flattened_size, out_features= 2048),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(in_features= 2048, out_features = n_outputs)

    def forward(self, x): # x shape -> (batch, depth, height, width)
        features = self.conv_block_3(self.inception_block_2(self.inception_block_1(x)))
        return self.fc2(self.fc1(features))
        
    def forward_multi_view(self, x): # x shape -> (views, depth, height, width)
        features = self.conv_block_3(self.inception_block_2(self.inception_block_1(x)))
        features = self.fc1(features) # x shape -> (views, features)
        pooled_features = features.max(dim = 0).values # x shape -> (features)
        return self.fc2(pooled_features.unsqueeze(0)) 

class MVCNN(nn.Module):
    def __init__(self, checkpoint_path = None):
        super(MVCNN, self).__init__()

        self.alexnet = torchvision.models.alexnet()
        if checkpoint_path:
            self.alexnet.load_state_dict(torch.load(checkpoint_path))
        self.alexnet.classifier[-1] = nn.Linear(in_features = 4096, out_features = 4, bias = True)

        for param in self.alexnet.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.alexnet(x)

    def forward_multi_view(self, x): # x shape -> (views, depth, height, width)
        # Based on the forward method in https://pytorch.org/vision/main/_modules/torchvision/models/alexnet.html#alexnet
        features = self.alexnet.avgpool(self.alexnet.features(x))
        features = torch.flatten(features, 1)
        features = self.alexnet.classifier[:-1](features) # x shape -> (views, features)
        pooled_features = features.max(dim = 0).values # x shape -> (features)
        return self.alexnet.classifier[-1](pooled_features.unsqueeze(0))
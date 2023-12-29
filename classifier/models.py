import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from IPython.core.debugger import set_trace

class ConvNet1D(nn.Module):

    def __init__(self, n, num_classes, debug=False):
        super().__init__()
        self.debug = debug
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        
        self.conv1 = nn.Conv1d(in_channels=1,
                               out_channels=64, 
                               kernel_size=3, 
                               stride=1,
                               padding=1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.maxpool1 = nn.MaxPool1d(3, stride=3)
        self.conv2 = nn.Conv1d(in_channels=64,
                       out_channels=128, 
                       kernel_size=3, 
                       stride=1,
                       padding=1)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.maxpool2 = nn.MaxPool1d(3, stride=3)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128, num_classes)
        
    def forward(self,x):
        set_trace()
        if self.debug: print('input', x.shape)

        out = self.conv1(x)
        if self.debug: print('conv1', x.shape)

        out = self.batchnorm1(out)
        if self.debug: print('bn', out.shape)

        out = F.relu(out)
        if self.debug: print('relu1', out.shape)

        out = self.maxpool1(out)
        if self.debug: print('mp1', out.shape)       
        
        out = self.conv2(out)
        if self.debug: print('conv2', out.shape)
            
        out = self.batchnorm2(out)
        if self.debug: print('bn2', out.shape)

        out = F.relu(out)
        if self.debug: print('relu2', out.shape)
        
        out = self.maxpool2(out)
        if self.debug: print('mp2', out.shape)
            
        out = self.dropout(out)
        if self.debug: print('do1', out.shape)
            
        out = self.fc1(out)
        if self.debug: print('fc1', out.shape)

        out = F.softmax(out, dim=1)
        if self.debug: print('softmax', out.shape)
        
        # Remove unnecessary dimensions and change shape to match target tensor
        # [BATCH_SIZE, num_classes, 1, 1] --> [BATCH_SIZE,num_classes] to make predicitons
#         output = torch.squeeze(x)
        dim = out.size(1) * out.size(2)
        if self.debug: print('output dim:', dim)

        out = out.view(x.shape[0], dim)
        # out = torch.reshape(x, (-1, self.num_classes))
        if self.debug: print("output:\t", out.shape)

        return out
    
class Network(nn.Module):

    def __init__(self, n, num_classes, debug=False):
        super().__init__()
        self.debug = debug
        self.fc1 = nn.Linear(n, 64)
        self.b1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64,128)
        self.b2 = nn.BatchNorm1d(128)
        self.d1 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(128,num_classes)

    def forward(self,x):

        x = F.relu(self.fc1(x))
        x = self.b1(x)
        x = F.relu(self.fc2(x))
        x = self.b2(x)
        x = self.d1(x)
        x = torch.sigmoid(self.fc3(x))
        
        # Remove unnecessary dimensions and change shape to match target tensor
        # [BATCH_SIZE, num_classes, 1, 1] --> [BATCH_SIZE,num_classes] to make predicitons
        output = torch.squeeze(x)
        if self.debug: print("output:\t", output.shape)

        return x
# model: OCC_VGG19_v4_0_5
import torch
import torch.nn as nn
import numpy as np

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG16_v4_0': [64,'A', 64, 'M', 128, 128, 'M', 256, 256, 256,'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19_v4_1': [64, 64, 'M','A', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG19_v4_2': [64, 64, 'M', 128, 128, 'M','A', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG19_v4_3': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M','A', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

fc_n = 16

class VGG_OCC(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_OCC, self).__init__()
        self.features_A, self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)   # fully-connect nn.Linear(in_features, out_features)
        self.classifier_occluded = nn.Linear(fc_n,2)
        #self.activate = nn.ReLU(inplace=True)
    def forward(self, x):
        out_A = self.features_A(x)     # out:[1, 512, 1, 1]   [number of image, image_size, _, _]
        out = self.features(x)
        out_A = out_A.view(out.size(0), -1)
        out = out.view(out.size(0), -1)  # out:[1, 512]
        
        out_confidence = self.classifier(out)    # out:[1, 10]
        out_occluded = self.classifier_occluded(out_A)     # out:[1, 2]
        #out_occluded = self.activate(out_occluded)
        out = torch.cat((out_occluded,out_confidence),1) 
        #print('3',out.shape)
        return out

    def _make_layers(self, cfg):
        layers_share = []
        layers = []
        layers_A = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#                 np_M_shape = np.shape(layers)
#                 print('np_M_shape:',np_M_shape)
            
            elif x == 'A':
                layers_A += layers
                #layers_A += [nn.MaxPool2d(kernel_size=2, stride=2)]
                #for 'VGG19_v4_1' and 'VGG19_v4_2' 
                layers_A += [nn.Conv2d(64, fc_n, kernel_size=3, padding=1,stride = 4),
                           nn.BatchNorm2d(fc_n),
                           nn.ReLU(inplace=True)]
                layers_A += [nn.MaxPool2d(kernel_size=2, stride=8)]
                
                #layers_A += [nn.Conv2d(128, 256, kernel_size=3, padding=1,stride = 2),
                #           nn.BatchNorm2d(256),
                #           nn.ReLU(inplace=True)]
                #layers_A += [nn.MaxPool2d(kernel_size=2, stride=2)]
                
                #layers_A += [nn.Conv2d(256, 512, kernel_size=3, padding=1, stride = 2),
                #           nn.BatchNorm2d(512),
                #           nn.ReLU(inplace=True)]
                #layers_A += [nn.MaxPool2d(kernel_size=2, stride=2)]
            
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
#                 np_shape = np.shape(layers)
#                 np_size = np.size(layers)
#                 print('np_shape:',np_shape,'np_size',np_size)
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        layers_A += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers_A), nn.Sequential(*layers)
        
def test():
    net = VGG_OCC('VGG19_v4_0')
    x = torch.randn(1,3,32,32)
    y = net(x)
    print('node_12:  ',y.size())
    
#test()

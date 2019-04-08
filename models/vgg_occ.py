import torch
import torch.nn as nn
import numpy as np

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGGtest': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
}


class VGG_OCC(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_OCC, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)   # fully-connect nn.Linear(in_features, out_features)
        #self.classifier_occluded = nn.Linear(512,1)
    def forward(self, x):
        out = self.features(x)     # out:[1, 512, 1, 1]   [number of image, image_size, _, _]
        #print('1',out.shape)
        out = out.view(out.size(0), -1)  # out:[1, 512]
        #print('2',out.shape)
        out = self.classifier(out)    # out:[1, 10]


        # out_occluded = self.classifier_occluded(out)     # out:[1, 1]
        # out = torch.cat((out_confidence,out_occluded),1 ) 
        # #out:[1,10]
        #print('3',out.shape)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#                 np_M_shape = np.shape(layers)
#                 print('np_M_shape:',np_M_shape)
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
#                 np_shape = np.shape(layers)
#                 np_size = np.size(layers)
#                 print('np_shape:',np_shape,'np_size',np_size)
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
        
def test():
    net = VGG_OCC('VGG11')
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())
    
test()

import yaml
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
cpath = Path().absolute().joinpath('classification\config.yaml')
print(cpath)
config_file = Path(cpath)
with open(config_file) as file:
  config = yaml.safe_load(file)

class ResNet3D_18_Classifier(nn.Sequential):
    def __init__(self, pretrained, in_ch, out_ch, linear_ch=512, seed=None, early_layers_learning_rate=0,
                 ):
        
        '''
        in_ch = 1 or 3
        early_layers can be 'freeze' or 'lower_lr'
        '''
        super(ResNet3D_18_Classifier, self).__init__()
        if seed != None:
            print(f"Seed set to {seed}")
            torch.manual_seed(seed)
        

        self.model = torchvision.models.video.r3d_18(pretrained=pretrained)
        if not early_layers_learning_rate: # 
            print("Freezing layers")
            for p in self.model.parameters():
                p.requires_grad = False
        elif early_layers_learning_rate:
            print(f"Early layers will use a learning rate of {early_layers_learning_rate}")
        #Reshape
        print(f"Initializing network for {in_ch} channel input")
        if in_ch!=3:
            self.model.stem[0] =  nn.Conv3d(in_ch, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        
        self.model.fc = nn.Linear(linear_ch, out_ch)
        print(f"Linear layer initialized with {linear_ch} number of channels.")
        if out_ch == 1:
            self.out = nn.Sigmoid()
        else:
            self.out = nn.Softmax(dim=1)
        super(ResNet3D_18_Classifier, self).__init__(self.model,self.out)


from torch.autograd import Function

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class GradReverse(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class DomainDiscriminator(nn.Module):
    def __init__(self, in_features, num_domains):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, in_features//2),
            nn.ReLU(inplace=True),
            nn.Linear(in_features//2, num_domains)
        )
    def forward(self, x):
        return self.net(x)
    
from torch import nn
import torch
import torchvision

# Re-use your existing GRL and DomainDiscriminator
# (no changes to those)
# class GradientReversalFunction(Function): ...
# class GradReverse(nn.Module): ...
# class DomainDiscriminator(nn.Module): ...

class ResNet3D_18_DANN(nn.Module):
    """
    3D-ResNet18 + GRL + domain discriminator head.
    Returns (class_pred, class_logits, domain_logits).
    """
    def __init__(self, pretrained, in_ch, out_ch, linear_ch=512,
                 seed=None, early_layers_learning_rate=0,
                 num_domains=None, dann_lambda=1.0):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        # 1 Build the 3D ResNet feature extractor
        base = torchvision.models.video.r3d_18(pretrained=pretrained)
        if not early_layers_learning_rate:
            for p in base.parameters():
                p.requires_grad = False
        elif early_layers_learning_rate:
            # your existing lr-scaling code remains
            pass

        if in_ch != 3:
            base.stem[0] = nn.Conv3d(
                in_ch, 64, kernel_size=(3,7,7),
                stride=(1,2,2), padding=(1,3,3), bias=False
            )

        # strip off the final fc
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
        self.classifier = nn.Linear(linear_ch, out_ch)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

        # activation for class output
        self.out = nn.Sigmoid() if out_ch == 1 else nn.Softmax(dim=1)

        # 2 DANN heads
        self.grl = GradReverse(dann_lambda)
        # require user to pass num_domains
        assert num_domains is not None, "Set num_domains for DANN"
        self.domain_discriminator = DomainDiscriminator(linear_ch, num_domains)

    def forward(self, x):
        # x: [B, C, D, H, W]
        f = self.feature_extractor(x)          # [B, 512,1,1,1]
        f = f.view(f.size(0), -1)              # [B, 512]
        class_logits = self.classifier(f)      # [B, out_ch]
        class_pred = self.out(class_logits)    # [B, out_ch]
        # domain head via GRL
        d_in = self.grl(f)                     # [B, 512]
        domain_logits = self.domain_discriminator(d_in)  # [B, num_domains]
        return class_pred, class_logits, domain_logits
    
    
# class ResNet18Classifier(nn.Sequential):
#     def __init__(self, pretrained, in_ch, out_ch, seed=None, early_layers_learning_rate=0):
#         '''
#         in_ch = 1 or 3
#         early_layers can be 'freeze' or 'lower_lr'
#         '''
#         super(ResNet18Classifier, self).__init__()
#         self.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
#         # model.classifier[1]=nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1)) # Apply glorot initialization

#         if not early_layers_learning_rate: # 
#             print("Freezing layers")
#             for p in self.model.parameters():
#                 p.requires_grad = False
#         elif early_layers_learning_rate:
#             print(f"Early layers will use a learning rate of {early_layers_learning_rate}")
#         self.model.fc = nn.Linear(512, out_ch)

#         if isinstance(self.model.fc, nn.Linear):
#             torch.nn.init.xavier_uniform_(self.model.fc.weight)
#             if self.model.fc.bias is not None:
#                 torch.nn.init.zeros_(self.model.fc.bias)

#         if out_ch == 1:
#             self.out = nn.Sigmoid()
#         else:
#             self.out = nn.Softmax(dim=1)
#         super(ResNet18Classifier, self).__init__(self.model, 
#                                                  self.out)
                                                
# class SqueezeNetClassifier(nn.Sequential):
#     def __init__(self, pretrained, in_ch, out_ch, seed=None, early_layers='freeze'):
#         '''
#         in_ch = 1 or 3
#         early_layers can be 'freeze' or 'lower_lr'
#         '''
#         super(SqueezeNetClassifier, self).__init__()
#         model = torch.hub.load('pytorch/vision:v0.6.0', 'squeezenet1_0', pretrained=True)
#         model.classifier[1]=nn.Conv2d(512, out_ch, kernel_size=(1, 1), stride=(1, 1)) # Apply glorot initialization
#         if isinstance(model.classifier[1], nn.Conv2d):
#             torch.nn.init.xavier_uniform_(model.classifier[1].weight)
#             if model.classifier[1].bias is not None:
#                 torch.nn.init.zeros_(model.classifier[1].bias)
        
#         super(SqueezeNetClassifier, self).__init__(self.model)
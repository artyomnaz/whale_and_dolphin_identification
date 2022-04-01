import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

    def __repr__(self):
        return self.__class__.__name__ + \
            '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
            ', ' + 'eps=' + str(self.eps) + ')'


class ArcMarginProduct(nn.Module):
    """ArcFace model
    """

    def __init__(self, in_features, out_features, s=30.0,
                 m=0.50, easy_margin=False, ls_eps=0.0, device='cuda:0'):
        """initialization

        Args:
            in_features (int): input features amount
            out_features (int): output features amount
            s (float, optional): parameter. Defaults to 30.0.
            m (float, optional): parameter. Defaults to 0.50.
            easy_margin (bool, optional): parameter. Defaults to False.
            ls_eps (float, optional): parameter. Defaults to 0.0.
            device (str, optional): device. Defaults to 'cuda:0'.
        """
        super(ArcMarginProduct, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.weight = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        self.device = device

        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        """Forward function

        Args:
            input (torch.tensor): input tensor
            label (torch.tensor): label for the input tensor

        Returns:
            _type_: torch.tensor
        """
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size()).to(self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + \
                self.ls_eps / self.out_features

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class HappyModelBackbone(nn.Module):
    """Happy Model Backbone
    """

    def __init__(self, model_name='vgg16'):
        """_summary_

        Args:
            model_name (str, optional): _description_. Defaults to 'vgg16'.
        """
        super(HappyModelBackbone, self).__init__()

        if model_name == 'vgg16':
            self.pooling = GeM()
            self.pretrained_model = models.vgg16(pretrained=True)
            self.pretrained_model = torch.nn.Sequential(
                *(list(self.pretrained_model.children())[:-2] + [nn.AdaptiveAvgPool2d(1)]))
            self.backbone_features = 512
        elif model_name == 'resnet18':
            self.pooling = GeM()
            self.pretrained_model = models.resnet18(pretrained=True)
            self.pretrained_model = torch.nn.Sequential(
                *(list(self.pretrained_model.children())[:-2] + [nn.AdaptiveAvgPool2d(1)]))
            self.backbone_features = 512

    def forward(self, x):
        """Forward function

        Args:
            x (torch.tensor): embedding

        Returns:
            _type_: torch.tensor
        """
        # return self.pooling(self.pretrained_model(x))
        return self.pretrained_model(x)


class HappyWhaleModel(nn.Module):
    """ The main Happy Whale Model
    """

    def __init__(self, numClasses, noNeurons, embeddingSize, model_name='vgg16'):
        """initialization

        Args:
            numClasses (int): classes amount
            noNeurons (int): hidden neurons amount
            embeddingSize (int): output embedding size
            model_name (str): backbone model name
        """
        super(HappyWhaleModel, self).__init__()
        self.model_name = model_name
        self.backbone = HappyModelBackbone(self.model_name)
        self.backbone_features = self.backbone.backbone_features

        self.embedding = nn.Sequential(nn.Linear(self.backbone_features, noNeurons),
                                       nn.BatchNorm1d(noNeurons),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.2),

                                       nn.Linear(noNeurons, embeddingSize),
                                       nn.BatchNorm1d(embeddingSize),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.2))

        self.arcface = ArcMarginProduct(in_features=embeddingSize,
                                        out_features=numClasses,
                                        s=30.0, m=0.50, easy_margin=False, ls_eps=0.0)

    def forward(self, image, target=None):
        """Forward function

        Args:
            image (torch.tensor): input image tensor
            target (torch.tensor, optional): labels tensor. Defaults to None.

        Returns:
            _type_: torch.tensor
        """
        # embeddings retrieving
        features = self.backbone(image)
        features = features.view(-1, self.backbone_features)
        embedding = self.embedding(features)

        if target != None:
            # apply arcface features
            out = self.arcface(embedding, target)
            return out, embedding
        return embedding

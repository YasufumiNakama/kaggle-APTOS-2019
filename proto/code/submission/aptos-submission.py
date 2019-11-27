import pandas as pd
import numpy as np
import os

from collections import OrderedDict
import gc
from functools import partial
import cv2
import random
import math
import numbers

import scipy as sp
from functools import partial
from sklearn.metrics import cohen_kappa_score

from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as M
from torch.nn import functional as F
from torchvision.transforms import (Compose, ToTensor, Normalize, Resize, RandomHorizontalFlip)

sub = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')
folds = pd.read_csv("../input/aptos-folds/folds.csv")
N_CLASSES = 1

##################### quadratic_weighted_kappa ########################
def quadratic_weighted_kappa(y_hat, y):
    return cohen_kappa_score(y_hat, y, weights='quadratic')


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']

##################### DataSet ########################
def load_image(path, item):
    image = cv2.imread(f'{path}/{item}.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)

class TTADataset(Dataset):
    def __init__(self, root, ids, tta=4):
        self.root = root
        self.ids = ids
        
        self.transform = Compose([
            Resize((256, 256)),
            ToTensor(),
            Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
            
        self.tta = tta

    def __len__(self):
        return len(self.ids) * self.tta

    def __getitem__(self, idx):
        item_id = self.ids[idx % len(self.ids)]
        image = load_image(self.root, item_id)
        image = self.transform(image)
        return image, item_id

class ValidDataset(Dataset):
    def __init__(self, root, df):
        super().__init__()
        self._root = root
        self._df = df
        
        self.transform = Compose([
            Resize((256, 256)),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx: int):
        item = self._df.iloc[idx]
        image = load_image(self._root, item.id_code)
        image = self.transform(image)
        target = torch.tensor(self._df.loc[idx, 'diagnosis'])
        return image, target
    
##################### Models ########################
class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
                               stride=stride, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x
    
class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


class ResNet(nn.Module):
    def __init__(self, weights_path, net_cls=M.resnet101):
        super().__init__()
        self.net = net_cls()
        self.net.avgpool = AvgPool()
        self.net.fc = nn.Linear(self.net.fc.in_features, N_CLASSES)
        self.load_state_dict(torch.load(weights_path)['model'])

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        return self.net(x)


class DenseNet(nn.Module):
    def __init__(self, weights_path, net_cls=M.densenet121):
        super().__init__()
        self.net = net_cls()
        self.avg_pool = AvgPool()
        self.net.classifier = nn.Linear(self.net.classifier.in_features, N_CLASSES)
        self.load_state_dict(torch.load(weights_path)['model'])

    def fresh_params(self):
        return self.net.classifier.parameters()

    def forward(self, x):
        out = self.net.features(x)
        out = F.relu(out, inplace=True)
        out = self.avg_pool(out).view(out.size(0), -1)
        out = self.net.classifier(out)
        return out
    
class InceptionNet(nn.Module):
    def __init__(self, weights_path, net_cls=M.inception_v3):
        super().__init__()
        self.net = net_cls()
        self.net.fc = nn.Linear(self.net.fc.in_features, N_CLASSES)
        self.net.AuxLogits.fc = nn.Linear(self.net.AuxLogits.fc.in_features, N_CLASSES)
        self.load_state_dict(torch.load(weights_path)['model'])

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        return self.net(x)
    
resnet18 = partial(ResNet, net_cls=M.resnet18)
resnet34 = partial(ResNet, net_cls=M.resnet34)
resnet50 = partial(ResNet, net_cls=M.resnet50)
resnet101 = partial(ResNet, net_cls=M.resnet101)
resnet152 = partial(ResNet, net_cls=M.resnet152)

#densenet121 = partial(DenseNet, net_cls=M.densenet121)
#densenet169 = partial(DenseNet, net_cls=M.densenet169)
#densenet201 = partial(DenseNet, net_cls=M.densenet201)
#densenet161 = partial(DenseNet, net_cls=M.densenet161)

class SEResNet50(nn.Module):
    def __init__(self, weights_path, pretrained=False, dropout=False, net_cls=None):
        super().__init__()
        self.net = SENet(SEResNetBottleneck, [3,4,6,3], groups=1, reduction=16,
                         dropout_p=None, inplanes=64, input_3x3=False,
                         downsample_kernel_size=1, downsample_padding=0,
                         num_classes=N_CLASSES)
        self.net.avg_pool = AvgPool()
        self.load_state_dict(torch.load(weights_path)['model'])
        #self.net.last_linear = nn.Linear(self.net.last_linear.in_features, num_classes)

    def fresh_params(self):
        return self.net.last_linear.parameters()

    def forward(self, x):
        return self.net(x)


class SEResNet101(nn.Module):
    def __init__(self, weights_path, pretrained=False, dropout=False, net_cls=None):
        super().__init__()
        self.net = SENet(SEResNetBottleneck, [3,4,23,3], groups=1, reduction=16,
                         dropout_p=None, inplanes=64, input_3x3=False,
                         downsample_kernel_size=1, downsample_padding=0,
                         num_classes=N_CLASSES)
        self.net.avg_pool = AvgPool()
        self.load_state_dict(torch.load(weights_path)['model'])
        #self.net.last_linear = nn.Linear(self.net.last_linear.in_features, num_classes)

    def fresh_params(self):
        return self.net.last_linear.parameters()

    def forward(self, x):
        return self.net(x)


class SEResNet152(nn.Module):
    def __init__(self, weights_path, pretrained=False, dropout=False, net_cls=None):
        super().__init__()
        self.net = SENet(SEResNetBottleneck, [3,8,36,3], groups=1, reduction=16,
                         dropout_p=None, inplanes=64, input_3x3=False,
                         downsample_kernel_size=1, downsample_padding=0,
                         num_classes=N_CLASSES)
        self.net.avg_pool = AvgPool()
        self.load_state_dict(torch.load(weights_path)['model'])
        #self.net.last_linear = nn.Linear(self.net.last_linear.in_features, num_classes)

    def fresh_params(self):
        return self.net.last_linear.parameters()

    def forward(self, x):
        return self.net(x)


class SEResNeXt50_32x4d(nn.Module):
    def __init__(self, weights_path, pretrained=False, dropout=False, net_cls=None):
        super().__init__()
        self.net = SENet(SEResNeXtBottleneck, [3,4,6,3], groups=32, reduction=16,
                         dropout_p=None, inplanes=64, input_3x3=False,
                         downsample_kernel_size=1, downsample_padding=0,
                         num_classes=N_CLASSES)
        self.net.avg_pool = AvgPool()
        self.load_state_dict(torch.load(weights_path)['model'])
        #self.net.last_linear = nn.Linear(self.net.last_linear.in_features, num_classes)

    def fresh_params(self):
        return self.net.last_linear.parameters()

    def forward(self, x):
        return self.net(x)


class SEResNeXt101_32x4d(nn.Module):
    def __init__(self, weights_path, pretrained=False, dropout=False, net_cls=None):
        super().__init__()
        self.net = SENet(SEResNeXtBottleneck, [3,4,23,3], groups=32, reduction=16,
                         dropout_p=None, inplanes=64, input_3x3=False,
                         downsample_kernel_size=1, downsample_padding=0,
                         num_classes=N_CLASSES)
        self.net.avg_pool = AvgPool()
        self.load_state_dict(torch.load(weights_path)['model'])
        #self.net.last_linear = nn.Linear(self.net.last_linear.in_features, num_classes)

    def fresh_params(self):
        return self.net.last_linear.parameters()

    def forward(self, x):
        return self.net(x)


se_resnet50 = partial(SEResNet50, net_cls=None)
se_resnet101 = partial(SEResNet101, net_cls=None)
se_resnet152 = partial(SEResNet152, net_cls=None)
se_resnext50_32x4d = partial(SEResNeXt50_32x4d, net_cls=None)
se_resnext101_32x4d = partial(SEResNeXt101_32x4d, net_cls=None)

##################### Prediction ########################
def predictoin(model, df, mode="test"):
    model.cuda()
    model.eval()
    
    if mode=="test":
        test_dataset = TTADataset('../input/aptos2019-blindness-detection/test_images', df.id_code.values, tta=2)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
        all_predictions, all_ids = [], []
        with torch.no_grad():
            for inputs, ids in test_loader:
                all_ids.append(ids)
                device = "cuda:0"
                inputs = inputs.to(device, dtype=torch.float)
                predictions = model(inputs)
                all_predictions.append(predictions.cpu().numpy())
        all_predictions = np.concatenate(all_predictions)
        all_ids = np.concatenate(all_ids)
        test_preds = pd.DataFrame(data=all_predictions,
                                  index=all_ids,
                                  columns=map(str, range(N_CLASSES)))
        test_preds = test_preds.groupby(level=0).mean()
        return test_preds
        
    else:
        test_dataset = ValidDataset('../input/aptos2019-blindness-detection/train_images', df)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
        all_predictions, all_targets = [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                all_targets.append(targets.numpy().copy())
                device = "cuda:0"
                inputs = inputs.to(device, dtype=torch.float)
                predictions = model(inputs)
                all_predictions.append(predictions.cpu().numpy())
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        
        optR = OptimizedRounder()
        optR.fit(all_predictions, all_targets)
        coefficients = optR.coefficients()
        y_pred = optR.predict(all_predictions, coefficients)

        def get_score(y_pred):
            return quadratic_weighted_kappa(all_targets, y_pred)
        
        qwk = get_score(y_pred)
        #print("QWK: ", qwk)
        return coefficients

    del model, test_dataset, test_loader
    gc.collect();
    
##################### main ########################
def main():
    valid_fold = folds[folds['fold'] == 0].reset_index(drop=True)
    model = se_resnext101_32x4d('../input/aptos-weights1/se_resnext101_256_v42_fold0.pth')
    coefficients = predictoin(model=model, df=valid_fold, mode="valid")
    test_preds = predictoin(model=model, df=sub, mode="test")
    optR = OptimizedRounder()
    y_pred = optR.predict(test_preds, coefficients)
    sub['diagnosis'] = y_pred.astype(int)
    sub.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()
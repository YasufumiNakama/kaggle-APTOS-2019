#===========================================================
# prediction flags
#===========================================================

# ensemble or single prediction
ENSEMBLE = True

# models
SE_RESNEXT101_CAMARO = True
EFFICIENTNET_B5_NAKAMA = True

ENSEMBLE_WEIGHTS = [
    0.5,  # SEResNeXT101 (Camaro)
    0.5,  # EfficientNet-b5 (Nakama)
]

NUM_FOLDS = 5

N_JOBS = 2
SEED = 777


#===========================================================
# prediction params
#===========================================================

SE_RESNEXT101_PARAMS = {
    'batch_size': 64,
    'image_size': 320,
    'num_tta': 3,
    'coefficients': [0.52650521, 1.51066653, 2.43756789, 3.3111235 ],
}

EFFICIENTNET_B5_PARAMS = {
    'batch_size': 32,
    'image_size': 320,
    'num_tta': 4,
    'coefficients': [0.538267, 1.490078, 2.463506, 3.265204],
}


#===========================================================
# imports
#===========================================================

import sys
sys.path.append('../input/efficientnet-pytorch-repository/repository/lukemelas-EfficientNet-PyTorch-e5c8726')
sys.path.append('../input/pytorch-pretrained-models/repository/pretrained-models.pytorch-master')

import gc
import math
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy as sp
from PIL import Image
from sklearn.metrics import cohen_kappa_score

from albumentations import ImageOnlyTransform
from albumentations import Cutout, HorizontalFlip, OneOf, Rotate, VerticalFlip, RandomContrast, IAAAdditiveGaussianNoise
from albumentations import Compose as A_Compose
from albumentations import Normalize as A_Normalize
from albumentations import Resize as A_Resize
from albumentations.pytorch import ToTensor as A_ToTensor

from efficientnet_pytorch import EfficientNet

import pretrainedmodels

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomRotation, RandomHorizontalFlip, RandomVerticalFlip
from torchvision.transforms import Compose as T_Compose
from torchvision.transforms import Normalize as T_Normalize
from torchvision.transforms import Resize as T_Resize
from torchvision.transforms import ToTensor as T_ToTensor


#===========================================================
# utils
#===========================================================

@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')


def init_logger(log_file='inference.log'):
    from logging import getLogger, DEBUG, FileHandler,  Formatter,  StreamHandler
    
    log_format = '%(asctime)s %(levelname)s %(message)s'
    
    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(Formatter(log_format))
    
    file_handler = FileHandler(log_file)
    file_handler.setFormatter(Formatter(log_format))
    
    logger = getLogger('APTOS')
    logger.setLevel(DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    
    return logger

LOG_FILE = 'aptos-inference-ensemble.log'
LOGGER = init_logger(LOG_FILE)


def init_cudnn():
    torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.fastest = True


def seed_torch(seed=777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True


seed_torch(SEED)
init_cudnn()

device = torch.device('cuda')


#===========================================================
# qwk optimizer
#===========================================================

def quadratic_weighted_kappa(y_hat, y):
    return cohen_kappa_score(y_hat, y, weights='quadratic')


class OptimizedRounder():
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

        ll = quadratic_weighted_kappa(y, X_p)
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


#===========================================================
# dataset
#===========================================================

APTOS_DIR = Path('../input/aptos2019-blindness-detection')

ID_COLUMN = 'id_code'
TARGET_COLUMN = 'diagnosis'

APTOS_WEIGHTS_DIR_NAKAMA = Path('../input/aptos-efficientnetb5-finetune-2019test')
APTOS_WEIGHTS_DIR_CAMARO = Path('../input/seresnext-lb0842-models')


class APTOSTestDataset(Dataset):
    def __init__(self, image_dir, file_paths, transform=None, use_torchvision=False):
        self.image_dir = image_dir
        self.file_paths = file_paths
        self.transform = transform
        self.use_torchvision = use_torchvision
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = f'{self.image_dir}/{self.file_paths[idx]}.png'

        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            if self.use_torchvision:
                return self.transform(image)
            else:
                augmented = self.transform(image=image)
                return augmented['image']


#===========================================================
# transforms
#===========================================================

def crop_image1(img, tol=7):
    mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


def crop_image_from_gray(img, tol=7):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray_img > tol

    check_shape = img[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0]
    if (check_shape == 0):
        return img
    else:
        img1=img[:,:,0][np.ix_(mask.any(1), mask.any(0))]
        img2=img[:,:,1][np.ix_(mask.any(1), mask.any(0))]
        img3=img[:,:,2][np.ix_(mask.any(1), mask.any(0))]
        img = np.stack([img1, img2, img3],axis=-1)
    return img


class CropBens():
    def __init__(self, image_size, sigma_x=10):
        self.image_size = image_size
        self.sigma_x = sigma_x

    def __call__(self, img):
        img = crop_image_from_gray(img, tol=7)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), self.sigma_x), -4 ,128)
        return Image.fromarray(img)


def get_transforms_nakama(img_size):
    return A_Compose([
        A_Resize(img_size, img_size),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomContrast(0.5, p=0.5),
        IAAAdditiveGaussianNoise(p=0.25),
        A_Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        A_ToTensor(),
    ])


def get_transforms_camaro(img_size):
    return T_Compose([
        CropBens(img_size),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation((-120, 120)),
        T_ToTensor(),
        T_Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])  


#===========================================================
# Camaro model
#===========================================================

class CustomSENet(nn.Module):
    def __init__(self, model_name='se_resnet50'):
        assert model_name in ('senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d')
        super().__init__()
        
        self.net = pretrainedmodels.__dict__[model_name](pretrained=None)
        n_features = self.net.last_linear.in_features
        
        self.net.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.net.last_linear = nn.Linear(n_features, 1)
        
    def forward(self, x):
        return self.net(x)


#===========================================================
# Nakama model
#===========================================================

class ClassifierModule(nn.Sequential):
    def __init__(self, n_features):
        super().__init__(
            nn.BatchNorm1d(n_features),
            nn.Dropout(0.5),
            nn.Linear(n_features, n_features),
            nn.PReLU(),
            nn.BatchNorm1d(n_features),
            nn.Dropout(0.2),
            nn.Linear(n_features, 1),
        )


class CustomEfficientNet(nn.Module):
    def __init__(self, model_name='efficientnet-b0'):
        assert model_name in ('efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5')
        super().__init__()
        
        self.net = EfficientNet.from_name(model_name)
        n_features = self.net._fc.in_features
        
        self.net._fc = ClassifierModule(n_features)
        
    def forward(self, x):
        return self.net(x)


#===========================================================
# entry point
#===========================================================

def main():
    all_coefficients = []
    
    with timer('Load sample submission csv'):
        submission = pd.read_csv(APTOS_DIR / 'sample_submission.csv')
        test_ids = submission[ID_COLUMN].values

    if SE_RESNEXT101_CAMARO:
        with timer('SEResNeXT101 Camaro'):
            se_resnext101_predictions = []
            
            batch_size = SE_RESNEXT101_PARAMS['batch_size']
            image_size = SE_RESNEXT101_PARAMS['image_size']
            coefficients = SE_RESNEXT101_PARAMS['coefficients']
            all_coefficients.append(coefficients)
            num_tta = SE_RESNEXT101_PARAMS['num_tta']
            LOGGER.debug(f'  coefficients: {coefficients}')
            
            test_dataset = APTOSTestDataset(image_dir=APTOS_DIR / 'test_images',
                                            file_paths=test_ids,
                                            transform=get_transforms_camaro(image_size),
                                            use_torchvision=True)
            test_loader = DataLoader(test_dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=N_JOBS)
            
            model = CustomSENet('se_resnext101_32x4d')
            
            for fold in range(NUM_FOLDS):
                with timer(f'  * fold {fold}'):
                    model.load_state_dict(torch.load(APTOS_WEIGHTS_DIR_CAMARO / f'with_test_lb0849_soft_fold{fold}_best-model.pt')['model'])
                    model.to(device)

                    for param in model.parameters():
                        param.requires_grad = False
                    model.eval()

                    for _ in range(num_tta):
                        test_preds = np.zeros((len(test_dataset)), dtype=np.float32)
                        for i, images in enumerate(test_loader):
                            with torch.no_grad():
                                y_preds = model(images.to(device)).detach()
                            test_preds[i * batch_size: (i+1) * batch_size] = y_preds[:, 0].to('cpu').numpy()

                        se_resnext101_predictions.append(test_preds)
            
            se_resnext101_result = np.mean(se_resnext101_predictions, axis=0)
            
            del se_resnext101_predictions, test_preds
            gc.collect()
    
    if EFFICIENTNET_B5_NAKAMA:
        with timer('EfficientNet-b5 Nakama'):
            efficientnet_b5_predictions = []
            
            batch_size = EFFICIENTNET_B5_PARAMS['batch_size']
            image_size = EFFICIENTNET_B5_PARAMS['image_size']
            coefficients = EFFICIENTNET_B5_PARAMS['coefficients']
            all_coefficients.append(coefficients)
            num_tta = EFFICIENTNET_B5_PARAMS['num_tta']
            LOGGER.debug(f'  coefficients: {coefficients}')
            
            test_dataset = APTOSTestDataset(image_dir=APTOS_DIR / 'test_images',
                                            file_paths=test_ids,
                                            transform=get_transforms_nakama(image_size))
            test_loader = DataLoader(test_dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=N_JOBS)

            model = CustomEfficientNet('efficientnet-b5')
            
            for fold in range(NUM_FOLDS):
                with timer(f'  * fold {fold}'):
                    model.load_state_dict(torch.load(APTOS_WEIGHTS_DIR_NAKAMA / f'efficientnet-b5_fold{fold}_best_pseudo_lb0849_hard.pth'))
                    model.to(device)

                    for param in model.parameters():
                        param.requires_grad = False
                    model.eval()

                    for _ in range(num_tta):
                        test_preds = np.zeros((len(test_dataset)), dtype=np.float32)
                        for i, images in enumerate(test_loader):
                            with torch.no_grad():
                                y_preds = model(images.to(device)).detach()
                            test_preds[i * batch_size: (i+1) * batch_size] = y_preds[:, 0].to('cpu').numpy()

                        efficientnet_b5_predictions.append(test_preds)
            
            efficientnet_b5_result = np.mean(efficientnet_b5_predictions, axis=0)
            
            del efficientnet_b5_predictions, test_preds
            gc.collect()
            

    with timer('Ensemble and create submission file'):
        optimized_rounder = OptimizedRounder()
        
        if ENSEMBLE:
            results, weights = [], []

            if SE_RESNEXT101_CAMARO:
                results.append(se_resnext101_result)
                weights.append(ENSEMBLE_WEIGHTS[0])
            
            if EFFICIENTNET_B5_NAKAMA:
                results.append(efficientnet_b5_result)
                weights.append(ENSEMBLE_WEIGHTS[1])
                
            results = np.average(results, weights=weights, axis=0)
            
            submission[TARGET_COLUMN] = results
            submission.to_csv('soft_labels.csv', index=False)
            
            coefficients = np.average(all_coefficients, weights=weights, axis=0)
            results = optimized_rounder.predict(results, coefficients)
            submission[TARGET_COLUMN] = results.astype(int)
        else:
            if SE_RESNEXT101_CAMARO:
                se_resnext101_result = optimized_rounder.predict(se_resnext101_result, coefficients)
                submission[TARGET_COLUMN] = se_resnext101_result.astype(int)
                    
            if EFFICIENTNET_B5_NAKAMA:
                efficientnet_b5_result = optimized_rounder.predict(efficientnet_b5_result, coefficients)
                submission[TARGET_COLUMN] = efficientnet_b5_result.astype(int)

        submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
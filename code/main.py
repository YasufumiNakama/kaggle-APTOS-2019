import argparse
from itertools import islice
import json
from pathlib import Path
import shutil
import warnings
from typing import Dict

import numpy as np
import pandas as pd
import scipy as sp
from functools import partial
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score
from sklearn.exceptions import UndefinedMetricWarning
import torch
import torch.nn.functional as F
from torch import nn, cuda
from torch.optim import Adam
import tqdm

from . import models
from .dataset import TrainDataset, TTADataset, get_ids, DATA_ROOT
from .transforms import train_transform, test_transform
from .utils import (
    write_event, load_model, mean_df, ThreadingDataLoader as DataLoader,
    ON_KAGGLE)


def quadratic_weighted_kappa(y_hat, y):
    return torch.tensor(cohen_kappa_score(torch.argmax(y_hat, 1), y, weights='quadratic'), device='cuda:0')


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

        ll = metrics.cohen_kappa_score(y, X_p, weights='quadratic')
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


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('mode', choices=['train', 'validate', 'predict_valid', 'predict_test'])
    arg('run_root')
    arg('--model', default='resnet101')
    arg('--loss', default='bce')
    arg('--pretrained', type=int, default=1)
    arg('--batch-size', type=int, default=64)
    arg('--step', type=int, default=1)
    arg('--workers', type=int, default=2 if ON_KAGGLE else 4)
    arg('--lr', type=float, default=1e-4)
    arg('--patience', type=int, default=4)
    arg('--clean', action='store_true')
    arg('--n-epochs', type=int, default=100)
    arg('--epoch-size', type=int)
    arg('--tta', type=int, default=4)
    arg('--use-sample', action='store_true', help='use a sample of the dataset')
    arg('--debug', action='store_true')
    arg('--limit', type=int)
    arg('--fold', type=int, default=0)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    folds = pd.read_csv('folds.csv')
    train_root = DATA_ROOT / ('train_sample' if args.use_sample else 'train')
    if args.use_sample:
        folds = folds[folds['Id'].isin(set(get_ids(train_root)))]
    train_fold = folds[folds['fold'] != args.fold]
    valid_fold = folds[folds['fold'] == args.fold]
    if args.limit:
        train_fold = train_fold[:args.limit]
        valid_fold = valid_fold[:args.limit]

    def make_loader(df: pd.DataFrame, image_transform) -> DataLoader:
        return DataLoader(
            TrainDataset(train_root, df, image_transform, debug=args.debug),
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )

    criterion = nn.MSELoss()

    model = getattr(models, args.model)(num_classes=1, pretrained=args.pretrained)
    use_cuda = cuda.is_available()
    fresh_params = list(model.fresh_params())
    all_params = list(model.parameters())
    if use_cuda:
        model = model.cuda()

    if args.mode == 'train':
        if run_root.exists() and args.clean:
            shutil.rmtree(run_root)
        run_root.mkdir(exist_ok=True, parents=True)
        (run_root / 'params.json').write_text(
            json.dumps(vars(args), indent=4, sort_keys=True))

        train_loader = make_loader(train_fold, train_transform)
        valid_loader = make_loader(valid_fold, test_transform)
        print(f'{len(train_loader.dataset):,} items in train, '
              f'{len(valid_loader.dataset):,} in valid')

        train_kwargs = dict(
            args=args,
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            valid_loader=valid_loader,
            patience=args.patience,
            init_optimizer=lambda params, lr: Adam(params, lr),
            use_cuda=use_cuda,
        )

        if args.pretrained:
            if train(params=fresh_params, n_epochs=1, **train_kwargs):
                train(params=all_params, **train_kwargs)
        else:
            train(params=all_params, **train_kwargs)

    elif args.mode == 'validate':
        valid_loader = make_loader(valid_fold, test_transform)
        load_model(model, run_root / 'model.pth')
        validation(model, criterion, tqdm.tqdm(valid_loader, desc='Validation'),
                   use_cuda=use_cuda)


def train(args, model: nn.Module, criterion, *, params,
          train_loader, valid_loader, init_optimizer, use_cuda,
          n_epochs=None, patience=2, max_lr_changes=2) -> bool:
    lr = args.lr
    n_epochs = n_epochs or args.n_epochs
    params = list(params)
    optimizer = init_optimizer(params, lr)

    run_root = Path(args.run_root)
    model_path = run_root / 'model.pth'
    best_model_path = run_root / 'best-model.pth'
    if model_path.exists():
        state = load_model(model, model_path)
        epoch = state['epoch']
        step = state['step']
        best_valid_loss = state['best_valid_loss']
        best_valid_qwk = state['best_valid_qwk']
    else:
        epoch = 1
        step = 0
        best_valid_loss = float('inf')
        best_valid_qwk = 0
    lr_changes = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        'best_valid_loss': best_valid_loss,
        'best_valid_QWK': best_valid_qwk
    }, str(model_path))

    report_each = 10000
    log = run_root.joinpath('train.log').open('at', encoding='utf8')
    valid_losses = []
    lr_reset_epoch = epoch
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        tq = tqdm.tqdm(total=(args.epoch_size or
                              len(train_loader) * args.batch_size))
        tq.set_description(f'Epoch {epoch}, lr {lr}')
        losses = []
        tl = train_loader
        if args.epoch_size:
            tl = islice(tl, args.epoch_size // args.batch_size)
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = _reduce_loss(criterion(outputs, targets))
                batch_size = inputs.size(0)
                (batch_size * loss).backward()
                if (i + 1) % args.step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    step += 1
                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss=f'{mean_loss:.3f}')
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
            write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader, use_cuda)
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_qwk = valid_metrics['valid_qwk']
            valid_losses.append(valid_loss)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                # shutil.copy(str(model_path), str(best_model_path))
            if best_valid_qwk < valid_qwk:
                best_valid_qwk = valid_qwk
                shutil.copy(str(model_path), str(best_model_path))
                print(f'Saved best model at Epoch {epoch}, best_valid_qwk {best_valid_qwk}')
            elif (patience and epoch - lr_reset_epoch > patience and
                  min(valid_losses[-patience:]) > best_valid_loss):
                # "patience" epochs without improvement
                lr_changes += 1
                if lr_changes > max_lr_changes:
                    break
                lr /= 5
                print(f'lr updated to {lr}')
                lr_reset_epoch = epoch
                optimizer = init_optimizer(params, lr)
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return False
    return True


def validation(
        model: nn.Module, criterion, valid_loader, use_cuda,
        ) -> Dict[str, float]:
    model.eval()
    all_losses, all_predictions, all_targets = [], [], []
    with torch.no_grad():
        for inputs, targets in valid_loader:
            all_targets.append(targets.numpy().copy())
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            all_losses.append(_reduce_loss(loss).item())
            predictions = torch.sigmoid(outputs)
            all_predictions.append(predictions.cpu().numpy())
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    optR = OptimizedRounder()
    optR.fit(all_predictions, all_targets)
    coefficients = optR.coefficients()
    y_pred = optR.predict(all_predictions, coefficients)

    def get_score(y_pred):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UndefinedMetricWarning)
            return quadratic_weighted_kappa(all_targets, y_pred)

    metrics = {}
    metrics['valid_loss'] = np.mean(all_losses)
    metrics['valid_qwk'] = get_score(y_pred)

    return metrics


def _reduce_loss(loss):
    return loss.sum() / loss.shape[0]


if __name__ == '__main__':
    main()

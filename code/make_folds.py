import pandas as pd
import argparse
from sklearn.model_selection import StratifiedKFold

from .dataset import DATA_ROOT


def make_folds(n_folds: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(DATA_ROOT / 'train.csv')
    df['fold'] = 0
    skf = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)
    for n, (train_index, val_index) in enumerate(skf.split(df, df['diagnosis'])):
        df.loc[val_index, 'fold'] = int(n)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    df = make_folds(n_folds=args.n_folds, seed=args.seed)
    df.to_csv('folds.csv', index=None)


if __name__ == '__main__':
    main()

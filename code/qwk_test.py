from sklearn.metrics import cohen_kappa_score


def quadratic_weighted_kappa(y_hat, y):
    return cohen_kappa_score(y_hat, y, weights='quadratic')


if __name__=="__main__":
    y_train = [0, 1, 2, 3, 4, 1, 1, 1, 2, 3]
    y_pred = [1, 3, 2, 3, 4, 1, 1, 1, 4, 1]
    score = quadratic_weighted_kappa(y_pred, y_train)
    print(score)

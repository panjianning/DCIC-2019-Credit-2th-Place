# Gotcha
# 2019

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import time
from contextlib import contextmanager
from sklearn.model_selection import StratifiedKFold, KFold
import lightgbm as lgb
import xgboost as xgb
import catboost as ctb
from sklearn.linear_model import HuberRegressor


@contextmanager
def timer(name):
    t = time.time()
    yield
    print('[%s] done in %.2f seconds' % (name, time.time() - t))


def make_dir(path):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        return True
    return False


def score(y_true, y_pred):
    return 1 / (1 + mean_absolute_error(y_true, y_pred))


################################# custom loss functions and metrics ##################################


def mae_loss(y_true, y_pred):
    x = y_pred - y_true
    grad = np.sign(x)
    hess = np.zeros_like(x)
    return grad, hess


def mse_loss(y_true, y_pred):
    x = y_pred - y_true
    grad = x
    hess = np.ones_like(x)
    return grad, hess


def fair_loss(y_true, y_pred, faic_c):
    x = y_pred - y_true
    grad = faic_c * x / (np.abs(x) + faic_c)
    hess = faic_c ** 2 / (np.abs(x) + faic_c) ** 2
    return grad, hess


def pseudo_huber_loss(y_true, y_pred, delta):
    x = y_pred - y_true
    scale = 1 + (x / delta) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = x / scale_sqrt
    hess = 1 / (scale * scale_sqrt)
    return grad, hess


def mae_fair_loss(y_true, y_pred, fair_c):
    grad_mae, hess_mae = mae_loss(y_true, y_pred)
    grad_fair, hess_fair = fair_loss(y_true, y_pred, fair_c)
    grad = 0.5 * grad_mae + 0.5 * grad_fair
    hess = 0.5 * hess_mae + 0.5 * hess_fair
    return grad, hess


def mae_huber_loss(y_true, y_pred, huber_delta):
    grad_mae, hess_mae = mae_loss(y_true, y_pred)
    grad_huber, hess_huber = pseudo_huber_loss(y_true, y_pred, huber_delta)
    grad = 0.5 * grad_mae + 0.5 * grad_huber
    hess = 0.5 * hess_mae + 0.5 * hess_huber
    return grad, hess


def fair_huber_loss(y_true, y_pred, fair_c, huber_delta):
    grad_fair, hess_fair = fair_loss(y_true, y_pred, fair_c)
    grad_huber, hess_huber = pseudo_huber_loss(y_true, y_pred, huber_delta)
    grad = 0.5 * grad_fair + 0.5 * grad_huber
    hess = 0.5 * hess_fair + 0.5 * hess_huber
    return grad, hess


def pseudo_huber_metric(y_true, y_pred, delta):
    x = y_pred - y_true
    return 'huber%d' % delta, np.mean(delta ** 2 * (np.sqrt((x / delta) ** 2 + 1) - 1)), False


def mae_fair_metric(y_true, y_pred, fair_c):
    x = y_pred - y_true
    tmp = np.abs(x) / fair_c
    fair = fair_c ** 2 * (tmp - np.log(tmp + 1))
    mae = np.abs(x)
    return 'mae_fair%d' % fair_c, np.mean(0.5 * fair + 0.5 * mae), False


def mae_huber_metric(y_true, y_pred, huber_delta):
    x = y_pred - y_true
    huber = pseudo_huber_metric(y_true, y_pred, huber_delta)[1]
    mae = np.abs(x)
    return 'mae_huber%d' % huber_delta, np.mean(0.5 * huber + 0.5 * mae), False


def fair_huber_metric(y_true, y_pred, fair_c, huber_delta):
    x = y_pred - y_true
    huber = pseudo_huber_metric(y_true, y_pred, huber_delta)[1]
    tmp = np.abs(x) / fair_c
    fair = fair_c ** 2 * (tmp - np.log(tmp + 1))
    return 'mae_fair%d_huber%d' % (fair_c, huber_delta), np.mean(0.5 * huber + 0.5 * fair), False


def mae_metric(y_true, y_pred):
    return 'mae', np.mean(np.abs(y_true - y_pred)), False


def mse_metric(y_true, y_pred):
    return 'mse', np.mean((y_true - y_pred) ** 2), False


def fn_identity(x):
    return x


def kf_lgbm(x, y, x_test, output_dir, name="mae_fair30",
            n_folds=10, stratify=True, split_seed=8888,
            fn_reverse_transform=fn_identity,
            boosting_type="gbdt", base_score=None, sample_weight=None,
            objective="mae_fair", eval_metric="mae_fair",
            fair_c=30, huber_delta=20, n_estimators=3000, learning_rate=0.01,
            num_leaves=31, max_depth=5, max_bin=255, reg_alpha=2.0,
            reg_lambda=5.0, colsample_bytree=0.5, subsample=0.8,
            subsample_freq=2, min_child_samples=20, min_split_gain=1,
            categorical_feature=['用户话费敏感度'], early_stopping_rounds=80, verbose=200,
            **kwargs):
    if objective == "mae_fair":
        def fn_objective(y_true, y_pred):
            return mae_fair_loss(y_true, y_pred, fair_c)
    elif objective == "mae_huber":
        def fn_objective(y_true, y_pred):
            return mae_huber_loss(y_true, y_pred, huber_delta)
    elif objective == "fair_huber":
        def fn_objective(y_true, y_pred):
            return fair_huber_loss(y_true, y_pred, fair_c, huber_delta)
    else:
        fn_objective = objective
    objective = fn_objective

    num_training_samples = x.shape[0]
    num_testing_samples = x_test.shape[0]

    test_pred = np.zeros(num_testing_samples)
    oof_train_pred = np.zeros(num_training_samples)
    scores = []
    fold_idx = 1

    if stratify:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=split_seed)
        idx = y.argsort()
        y_lab = np.repeat(list(range(num_training_samples // 20)), 20)
        y_lab = np.asarray(sorted(list(zip(idx, y_lab))))[:, -1].astype(np.int32)
        splits = kf.split(x, y_lab)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=split_seed)
        splits = kf.split(x)

    model = None
    for train_idx, valid_idx in splits:
        print()
        print("=" * 50, "Fold %d" % fold_idx, "=" * 50)
        fold_idx += 1
        if not isinstance(x, pd.DataFrame):
            x_train, x_valid = x[train_idx], x[valid_idx]
        else:
            x_train, x_valid = x.iloc[train_idx], x.iloc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        if sample_weight is not None:
            sample_weight_train = sample_weight[train_idx]
        else:
            sample_weight_train = None

        if base_score is not None:
            init_score = base_score
        else:
            init_score = np.median(y_train)

        if boosting_type == 'rf':
            print('boosting_type is rf, ignore base_score')
            init_score = 0

        if eval_metric == "mae_fair":
            def fn_eval_metric(y_true, y_pred):
                y_true = fn_reverse_transform(y_true + init_score)
                y_pred = fn_reverse_transform(y_pred + init_score)
                return mae_fair_metric(y_true, y_pred, fair_c)
        elif eval_metric == "mae_huber":
            def fn_eval_metric(y_true, y_pred):
                y_true = fn_reverse_transform(y_true + init_score)
                y_pred = fn_reverse_transform(y_pred + init_score)
                return mae_huber_metric(y_true, y_pred, huber_delta)
        elif eval_metric == "fair_huber":
            def fn_eval_metric(y_true, y_pred):
                y_true = fn_reverse_transform(y_true + init_score)
                y_pred = fn_reverse_transform(y_pred + init_score)
                return fair_huber_metric(y_true, y_pred, fair_c, huber_delta)
        elif eval_metric == "mae":
            def fn_eval_metric(y_true, y_pred):
                y_true = fn_reverse_transform(y_true + init_score)
                y_pred = fn_reverse_transform(y_pred + init_score)
                return mae_metric(y_true, y_pred)
        else:
            fn_eval_metric = eval_metric
        eval_metric = fn_eval_metric

        model = lgb.LGBMRegressor(boosting_type=boosting_type,
                                  learning_rate=learning_rate,
                                  num_leaves=num_leaves,
                                  max_depth=max_depth,
                                  n_estimators=n_estimators,
                                  max_bin=max_bin,
                                  objective=objective,
                                  reg_alpha=reg_alpha,
                                  reg_lambda=reg_lambda,
                                  colsample_bytree=colsample_bytree,
                                  subsample=subsample,
                                  subsample_freq=subsample_freq,
                                  min_child_samples=min_child_samples,
                                  min_split_gain=min_split_gain,
                                  metric=['mae'], **kwargs)

        init_score_ = None if boosting_type == 'rf' else np.ones_like(y_train) * init_score

        model.fit(x_train, y_train, eval_set=[(x_train, y_train - init_score),
                                              (x_valid, y_valid - init_score)],
                  eval_names=['train', 'test'],
                  sample_weight=sample_weight_train,
                  eval_metric=eval_metric,
                  verbose=verbose, early_stopping_rounds=early_stopping_rounds,
                  categorical_feature=categorical_feature,
                  init_score=init_score_)

        val_pred = model.predict(x_valid, num_iteration=model.best_iteration_) + init_score
        val_pred = fn_reverse_transform(val_pred)
        oof_train_pred[valid_idx] = val_pred

        test_pred_fold = model.predict(x_test, num_iteration=model.best_iteration_) + init_score
        test_pred_fold = fn_reverse_transform(test_pred_fold)
        test_pred += test_pred_fold / n_folds

        scores.append(score(val_pred, fn_reverse_transform(y_valid)))

    make_dir(output_dir + '/')
    np.save(os.path.join(output_dir, 'val.%s.npy' % name), oof_train_pred)
    np.save(os.path.join(output_dir, 'test.%s.npy' % name), test_pred)

    print("=" * 100)
    print('\t'.join(map(str, scores)))
    print('min score: %.6f' % np.min(scores))
    print('max score: %.6f' % np.max(scores))
    print('median score: %.6f' % np.median(scores))
    print('mean score: %.6f' % np.mean(scores))
    print(test_pred[:10])
    return model


def kf_xgbm(x, y, x_test, output_dir, name="mae_fair30",
            n_folds=10, stratify=True, split_seed=8888,
            fn_reverse_transform=fn_identity,
            fair_c=30, huber_delta=20,
            objective="mae_fair", eval_metric="mae_fair",
            n_estimators=3000, learning_rate=0.01,
            max_depth=5, reg_alpha=2.0, reg_lambda=5.0,
            colsample_bytree=0.5, subsample=0.8,
            max_leaves=31, min_child_weight=20,
            tree_method="hist", grow_policy="depthwise",
            base_score=None, verbose=200, early_stopping_rounds=80,
            **kwargs):
    if objective == "mae_fair":
        def fn_objective(y_true, y_pred):
            return mae_fair_loss(y_true, y_pred, fair_c)
    elif objective == "mae_huber":
        def fn_objective(y_true, y_pred):
            return mae_huber_loss(y_true, y_pred, huber_delta)
    elif objective == "fair_huber":
        def fn_objective(y_true, y_pred):
            return fair_huber_loss(y_true, y_pred, fair_c, huber_delta)
    else:
        fn_objective = objective
    objective = fn_objective

    num_training_samples = x.shape[0]
    num_testing_samples = x_test.shape[0]

    test_pred = np.zeros(num_testing_samples)
    oof_train_pred = np.zeros(num_training_samples)
    scores = []
    fold_idx = 1

    if stratify:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=split_seed)
        idx = y.argsort()
        y_lab = np.repeat(list(range(num_training_samples // 20)), 20)
        y_lab = np.asarray(sorted(list(zip(idx, y_lab))))[:, -1].astype(np.int32)
        splits = kf.split(x, y_lab)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=split_seed)
        splits = kf.split(x)

    model = None
    for train_idx, valid_idx in splits:
        print()
        print("=" * 50, "Fold %d" % fold_idx, "=" * 50)
        fold_idx += 1
        if not isinstance(x, pd.DataFrame):
            x_train, x_valid = x[train_idx], x[valid_idx]
        else:
            x_train, x_valid = x.iloc[train_idx], x.iloc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        if base_score is not None:
            init_score = base_score
        else:
            init_score = np.median(y_train)

        if eval_metric == "mae_fair":
            def fn_eval_metric(y_pred, y_true):
                y_true = fn_reverse_transform(y_true.get_label())
                y_pred = fn_reverse_transform(y_pred)
                return mae_fair_metric(y_true, y_pred, fair_c)[:2]
        elif eval_metric == "mae_huber":
            def fn_eval_metric(y_pred, y_true):
                y_true = fn_reverse_transform(y_true.get_label())
                y_pred = fn_reverse_transform(y_pred)
                return mae_huber_metric(y_true, y_pred, huber_delta)[:2]
        elif eval_metric == "fair_huber":
            def fn_eval_metric(y_pred, y_true):
                y_true = fn_reverse_transform(y_true.get_label())
                y_pred = fn_reverse_transform(y_pred)
                return fair_huber_metric(y_true, y_pred, fair_c, huber_delta)[:2]
        elif eval_metric == "mae":
            def fn_eval_metric(y_pred, y_true):
                y_true = fn_reverse_transform(y_true.get_label())
                y_pred = fn_reverse_transform(y_pred)
                return mae_metric(y_true, y_pred)[:2]
        else:
            fn_eval_metric = eval_metric
        eval_metric = fn_eval_metric

        model = xgb.XGBRegressor(learning_rate=learning_rate,
                                 max_leaves=max_leaves,
                                 max_depth=max_depth,
                                 n_estimators=n_estimators,
                                 objective=objective,
                                 reg_alpha=reg_alpha,
                                 reg_lambda=reg_lambda,
                                 colsample_bytree=colsample_bytree,
                                 subsample=subsample,
                                 tree_method=tree_method,
                                 min_child_weight=min_child_weight,
                                 base_score=init_score,
                                 eval_metric='mae',
                                 grow_policy=grow_policy,
                                 **kwargs)

        model.fit(x_train, y_train, eval_set=[(x_train, y_train),
                                              (x_valid, y_valid)],
                  eval_metric=eval_metric, verbose=verbose,
                  early_stopping_rounds=early_stopping_rounds)

        val_pred = model.predict(x_valid, ntree_limit=model.best_iteration)
        val_pred = fn_reverse_transform(val_pred)
        oof_train_pred[valid_idx] = val_pred

        test_pred_fold = model.predict(x_test, ntree_limit=model.best_iteration)
        test_pred_fold = fn_reverse_transform(test_pred_fold)
        test_pred += test_pred_fold / n_folds

        scores.append(score(val_pred, fn_reverse_transform(y_valid)))

    make_dir(output_dir + '/')
    np.save(os.path.join(output_dir, 'val.%s.npy' % name), oof_train_pred)
    np.save(os.path.join(output_dir, 'test.%s.npy' % name), test_pred)

    print("=" * 100)
    print('\t'.join(map(str, scores)))
    print('min score: %.6f' % np.min(scores))
    print('max score: %.6f' % np.max(scores))
    print('median score: %.6f' % np.median(scores))
    print('mean score: %.6f' % np.mean(scores))
    print(test_pred[:10])
    return model


def kf_ctbm(x, y, x_test, output_dir, name="ctb",
            n_folds=10, stratify=True, split_seed=8888,
            fn_reverse_transform=fn_identity,
            cat_features_idx=None,
            verbose=200, early_stopping_rounds=80,
            **kwargs):
    num_training_samples = x.shape[0]
    num_testing_samples = x_test.shape[0]

    test_pred = np.zeros(num_testing_samples)
    oof_train_pred = np.zeros(num_training_samples)
    scores = []
    fold_idx = 1

    if stratify:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=split_seed)
        idx = y.argsort()
        y_lab = np.repeat(list(range(num_training_samples // 20)), 20)
        y_lab = np.asarray(sorted(list(zip(idx, y_lab))))[:, -1].astype(np.int32)
        splits = kf.split(x, y_lab)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=split_seed)
        splits = kf.split(x)

    model = None
    for train_idx, valid_idx in splits:
        print()
        print("=" * 50, "Fold %d" % fold_idx, "=" * 50)
        fold_idx += 1
        if not isinstance(x, pd.DataFrame):
            x_train, x_valid = x[train_idx], x[valid_idx]
        else:
            x_train, x_valid = x.iloc[train_idx], x.iloc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        model = ctb.CatBoostRegressor(**kwargs)
        model.fit(x_train, y_train,
                  eval_set=[(x_valid, y_valid)],
                  cat_features=cat_features_idx,
                  verbose=verbose,
                  early_stopping_rounds=early_stopping_rounds,
                  use_best_model=True)

        val_pred = model.predict(x_valid)
        val_pred = fn_reverse_transform(val_pred)
        oof_train_pred[valid_idx] = val_pred

        test_pred_fold = model.predict(x_test)
        test_pred_fold = fn_reverse_transform(test_pred_fold)
        test_pred += test_pred_fold / n_folds

        scores.append(score(val_pred, fn_reverse_transform(y_valid)))

    make_dir(output_dir + '/')
    np.save(os.path.join(output_dir, 'val.%s.npy' % name), oof_train_pred)
    np.save(os.path.join(output_dir, 'test.%s.npy' % name), test_pred)

    print("=" * 100)
    print('\t'.join(map(str, scores)))
    print('min score: %.6f' % np.min(scores))
    print('max score: %.6f' % np.max(scores))
    print('median score: %.6f' % np.median(scores))
    print('mean score: %.6f' % np.mean(scores))
    print(test_pred[:10])
    return model


def kf_sklearn(x, y, x_test, output_dir, model_class=HuberRegressor,
               fn_reverse_transform=fn_identity,
               name="huber", n_folds=10, split_seed=8888, stratify=True,
               **kwargs):
    num_training_samples = x.shape[0]
    num_testing_samples = x_test.shape[0]

    test_pred = np.zeros(num_testing_samples)
    oof_train_pred = np.zeros(num_training_samples)
    scores = []
    fold_idx = 1

    if stratify:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=split_seed)
        idx = y.argsort()
        y_lab = np.repeat(list(range(num_training_samples // 20)), 20)
        y_lab = np.asarray(sorted(list(zip(idx, y_lab))))[:, -1].astype(np.int32)
        splits = kf.split(x, y_lab)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=split_seed)
        splits = kf.split(x)

    model = None
    for train_idx, valid_idx in splits:
        print()
        print("=" * 50, "Fold %d" % fold_idx, "=" * 50)
        fold_idx += 1
        if not isinstance(x, pd.DataFrame):
            x_train, x_valid = x[train_idx], x[valid_idx]
        else:
            x_train, x_valid = x.iloc[train_idx], x.iloc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        model = model_class(**kwargs)
        model.fit(x_train, y_train)

        val_pred = model.predict(x_valid)
        val_pred = fn_reverse_transform(val_pred)
        oof_train_pred[valid_idx] = val_pred

        test_pred_fold = model.predict(x_test)
        test_pred_fold = fn_reverse_transform(test_pred_fold)
        test_pred += test_pred_fold / n_folds

        scores.append(score(val_pred, fn_reverse_transform(y_valid)))

    make_dir(output_dir + '/')
    np.save(os.path.join(output_dir, 'val.%s.npy' % name), oof_train_pred)
    np.save(os.path.join(output_dir, 'test.%s.npy' % name), test_pred)

    print("=" * 100)
    print('\t'.join(map(str, scores)))
    print('min score: %.6f' % np.min(scores))
    print('max score: %.6f' % np.max(scores))
    print('median score: %.6f' % np.median(scores))
    print('mean score: %.6f' % np.mean(scores))
    print(test_pred[:10])
    return model

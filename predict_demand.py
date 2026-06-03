```python
import numpy as np
import pandas as pd
import warnings

from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')


# =========================
# LOAD DATA
# =========================
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# =========================
# TIME PROCESSING
# =========================
def pts(df):
    s = df['timestamp'].astype(str).str.strip().str.split(':', expand=True)

    h = pd.to_numeric(s[0], errors='coerce').fillna(0).astype(int)
    m = pd.to_numeric(s[1], errors='coerce').fillna(0).astype(int)

    ts = h * 60 + m

    return ts, ts // 15


for df in [train, test]:
    df['ts_min'], df['time_slot'] = pts(df)


# =========================
# SPLIT DAYS
# =========================
d48 = train[train['day'] == 48].copy().reset_index(drop=True)
d49t = train[train['day'] == 49].copy().reset_index(drop=True)


# =========================
# LAG FEATURES
# =========================
lm = d48.set_index(['geohash', 'time_slot'])['demand']
gm = d48['demand'].mean()

BACK = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24]
FWD = [1, 2, 3, 4, 5, 6, 8]

# Backward lags
for lag in BACK:
    for df in [d48, d49t, test]:
        df[f'lm{lag}'] = df.apply(
            lambda r: lm.get((r['geohash'], r['time_slot'] - lag), np.nan),
            axis=1
        )

# Forward lags
for lag in FWD:

    for df in [d49t, test]:
        df[f'lp{lag}'] = df.apply(
            lambda r: lm.get((r['geohash'], r['time_slot'] + lag), np.nan),
            axis=1
        )

    d48[f'lp{lag}'] = np.nan


# =========================
# GEOHASH STATISTICS
# =========================
fg = (
    d48.groupby('geohash')['demand']
    .agg(
        g_mu='mean',
        g_sd='std',
        g_med='median',
        g_p25=lambda x: x.quantile(0.25),
        g_p75=lambda x: x.quantile(0.75)
    )
    .reset_index()
)

fg['g_iqr'] = fg['g_p75'] - fg['g_p25']

gc = [c for c in fg.columns if c != 'geohash']


# =========================
# OOF GEOHASH FEATURES
# =========================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for c in gc:
    d48[c] = np.nan

for ti, vi in kf.split(d48):

    fs = (
        d48.iloc[ti]
        .groupby('geohash')['demand']
        .agg(
            g_mu='mean',
            g_sd='std',
            g_med='median',
            g_p25=lambda x: x.quantile(0.25),
            g_p75=lambda x: x.quantile(0.75)
        )
        .reset_index()
    )

    fs['g_iqr'] = fs['g_p75'] - fs['g_p25']

    va = d48.iloc[vi][['geohash']].merge(
        fs,
        on='geohash',
        how='left'
    )

    for c in gc:
        d48.loc[d48.index[vi], c] = va[c].values

for c in gc:
    d48[c] = d48[c].fillna(gm)


# =========================
# APPLY GEOHASH FEATURES
# =========================
for df in [d49t, test]:

    df.drop(columns=gc, errors='ignore', inplace=True)

    tmp = df.merge(fg, on='geohash', how='left')

    for c in gc:
        df[c] = tmp[c].fillna(gm).values


# =========================
# GEOHASH + TIMESLOT FEATURES
# =========================
fgts = (
    d48.groupby(['geohash', 'time_slot'])['demand']
    .agg(
        gts_mu='mean',
        gts_sd='std'
    )
    .reset_index()
)

for c in ['gts_mu', 'gts_sd']:
    d48[c] = np.nan

for ti, vi in kf.split(d48):

    fgt = (
        d48.iloc[ti]
        .groupby(['geohash', 'time_slot'])['demand']
        .agg(
            gts_mu='mean',
            gts_sd='std'
        )
        .reset_index()
    )

    va = d48.iloc[vi][['geohash', 'time_slot']].merge(
        fgt,
        on=['geohash', 'time_slot'],
        how='left'
    )

    d48.loc[d48.index[vi], 'gts_mu'] = va['gts_mu'].values
    d48.loc[d48.index[vi], 'gts_sd'] = va['gts_sd'].values

d48['gts_mu'] = d48['gts_mu'].fillna(d48['g_mu'])
d48['gts_sd'] = d48['gts_sd'].fillna(d48['g_sd'])


# =========================
# APPLY GTS FEATURES
# =========================
for df in [d49t, test]:

    df.drop(columns=['gts_mu', 'gts_sd'], errors='ignore', inplace=True)

    tmp = df.merge(
        fgts,
        on=['geohash', 'time_slot'],
        how='left'
    )

    df['gts_mu'] = tmp['gts_mu'].fillna(df['g_mu']).values
    df['gts_sd'] = tmp['gts_sd'].fillna(df['g_sd']).values


# =========================
# FEATURE ENGINEERING
# =========================
gt = train['Temperature'].median()

wx = {
    'Sunny': 0,
    'Rainy': 1,
    'Foggy': 2,
    'Snowy': 3
}

rd = {
    'Residential': 0,
    'Street': 1,
    'Highway': 2
}

for df in [d48, d49t, test]:

    df['road_enc'] = df['RoadType'].map(rd).fillna(0).astype(int)

    df['wx_enc'] = df['Weather'].map(wx).fillna(0).astype(int)

    df['lv'] = (df['LargeVehicles'] == 'Allowed').astype(int)

    df['lm_f'] = (df['Landmarks'] == 'Yes').astype(int)

    df['temp'] = df['Temperature'].fillna(gt)

    df['temp2'] = df['temp'] ** 2

    df['is_hw'] = (df['NumberofLanes'] >= 4).astype(int)

    df['lanes2'] = df['NumberofLanes'] ** 2

    df['ts_sin'] = np.sin(2 * np.pi * df['time_slot'] / 96)

    df['ts_cos'] = np.cos(2 * np.pi * df['time_slot'] / 96)

    df['hr'] = df['ts_min'] // 60

    df['is_mpk'] = df['hr'].between(7, 9).astype(int)

    df['is_epk'] = df['hr'].between(17, 19).astype(int)

    df['ratio1'] = df['gts_mu'] / (df['g_mu'] + 1e-8)


# =========================
# FEATURE LIST
# =========================
FEATS = (
    [f'lm{l}' for l in BACK] +
    [f'lp{l}' for l in FWD] +
    gc +
    [
        'gts_mu',
        'gts_sd',
        'road_enc',
        'wx_enc',
        'lv',
        'lm_f',
        'temp',
        'temp2',
        'is_hw',
        'lanes2',
        'time_slot',
        'ts_sin',
        'ts_cos',
        'hr',
        'is_mpk',
        'is_epk',
        'ratio1',
        'NumberofLanes'
    ]
)


# =========================
# TRAIN DATA
# =========================
atr = pd.concat([d48, d49t], ignore_index=True)

X = atr[FEATS].astype(float).fillna(-999)
y = atr['demand'].astype(float)

XT = test[FEATS].astype(float).fillna(-999)

print(f'X = {X.shape}')


# =========================
# MODEL SETUP
# =========================
kf2 = KFold(n_splits=5, shuffle=True, random_state=42)

olf = np.zeros(len(X))
prl = np.zeros(len(XT))

oxb = np.zeros(len(X))
prx = np.zeros(len(XT))


# LightGBM Params
lp = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.03,
    'num_leaves': 255,
    'min_child_samples': 5,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'reg_alpha': 0.05,
    'reg_lambda': 0.1,
    'n_jobs': 4,
    'verbose': -1,
    'seed': 42
}

# XGBoost Params
xp = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.05,
    'max_depth': 8,
    'min_child_weight': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.05,
    'reg_lambda': 1.0,
    'n_jobs': 4,
    'seed': 42,
    'verbosity': 0
}


# =========================
# TRAINING LOOP
# =========================
for fold, (ti, vi) in enumerate(kf2.split(X), 1):

    Xtr = X.iloc[ti]
    Xva = X.iloc[vi]

    ytr = y.iloc[ti]
    yva = y.iloc[vi]

    # ---------------------
    # LightGBM
    # ---------------------
    dt = lgb.Dataset(Xtr, ytr)
    dv = lgb.Dataset(Xva, yva, reference=dt)

    ml = lgb.train(
        lp,
        dt,
        num_boost_round=2000,
        valid_sets=[dv],
        callbacks=[
            lgb.early_stopping(150, verbose=False),
            lgb.log_evaluation(99999)
        ]
    )

    olf[vi] = ml.predict(
        Xva,
        num_iteration=ml.best_iteration
    )

    prl += (
        ml.predict(
            XT,
            num_iteration=ml.best_iteration
        ) / 5
    )

    # ---------------------
    # XGBoost
    # ---------------------
    dtr = xgb.DMatrix(Xtr, ytr)
    dva = xgb.DMatrix(Xva, yva)

    mx = xgb.train(
        xp,
        dtr,
        num_boost_round=2000,
        evals=[(dva, 'val')],
        early_stopping_rounds=100,
        verbose_eval=False
    )

    oxb[vi] = mx.predict(dva)

    prx += (
        mx.predict(xgb.DMatrix(XT)) / 5
    )

    print(
        f'F{fold}: '
        f'LGB = {r2_score(yva, olf[vi]):.4f}  '
        f'XGB = {r2_score(yva, oxb[vi]):.4f}  '
        f'iter_l = {ml.best_iteration}  '
        f'iter_x = {mx.best_iteration}'
    )


# =========================
# OOF SCORES
# =========================
lr2 = r2_score(y, olf)
xr2 = r2_score(y, oxb)

print(f'LGB OOF : {lr2:.4f}')
print(f'XGB OOF : {xr2:.4f}')


# =========================
# BLENDING
# =========================
bw = -1
br = -1

for w in np.arange(0.2, 1.0, 0.02):

    r2 = r2_score(
        y,
        w * olf + (1 - w) * oxb
    )

    if r2 > br:
        br = r2
        bw = w

print(
    f'Best blend w_LGB = {bw:.2f}  '
    f'Blend OOF R2 = {br:.4f}  '
    f'Score = {max(0, 100 * br):.2f}'
)


# =========================
# FINAL PREDICTIONS
# =========================
fp = np.clip(
    bw * prl + (1 - bw) * prx,
    0,
    1.0
)

sub = pd.DataFrame({
    'Index': test['Index'],
    'demand': fp
})

sub.to_csv('submission.csv', index=False)

print(f'Saved. {sub.shape}')
```

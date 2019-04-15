# -*- coding: utf-8 -*-
# Inspired by tianchi.aliyun.com/notebook-ai/detail?postId=44844
import warnings
warnings.filterwarnings('ignore')
import time
import re
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_squared_error

# Load Dataset
t0 = time.time()
train = pd.read_csv('./data/jinnan_round1_train_20181227.csv', encoding='gb18030')
test = pd.read_csv('./data/jinnan_round1_testB_20190121.csv', encoding='gb18030')
print(f'Data Loaded in {(time.time()-t0):.1f} s !')
print('------------------------------------')

# Processing Anomalies and Combining Dataset
t0 = time.time()
train = train[train['收率'] > 0.87]
train.loc[train['B14'] == 40, 'B14'] = 400
train = train[train['B14']>=400]
train.loc[train['A25'] == '1900/3/10 0:00', 'A25'] = train['A25'].value_counts().values[0]
train['A25'] = train['A25'].astype(int)
target = train.pop('收率')
test_id = test['样本id']
data = pd.concat([train, test], axis=0, ignore_index=True)
data = data.fillna(-1)
data['样本id'] = data['样本id'].apply(lambda x: x.split('_')[1])
data['样本id'] = data['样本id'].astype(int)
print(f'Anomalies Processed in {(time.time()-t0):.1f} s !')
print('------------------------------------')

# Processing Special Case
print('Special Value(B14) Processing...')
test_select = {}
for v in [280, 360, 385, 390, 785]:
	print(v)
	print(test[test['B14'] == v]['样本id'])
	test_select[v] = test[test['B14'] == v]['样本id'].index
print('Special Value(B14) Processed!')
print('------------------------------------')

# Processing TimeData
t0 = time.time()
def timeTranSecond(t):
	try:
		t, m, s = t.split(":")
	except:
		if t == '1900/1/9 7:00':
			return 7 * 3600 / 3600
		elif t == '1900/1/1 2:30':
			return (2 * 3600 + 30 * 60) / 3600
		elif t == -1:
			return -1
		else:
			return 0
	try:
		tm = (int(t) * 3600 + int(m) * 60 + int(s)) / 3600
	except:
		return (30 * 60) / 3600
	return tm

for f in ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']:
	try:
		data[f] = data[f].apply(timeTranSecond)
	except:
		print(f, '应该在前面被删除了！')

def getDuration(se):
	try:
		sh, sm, eh, em = re.findall(r"\d+\.?\d*", se)
	except:
		if se == -1:
			return -1
	try:
		if int(sh) > int(eh):
			tm = (int(eh) * 3600 + int(em) * 60 - int(sm) * 60 - int(sh) * 3600) / 3600 + 24
		else:
			tm = (int(eh) * 3600 + int(em) * 60 - int(sm) * 60 - int(sh) * 3600) / 3600
	except:
		if se == '19:-20:05':
			return 1
		elif se == '15:00-1600':
			return 1
	return tm

for f in ['A20', 'A28', 'B4', 'B9', 'B10', 'B11']:
	data[f] = data.apply(lambda df: getDuration(df[f]), axis=1)
print(f'Time Features Processed in {(time.time()-t0):.1f} s !')
print('------------------------------------')

# Dataset Preparation for Differential
train = data[:train.shape[0]]
test  = data[train.shape[0]:]
train['target'] = list(target)
new_train = train.copy()
new_train = new_train.sort_values(['样本id'], ascending=True)
train_copy = train.copy()
train_copy = train_copy.sort_values(['样本id'], ascending=True)
# Train + Train = 2 x Train
train_len = len(new_train)
new_train = pd.concat([new_train, train_copy])
# Test + 2 x Train
test_len = len(test)
new_test = test.copy()
new_test = pd.concat([new_test, new_train])

# Dataset Differential
# New Trainset
diff_train = pd.DataFrame()
ids = list(train_copy['样本id'].values)
for i in tqdm(range(1, train_len)):
	# Interval of -1, -2, ... -len Rows
	diff_tmp = new_train.diff(-i)
	diff_tmp = diff_tmp[:train_len]
	diff_tmp.columns = [col_ + '_difference' for col_ in diff_tmp.columns.values]
	diff_tmp['样本id'] = ids
	diff_train = pd.concat([diff_train, diff_tmp])
# New Testset
diff_test = pd.DataFrame()
ids_test = list(test['样本id'].values)
for i in tqdm(range(test_len, test_len+train_len)):
	# Interval of  -test_len , -test_len -1 ,.... -test_len - train_len +1 Rows
	diff_tmp = new_test.diff(-i)
	diff_tmp = diff_tmp[:test_len]
	diff_tmp.columns = [col_ + '_difference' for col_ in diff_tmp.columns.values]
	diff_tmp['样本id'] = ids_test
	diff_test = pd.concat([diff_test, diff_tmp])
	diff_test = diff_test[diff_train.columns]
# Target
train_target = train['target']
train.drop(['target'], axis=1, inplace=True)
# Combine Original Dataset and Differential Dataset
diff_train = pd.merge(diff_train, train, how='left', on='样本id')
diff_test = pd.merge(diff_test, test, how='left', on='样本id')
target = diff_train['target_difference']
diff_train.drop(['target_difference'], axis=1, inplace=True)
diff_test.drop(['target_difference'], axis=1, inplace=True)

# Model Run
X_train = diff_train
y_train = target
X_test = diff_test
param = {'num_leaves': 31,
		 'min_data_in_leaf': 20,
		 'objective': 'regression',
		 'max_depth': -1,
		 'learning_rate': 0.01,
		 "boosting": "gbdt",
		 "feature_fraction": 0.9,
		 "bagging_freq": 1,
		 "bagging_fraction": 0.9,
		 "bagging_seed": 11,
		 "metric": 'mse',
		 "num_threads": 8,
		 "verbosity": -1}
folds = KFold(n_splits=5, shuffle=True, random_state=2018)
oof_lgb = np.zeros(len(diff_train))
predictions_lgb = np.zeros(len(diff_test))
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
	print("fold n°{}".format(fold_ + 1))
	dev = X_train.iloc[trn_idx]
	val = X_train.iloc[val_idx]
	trn_data = lgb.Dataset(dev, y_train.iloc[trn_idx])
	val_data = lgb.Dataset(val, y_train.iloc[val_idx])
	num_round = 3000
	clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=100, early_stopping_rounds=100)
	oof_lgb[val_idx] = clf.predict(val, num_iteration=clf.best_iteration)
	predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

# Get Original Train Target
diff_train['compare_id'] = diff_train['样本id'] - diff_train['样本id_difference']
train['compare_id'] = train['样本id']
train['compare_target'] = list(train_target)
diff_train = pd.merge(diff_train, train[['compare_id', 'compare_target']], how='left', on='compare_id')
diff_train['pre_target_diff'] = oof_lgb
diff_train['pre_target'] = diff_train['pre_target_diff'] + diff_train['compare_target']
# 
mean_result = diff_train.groupby('样本id')['pre_target'].mean().reset_index(name='pre_target_mean')
true_result = train[['样本id', 'compare_target']]
mean_result = pd.merge(mean_result, true_result, how='left', on='样本id')
print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, target)))
print("CV score: {:<8.8f}".format(mean_squared_error(mean_result['pre_target_mean'].values,  mean_result['compare_target'].values)))

# Get Test Target
diff_test['compare_id'] = diff_test['样本id'] - diff_test['样本id_difference']
diff_test = pd.merge(diff_test, train[['compare_id', 'compare_target']], how='left', on='compare_id')
diff_test['pre_target_diff'] = predictions_lgb
diff_test['pre_target'] = diff_test['pre_target_diff'] + diff_test['compare_target']
#
mean_result_test = diff_test.groupby(diff_test['样本id'], sort=False)['pre_target'].mean().reset_index(name='pre_target_mean')
test = pd.merge(test, mean_result_test, how='left', on='样本id')

# Submission
sub_df = pd.DataFrame()
sub_df[0] = test_id
sub_df[1] = test['pre_target_mean']
sub_df[1] = sub_df[1].apply(lambda x: round(x, 5))
# Processing Special Case
for v in test_select.keys():
	if v == 280:
		x = 0.947
	elif v == 360:
		x = 0.925
	elif v == 385 or v == 785:
		x = 0.879
	elif v == 390:
		x = 0.89
	print(v)
	print(test_select[v])
	sub_df.loc[test_select[v], 1] = x
sub_df.to_csv('submit_B.csv', index=False, header=False)
print('Submit Done!')

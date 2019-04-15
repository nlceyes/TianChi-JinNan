# -*- coding: utf-8 -*-
# Intial Run @20190218, Parameter Tuned Run @20190219
# Rev.1 @20180218: Categorical Features with pd.get_dummies - Seems Useless?
# Rev.2 @20180218: Adding Features
# Rev.3 @20180218: Adding Rules (B14=385,0.878)
# Rev.4 @20180218: Adding Features of B14 Statistical Info
# Rev.5 @20180218: Adding Stacking Method - Seems Useless? Using Avg Method
import time
from contextlib import contextmanager
import numpy as np
import pandas as pd
# from pandas.core.common import SettingWithCopyWarning
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import BayesianRidge, LinearRegression
# import warnings
# warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
# warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
	t0 = time.time()
	yield
	print('{} - Done in {:.1f}s'.format(title, time.time()-t0))
	print('------------------------------------')

# Time Feature Processing
def get_time(t):
	if pd.isnull(t):
		pass
	else:
		# Get Valid Time and Split into Hrs and Mins
		if '；' in t:
			temp = t.split('；')
		elif '::' in t:
			temp = t.split('::')
		elif '"' in t:
			temp = t.split('"')
		elif ';' in t:
			temp = t.split(';')
		elif ':' in t:
			temp = t.split(':')
		else:
			temp = t.split(':')
			# print(f'New Separate Character Found! - {t}')
			# print(col)
		# Hrs and Mins to Number
		if len(temp) > 1:
			hours = temp[0]
			if len(temp[1]) == 0:
				mins = 0
			else:
				mins = temp[1] if len(temp[1])<3 else temp[1][:2]
		else:
			hours = temp[0][:2]
			mins = temp[0][2:]
		time_int = int(hours) + int(mins)/60
		return time_int

def get_period(t):
	if pd.isnull(t):
		pass
	else:
		start, end = t.split('-')
		start = get_time(start)
		end = get_time(end)
		if end >= start:
			time_period = end - start
		else:
			time_period = 24 + end - start
		return time_period

# Loading and Preprocessing Dataset
def train_test():
	# Loading Dataset and Target
	df_submit = pd.read_csv('./data/FuSai.csv', encoding='gb18030')
	df_opt = pd.read_csv('./data/optimize.csv', encoding='gb18030')
	df_submit = df_submit.append(df_opt)
	df_submit['收率'] = 0
	df_train = pd.read_csv('./data/jinnan_round1_train_20181227.csv', encoding='gb18030')
	df_A = pd.read_csv('./data/jinnan_round1_testA_20181227.csv', encoding='gb18030')
	df_A_ans = pd.read_csv('./data/jinnan_round1_ansA_20190125.csv', header=None)
	df_A['收率'] = df_A_ans[1]
	df_B = pd.read_csv('./data/jinnan_round1_testB_20190121.csv', encoding='gb18030')
	df_B_ans = pd.read_csv('./data/jinnan_round1_ansB_20190125.csv', header=None)
	df_B['收率'] = df_B_ans[1]
	df_C = pd.read_csv('./data/jinnan_round1_test_20190201.csv', encoding='gb18030')
	df_C_ans = pd.read_csv('./data/jinnan_round1_ans_20190201.csv', header=None)
	df_C['收率'] = df_C_ans[1]
	print('Original:\nTrain samples: {}, Test samples: {},{},{}'.format(len(df_train), len(df_A), len(df_B), len(df_C)))
	# Anomaly Processing
	df_train.loc[df_train['A5']=='1900/1/29 0:00', 'A5'] = '0:00' # sample_223
	df_train.loc[df_train['A5']=='1900/1/21 0:00', 'A5'] = '0:00' # sample_1023,1027
	df_A.loc[df_A['A5']=='1900/1/22 0:00', 'A5'] = '0:00' # sample_1229
	df_train.loc[df_train['A9']=='1900/1/9 7:00', 'A9'] = '7:00' # sample_937
	df_train.loc[df_train['A9']=='700', 'A9'] = '7:00' # sample_496
	df_train.loc[df_train['A11']=='1900/1/1 2:30', 'A11'] = '2:30' # sample_1067
	df_train.loc[df_train['A11']==':30:00', 'A11'] = '00:30:00' # sample_130
	df_train.loc[df_train['A16']=='1900/1/12 0:00', 'A16'] = '0:00' # sample_933
	df_train.loc[df_train['A25']=='1900/3/10 0:00', 'A25'] = df_train['A25'].value_counts().index[0] # sample_1590
	df_train['A25'] = df_train['A25'].astype(int)
	df_train.loc[df_train['A26']=='1900/3/13 0:00', 'A26'] = '0:00' # sample_1350
	# df_train.to_csv('fs.csv', index=False)
	# Drop Outliers
	df_train = df_train[df_train['A17']>89] # sample_1469
	df_train = df_train[df_train['A22']>3.5] # sample_602
	df_train = df_train[df_train['B1']!=3.5] # sample_102
	df_train = df_train[df_train['B14']>290] # 13(2)
	df_train = df_train[df_train['收率']>0.835] # 3
	df_A = df_A[df_A['A19']<700] # sample_54
	df_A = df_A[df_A['B14']>280] # 1, sample_1043
	df_A = df_A[df_A['B14']<785] # 1, sample_316
	df_B = df_B[df_B['收率']>0.823] # 1
	df_C = df_C[df_C['B14']>320] # 3(1)
	print('Processed:\nTrain samples: {}, Test samples: {},{},{}'.format(len(df_train), len(df_A), len(df_B), len(df_C)))
	# Time Feature Processing
	cols_time = ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']
	cols_period = ['A20', 'A28', 'B4', 'B9', 'B10', 'B11']
	for col in cols_time:
		df_train[col] = df_train[col].apply(get_time)
		df_A[col] = df_A[col].apply(get_time)
		df_B[col] = df_B[col].apply(get_time)
		df_C[col] = df_C[col].apply(get_time)
		df_submit[col] = df_submit[col].apply(get_time)
	for col in cols_period:
		df_train[col] = df_train[col].apply(get_period)
		df_A[col] = df_A[col].apply(get_period)
		df_B[col] = df_B[col].apply(get_period)
		df_C[col] = df_C[col].apply(get_period)
		df_submit[col] = df_submit[col].apply(get_period)
	return df_train, df_A, df_B, df_C, df_submit

# additional features - Rev.2
def additional_features(df_train, df_A, df_B, df_C, df_submit):
	train_len = len(df_train)
	test_A_len = len(df_A)
	test_B_len = len(df_B)
	test_C_len = len(df_C)
	df = df_train.append([df_A, df_B, df_C, df_submit])
	# df.to_csv('df_all.csv', index=False)
	# Ingredient Ratio
	df['B12'] = df['B12'].fillna(df['B12'].value_counts().index[0])
	df['B1'] = df['B1'].fillna(df['B1'].value_counts().index[0])
	df['A2'] = df['A2'].fillna(0)
	df['A3'] = df['A3'].fillna(0)
	df['B14_B12'] = df['B14'] / df['B12']
	df['B14_B1'] = df['B14'] / df['B1']
	df['B14_A19'] = df['B14'] / df['A19']
	df['B14_A4'] = df['B14'] / df['A4']
	df['B14_A1'] = df['B14'] / df['A1']
	df['b14_rate0'] = df['B14'] / (df['A1']+df['A2']+df['A3']+df['A4']+df['A19']+df['B1']+df['B12'])
	df['b12_rate0'] = df['B12']/ (df['A1']+df['A2']+df['A3']+df['A4']+df['A19']+df['B1']+df['B12'])
	df['b1_rate0'] = df['B1']/ (df['A1']+df['A2']+df['A3']+df['A4']+df['A19']+df['B1']+df['B12'])
	df['a19_rate0'] = df['A19']/ (df['A1']+df['A2']+df['A3']+df['A4']+df['A19']+df['B1']+df['B12'])
	df['a4_rate0'] = df['A4']/ (df['A1']+df['A2']+df['A3']+df['A4']+df['A19']+df['B1']+df['B12'])
	# Time Related Diff
	df['A21'] = df['A21'].fillna(df['A21'].value_counts().index[0])
	df['A25'] = df['A25'].fillna(df['A25'].value_counts().index[0])
	df['A27'] = df['A27'].fillna(df['A27'].value_counts().index[0])
	df['B5'] = df['B5'].fillna(df['B5'].value_counts().index[0])
	df['B8'] = df['B8'].fillna(df['B8'].value_counts().index[0])
	df['A9-A5'] = df['A9'] - df['A5']
	df['A10-A6'] = df['A10'] - df['A6']
	df['A10-A6_A9-A5'] = df['A10-A6'] / df['A9-A5']
	df['A12-A10'] = df['A12'] - df['A10']
	df['A15-A12'] = df['A15'] - df['A12']
	df['A17-A15'] = df['A17'] - df['A15']
	df['A25-A21'] = df['A25'] - df['A21']
	df['A27-A25'] = df['A27'] - df['A25']
	df['B7-B5'] = df['B7'] - df['B5']
	df['B8-B6'] = df['B8'] - df['B6']
	df['B8-B6_B7-B5'] = df['B8-B6'] / df['B7-B5']
	df_train = df[:train_len]
	df_A = df[train_len : train_len+test_A_len]
	df_B = df[train_len+test_A_len : train_len+test_A_len+test_B_len]
	df_C = df[train_len+test_A_len+test_B_len : train_len+test_A_len+test_B_len+test_C_len]
	df_submit = df[train_len+test_A_len+test_B_len+test_C_len:]
	return df_train, df_A, df_B, df_C, df_submit

# Split Train and Valid
def dataset_split(df_train, df_A, df_B, df_C):
	train_df = [df_train.append([df_B, df_C]), df_train.append([df_A, df_C]), df_train.append([df_A, df_B])]
	test_df = [df_A, df_B, df_C]
	return train_df, test_df

# LightGBM
def kfold_lightgbm(train_df, test_df, params, feats, df_submit, debug=False, B14_SI=True):
	ids = ['A', 'B', 'C']
	oof_preds = np.zeros((0,1))
	oof_target = np.zeros((0,1))
	sub_preds = np.zeros(df_submit.shape[0])
	# k-fold, k=3
	for id, train_xy, valid_xy in zip(ids, train_df, test_df):
		train_x, train_y = train_xy[feats], train_xy['收率']
		valid_x, valid_y = valid_xy[feats], valid_xy['收率']
		test_x = df_submit[feats]
		# Rev.4
		if B14_SI:
			df_b14_si = train_xy.groupby(['B14'], as_index=False)['收率'].agg({'gb_B14_tar'+'_mean': 'mean', 'gb_B14_tar'+'_max': 'max', 'gb_B14_tar'+'_min': 'min'})
			train_x = pd.merge(train_x, df_b14_si, how='left', on=['B14'])
			valid_x = pd.merge(valid_x, df_b14_si, how='left', on=['B14'])
			test_x = pd.merge(test_x, df_b14_si, how='left', on=['B14'])
		lgb_train = lgb.Dataset(train_x, label=train_y, free_raw_data=False)
		lgb_test = lgb.Dataset(valid_x, label=valid_y, free_raw_data=False)
		clf = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_test], valid_names=['train', 'test'],
						num_boost_round=10000, early_stopping_rounds=100, verbose_eval=False)
		sub_preds += clf.predict(test_x, num_iteration=clf.best_iteration) / 3
		oof_pred = clf.predict(valid_x, num_iteration=clf.best_iteration)
		oof_preds = np.vstack((oof_preds, oof_pred.reshape(-1, 1)))
		oof_target = np.vstack((oof_target, np.asarray(valid_y).reshape(-1, 1)))
	score_test = mean_squared_error(oof_target, oof_preds)
	print(f'Final CV: {score_test:.8f} !')
	if debug:
		output = pd.DataFrame()
		output['target'] = list(oof_target)
		output['pred'] = list(oof_preds)
		output.to_csv('pred_lgb.csv')
	return oof_preds, oof_target, sub_preds

# XGBoost
def kfold_xgboost(train_df, test_df, params, feats, df_submit, debug=False):
	ids = ['A', 'B', 'C']
	oof_preds = np.zeros((0,1))
	oof_target = np.zeros((0,1))
	sub_preds = np.zeros(df_submit.shape[0])
	# k-fold, k=3
	for id, train_xy, valid_xy in zip(ids, train_df, test_df):
		train_x, train_y = train_xy[feats], train_xy['收率']
		valid_x, valid_y = valid_xy[feats], valid_xy['收率']
		test_x = df_submit[feats]
		# Rev.4
		df_b14_si = train_xy.groupby(['B14'], as_index=False)['收率'].agg({'gb_B14_tar'+'_mean': 'mean', 'gb_B14_tar'+'_max': 'max', 'gb_B14_tar'+'_min': 'min'})
		train_x = pd.merge(train_x, df_b14_si, how='left', on=['B14'])
		valid_x = pd.merge(valid_x, df_b14_si, how='left', on=['B14'])
		test_x = pd.merge(test_x, df_b14_si, how='left', on=['B14'])
		xgb_train = xgb.DMatrix(train_x, label=train_y)
		xgb_test = xgb.DMatrix(valid_x, label=valid_y)
		watchlist = [(xgb_train, 'train'), (xgb_test, 'valid_data')]
		clf = xgb.train(params=params, dtrain=xgb_train, num_boost_round=10000, evals=watchlist, early_stopping_rounds=100, verbose_eval=False)
		sub_preds += clf.predict(xgb.DMatrix(test_x), ntree_limit=clf.best_ntree_limit) / 3
		oof_pred = clf.predict(xgb.DMatrix(valid_x), ntree_limit=clf.best_ntree_limit)
		oof_preds = np.vstack((oof_preds, oof_pred.reshape(-1, 1)))
		oof_target = np.vstack((oof_target, np.asarray(valid_y).reshape(-1, 1)))
	score_test = mean_squared_error(oof_target, oof_preds)
	print(f'Final CV: {score_test:.8f} !')
	if debug:
		output = pd.DataFrame()
		output['target'] = list(oof_target)
		output['pred'] = list(oof_preds)
		output.to_csv('pred_xgb.csv')
	return oof_preds, oof_target, sub_preds

def main(debug=False):
	with timer('Train & Test'):
		train, test_A, test_B, test_C, df_submit = train_test()
	with timer('Additional Features'):
		train_ad, test_A_ad, test_B_ad, test_C_ad, df_submit_ad = additional_features(train, test_A, test_B, test_C, df_submit)
	with timer('Split Train & Test'):
		feats_exclude = ['样本id', 'A1', 'A2', 'A3', 'A7', 'A8', 'A11', 'A13', 'A14', 'A16', 'A18', 'A23', 'A26', 'B2', 'B3', 'B13', '收率']
		feats = [col for col in train.columns if col not in feats_exclude]
		feats_ad = [col for col in train_ad.columns if col not in feats_exclude]
		train_df, test_df = dataset_split(train, test_A, test_B, test_C)
		train_df_ad, test_df_ad = dataset_split(train_ad, test_A_ad, test_B_ad, test_C_ad)
	with timer('Run LightGBM_r0 with kfold'):
		params = {'num_leaves': 47,
				  'min_data_in_leaf': 2, 
				  'max_depth': -1,
				  'learning_rate': 0.05,
				  'boosting': 'gbdt',
				  'objective':'regression',
				  'feature_fraction': 1.0,
				  'bagging_freq': 1,
				  'bagging_fraction': 0.9,
				  'metric': 'mse',
				  'lambda_l1': 0,
				  'lambda_l2': 0,
				  'verbosity': -1,
				  'num_threads': 2}
		oof_preds_lgb_r0, oof_target, sub_preds_lgb_r0 = kfold_lightgbm(train_df, test_df, params, feats, df_submit, debug=debug, B14_SI=False)
	with timer('Run LightGBM_r4 with kfold'):
		params = {'num_leaves': 47,
				  'min_data_in_leaf': 2, 
				  'max_depth': -1,
				  'learning_rate': 0.1,
				  'boosting': 'gbdt',
				  'objective':'regression',
				  'feature_fraction': 0.8,
				  'bagging_freq': 1,
				  'bagging_fraction': 1.0,
				  'metric': 'mse',
				  'lambda_l1': 0,
				  'lambda_l2': 0.1,
				  'verbosity': -1,
				  'num_threads': 2}
		oof_preds_lgb_r4, oof_target, sub_preds_lgb_r4 = kfold_lightgbm(train_df, test_df, params, feats, df_submit, debug=debug)
	with timer('Run XGBoost_r4_IR+TD with kfold'):
		params = {'learning_rate': 0.05,
				  'max_depth': 7, 
				  'subsample': 0.8,
				  'colsample_bytree': 0.8,
				  'gamma': 0.0,
				  'min_child_weight': 1,
				  'reg_alpha': 0,
				  'reg_lambda': 2,
				  'objective': 'reg:linear',
				  'eval_metric': 'rmse',
				  'silent': True,
				  'seed': 0,
				  'nthread': 4}
		oof_preds_xgb_r4IT, oof_target, sub_preds_xgb_r4IT = kfold_xgboost(train_df_ad, test_df_ad, params, feats_ad, df_submit_ad, debug=debug)
	with timer('Stacking Above Models'):
		oof_preds_avg = (oof_preds_lgb_r0+oof_preds_lgb_r4+oof_preds_xgb_r4IT)/3
		score_test = mean_squared_error(oof_target, oof_preds_avg)
		print(f'Final CV: {score_test:.8f} !')
		sub_preds_avg = (sub_preds_lgb_r0+sub_preds_lgb_r4+sub_preds_xgb_r4IT)/3
		df_submit['收率'] = sub_preds_avg
		print('Special Value(B14) Processing...')
		v = 385
		print('Special Value(B14):', v)
		submit_select = df_submit[df_submit['B14'] == v]['样本id'].index
		df_submit.loc[submit_select, '收率'] = 0.878
		print('Special Value(B14) Processed!')
		df_submit[['样本id', '收率']][:-1].to_csv('submit_FuSai.csv', index=False, header=False)
		df_submit[['样本id', '收率']][-1:].to_csv('submit_optimize.csv', index=False, header=False)

if __name__ == '__main__':
	with timer('Full model run'):
		main(debug=False)
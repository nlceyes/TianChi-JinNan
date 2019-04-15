# -*- coding: utf-8 -*-
# EDA and Dataset Preprocessing - Inspired by zhuanlan.zhihu.com/p/53728786
# import warnings
# warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
t0 = time.time()
train = pd.read_csv('jinnan_round1_train_20181227.csv', encoding='gb18030')
test_A = pd.read_csv('jinnan_round1_testA_20181227.csv', encoding='gb18030')
test_A_ans = pd.read_csv('jinnan_round1_ansA_20190125.csv', header=None)
test_A['收率'] = test_A_ans[1]
test_B = pd.read_csv('jinnan_round1_testB_20190121.csv', encoding='gb18030')
test_B_ans = pd.read_csv('jinnan_round1_ansB_20190125.csv', header=None)
test_B['收率'] = test_B_ans[1]
test_C = pd.read_csv('jinnan_round1_test_20190201.csv', encoding='gb18030')
test_C_ans = pd.read_csv('jinnan_round1_ans_20190201.csv', header=None)
test_C['收率'] = test_C_ans[1]
print(f'Data Loaded in {(time.time()-t0):.1f} s !')
print('------------------------------------')

# Anomaly Processing
# A5
train.loc[train['A5']=='1900/1/29 0:00', 'A5'] = '0:00' # sample_223
train.loc[train['A5']=='1900/1/21 0:00', 'A5'] = '0:00' # sample_1023,1027
test_A.loc[test_A['A5']=='1900/1/22 0:00', 'A5'] = '0:00' # sample_1229
# A9
train.loc[train['A9']=='1900/1/9 7:00', 'A9'] = '7:00' # sample_937
train.loc[train['A9']=='700', 'A9'] = '7:00' # sample_496
# A11
train.loc[train['A11']=='1900/1/1 2:30', 'A11'] = '2:30' # sample_1067
train.loc[train['A11']==':30:00', 'A11'] = '00:30:00' # sample_130
# A16
train.loc[train['A16']=='1900/1/12 0:00', 'A16'] = '0:00' # sample_933
# A25
train.loc[train['A25']=='1900/3/10 0:00', 'A25'] = 50 # sample_1590
# A26
train.loc[train['A26']=='1900/3/13 0:00', 'A26'] = '0:00' # sample_1350
print('Anomaly Data Corrected!')
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
			print(f'New Separate Character Found! - {t}')
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

cols_time = ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']
cols_period = ['A20', 'A28', 'B4', 'B9', 'B10', 'B11']
for col in cols_time:
	train[col] = train[col].apply(get_time)
	test_A[col] = test_A[col].apply(get_time)
	test_B[col] = test_B[col].apply(get_time)
	test_C[col] = test_C[col].apply(get_time)
for col in cols_period:
	train[col] = train[col].apply(get_period)
	test_A[col] = test_A[col].apply(get_period)
	test_B[col] = test_B[col].apply(get_period)
	test_C[col] = test_C[col].apply(get_period)
print('Time Feature Processed!')
print('------------------------------------')

train.to_csv('train_rev.csv', index=False)
test_A.to_csv('test_A_rev.csv', index=False)
test_B.to_csv('test_B_rev.csv', index=False)
test_C.to_csv('test_C_rev.csv', index=False)

# Target Analysis
train_size = train.shape[0]
test_A_size = test_A.shape[0]
test_B_size = test_B.shape[0]
test_C_size = test_C.shape[0]
plt.scatter(range(train_size), np.sort(train['收率']))
plt.show()
train['收率'].hist(bins=100)
plt.show()

# General Summary
sum_table = []
# col_features = [col for col in test_A.columns if col != '样本id']
# col_remove = ['样本id', 'A1', 'A2', 'A3', 'A7', 'A8', 'A11', 'A13', 'A14', 'A16', 'A18', 'A23', 'A26', 'B2', 'B3', 'B13', '收率']
col_remove = ['样本id', 'A1', 'A2', 'A3', 'A7', 'A8', 'A13', 'A18', 'A23', 'A26', 'B2', 'B3', 'B13', '收率']
col_features = [col for col in test_A.columns if col not in col_remove]

for col in col_features:
	sum_table.append((col, train[col].dtype, train[col].nunique(), train[col].isnull().sum()/train_size, train[col].value_counts(dropna=False, normalize=True).values[0], 
											 test_A[col].nunique(), test_A[col].isnull().sum()/test_A_size, test_A[col].value_counts(dropna=False, normalize=True).values[0], 
											 test_B[col].nunique(), test_B[col].isnull().sum()/test_B_size, test_B[col].value_counts(dropna=False, normalize=True).values[0], 
											 test_C[col].nunique(), test_C[col].isnull().sum()/test_C_size, test_C[col].value_counts(dropna=False, normalize=True).values[0]))
sum_cols = ['Feature', 'Type', 'Values(Train)', 'Null_Rate(Train)', 'Major_Value_Rate(Train)', 'Values(Test_A)', 'Null_Rate(Test_A)', 'Major_Value_Rate(Test_A)', 'Values(Test_B)', 'Null_Rate(Test_B)', 'Major_Value_Rate(Test_B)', 'Values(Test_C)', 'Null_Rate(Test_C)', 'Major_Value_Rate(Test_C)']
sum_table = pd.DataFrame(sum_table, columns=sum_cols)
sum_table.to_csv('Feature_Sum.csv', index=False)
print(sum_table)
sns.heatmap(abs(train[col_features].corr()), annot=True)
plt.show()
print('------------------------------------')

# Drop Outliers
print('Trainset:')
print(train.shape[0])
train = train[train['A17']>89] # sample_1469
print(train.shape[0])
train = train[train['A22']>3.5] # sample_602
print(train.shape[0])
train = train[train['B1']!=3.5] # sample_102
print(train.shape[0])
train = train[train['B14']>290] # 13(2)
print(train.shape[0])
train = train[train['收率']>0.835] # 3
print(train.shape[0])
print('Testset_A:')
print(test_A.shape[0])
test_A = test_A[test_A['A19']<700] # sample_54
print(test_A.shape[0])
test_A = test_A[test_A['B14']>280] # 1
print(test_A.shape[0])
test_A = test_A[test_A['B14']<785] # 1
print(test_A.shape[0])
print('Testset_B:')
print(test_B.shape[0])
test_B = test_B[test_B['收率']>0.823] # 1
print(test_B.shape[0])
print('Testset_C:')
print(test_C.shape[0])
test_C = test_C[test_C['B14']>320] # 3(1)
print(test_C.shape[0])

######
# train[['样本id']+col_features+['收率']].to_csv('train_drop_anomaly.csv', index=False)
# test[['样本id']+col_features].to_csv('test_drop_anomaly.csv', index=False)

# Feature Distribution in Dataset
def plot_kde(train, test_A, test_B, test_C, col):
	fig, ax =plt.subplots(1, 5)
	sns.kdeplot(train[col], color='g', ax=ax[0])
	sns.kdeplot(test_A[col], color='r', ax=ax[1])
	sns.kdeplot(test_B[col], color='y', ax=ax[2])
	sns.kdeplot(test_C[col], color='m', ax=ax[3])
	sns.kdeplot(train[col], color='g', ax=ax[4])
	sns.kdeplot(test_A[col], color='r', ax=ax[4])
	sns.kdeplot(test_B[col], color='y', ax=ax[4])
	sns.kdeplot(test_C[col], color='m', ax=ax[4])
	plt.title('Distribution_' + col)
	plt.show()
for col in col_features:
	plot_kde(train, test_A, test_B, test_C, col)

# Feature Significance
# for fea in fea_list:
	# plt.figure(1)
	# plt.subplot(121)
	# safe_user = train[train['label']==0][fea].value_counts()
	# risk_user= train[train['label']==1][fea].value_counts()
	# ratio = risk_user / safe_user
	# ratio.plot(kind='bar')
	# plt.subplot(122)
	# sns.kdeplot(train[fea], color='y')
	# sns.kdeplot(test[fea], color='b')
	# plt.show()

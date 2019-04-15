# -*- coding: utf-8 -*-
# Find Duplicate Samples
# import warnings
# warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
t0 = time.time()
df_all = pd.read_csv('all_rev.csv')
col_remove = ['id', 'target']
col_retain = [col for col in df_all.columns if col not in col_remove]
df = df_all[col_retain].copy()
df = df.astype('str')
# Duplicate Search
total_iters = len(df)
for i in range(total_iters):
	dup_list = []
	temp1_id = df_all.ix[i]['id']
	dup_list.append(temp1_id)
	temp1_list = list(df.ix[i])
	temp1 = ''.join(temp1_list)
	for j in range(i+1, total_iters):
		temp2_list = list(df.ix[j])
		temp2 = ''.join(temp2_list)
		if temp1 == temp2:
			temp2_id = df_all.ix[j]['id']
			dup_list.append(temp2_id)
	if len(dup_list) > 1:
		print(dup_list)
print('------------------------------------')
print(f'Done in {(time.time()-t0):.1f} s !')
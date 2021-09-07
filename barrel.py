import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, ward
import mglearn
os.chdir('C:/Users/82102/Desktop/bigcon/01_제공데이터')

HTS_df_2018 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_HTS_2018.csv',encoding='cp949')
HTS_df_2019 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_HTS_2019.csv',encoding='cp949')
HTS_df_2020 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_HTS_2020.csv',encoding='cp949')
HTS_df_2021 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_HTS_2021.csv',encoding='cp949')

HTS_df_total = pd.concat([HTS_df_2018,HTS_df_2019,HTS_df_2020,HTS_df_2021],ignore_index=True)
HTS_df_total.drop(['GYEAR', 'G_ID', 'PIT_ID', 'PCODE', 'T_ID', 'INN', 'STADIUM','HIT_RESULT', 'PIT_VEL'],axis=1,inplace=True)
HTS_df_total = HTS_df_total[['HIT_VEL', 'HIT_ANG_VER']]


HTS_df_int = HTS_df_total.astype(int)
HTS_df_int[['HIT_VEL']].value_counts()
HTS_df_int[['HIT_ANG_VER']].value_counts()

HTS_df_int.describe()
HTS_df_int = HTS_df_int.assign(label = 0)


sns.distplot(HTS_df_int['HIT_VEL'])
sns.distplot(HTS_df_int['HIT_ANG_VER'])
HIT_VEL_list = []
HIT_ANG_VER_list = []

for j in HTS_df_int['HIT_VEL']:
    if not j in HIT_VEL_list:
        HIT_VEL_list.append(j)

for j in HTS_df_int['HIT_ANG_VER']:
    if not j in HIT_VEL_list:
        HIT_ANG_VER_list.append(j)
cnt = 0
for i in HIT_VEL_list:
    for j in HIT_ANG_VER_list:
        cnt += 1
        for k in range(len(HTS_df_int)):
            if i == HTS_df_int.loc[k][0] and j == HTS_df_int.loc[k][1]:
                HTS_df_int.loc[k,'label'] = cnt

# =============================================================================
# standard_scaler = StandardScaler()
# HTS_scaled_df = pd.DataFrame(standard_scaler.fit_transform(HTS_df_total), columns=HTS_df_total.columns)
# 
# dbscan = DBSCAN(eps=0.1,min_samples=25)
# clusters = pd.DataFrame(dbscan.fit_predict(HTS_scaled_df))    
# clusters.columns=['predict']
# clusters.value_counts()
# 
# 
# r = pd.concat([HTS_scaled_df,clusters],axis=1)
# 
# sns.pairplot(r,hue='predict')
# plt.show()
# 
# 
# plt.scatter(HTS_scaled_df[:, 0], HTS_scaled_df[:, 1], c=clusters, cmap=mglearn.cm2, s=60, edgecolors='black')
# plt.xlabel("attr 0")
# plt.ylabel("attr 1")
# 
# 
# plt.scatter(HTS_df_total.index,HTS_df_total[['HIT_ANG_VER']])
# plt.show()
# =============================================================================
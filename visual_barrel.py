import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, ward
import scipy
from scipy import stats
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns

os.chdir('C:/Users/KIMJUNYOUNG/Desktop/BC_data/train_data/')
df = pd.read_csv('zzin_df.csv')
df.drop(['index'], axis = 1, inplace=True)

# 전체 데이터 산점도
plt.figure(figsize=(18, 10))
plt.scatter(x = 'HIT_VEL',y='HIT_ANG_VER', data = df)
plt.xlim(140,185)
plt.ylim(5,55)
plt.xlabel('HIT_VEL')
plt.ylabel('HIT_ANG_VER')
plt.show()

# HIT_VEL을 기준으로 정렬
df = df.sort_values(by = 'HIT_VEL', ascending = True, ignore_index = True)
len(df) #6558
min(df.HIT_VEL) # 144.63
max(df.HIT_VEL) # 179.25
# =============================================================================
# cut_df = pd.cut(df.HIT_VEL,36)
# grouped_cut_df = df.HIT_VEL.groupby(cut_df)
# summary_df = grouped_cut_df.agg(['count', 'mean', 'std', 'min', 'max'])
# =============================================================================

# 50개씩 나누기
qcut_df = pd.qcut(df.HIT_VEL, len(df)//50, labels=False)
df['interval'] = qcut_df

# 최소를 기준으로 간격 설정
group_qcut_df = df.HIT_VEL.groupby(qcut_df)
min_interval_df = group_qcut_df.agg(['min'])

# lmplot을 사용하기 위해 새로운 dataframe 만들기
interval_max = df['HIT_ANG_VER'].groupby(df['interval']).max()
interval_min = df['HIT_ANG_VER'].groupby(df['interval']).min()
interval_90 = pd.DataFrame((interval_max - interval_min) * 0.90 + interval_min)
interval_90.columns = ['HIT_ANG_VER']
interval_90['sep'] = ['upper_limit' for _ in range(len(interval_90))]
interval_90['HIT_VEL'] = min_interval_df
interval_10 = pd.DataFrame((interval_max - interval_min) * 0.10 + interval_min)
interval_10.columns = ['HIT_ANG_VER']
interval_10['sep'] = ['lower_limit' for _ in range(len(interval_10))]
interval_10['HIT_VEL'] = min_interval_df
interval_df = pd.concat([interval_90, interval_10], axis=0)


# 위에서 만든 dataframe 시각화
sns.lmplot(x = 'HIT_VEL', y = 'HIT_ANG_VER', hue = 'sep', data = interval_df)
plt.show()

# =============================================================================
# pvalue 확인과 기울기, y절편 구하기
# =============================================================================
# slope=0.15562915209790212, intercept=32.48832867132867, rvalue=0.8053780797060072, pvalue=6.059472088722152e-16, stderr=0.014431146341511774
up_slope, up_intercept, up_rvalue, up_pvalue, up_stderr = scipy.stats.linregress(interval_90.HIT_VEL, interval_90.HIT_ANG_VER)

# slope=-0.14348317307692315, intercept=23.852230769230772, rvalue=-0.930499526073186, pvalue=3.564601510033978e-29, stderr=0.007116120181029661
low_slope, low_intercept, low_rvalue, low_pvalue, low_stderr = scipy.stats.linregress(interval_10.HIT_VEL, interval_10.HIT_ANG_VER)

def limit_line(x):
    up_slope = 0.15562915209790212
    up_intercept = 32.48832867132867
    low_slope = -0.14348317307692315
    low_intercept = 23.852230769230772
    
    return up_slope * x + up_intercept, low_slope * x + low_intercept
    
limit_line(159)




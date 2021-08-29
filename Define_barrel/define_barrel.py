import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
import scipy
from scipy import stats
# =============================================================================
# 사용된 함수들 정리
# =============================================================================
def numeric_hit_result(x):
    return hit_result[x]

def numeric_stadium(x):
    return stadium[x]

def clustering(dataset):
    sam_data = dataset.iloc[:, 1:3].values
    agg_clustering = AgglomerativeClustering(n_clusters=500, linkage='average')
    labels = agg_clustering.fit_predict(sam_data)
    dataset = pd.concat([dataset,pd.DataFrame(labels)],axis = 1)
    dataset.rename(columns={0:'label'},inplace=True)
    
    return dataset

def barrel_selection(dataset):
    
    barrel_df = pd.DataFrame()
    hit_list = [1,2,3,4,7,15]
    one_hit_list  = [1,4,7]
    
    for idx in range(0,500):
        boonmo = 0
        HIT_boonza = 0
        SLG_boonza = 0
        HIT_rate = 0
        SLG_per = 0
        label_HTS = dataset[(dataset['label'] == idx)]
        except_1617 = label_HTS[(label_HTS['HIT_RESULT'] < 16)]
        boonmo = len(except_1617)
        hit_cnt = label_HTS[(label_HTS['HIT_RESULT'].isin(hit_list))]
        HIT_boonza = len(hit_cnt)
        one_hit = hit_cnt[(hit_cnt['HIT_RESULT'].isin(one_hit_list))]
        two_hit = hit_cnt[(hit_cnt['HIT_RESULT'] == 2)]
        three_hit = hit_cnt[(hit_cnt['HIT_RESULT'] == 3)]
        four_hit = hit_cnt[(hit_cnt['HIT_RESULT'] == 15)]
        SLG_boonza = 1*len(one_hit) + 2*len(two_hit) + 3*len(three_hit) + 4*len(four_hit)
        if boonmo == 0:
            HIT_rate = 0
            SLG_per = 0
        else:
            HIT_rate = HIT_boonza/boonmo
            SLG_per = SLG_boonza/boonmo
        barrel_sample_df = pd.DataFrame({'label':idx,'HIT_rate':HIT_rate,'SLG_per':SLG_per},index = [0])
        barrel_df = pd.concat([barrel_df,barrel_sample_df], ignore_index = True)
        
    good_barrel = barrel_df[(barrel_df['HIT_rate'] >= 0.5) & (barrel_df['SLG_per'] >= 1.5)]
    lable_list = list(good_barrel['label'])
     
    real_barrel_df = dataset[(dataset['label'].isin(lable_list))]

    return real_barrel_df

def limit_line(x):
    global up_slope, up_intercept, low_slope, low_intercept
    
    return up_slope * x + up_intercept, low_slope * x + low_intercept

def check_barrel(df):
    upper, lower = limit_line(df.HIT_VEL)
    if upper >= df.HIT_ANG_VER and lower <= df.HIT_ANG_VER:
        return 1
    else:
        return 0
    
    
# =============================================================================
# 사용된 경로 설정
# 예시) 입력방법
# user_path = 'C:/Users/KIMJUNYOUNG/Desktop/BC_data/train_data/'
# =============================================================================
user_path = input()
# =============================================================================
# csv 파일에서 타구결과와 경기장을 수치화시킴
# =============================================================================
# # 타격 결과 항목(총 17가지)
#   1> 1루타
#   2> 2루타
#   3> 3루타
#   4> 내야안타(1루타)
#   5> 땅볼아웃
#   6> 번트아웃
#   7> 번트안타
#   8> 병살타
#   9> 삼중살타
#   10> 야수선택
#   11> 인필드플라이
#   12> 직선타
#   13> 파울플라이
#   14> 플라이
#   15> 홈런
#   16> 희생번트
#   17> 희생플라이
# =============================================================================
# =============================================================================
os.chdir(user_path)
hts_2018 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_HTS_2018.csv', encoding='CP949')
hts_2019 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_HTS_2019.csv', encoding='CP949')
hts_2020 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_HTS_2020.csv', encoding='CP949')
hts_2021 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_HTS_2021.csv', encoding='CP949')

hit_result = {'1루타':1, '2루타':2, '3루타':3, '내야안타(1루타)':4, '땅볼아웃':5, '번트아웃':6,'번트안타':7,'병살타':8,
              '삼중살타':9, '야수선택':10,'인필드플라이':11, '직선타':12,'파울플라이':13,'플라이':14,'홈런':15,'희생번트':16,'희생플라이':17}


hts_2018['HIT_RESULT'] = hts_2018['HIT_RESULT'].apply(numeric_hit_result)
hts_2019['HIT_RESULT'] = hts_2019['HIT_RESULT'].apply(numeric_hit_result)
hts_2020['HIT_RESULT'] = hts_2020['HIT_RESULT'].apply(numeric_hit_result)
hts_2021['HIT_RESULT'] = hts_2021['HIT_RESULT'].apply(numeric_hit_result)

# STADIUM count
hts_2018.STADIUM.value_counts() # 잠실, 수원, 문학, 광주, 고척, "마산", 대구, 사직, 대전
hts_2019.STADIUM.value_counts() # 잠실, 수원, 문학, 광주, 고척, "창원", 대구, 사직, 대전
hts_2020.STADIUM.value_counts() # 잠실, 수원, 문학, 광주, 고척, "창원", 대구, 사직, 대전
hts_2021.STADIUM.value_counts() # 잠실, 수원, 문학, 광주, 고척, "창원", 대구, 사직, 대전

stadium = {'잠실':1, '수원':2, '문학':3, '광주':4,'고척':5,'마산':6,'창원':7,'대구':8,'사직':9,'대전':10}


hts_2018['STADIUM'] = hts_2018['STADIUM'].apply(numeric_stadium)
hts_2019['STADIUM'] = hts_2019['STADIUM'].apply(numeric_stadium)
hts_2020['STADIUM'] = hts_2020['STADIUM'].apply(numeric_stadium)
hts_2021['STADIUM'] = hts_2021['STADIUM'].apply(numeric_stadium)

hts_2018.to_csv('hts_2018.csv', index = False)
hts_2019.to_csv('hts_2019.csv', index = False)
hts_2020.to_csv('hts_2020.csv', index = False)
hts_2021.to_csv('hts_2021.csv', index = False)

# =============================================================================
# 데이터가 너무 많아 4개의 데이터로 분할
# 필요없는 columns 제거
# =============================================================================

hts_2018 = pd.read_csv('hts_2018.csv')
hts_2019 = pd.read_csv('hts_2019.csv')
hts_2020 = pd.read_csv('hts_2020.csv')
hts_2021 = pd.read_csv('hts_2021.csv')

hts = pd.concat([hts_2018, hts_2019, hts_2020, hts_2021], ignore_index=True)

hts.drop(['GYEAR', 'G_ID', 'T_ID', 'INN', 'STADIUM', 'PIT_VEL', 'PCODE'], axis=1, inplace=True)

hts_shuffled=hts.sample(frac=1).reset_index(drop=True)

len_hts_shuffled = len(hts_shuffled)//4

hts_1 = hts_shuffled.iloc[:len_hts_shuffled]
hts_2 = hts_shuffled.iloc[len_hts_shuffled : len_hts_shuffled*2]
hts_3 = hts_shuffled.iloc[len_hts_shuffled*2 : len_hts_shuffled*3]
hts_4 = hts_shuffled.iloc[len_hts_shuffled*3:]

hts_1.reset_index(drop=True, inplace=True)
hts_2.reset_index(drop=True, inplace=True)
hts_3.reset_index(drop=True, inplace=True)
hts_4.reset_index(drop=True, inplace=True)

hts_1.to_csv('hts_1.csv', index = False)
hts_2.to_csv('hts_2.csv', index = False)
hts_3.to_csv('hts_3.csv', index = False)
hts_4.to_csv('hts_4.csv', index = False)

# =============================================================================
# clustering하기
# barrel를 찾기 위해 타율과 장타율 계산하기
# =============================================================================
os.chdir(user_path)
HTS_one = pd.read_csv('hts_1.csv')
HTS_two = pd.read_csv('hts_2.csv')
HTS_three = pd.read_csv('hts_3.csv')
HTS_four = pd.read_csv('hts_4.csv')

split_HTS_list = [HTS_one,HTS_two,HTS_three,HTS_four]

total_barrel_df = pd.DataFrame()
for i in range(len(split_HTS_list)):
    split_HTS_list[i] = clustering(split_HTS_list[i])
    total_barrel_df = pd.concat([total_barrel_df,barrel_selection(split_HTS_list[i])])


# 시각화 이후 이상치 제거(눈에 보이는 이상치 제거)
plt.figure(figsize=(18, 10))
plt.scatter(x = 'HIT_VEL',y='HIT_ANG_VER', data = total_barrel_df)
plt.xlabel('HIT_VEL')
plt.ylabel('HIT_ANG_VER')
plt.show()

total_barrel_df = total_barrel_df[(total_barrel_df['HIT_ANG_VER'] > 0) & (total_barrel_df['HIT_VEL'] > 140)]
total_barrel_df = total_barrel_df.reset_index()
total_barrel_df.to_csv('total_barrel_df.csv', index = False)

# =============================================================================
# barrel 시각화
# 타구속도에 해당하는 타구 각도 상한값 하한값 구하기
# =============================================================================

os.chdir(user_path)
total_barrel_df = pd.read_csv('total_barrel_df.csv')
total_barrel_df.drop(['index'], axis = 1, inplace=True)

# 전체 데이터 산점도
plt.figure(figsize=(18, 10))
plt.scatter(x = 'HIT_VEL',y='HIT_ANG_VER', data = total_barrel_df)
plt.xlim(140,185)
plt.ylim(5,55)
plt.xlabel('HIT_VEL')
plt.ylabel('HIT_ANG_VER')
plt.show()

# HIT_VEL을 기준으로 정렬
total_barrel_df = total_barrel_df.sort_values(by = 'HIT_VEL', ascending = True, ignore_index = True)
len(total_barrel_df) #6558
min(total_barrel_df.HIT_VEL) # 144.63
max(total_barrel_df.HIT_VEL) # 179.25


# 50개씩 나누기
qcut_df = pd.qcut(total_barrel_df.HIT_VEL, len(total_barrel_df)//50, labels=False)
total_barrel_df['interval'] = qcut_df

# 최소를 기준으로 간격 설정
group_qcut_df = total_barrel_df.HIT_VEL.groupby(qcut_df)
min_interval_df = group_qcut_df.agg(['min'])

# lmplot을 사용하기 위해 새로운 dataframe 만들기
interval_max = total_barrel_df['HIT_ANG_VER'].groupby(total_barrel_df['interval']).max()
interval_min = total_barrel_df['HIT_ANG_VER'].groupby(total_barrel_df['interval']).min()
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

# =============================================================================
# 해당 타구가 barrel 인지 아닌지 확인
# 해당 타구가 barrel 타구 이면 1
# 해당 타구가 barrel 타구가 아니라면 0
# =============================================================================
os.chdir(user_path)

hts = pd.concat([hts_2018, hts_2019, hts_2020, hts_2021], ignore_index=True)

hts.drop(['G_ID', 'PIT_ID', 'T_ID', 'INN', 'STADIUM', 'PIT_VEL', 'HIT_RESULT'], axis=1, inplace=True)

sort_hts = hts.sort_values(by=['PCODE','GYEAR'],ascending=True,ignore_index=True)

pcode_list = []
for i in range(len(sort_hts)):
    if not int(sort_hts.loc[i]['PCODE']) in pcode_list:
        pcode_list.append(int(sort_hts.loc[i]['PCODE']))  

sort_hts['barrel'] = sort_hts.apply(check_barrel, axis = 1)

sort_hts.to_csv('sort_hts.csv', index = False)

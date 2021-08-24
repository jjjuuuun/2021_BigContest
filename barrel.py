import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, ward
import mglearn
from sklearn.cluster import AgglomerativeClustering
# =============================================================================
# import scipy.cluster.hierarchy as shc
# =============================================================================
os.chdir('C:/Users/82102/Desktop/bigcon/01_제공데이터')

#분활된 HTS데이터를 읽어와요
HTS_one = pd.read_csv('hts_1.csv')
HTS_two = pd.read_csv('hts_2.csv')
HTS_three = pd.read_csv('hts_3.csv')
HTS_four = pd.read_csv('hts_4.csv')
split_HTS = [HTS_one,HTS_two,HTS_three,HTS_four]

#Agglormerative clustering을 하는 함수에요
def clustering(dataset):
    sam_data = dataset.iloc[:, 1:3].values
    agg_clustering = AgglomerativeClustering(n_clusters=500, linkage='average')
    labels = agg_clustering.fit_predict(sam_data)
    oncat([dataset,pd.DataFrame(labels)],axis = 1)
    dataset.rename(columns={0:'label'},inplace=True)
    return dataset

#배럴타구에 적합한 타구들을 추출하는 함수에요
def barrel_selection(dataset):
    
    barrel_df = pd.DataFrame()
    hit_list = [1,2,3,4,7,15]
    out_list = [5,6,7,8,9,10,11,12,13,14]
    #-----------------------------------------------------------------------
    #이 씹새끼가 아주 문제에요
    #각 타구의 타율과 장타율을 계산해요
    for i in range(0,500):
        boonmo = 0
        HIT_boonza = 0
        SLG_boonza = 0
        for j in range(len(dataset)):
            if i == dataset.loc[j]['label']:
                if dataset.loc[j]['HIT_RESULT'] in range(1,16):
                    boonmo += 1
                    if dataset.loc[j]['HIT_RESULT'] in hit_list:
                        HIT_boonza += 1
                        if dataset.loc[j]['HIT_RESULT'] in range(1,4):
                            SLG_boonza += HTS_one.loc[j]['HIT_RESULT']*1
                        elif dataset.loc[j]['HIT_RESULT'] != 15:
                            SLG_boonza += 1
                        elif dataset.loc[j]['HIT_RESULT'] == 15:
                            SLG_boonza += 4
        if boonmo == 0:
            HIT_rate = 0
            SLG_per = 0
        else:
            HIT_rate = HIT_boonza/boonmo
            SLG_per = SLG_boonza/boonmo
        barrel_sample_df = pd.DataFrame({'label':i,'HIT_rate':HIT_rate,'SLG_per':SLG_per},index = [0])
        barrel_df = pd.concat([barrel_df,barrel_sample_df],ignore_index = True)
        print(i)
    #-------------------------------------------------------------------------
    #타율 0.5, 장타율 1.5 이상인 타구들을 선별해요
    barrel_list = []            
    for i in range(0,500):
        if barrel_df.loc[i]['HIT_rate'] > 0.5 and barrel_df.loc[i]['SLG_per'] > 1.5:
            barrel_list.append(i)

    real_barrel_df = pd.DataFrame()                
    for i in range(len(HTS_one)):
        if HTS_one.loc[i]['label'] in barrel_list:
            real_barrel_df = real_barrel_df.append(HTS_one.loc[i], ignore_index=True)

    return real_barrel_df

#클러스터링을 실행하고
#타구들을 선별해요
zzin_barrel_df = pd.DataFrame()
for i in range(len(split_HTS)):
    split_HTS[i] = clustering(split_HTS[i])
    zzin_barrel_df = pd.cancat[zzin_barrel_df,barrel_selection(split_HTS[i])]

#선별된 타구들의 산점도를 찍어봐요 ㅎㅎ
scatter_plot = zzin_barrel_df.plot.scatter(x='HIT_VEL',y='HIT_ANG_VER')
scatter_plot.plot()
plt.show()
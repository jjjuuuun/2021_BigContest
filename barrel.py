import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
os.chdir('C:/Users/82102/Desktop/bigcon/01_제공데이터')

#분활된 HTS데이터를 읽어와요
HTS_one = pd.read_csv('hts_1.csv')
HTS_two = pd.read_csv('hts_2.csv')
HTS_three = pd.read_csv('hts_3.csv')
HTS_four = pd.read_csv('hts_4.csv')
split_HTS = [HTS_one,HTS_two,HTS_three,HTS_four]
#=============================함수========================================
#산점도를 찍는 함수에요
def scatter_graph(dataset,x_line,y_line):
    scatter_plot = dataset.plot.scatter(x=x_line,y=y_line)
    scatter_plot.plot() 
    plt.show()
    
#Agglormerative clustering을 하는 함수에요
def clustering(dataset):
    sam_data = dataset.iloc[:, 1:3].values
    agg_clustering = AgglomerativeClustering(n_clusters=500, linkage='average')
    labels = agg_clustering.fit_predict(sam_data)
    dataset = pd.concat([dataset,pd.DataFrame(labels)],axis = 1)
    dataset.rename(columns={0:'label'},inplace=True)
    return dataset

#배럴타구에 적합한 타구들을 추출하는 함수에요
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
        barrel_df = pd.concat([barrel_df,barrel_sample_df],ignore_index = True)

    good_barrel = barrel_df[(barrel_df['HIT_rate'] >= 0.5) & (barrel_df['SLG_per'] >= 1.5)]
    lable_list = list(good_barrel['label'])
     
    real_barrel_df = dataset[(dataset['label'].isin(lable_list))]
    
    return real_barrel_df

#=============================================================================

#클러스터링을 실행하고
#타구들을 선별해요
zzin_barrel_df = pd.DataFrame()
for i in range(len(split_HTS)):
    split_HTS[i] = clustering(split_HTS[i])
    zzin_barrel_df = pd.concat([zzin_barrel_df,barrel_selection(split_HTS[i])])
zzin_barrel_df = zzin_barrel_df.reset_index()

#선별된 타구들의 산점도를 찍어봐요 ㅎㅎ
scatter_graph(zzin_barrel_df,'HIT_VEL','HIT_ANG_VER')

#이상치를 제거해요
zzin_barrel_df = zzin_barrel_df[(zzin_barrel_df['HIT_ANG_VER'] > 0) & (zzin_barrel_df['HIT_VEL'] > 140)]

#다시 산점도를 찍어요 ㅎㅎ
scatter_graph(zzin_barrel_df,'HIT_VEL','HIT_ANG_VER')
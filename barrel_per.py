import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# user_path = 'C:/Users/KIMJUNYOUNG/Desktop/BC_data/train_data/'
# =============================================================================
user_path = input()
os.chdir(user_path)

except_list = [76232, 68050, 75847, 67341, 79192, 78224, 78513, 76290, 79215, 67872]

batter_2018 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_타자 기본_2018.csv')
batter_2019 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_타자 기본_2019.csv')
batter_2020 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_타자 기본_2020.csv')
batter_2021 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_타자 기본_2021.csv')
except_batter_2021 = batter_2021[~(batter_2021['PCODE'].isin(except_list))]



sort_hts = pd.read_csv('sort_hts.csv')
sort_hts = sort_hts.sort_values(by = 'GYEAR', ascending = True, ignore_index = True)
hts_18 = sort_hts.iloc[:35029]
hts_19 = sort_hts.iloc[35029:68279]
hts_20 = sort_hts.iloc[68279:102781]
hts_21 = sort_hts.iloc[102781:]
except_hts_21 = hts_21[~(hts_21['PCODE'].isin(except_list))]


up_slope = 0.15562915209790212
up_intercept = 32.48832867132867
low_slope = -0.14348317307692315
low_intercept = 23.852230769230772

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
# hts_2018 = pd.read_csv('hts_2018.csv')
# hts_2019 = pd.read_csv('hts_2019.csv')
# hts_2020 = pd.read_csv('hts_2020.csv')
# hts_2021 = pd.read_csv('hts_2021.csv')
# 
# hts = pd.concat([hts_2018, hts_2019, hts_2020, hts_2021], ignore_index=True)
# 
# hts.drop(['G_ID', 'PIT_ID', 'T_ID', 'INN', 'PIT_VEL', 'HIT_RESULT'], axis=1, inplace=True)
# 
# sort_hts = hts.sort_values(by=['PCODE','GYEAR', 'STADIUM'],ascending=True,ignore_index=True)
# 
# # =============================================================================
# # pcode_list = []
# # for i in range(len(sort_hts)):
# #     if not int(sort_hts.loc[i]['PCODE']) in pcode_list:
# #         pcode_list.append(int(sort_hts.loc[i]['PCODE']))  
# # =============================================================================
# 
# sort_hts['barrel'] = sort_hts.apply(check_barrel, axis = 1)
# =============================================================================

def append_barrel(df):
    barrel_per_df = pd.DataFrame(columns = ['PCODE', 'HIT_VEL', 'HIT_ANG_VER', 'len_barrel', 'barrel_per'])
    # df['barrel'] = df.apply(barrel, axis = 1)
    pcode_list = set(list(df.PCODE))
    idx = 0    
    for pcode in pcode_list:
        pcode_df = df[(df['PCODE'] == pcode)]
        len_barrel = len(pcode_df[(pcode_df['barrel'] == 1)])
        barrel_per = len_barrel / len(pcode_df) 
        hit_vel = pcode_df['HIT_VEL'].mean()
        hit_ang_ver = pcode_df['HIT_ANG_VER'].mean()
        barrel_per_df.loc[idx] = pcode, hit_vel, hit_ang_ver, len_barrel, barrel_per
        idx += 1
    print(len(pcode_list))
    return barrel_per_df

barrel_18 = append_barrel(hts_18)
barrel_19 = append_barrel(hts_19)
barrel_20 = append_barrel(hts_20)
barrel_21 = append_barrel(hts_21)
except_barrel_21 = append_barrel(except_hts_21)


batter_2018 = batter_2018[(batter_2018['PA'] >= 446)]
batter_2019 = batter_2019[(batter_2019['PA'] >= 446)]
batter_2020 = batter_2020[(batter_2020['PA'] >= 446)]
batter_2021 = batter_2021[(batter_2021['PA'] >= 190)]
except_batter_2021 = except_batter_2021[(except_batter_2021['PA'] >= 190)]

batter_2018 = pd.merge(batter_2018, barrel_18, left_on = 'PCODE', right_on='PCODE', how = 'inner')
batter_2019 = pd.merge(batter_2019, barrel_19, left_on = 'PCODE', right_on='PCODE', how = 'inner')
batter_2020 = pd.merge(batter_2020, barrel_20, left_on = 'PCODE', right_on='PCODE', how = 'inner')
batter_2021 = pd.merge(batter_2021, barrel_21, left_on = 'PCODE', right_on='PCODE', how = 'inner')
except_batter_0221 = pd.merge(except_batter_2021, except_barrel_21, left_on = 'PCODE', right_on='PCODE', how = 'inner')

# 상관계수 확인
bat = pd.concat([batter_2018, batter_2019, batter_2020, batter_2021], ignore_index=True)
except_bat = pd.concat([batter_2018, batter_2019, batter_2020, except_batter_0221], ignore_index=True)

def obp(df):
    obp = (df.HIT + df.BB + df.HP) / (df.AB + df.BB + df.HP + df.SF)
    return obp

bat['OBP'] = bat.apply(obp, axis = 1)
bat['OPS'] = bat.SLG + bat.OBP
bat['barrel_per_pa'] = bat['len_barrel'] / bat['PA']
bat['barrel_per_ab'] = bat['len_barrel'] / bat['AB']

except_bat['OBP'] = except_bat.apply(obp, axis = 1)
except_bat['OPS'] = except_bat.SLG + except_bat.OBP
except_bat['barrel_per_pa'] = except_bat['len_barrel'] / except_bat['PA']
except_bat['barrel_per_ab'] = except_bat['len_barrel'] / except_bat['AB']

bat.to_csv('bat_all.csv', index = False)

# 해당 10명 선수 제외
bat_except_all = bat[~(bat['PCODE'].isin(except_list))]

bat_except_all.to_csv('bat_except_all.csv', index = False)
except_bat.to_csv('bat_except_21.csv', index = False)




bat_corr = bat.corr()
bat_corr_ops = bat.corr().OPS
bat_corr_ops.drop(['OPS'], inplace = True)
top_corr_ops = bat_corr_ops.sort_values(ascending = False).head(10)

bat.drop(['GYEAR', 'PCODE', 'GAMENUM', 'PA', 'AB', 'GD', 'SF', 'KK'], axis = 1 ,inplace= True)

plt.bar(x = top_corr_ops.index, height = top_corr_ops.values)
plt.tight_layout()
plt.show()

sns.heatmap(bat.corr(), annot=True)
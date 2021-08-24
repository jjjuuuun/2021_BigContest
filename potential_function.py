import pandas as pd
import os
import numpy as np

os.chdir('C:/Users/KIMJUNYOUNG/Desktop/BC_data/train_data/')
print(os.listdir())
print(os.getcwd())
player_2018 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_선수_2018.csv', encoding='CP949')
player_2019 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_선수_2019.csv', encoding='CP949')
player_2020 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_선수_2020.csv', encoding='CP949')
player_2021 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_선수_2021.csv', encoding='CP949')

# 포지션이 투수인 player를 제거
batter_2018 = player_2018[~(player_2018['POSITION'] == '투')]
batter_2019 = player_2019[~(player_2019['POSITION'] == '투')]
batter_2020 = player_2020[~(player_2020['POSITION'] == '투')]
batter_2021 = player_2021[~(player_2021['POSITION'] == '투')]

# (324, 7) (335, 7) (344, 7) (311, 7)
print(batter_2018.shape, batter_2019.shape, batter_2020.shape, batter_2021.shape)

batter_2018.reset_index(drop=True, inplace = True)
batter_2019.reset_index(drop=True, inplace = True)
batter_2020.reset_index(drop=True, inplace = True)
batter_2021.reset_index(drop=True, inplace = True)

# column 'MONEY'에서 만원과 달러와 같은 단위는 제거하고 dtype을 int로 바꿈
def int_money(df):
    for idx in range(df.shape[0]):
        try:
            if '만원' in df.MONEY[idx]:
                df.MONEY[idx] = df.MONEY[idx].replace('만원', '')
            elif '달러' in df.MONEY[idx]:
                df.MONEY[idx] = df.MONEY[idx].replace('달러', '')
        except:
            df.MONEY[idx] = '0'

batter_list = [batter_2018, batter_2019, batter_2020, batter_2021]
for df in batter_list:
    int_money(df)

batter_2018 = batter_2018.astype({'MONEY':'int'})
batter_2019 = batter_2019.astype({'MONEY':'int'})
batter_2020 = batter_2020.astype({'MONEY':'int'})
batter_2021 = batter_2021.astype({'MONEY':'int'})

# MONEY가 NaN이어서 제거(여기서는 MONEY가 0인 타자를 제거)
batter_2018 = batter_2018[~(batter_2018['MONEY'] == 0)]
batter_2019 = batter_2019[~(batter_2019['MONEY'] == 0)]
batter_2020 = batter_2020[~(batter_2020['MONEY'] == 0)]
batter_2021 = batter_2021[~(batter_2021['MONEY'] == 0)]

batter_2018.reset_index(drop=True, inplace = True)
batter_2019.reset_index(drop=True, inplace = True)
batter_2020.reset_index(drop=True, inplace = True)
batter_2021.reset_index(drop=True, inplace = True)

# =============================================================================
# before : 전년도, after : 다음년도
# 전년도(before)와 그 다음년도(after)에 경기를 치뤄 연봉이 집계된 타자들의 potential을 new_df로 또는 after 년도에만 경기를 치룬 타자들의 potential을 new_df로 저장
# 전년도(before)에는 경기를 치루지 않았지만 그 다음년도(after)에는 경기를 치룬 타자들의 potential을 0.0으로 assign하여 new_non_df로 저장
# =============================================================================
def potential(before, after):
    new_df = pd.DataFrame(index=range(after.shape[0]), columns = ['PCODE','POTENTIAL'])
    new_non_df = pd.DataFrame(index=range(before.shape[0]), columns = ['PCODE', 'MONEY'])
    for idx in range(after.shape[0]):
        after_pcode = after.PCODE[idx]
        before_pcode = before[(before['PCODE'] == after_pcode)]
        if before_pcode.empty == False: # after년도에 경기를 치룬 타자가 before년도에도 경기를 치루었다면 before_pcode에는 그 타자의 정보가 dataframe으로 들어가 있다.
            after_money = after.MONEY[idx]
            before_money = before_pcode.MONEY.iloc[0]
            potential = round((after_money - before_money) / before_money, 3)
        else: # before년도에는 경기를 치루지 않았지만 after년도에는 경기를 치룬사람의 potential을 0.0으로 assign
            potential = 0.0
        new_df.PCODE[idx] = after_pcode
        new_df.POTENTIAL[idx] = potential
    for idx in range(before.shape[0]): # after년도에는 경기를 치루지 않았지만 before년도에는 경기를 치룬 타자의 PCDOE와 MONEY를 new_non_df에 저장
        before_pcode = before.PCODE[idx]
        after_pcode = after[(after['PCODE'] == before_pcode)]
        if after_pcode.empty == True:
            new_non_df.PCODE[idx] = before_pcode
            new_non_df.MONEY[idx] = before.MONEY[idx]
    new_non_df.dropna(axis = 0, inplace=True)
    new_non_df.reset_index(drop=True, inplace=True)
    return new_df, new_non_df


# =============================================================================
# before : 이전년도 after : 그 후 년도
# 위의 함수 potential로부터 new_non_df를 받아 계산
# 예시) non_potential(non_potential_18_19, batter_2020)
# 18년도에 경기를 치루었지만 19년도에 경기를 치루지 않은 타자가 20년도에 경기를 치루었다면 potential를 계산해서 new_df로 저장
# 그렇지 않고 20년에도 경기를 치루지 않았다면 다음년도인 21년도와 비교를 위해 new_non_df로 저장
# =============================================================================
def non_potential(before, after):
    new_df = pd.DataFrame(index=range(before.shape[0]), columns = ['PCODE','POTENTIAL'])
    new_non_df = pd.DataFrame(index=range(before.shape[0]), columns = ['PCODE', 'MONEY'])
    for idx in range(before.shape[0]):
        before_pcode = before.PCODE[idx]
        after_pcode = after[(after['PCODE'] == before_pcode)]
        if after_pcode.empty == False:
            before_money = before.MONEY[idx]
            after_money = after_pcode.MONEY.iloc[0]
            potential = round((after_money - before_money) / before_money, 3)
        else:
            potential = None
        new_df.PCODE[idx] = before_pcode
        new_df.POTENTIAL[idx] = potential
        if potential == None:
            new_non_df.PCODE[idx] = before_pcode
            new_non_df.MONEY[idx] = before.MONEY[idx]
    new_df.dropna(axis=0, inplace=True)
    new_df.reset_index(drop=True, inplace=True)
    new_non_df.dropna(axis=0, inplace=True)
    new_non_df.reset_index(drop=True, inplace=True)
    return new_df, new_non_df

# =============================================================================
# potential_18_19     : 18, 19년도 모두 경기를 치룬 타자 그리고 19년도만 경기를 치룬 타자의 19년도 potential
# non_potential_18_19 : 18년도에 경기를 치루고 19년도에는 경기를 치루지 않은 타자들의 정보(PCODE, MONEY)
# potential_19_20     : 19, 20년도 모두 경기를 치룬 타자 그리고 20년도만 경기를 치룬 타자의 20년도 potential
# non_potential_19_20 : 19년도에 경기를 치루고 20년도에는 경기를 치루지 않은 타자들의 정보(PCODE, MONEY)
# potential_20_21     : 20, 21년도 모두 경기를 치룬 타자 그리고 21년도만 경기를 치룬 타자의 21년도 potential
# non_potential_20_21 : 20년도에 경기를 치루고 21년도에는 경기를 치루지 않은 타자들의 정보(PCODE, MONEY)
#
#
# potential_18_20     : 18년도에는 경기를 치루고 19년도에는 경기를 치루지 않고 20년도에는 경기를 치룬 타자의 20년도 potential
# non_potential_18_20 : 18년도에는 경기를 치루고 19년도 & 20년도에는 경기를 치루지 않은 타자들의 정보(PCODE, MONEY)
# potential_18_21     : 18년도에는 경기를 치루고 19년도 & 20년도에는 경기를 치루지 않고 21년도에는 경기를 치룬 타자의 21년도 potential
# non_potential_18_21 : 18년도에는 경기를 치루고 19년도 & 20년도 & 21년도에는 경기를 치루지 않은 타자들의 정보(PCODE, MONEY)
# potential_19_21     : 19년도에는 경기를 치루고 20년도에는 경기를 치루지 않고 21년도에는 경기를 치룬 타자의 21년도 potential
# non_potential_19_21 : 19년도에는 경기를 치루고 20년도 & 21년도에는 경기를 치루지 않은 타자들의 정보(PCODE, MONEY)
# =============================================================================
potential_18_19, non_potential_18_19 = potential(batter_2018, batter_2019)
potential_19_20, non_potential_19_20 = potential(batter_2019, batter_2020)
potential_20_21, non_potential_20_21 = potential(batter_2020, batter_2021)

potential_18_20, non_potential_18_20 = non_potential(non_potential_18_19, batter_2020)
potential_18_21, non_potential_18_21 = non_potential(non_potential_18_20, batter_2021)
potential_19_21, non_potential_19_21 = non_potential(non_potential_19_20, batter_2021)

# 19년도의 potential, 20년도의 potential, 21년도의 potential를 concat으로 합치기
potential_2019 = potential_18_19
potential_2020 = pd.concat([potential_19_20, potential_18_20], ignore_index=True)
potential_2021 = pd.concat([potential_20_21, potential_18_21, potential_19_21], ignore_index=True)

# 혹시나 있을 중복값 확인
dup_potential_2020 = potential_2020.duplicated(['PCODE'])
dup_potential_2021 = potential_2021.duplicated(['PCODE'])


# 아직 csv 파일 만들지 않음
potential_2019.to_csv('potential_2019.csv', index = False)
potential_2020.to_csv('potential_2020.csv', index = False)
potential_2021.to_csv('potential_2021.csv', index = False)
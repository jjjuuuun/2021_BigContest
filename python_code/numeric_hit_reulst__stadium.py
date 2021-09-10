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

import pandas as pd
import os

os.chdir('C:/Users/KIMJUNYOUNG/Desktop/BC_data/train_data/')
hts_2018 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_HTS_2018.csv', encoding='CP949')
hts_2019 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_HTS_2019.csv', encoding='CP949')
hts_2020 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_HTS_2020.csv', encoding='CP949')
hts_2021 = pd.read_csv('2021 빅콘테스트_데이터분석분야_챔피언리그_스포츠테크_HTS_2021.csv', encoding='CP949')

hit_result = {'1루타':1, '2루타':2, '3루타':3, '내야안타(1루타)':4, '땅볼아웃':5, '번트아웃':6,'번트안타':7,'병살타':8,
              '삼중살타':9, '야수선택':10,'인필드플라이':11, '직선타':12,'파울플라이':13,'플라이':14,'홈런':15,'희생번트':16,'희생플라이':17}

def numeric_hit_result(x):
    return hit_result[x]

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

def numeric_stadium(x):
    return stadium[x]

hts_2018['STADIUM'] = hts_2018['STADIUM'].apply(numeric_stadium)
hts_2019['STADIUM'] = hts_2019['STADIUM'].apply(numeric_stadium)
hts_2020['STADIUM'] = hts_2020['STADIUM'].apply(numeric_stadium)
hts_2021['STADIUM'] = hts_2021['STADIUM'].apply(numeric_stadium)

hts_2018.to_csv('hts_2018.csv', index = False)
hts_2019.to_csv('hts_2019.csv', index = False)
hts_2020.to_csv('hts_2020.csv', index = False)
hts_2021.to_csv('hts_2021.csv', index = False)
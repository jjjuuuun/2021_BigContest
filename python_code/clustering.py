from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

os.chdir('C:/Users/KIMJUNYOUNG/Desktop/BC_data/train_data/')
hts_2018 = pd.read_csv('hts_2018.csv')
hts_2019 = pd.read_csv('hts_2019.csv')
hts_2020 = pd.read_csv('hts_2020.csv')
hts_2021 = pd.read_csv('hts_2021.csv')

hts = pd.concat([hts_2018, hts_2019, hts_2020, hts_2021], ignore_index=True)

hts.drop(['GYEAR', 'G_ID', 'PIT_ID', 'PCODE', 'T_ID', 'INN', 'STADIUM','HIT_RESULT', 'PIT_VEL'], axis=1, inplace=True)

# 스케일링
hts_copy = hts.copy()
scaler = MinMaxScaler()
hts = scaler.fit_transform(hts)
hts = pd.DataFrame(hts, columns = hts_copy.columns, index = list(hts_copy.index.values))

# =============================================================================
# inertia
# 군집내 데이터들과 중심과의 거리의 합으로 군집의 응집도를 나타내는 값이다.
# 값이 작을 수록 군집화가 잘되었다고 평가할 수 있다.
# 군집 단위 별로 interia 값을 조회한 후 급격히 떨어지는 지점이 적정 군집수라 판단 할 수 있다.
# =============================================================================

# =============================================================================
# Elbow Method
# Cluster 간의 거리의 합을 나타내는 interia가 급격히 떨어지는 구간이 생기는데
# 이 지점의 K 값을 군집의 개수로 사용
# =============================================================================
inertia_list = []
k_range = range(2,15)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=200)
    kmeans.fit(hts)
    inertia = kmeans.inertia_
    print('k :', k,'inertia :', inertia)
    inertia_list.append(inertia)

inertia_arr = np.array(inertia_list)

# inertia_list 기울기 구하기
gradient_list = []
for idx in range(len(inertia_list)-1):
    gradient = inertia_list[idx+1] - inertia_list[idx]
    gradient_list.append(gradient)

# gradient_list 기울기 구하기(inertia_list 기울기의 변화율)
max_double_gradient = 0
idx_double_gradient = 0
for idx in range(len(gradient_list)-1):
    double_gradient = gradient_list[idx+1] - gradient_list[idx]
    if idx == 0:
        max_double_gradient = double_gradient
        idx_double_gradient = idx
    elif max_double_gradient < double_gradient:
        max_double_gradient = double_gradient
        idx_double_gradient = idx
    print(double_gradient, max_double_gradient, idx_double_gradient)

# Elbow Method 시각화
plt.plot(k_range, inertia_arr)
plt.vlines(idx_double_gradient+2, ymin=inertia_arr.min()*0.999, ymax=inertia_arr.max()*1.0003, linestyles='--', colors='g')
plt.vlines(idx_double_gradient+3, ymin=inertia_arr.min()*0.999, ymax=inertia_arr.max()*1.0003, linestyles='--', colors='r')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Elbow Method 결과
# 위 그래프에 의하면 K 값은 2 또는 3이 적당해 보인다.

# =============================================================================
# 손실함수
# K-Means Clustering으로 만든 feature를 Model에 사용하기 때문에 K 값을 하나의 hyperparameter로 보고
# 평가 점수가 가장 좋게 나오는 K를 선택해서 사용
# =============================================================================

# =============================================================================
# Silhouette Method
# 군집타당성지표인 실루엣 점수를 사용한다
# 1에 가까울수록 적절한 군집화가 되었다고 판단한다.
# =============================================================================
k_range = range(2,30)

best_n = -1
best_silhouette_score = -1

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=200)
    kmeans.fit(hts)
    clusters = kmeans.predict(hts)
    
    score = silhouette_score(hts, clusters)
    print('k :', k,'score :', score)
    
    if score > best_silhouette_score:
        best_n = k
        best_silhouette_score = score
        
print('best n :', best_n,'best score :', best_silhouette_score)


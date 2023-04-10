import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# 데이터 파일을 읽어들입니다.
with open('/content/drive/MyDrive/2023_AI_TSP.csv', "r", encoding='utf-8-sig') as file:
    reader = csv.reader(file)
    data = [tuple(map(float, row)) for row in reader]

start_point = data[0]

# 데이터를 numpy 배열로 변환합니다.
data_np = np.array(data)

# K-means 알고리즘을 사용하여 데이터를 3개의 클러스터로 나눕니다.
kmeans = KMeans(n_clusters=3, random_state=0).fit(data_np)

# 각 점에 대한 클러스터 레이블을 얻습니다.
labels = kmeans.labels_

# 결과를 저장할 데이터프레임을 생성합니다.
result_df = pd.DataFrame(data_np, columns=['x', 'y'])
result_df['cluster'] = labels

# 클러스터별로 데이터를 분할합니다.
cluster_0 = result_df[result_df['cluster'] == 0]
cluster_1 = result_df[result_df['cluster'] == 1]
cluster_2 = result_df[result_df['cluster'] == 2]

# 클러스터별 데이터를 출력합니다.
print("Cluster 0:\n", cluster_0)
print("Cluster 1:\n", cluster_1)
print("Cluster 2:\n", cluster_2)


# 각 클러스터의 좌표를 추출합니다.
x0, y0 = cluster_0['x'], cluster_0['y']
x1, y1 = cluster_1['x'], cluster_1['y']
x2, y2 = cluster_2['x'], cluster_2['y']

# 그래프를 그리고 각 클러스터의 좌표를 다른 색상으로 표시합니다.
plt.figure(figsize=(10, 8))
plt.scatter(x0, y0, c='red', label='Cluster 0')
plt.scatter(x1, y1, c='blue', label='Cluster 1')
plt.scatter(x2, y2, c='green', label='Cluster 2')

# 시작 좌표를 검은색 별표로 표시합니다.
plt.scatter(start_point[0], start_point[1], c='black', marker='*', s=150, label='Start Point')

# 축 레이블 및 범례를 추가하고 그래프를 표시합니다.
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='best')
plt.show()

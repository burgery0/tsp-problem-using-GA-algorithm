import numpy as np
import csv

cities = []

sol = []

def distance(x, y):
    dist = np.linalg.norm(np.array(x)-np.array(y))
    return dist


with open('/content/drive/MyDrive/result.csv', mode='r', encoding='cp949', newline='') as solution:
    reader = csv.reader(solution)
    for row in reader:
        sol.append(int(row[0]))

idx = sol.index(0)
front = sol[idx:]
back = sol[0:idx]
sol = front + back
sol.append(int(0))

with open('/content/drive/MyDrive/2023_AI_TSP.csv', mode='r', encoding='utf-8-sig', newline='') as tsp:
    reader = csv.reader(tsp)
    for row in reader:
        cities.append(row)

total_cost = 0

for idx in range(len(sol)-1):
    pos_city_1 = [float(cities[sol[idx]][0]), float(cities[sol[idx]][1])]
    pos_city_2 = [float(cities[sol[idx+1]][0]), float(cities[sol[idx+1]][1])]
    dist = distance(pos_city_1, pos_city_2)
    total_cost += dist

print('final cost : ' + str(total_cost))

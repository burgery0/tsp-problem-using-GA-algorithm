##  1. 완전 무작위 탐색 (생성): 해를 일정 시간 동안 무작위로 생성해서 좋은 것을 선택하는 방법
##  2. 최소 GA iteration 횟수정도는 보장되어야 합리적이겠지요.
##  3. 시작 노드의 경우 ‘2023_AI_TSP.csv’의 첫 번째 행 값으로 할 것
##  4. 설명을 위해선 객체지향 설계를 지향함 (e.g., 기능 별로 클래스를 구현)

import random
import time
import csv
import matplotlib.pyplot as plt
import math


class tsp_random_search:
    def __init__(self, csvfile, max_time=10, n=1000):
        self.max_time = max_time  # 최대 실행 시간
        self.n = n  # 데이터셋 개수
        self.cities = self._read_csv(csvfile)  # 도시 위치 리스트
        self.start = self.cities[0]  # 시작 지점을 첫 번째 좌표로 고정
        self.cities = self.cities[1:] # 두번째 좌표부터 끝까지 추출
        random.shuffle(self.cities) # 무작위로 섞음
        self.cities.insert(0, self.start) # 맨 앞에 첫 번째 좌표 추가
        self.count = 0  # 현재 반복 횟수
        self.best_solution = None  # 현재까지 찾은 최적의 해
        self.best_distance = math.inf  # 현재까지 찾은 최적의 거리
        self.best_distances = []  # 최적 거리값 리스트

    def _read_csv(self, csvfile):
        with open(csvfile, encoding='utf-8-sig') as csvfile:
            newfile = csv.reader(csvfile)
            cities = []
            for i in newfile:
                cities.append((float(i[0]), float(i[1])))
            return cities

    def _calc_distance(self, solution): # 모든 경로의 유클리디안 distance를 계산
        distance = 0
        for i in range(self.n-1):
            distance += math.sqrt(
                (((self.cities[solution[i]][0] - self.cities[solution[i + 1]][0]) ** 2 +
                  (self.cities[solution[i]][1] - self.cities[solution[i + 1]][1]) ** 2)))
        distance += math.sqrt(((self.cities[solution[self.n - 1]][0] - self.cities[solution[0]][0]) ** 2 +
                               (self.cities[solution[self.n - 1]][1] - self.cities[solution[0]][1]) ** 2))
        return distance

    def solve(self):
        start_time = time.time()  # 시작 시간을 저장하는 변수
        # 무작위 탐색 루프
        while True:
            self.count += 1

            solution = list(range(1, self.n)) # 무작위 해 생성
            random.shuffle(solution) #무작위로 섞음
            solution = [0] + solution #맨 앞에 첫번째 도시를 배치


            distance = self._calc_distance(solution) # 무작위 해의 유클리디안 distance 값 계산


            if distance < self.best_distance: # 현재까지 찾은 최적의 해와 비교
                self.best_solution = solution
                self.best_distance = distance
                self.best_distances.append(self.best_distance)  # 최적 거리값 리스트에 추가

            if self.count % 20 == 0:  # 최소 GA iteration 횟수 보장을 위해 20번 반복마다 최적해 shuffle
                solution = self.best_solution[:]
                random.shuffle(solution)
                distance = self.best_distance

            if time.time() - start_time >= self.max_time:  # max_time에 도달시 종료
                break

        print(f"최적의 경로: {self.best_solution}")
        print(f"최적의 거리값: {self.best_distance}")
        print(f"list : {self.best_distances}")

        # 최적해 거리값 시각화
        plt.plot(self.best_distances)
        plt.title("Random Search")
        plt.ylabel("Distance")
        plt.show()


tsp = tsp_random_search('2023_AI_TSP.csv')

tsp.solve()

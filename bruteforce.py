import random
import time

def generate_random_tour(city_list):
    tour = city_list[:]
    random.shuffle(tour)
    return tour

def calculate_distance(city1, city2):
    x_distance = abs(city1[0] - city2[0])
    y_distance = abs(city1[1] - city2[1])
    distance = (x_distance ** 2 + y_distance ** 2) ** 0.5
    return distance

def calculate_tour_distance(city_list, tour):
    tour_distance = 0
    for i in range(len(tour)):
        city1 = city_list[tour[i]]
        city2 = city_list[tour[(i+1) % len(tour)]]
        tour_distance += calculate_distance(city1, city2)
    return tour_distance

def random_search(city_list, time_limit):
    start_time = time.time()
    best_tour = generate_random_tour(city_list)
    best_distance = calculate_tour_distance(city_list, best_tour)
    
    iterations = 1
    while time.time() - start_time < time_limit:
        tour = generate_random_tour(city_list)
        distance = calculate_tour_distance(city_list, tour)
        if distance < best_distance:
            best_tour = tour
            best_distance = distance
        
        iterations += 1
    
    # 최소 GA iteration 횟수를 보장하기 위해 남은 시간동안 iteration을 진행합니다.
    while time.time() - start_time < time_limit and iterations < 1000:
        tour = generate_random_tour(city_list)
        distance = calculate_tour_distance(city_list, tour)
        if distance < best_distance:
            best_tour = tour
            best_distance = distance
        
        iterations += 1
    
    return best_tour, best_distance

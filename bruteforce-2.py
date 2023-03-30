import random
import time

def generate_random_solution(num_cities):
    cities = list(range(num_cities))
    random.shuffle(cities)
    return cities

def calculate_path_length(path, distances):
    total_length = 0
    for i in range(len(path) - 1):
        total_length += distances[path[i]][path[i+1]]
    total_length += distances[path[-1]][path[0]]
    return total_length

def random_search(distances, num_cities, time_limit):
    start_time = time.time()
    best_path = generate_random_solution(num_cities)
    best_length = calculate_path_length(best_path, distances)
    num_iterations = 1
    
    while time.time() - start_time < time_limit:
        path = generate_random_solution(num_cities)
        length = calculate_path_length(path, distances)
        if length < best_length:
            best_length = length
            best_path = path
        num_iterations += 1
        
    return best_path, best_length, num_iterations

import pandas as pd
import numpy as np
import random

df = pd.read_csv('/content/drive/My Drive/2023_AI_TSP.csv', header=None)
coords_cache = {i: tuple(df.iloc[i]) for i in range(len(df))}

start_node = tuple(df.iloc[0].tolist())
print(start_node[0])
def create_population(n):
    population = []
    for i in range(n):
        path = list(range(1, len(df)))
        random.shuffle(path)
        path.insert(0, start_node[0]) # 시작지점 고정
        population.append(path)
    return population

def fitness(path):
    total_distance = 0
    for i in range(len(path)):
        current_city = path[i]
        if i == len(path)-1:
            next_city = path[0]
        else:
            next_city = path[i+1]
        current_city_coords = coords_cache[current_city]
        next_city_coords = coords_cache[next_city]
        distance = np.sqrt((current_city_coords[0]-next_city_coords[0])**2 + (current_city_coords[1]-next_city_coords[1])**2)
        total_distance += distance
    
    
    return 1/total_distance

def selection(population, n_parents):
    fitness_scores = [fitness(path) for path in population]
    parents_indices = np.random.choice(len(population), n_parents, replace=False, p=np.array(fitness_scores)/sum(fitness_scores))
    parents = [population[i] for i in parents_indices]
    return parents

def order_crossover(parents):
    parent1, parent2 = parents
    length = len(parent1)
    start_index = random.randint(0, length - 1)
    end_index = random.randint(start_index + 1, length)
    child = [-1] * length
    child[start_index:end_index] = parent1[start_index:end_index]
    remaining_cities = [city for city in parent2 if city not in child]
    j = 0
    for i in range(length):
        if child[i] == -1:
            child[i] = remaining_cities[j]
            j += 1
    return child

def mutation(child, mutation_rate):
    for i in range(len(child)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(child)-1)
            child[i], child[j] = child[j], child[i]
    return child

def replacement(population, offspring):
    combined_population = population + offspring
    fitness_scores = [fitness(path) for path in combined_population]
    sorted_indices = np.argsort(fitness_scores)[::-1]
    new_population = [combined_population[i] for i in sorted_indices[:len(population)]]
    return new_population

def genetic_algorithm(n_generations, population_size, n_parents, mutation_rate):
    population = create_population(population_size-1) # 시작빼고 생성
    population.insert(0, list(range(len(df))))  # 시작지점 추가
    best_fitness = 0
    best_path = []
    for i in range(n_generations):
        parents = selection(population, n_parents)
        offspring = []
        for j in range(population_size):
            child = order_crossover(random.sample(parents, 2))
            child = mutation(child, mutation_rate)
            offspring.append(child)
        population = replacement(population, offspring)
        current_best_fitness = max([fitness(path) for path in population])
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_path = population[np.argmax([fitness(path) for path in population])]
        print(f'{best_path}')
        print(f'Generation {i+1} - Best Fitness: {best_fitness:f}') #.4f -> f
    return best_path

best_path = genetic_algorithm(n_generations=20, population_size=200, n_parents=20, mutation_rate=0.2)







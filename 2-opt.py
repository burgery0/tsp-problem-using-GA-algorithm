import csv,time
import random
import numpy as np
from collections import defaultdict
start_time = time.time()

# 파일에서 데이터를 읽어오기
with open('/content/drive/MyDrive/2023_AI_TSP.csv', "r", encoding='utf-8-sig') as file:
    reader = csv.reader(file)
    data = [tuple(map(float, row)) for row in reader]

start_point = data[0]

class TreeNode:
    def __init__(self, path):
        self.path = path
        self.left = None
        self.right = None

    def evaluate_path(self, data):
        """
        현재 노드의 경로를 평가하는 함수
        """
        path = self.path
        total_distance = 0
        for i in range(len(path) - 1):
            idx1, idx2 = int(path[i]), int(path[i + 1])
            total_distance += distance(data[idx1], data[idx2])
        idx1, idx2 = int(path[-1]), int(path[0])
        total_distance += distance(data[idx1], data[idx2])
        return total_distance

    def get_inorder(self):
        """
        노드를 중위순회하며 노드의 값들을 리스트에 저장하여 반환
        """
        if self.left is not None:
            left_inorder = self.left.get_inorder()
        else:
            left_inorder = []

        if self.right is not None:
            right_inorder = self.right.get_inorder()
        else:
            right_inorder = []

        return left_inorder + [self.path[0]] + right_inorder

    def subtree_replace(self, target, replacement):
        """
        타겟 노드와 대체할 노드를 입력으로 받아 타겟 노드의 부모 노드를 찾아 대체할 노드를 대입
        """
        if self.left is not None:
            self.left = self.left.subtree_replace(target, replacement)
        if self.right is not None:
            self.right = self.right.subtree_replace(target, replacement)
        if self.path[0] == target:
            return TreeNode([replacement] + self.path[1:])
        return self

def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def fitness(tree_node):
    individual = tree_node.path
    total_distance = 0
    for i in range(len(individual) - 1):
        idx1, idx2 = int(individual[i]), int(individual[i + 1])
        total_distance += distance(data[idx1], data[idx2])
    idx1, idx2 = int(individual[-1]), int(individual[0])
    total_distance += distance(data[idx1], data[idx2])
    return total_distance  # 거리가 짧을수록 적합도가 높음

def create_initial_population(population_size, data):
    population = []
    for _ in range(population_size):
        path = list(range(1, len(data)))  # 모든 지점들의 인덱스
        random.shuffle(path)  # 무작위로 섞기
        path.insert(0, 0)  # 시작 노드는 고정으로 추가
        population.append(TreeNode(path))
    return population

def tournament_selection(population, tournament_size):
    selected_indices = random.choices(range(len(population)), k=tournament_size)
    selected_individuals = [(i, population[i]) for i in selected_indices if i < len(population)]
    if not selected_individuals:
        selected_individuals = [(i, population[i]) for i in range(len(population))]
    best_individual = min(selected_individuals, key=lambda x: fitness(x[1]))
    return best_individual[0]

def subtree_crossover(parent1, parent2):
    # 랜덤하게 부모 노드 선택
    crossover_point1 = random.choice(parent1.get_inorder())
    crossover_point2 = random.choice(parent2.get_inorder())

    # 선택한 노드들의 서브트리 교환
    child1 = parent1.subtree_replace(crossover_point1, crossover_point2)
    child2 = parent2.subtree_replace(crossover_point2, crossover_point1)

    return child1, child2

def add_node(tree_node, node_value):
    if isinstance(tree_node, TreeNode) and node_value not in tree_node.path:
        new_node = TreeNode(tree_node.path + [node_value])
        tree_node.add_child(new_node)
        return new_node
    elif isinstance(tree_node, list) and node_value not in tree_node:
        new_node = [node_value]
        tree_node.append(new_node)
        return new_node
    return None

def add_subtree(tree_node, crossover_node, donor_path):
    if isinstance(tree_node, TreeNode):
        subtree_root = add_node(tree_node, crossover_node)
    elif isinstance(tree_node, list):
        subtree_root = tree_node
    else:
        return

    if subtree_root is None:
        for child in tree_node.children:
            if child.path[-1] == crossover_node:
                subtree_root = child
                break

    for i, node_value in enumerate(donor_path):
        if node_value == crossover_node:
            continue
        if isinstance(subtree_root, TreeNode):
            subtree_root = add_node(subtree_root, node_value)
        elif isinstance(subtree_root, list) and node_value not in subtree_root:
            subtree_root.append(node_value)


def mutation(tree_node, mutation_rate):
    individual = tree_node.path
    for i in range(1, len(individual) - 1):  # 첫 번째 위치를 건너뛰고 범위 선택
        if random.random() < mutation_rate:
            j = random.randint(1, len(individual) - 2)
            individual[i], individual[j] = individual[j], individual[i]
    return TreeNode(individual)

def evaluate_population(population):
    fitness_values = []
    for tree_node in population:
        fitness_values.append(fitness(tree_node))
    return fitness_values

def local_search(tree_node):
    
    individual = tree_node.path
    n = len(individual) - 1
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue
                new_individual = individual[:i] + individual[i:j][::-1] + individual[j:]
                new_fitness = fitness(TreeNode(new_individual))
                if new_fitness < fitness(tree_node):
                    tree_node = TreeNode(new_individual)
                    individual = new_individual
                    improved = True
                    print(f"거리: {new_fitness} 지역 최적화 진행 상황: {tree_node.path}")  # 진행 상황 출력
                    break
            if improved:
                break
    return tree_node


def replacement(population, new_population, fitness_values, new_fitness_values):
    combined_population = population + new_population
    combined_fitness_values = fitness_values + new_fitness_values

    sorted_indices = np.argsort(combined_fitness_values)
    new_indices = sorted_indices[:len(population)]

    updated_population = [combined_population[i] for i in new_indices]
    updated_fitness_values = [combined_fitness_values[i] for i in new_indices]

    # 지역 최적화 적용 (상위 개체에만 적용)
    num_local_search = int(len(updated_population) * 0.02)  # 상위 10% 개체에만 적용
    for i in range(num_local_search):
      updated_population[i] = local_search(updated_population[i])

    if updated_fitness_values[0] != fitness_values[0]:
        population, fitness_values = updated_population, updated_fitness_values

    return population, fitness_values

def check_termination(population, threshold=0.8):
    quality_counts = defaultdict(int)

    for individual in population:
        tour_len = fitness(individual)
        quality_counts[tour_len] += 1

    max_count = max(quality_counts.values())
    return max_count / len(population) >= threshold



if __name__ == "__main__":
    population_size = 100
    generations = 1000
    mutation_rate = 0.01
    tournament_size = 5

    population = create_initial_population(population_size, data)
    fitness_values = evaluate_population(population)

    for generation in range(generations):

        new_population = []

        for _ in range(population_size):

            

            parent1_idx = tournament_selection(population, tournament_size)
            parent2_idx = tournament_selection(population, tournament_size)

            child1, child2 = subtree_crossover(population[parent1_idx], population[parent2_idx])

            child1 = mutation(child1, mutation_rate)
            child2 = mutation(child2, mutation_rate)

            new_population.append(child1)
            new_population.append(child2)

        new_fitness_values = evaluate_population(new_population)
        population, fitness_values = replacement(population, new_population, fitness_values, new_fitness_values)

        best_individual_idx = np.argmin(fitness_values)
        best_individual = population[best_individual_idx]
        best_fitness = fitness_values[best_individual_idx]
        print(f"Generation {generation}: Total distance: {best_fitness} Path: {best_individual.path}")

        if check_termination(population):
            print(f"Termination condition met at generation {generation}")
            break

    best_individual_idx = np.argmin(fitness_values)
    best_individual = population[best_individual_idx]
    best_fitness = fitness_values[best_individual_idx]

    print(f"Best solution found at generation {generation}:")
    print(f"Path: {best_individual.path}")
    print(f"Total distance: {best_fitness}")


with open('/content/drive/MyDrive/best_path.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([0])  # 시작 노드를 첫 행에 추가
    for city_idx in best_individual.path[1:]:  # 시작 노드는 이미 추가되었으므로 생략
        writer.writerow([city_idx])

end_time = time.time()
print("실행 시간 :", end_time - start_time)

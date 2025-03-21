import random
import numpy as np

class WeddingSeatingGA:
    def __init__(self, guests, relationships, max_per_table):
        # Thông tin cơ bản
        self.guests = guests
        self.relationships = relationships
        self.max_per_table = max_per_table
        self.num_guests = len(guests)
        self.num_tables = (self.num_guests + max_per_table - 1) // max_per_table
        
        # Tham số GA - đã rút gọn
        self.population_size = 20
        self.num_generations = 25
        self.mutation_rate = 0.2
        self.elite_size = 2
        
        # Điểm số quan hệ
        self.relationship_scores = {
            'vợ/chồng/người yêu': 2000,
            'anh/chị/em ruột': 900,
            'cha/mẹ - con cái': 700,
            'anh chị em họ': 500,
            'dì/chú/bác - cháu': 300,
            'bạn bè': 100,
            'không quen biết': 0
        }
        
        # Khởi tạo quần thể
        self.population = self.initialize_population()
    
    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            # Tạo sắp xếp ngẫu nhiên
            assignment = list(range(self.num_guests))
            random.shuffle(assignment)
            # Phân bổ vào các bàn
            seating_plan = []
            for i in range(self.num_tables):
                start_idx = i * self.max_per_table
                end_idx = min((i + 1) * self.max_per_table, self.num_guests)
                table = assignment[start_idx:end_idx]
                seating_plan.append(table)
            population.append(seating_plan)
        return population
    
    def calculate_fitness(self, seating_plan):
        total_score = 0
        # Tính điểm cho từng bàn
        for table in seating_plan:
            # Xét từng cặp khách trong bàn
            for i in range(len(table)):
                for j in range(i + 1, len(table)):
                    guest1 = self.guests[table[i]]
                    guest2 = self.guests[table[j]]
                    # Kiểm tra mối quan hệ
                    relation_key = (guest1, guest2) if (guest1, guest2) in self.relationships else \
                                  (guest2, guest1) if (guest2, guest1) in self.relationships else None
                    if relation_key:
                        relation_type = self.relationships[relation_key]
                        total_score += self.relationship_scores.get(relation_type.lower(), 0)
        return total_score
    
    def crossover(self, parent1, parent2):
        # Chuyển đổi thành mảng 1 chiều
        parent1_flat = [guest for table in parent1 for guest in table]
        parent2_flat = [guest for table in parent2 for guest in table]
        # Chọn điểm cắt ngẫu nhiên
        crossover_point = random.randint(1, self.num_guests - 1)
        # Tạo con lai
        child_flat = parent1_flat[:crossover_point].copy()
        # Thêm các phần tử từ parent2 mà chưa có trong child
        for idx in parent2_flat:
            if idx not in child_flat:
                child_flat.append(idx)
        # Chuyển lại thành seating_plan
        child_seating = []
        for i in range(self.num_tables):
            start_idx = i * self.max_per_table
            end_idx = min((i + 1) * self.max_per_table, self.num_guests)
            table = child_flat[start_idx:end_idx]
            child_seating.append(table)
        return child_seating
    
    def mutate(self, seating_plan):
        if random.random() > self.mutation_rate:
            return seating_plan
        # Chuyển đổi thành mảng 1 chiều
        flat_assignment = [guest for table in seating_plan for guest in table]
        # Chọn hai vị trí ngẫu nhiên và đổi chỗ
        pos1, pos2 = random.sample(range(self.num_guests), 2)
        flat_assignment[pos1], flat_assignment[pos2] = flat_assignment[pos2], flat_assignment[pos1]
        # Chuyển lại thành seating_plan
        mutated_seating = []
        for i in range(self.num_tables):
            start_idx = i * self.max_per_table
            end_idx = min((i + 1) * self.max_per_table, self.num_guests)
            table = flat_assignment[start_idx:end_idx]
            mutated_seating.append(table)
        return mutated_seating
    
    def select_parent(self, fitness_scores):
        # Chọn cha mẹ sử dụng tournament selection - đơn giản hóa
        tournament_size = 2
        tournament_indices = random.sample(range(len(fitness_scores)), tournament_size)
        tournament_scores = [fitness_scores[i] for i in tournament_indices]
        return tournament_indices[tournament_scores.index(max(tournament_scores))]
    
    def evolve(self):
        # Tính điểm fitness cho quần thể
        fitness_scores = [self.calculate_fitness(individual) for individual in self.population]
        # Sắp xếp quần thể theo điểm fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        # Lưu lại elite
        elite = [self.population[idx].copy() for idx in sorted_indices[:self.elite_size]]
        # Tạo quần thể mới
        new_population = elite.copy()
        # Sinh các cá thể mới cho quần thể
        while len(new_population) < self.population_size:
            # Chọn cha mẹ
            parent1_idx = self.select_parent(fitness_scores)
            parent2_idx = self.select_parent(fitness_scores)
            # Đảm bảo khác nhau
            while parent2_idx == parent1_idx:
                parent2_idx = self.select_parent(fitness_scores)
            # Tạo con cái và đột biến
            child = self.crossover(self.population[parent1_idx], self.population[parent2_idx])
            child = self.mutate(child)
            new_population.append(child)
        self.population = new_population
        # Trả về cá thể tốt nhất
        best_idx = sorted_indices[0]
        return self.population[best_idx], fitness_scores[best_idx]
    
    def run(self):
        best_solution = None
        best_fitness = float('-inf')
        for generation in range(self.num_generations):
            solution, fitness = self.evolve()
            if fitness > best_fitness:
                best_solution = solution
                best_fitness = fitness
            # In tiến trình ít thường xuyên hơn
            if generation % 10 == 0:
                print(f"Thế hệ {generation}: Điểm tốt nhất = {best_fitness}")
        print(f"Giải pháp cuối cùng có điểm: {best_fitness}")
        return best_solution, best_fitness
    
    def format_solution(self, solution):
        result = []
        for i, table in enumerate(solution):
            guest_names = [self.guests[idx] for idx in table]
            result.append(f"Bàn {i+1}: {', '.join(guest_names)}")
        return result
    
    def analyze_solution(self, solution):
        results = []
        for table_idx, table in enumerate(solution):
            table_score = 0
            relations = []
            for i in range(len(table)):
                for j in range(i + 1, len(table)):
                    guest1 = self.guests[table[i]]
                    guest2 = self.guests[table[j]]
                    
                    relation_key = (guest1, guest2) if (guest1, guest2) in self.relationships else \
                                  (guest2, guest1) if (guest2, guest1) in self.relationships else None
                    if relation_key:
                        relation_type = self.relationships[relation_key]
                        score = self.relationship_scores.get(relation_type.lower(), 0)
                        table_score += score
                        if score > 0:
                            relations.append(f"{guest1} - {guest2}: {relation_type}")
            table_info = {
                "table_num": table_idx + 1,
                "score": table_score,
                "relationships": relations
            }
            results.append(table_info)
        return results

def main():
    # Danh sách khách mời (bao gồm người lạ)
    guests = [
        "Chú rể", "Cô dâu", 
        "Bố chú rể", "Mẹ chú rể", 
        "Bố cô dâu", "Mẹ cô dâu",
        "Anh chú rể", "Chị dâu",
        "Em gái cô dâu", "Bạn trai em gái cô dâu",
        "Bạn thân chú rể 1", "Bạn thân chú rể 2",
        "Bạn thân cô dâu 1", "Bạn thân cô dâu 2",
        "Chú của chú rể", "Cô của chú rể",
        "Dì của cô dâu", "Cậu của cô dâu",
        "Anh họ chú rể", "Chị họ cô dâu",
        # Khách lạ
        "Khách lạ 1", "Khách lạ 2", "Khách lạ 3"
    ]
    
    # Định nghĩa mối quan hệ
    relationships = {
        ("Chú rể", "Cô dâu"): "vợ/chồng/người yêu",
        ("Chú rể", "Bố chú rể"): "cha/mẹ - con cái",
        ("Chú rể", "Mẹ chú rể"): "cha/mẹ - con cái",
        ("Cô dâu", "Bố cô dâu"): "cha/mẹ - con cái",
        ("Cô dâu", "Mẹ cô dâu"): "cha/mẹ - con cái",
        ("Bố chú rể", "Mẹ chú rể"): "vợ/chồng/người yêu",
        ("Bố cô dâu", "Mẹ cô dâu"): "vợ/chồng/người yêu",
        ("Chú rể", "Anh chú rể"): "anh/chị/em ruột",
        ("Anh chú rể", "Chị dâu"): "vợ/chồng/người yêu",
        ("Cô dâu", "Em gái cô dâu"): "anh/chị/em ruột",
        ("Em gái cô dâu", "Bạn trai em gái cô dâu"): "vợ/chồng/người yêu",
        ("Chú rể", "Bạn thân chú rể 1"): "bạn bè",
        ("Chú rể", "Bạn thân chú rể 2"): "bạn bè",
        ("Bạn thân chú rể 1", "Bạn thân chú rể 2"): "bạn bè",
        ("Cô dâu", "Bạn thân cô dâu 1"): "bạn bè",
        ("Cô dâu", "Bạn thân cô dâu 2"): "bạn bè",
        ("Bạn thân cô dâu 1", "Bạn thân cô dâu 2"): "bạn bè",
        ("Chú rể", "Chú của chú rể"): "dì/chú/bác - cháu",
        ("Chú rể", "Cô của chú rể"): "dì/chú/bác - cháu",
        ("Cô dâu", "Dì của cô dâu"): "dì/chú/bác - cháu",
        ("Cô dâu", "Cậu của cô dâu"): "dì/chú/bác - cháu",
        ("Chú rể", "Anh họ chú rể"): "anh chị em họ",
        ("Cô dâu", "Chị họ cô dâu"): "anh chị em họ"
    }
    
    # Số người tối đa trên mỗi bàn
    max_per_table = 5
    
    # Chạy giải thuật
    ga = WeddingSeatingGA(guests, relationships, max_per_table)
    best_solution, best_fitness = ga.run()
    
    # In kết quả
    print("\nSơ đồ chỗ ngồi tối ưu:")
    for line in ga.format_solution(best_solution):
        print(line)
    
    # Phân tích kết quả
    print("\nPhân tích mức độ tương tác trong mỗi bàn:")
    analysis = ga.analyze_solution(best_solution)
    for table in analysis:
        print(f"Bàn {table['table_num']}:")
        print(f"  Điểm: {table['score']}")
        if table['relationships']:
            print("  Mối quan hệ:")
            for rel in table['relationships']:
                print(f"    - {rel}")
        else:
            print("  Không có mối quan hệ đặc biệt")
        print()

if __name__ == "__main__":
    main()
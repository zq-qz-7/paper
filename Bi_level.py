from collections import defaultdict, deque
import re
from itertools import product
import numpy as np
from math import inf
import datetime
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from time import perf_counter as pc


D_size = 19
P_size = 200
C_rate = 0.8
M_rate = 0.05
N_generation = 50

n_name = "SiouxFalls"
m_gap = 0.0001
l_gap = 0.0001
gap_interval = 50


class NODE:
    def __init__(self, node_id):
        self.node_id: int = node_id
        self.l_in: list[LINK] = []
        self.l_out: list[LINK] = []
        self.down_node: list[NODE] = []
        self.up_node: list[NODE] = []
        self.message_id: list[int] = []
        self.message: list[list[State]] = []
        self.message_prob: list[float] = []
        self.ett: float = 0  # Expected Traval Time
        self.vec_y: float = 0  # Number of travelers originating at node i

    def __repr__(self):
        return f"Node {self.node_id}"


class LINK:
    def __init__(self, link_id, head_node=None, tail_node=None, capacity=None,
                 length=None, fft=None, b=None, power=None):
        self.link_id: int = link_id
        self.head_node: NODE = head_node
        self.tail_node: NODE = tail_node
        self.capacity: float = capacity
        self.length: float = length
        self.fft: float = fft
        self.b: float = b
        self.power: float = power
        self.cost: float = 0
        self.flow: float = 0
        self.aux_flow: float = 0
        self.last_flow: float = 0
        self.state: list[State] = []

    def __repr__(self):
        return f"Link {self.link_id}"

    def update_cost(self):
        self.cost = self.fft * (1 + self.b * (self.flow / self.capacity) ** self.power)

    def obtain_cost(self, para):
        return self.fft * (1 + self.b * (para / self.capacity) ** self.power)


class State:
    def __init__(self, state_id, mother_link, prob, head_node, tail_node, capacity, length, fft, b, power):
        self.state_id: int = state_id
        self.mother_link: LINK = mother_link
        self.prob: float = prob
        self.head_node: NODE = head_node
        self.tail_node: NODE = tail_node
        self.capacity: float = capacity
        self.length: float = length
        self.fft: float = fft
        self.b: float = b
        self.power: float = power
        self.cost: float = 0
        self.flow: float = 0
        self.rho: float = 0  # the probability of leaving node i via link ij in state s
        self.aux_flow: float = 0

    def __repr__(self):
        return f"Link {self.mother_link.link_id} - state {self.state_id}"

    def update_cost(self):
        self.cost = self.fft * (1 + self.b * (self.flow / self.capacity) ** self.power)

    def obtain_cost(self, para):
        return self.fft * (1 + self.b * (para / self.capacity) ** self.power)


class POLICY:
    def __init__(self, des):
        self.des: NODE = des
        self.map: defaultdict[NODE:dict[int: NODE]] = defaultdict(dict)

    def __repr__(self):
        return f'{self.map}\n'


class ODPair:
    def __init__(self, origin, destination, demand):
        self.origin: NODE = origin
        self.destination: NODE = destination
        self.demand: float = demand

    def __repr__(self):
        return f"{self.origin.node_id}-{self.destination.node_id}: {self.demand}"


class Network:
    def __init__(self, name, strategy):
        self.name: str = name
        self.strategy: list = strategy
        self.Node: list[NODE] = []
        self.Link: list[LINK] = []
        self.link_State: list[State] = []
        self.Policy: dict[NODE: POLICY] = dict()
        self.OD: list[ODPair] = []
        self.Dest: list[NODE] = []
        self.num_node: int = 0
        self.num_link: int = 0
        self.num_link_states: int = 0
        self.read_net()
        self.read_od()

    def read_net(self):
        with open(f"TransportationNetworks\\{self.name}\\{self.name}_net.txt") as f:
            lines = f.readlines()
            pattern = re.compile(r'[0-9.a-zA-Z]+')
            lines = [pattern.findall(line) for line in lines if pattern.findall(line) != []]
            for i in range(len(lines)):
                line = lines[i]
                if "NUMBER" and "NODES" in line:
                    self.num_node = int(line[-1])
                if "NUMBER" and "LINKS" in line:
                    self.num_link = int(line[-1])
                if "capacity" in line:
                    lines = lines[i + 1:]
                    break
        self.Node = [NODE(i) for i in range(self.num_node + 1)]
        self.Link = [LINK(0)]
        for i in range(len(lines)):
            line = lines[i]
            head, tail = self.Node[int(line[1])], self.Node[int(line[0])]
            link = LINK(link_id=i + 1, head_node=head, tail_node=tail, capacity=float(line[2]),
                        length=int(line[3]), fft=float(line[4]), b=float(line[5]), power=int(line[6]))
            self.Link.append(link)
            head.l_in.append(link)
            tail.l_out.append(link)
            head.up_node.append(tail)
            tail.down_node.append(head)

            # Create Link_State
            if self.strategy[i]:
                state_1 = State(state_id=1, mother_link=link, prob=0.9, head_node=head, tail_node=tail,
                                capacity=float(line[2]) * 0.9, length=int(line[3]), fft=float(line[4]),
                                b=float(line[5]), power=int(line[6]))
                state_2 = State(state_id=2, mother_link=link, prob=0.1, head_node=head, tail_node=tail,
                                capacity=float(line[2]) * 0.1 * 0.5, length=int(line[3]), fft=float(line[4]),
                                b=float(line[5]), power=int(line[6]))
                link.state.extend((state_1, state_2))
                self.link_State.extend((state_1, state_2))
                self.num_link_states += 2
            else:
                state_provided = State(state_id=1, mother_link=link, prob=1, head_node=head, tail_node=tail,
                                       capacity=float(line[2]) * 0.9 * 0.9 + float(line[2]) * 0.1 * 0.5 * 0.1,
                                       length=int(line[3]), fft=float(line[4]), b=float(line[5]), power=int(line[6]))
                link.state.append(state_provided)
                self.link_State.append(state_provided)
                self.num_link_states += 1

            # Create Node_Message
        for node in self.Node[1:]:
            states = []
            for link in node.l_out:
                states.append(link.state)
            node.message = [list(comb) for comb in product(*states)]
            node.message_id = [i for i in range(len(node.message))]
            for link_states in node.message:
                prob = 1
                for i in range(len(link_states)):
                    prob *= link_states[i].prob
                node.message_prob.append(round(prob, 6))

    def read_od(self):
        with open(f"TransportationNetworks\\{self.name}\\{self.name}_trips.txt") as f:
            lines = f.readlines()
            pattern = re.compile(r'[0-9.a-zA-Z]+')
            lines = [pattern.findall(line) for line in lines if pattern.findall(line) != []]
            for i in range(len(lines)):
                line = lines[i]
                if "TOTAL" in line:
                    total_flow = float(line[-1])
                if "Origin" in line:
                    lines = lines[i:]
                    break
            for i in range(len(lines)):
                line = lines[i]
                if "Origin" in line:
                    origin = self.Node[int(line[-1])]
                else:
                    for j in range(len(line) // 2):
                        destination = self.Node[int(line[2 * j])]
                        demand = float(line[2 * j + 1])
                        self.OD.append(ODPair(origin, destination, demand))
                        if destination not in self.Dest:
                            self.Dest.append(destination)
            for des in self.Dest:
                self.Policy[des] = POLICY(des)
            if total_flow != sum([od.demand for od in self.OD]):
                print("demand is wrong!")

    def generate_b(self, des: NODE):
        #  Generate rho
        for link in self.Link[1:]:
            for state in link.state:
                state.rho = 0
        for node in self.Node[1:]:
            if node == des:
                continue
            for m_id in node.message_id:
                m_vec, m_prob = node.message, node.message_prob
                next_node = self.Policy[des].map[node][m_id]
                for state in m_vec[m_id]:
                    if state.head_node == next_node:
                        state.rho += m_prob[m_id]
                        break
        #  Generate y
        for node in self.Node[1:]:
            node.vec_y = 0
        for od in self.OD:
            if od.destination == des:
                od.origin.vec_y += od.demand
        #  Generate b
        vec_b = np.zeros(self.num_link_states)
        for i, state in enumerate(self.link_State):
            vec_b[i] = state.rho * state.tail_node.vec_y
        return vec_b

    def generate_TM(self):
        matrix_t = np.zeros(shape=(self.num_link_states, self.num_link_states))
        for row in self.link_State:
            i = self.link_State.index(row)
            for link in row.head_node.l_out:
                for state in link.state:
                    j = self.link_State.index(state)
                    matrix_t[i][j] = state.rho
        return matrix_t


class UE:
    def __init__(self, network, main_gap, ls_gap):
        self.network: Network = network
        self.main_gap: float = main_gap
        self.ls_gap: float = ls_gap
        self.total_time: float = 0
        self.gap_list = []
        self.run()
        self.total_travel_time()
        self.state_info()

    def TD_OSP(self, des: int):
        def reduce(xi_prime_in, lam_prime_in):
            lam_in = list(set(lam_prime_in))
            ele_prob = {element: 0 for element in lam_in}
            for element, prob in zip(lam_prime_in, xi_prime_in):
                ele_prob[element] += prob
            xi_in = list(ele_prob.values())
            lam_in = list(ele_prob.keys())
            return xi_in, lam_in

        # Step 1: initialize
        for node in self.network.Node[1:]:
            node.ett = inf
        self.network.Node[des].ett = 0
        sel = deque()
        for node in self.network.Node[des].up_node:
            sel.append(node)
        # Step 2
        while sel:
            node_i = sel.popleft()
            xi, lam = [1], [inf]
            for node_j in node_i.down_node:
                xi_prime, lam_prime = [], []
                link = None
                for ij in node_i.l_out:
                    if ij.head_node == node_j:
                        link = ij
                        break
                for l_state in link.state:
                    for k in range(len(xi)):
                        xi_prime.append(xi[k] * l_state.prob)
                        if l_state.cost + node_j.ett < lam[k]:
                            lam_prime.append(l_state.cost + node_j.ett)
                        else:
                            lam_prime.append(lam[k])
                xi, lam = reduce(xi_prime, lam_prime)
            cur = np.dot(xi, lam)
            if cur < node_i.ett:
                node_i.ett = cur
                sel.extend(node_i.up_node)
        # Step 3: Choose Optimal Policy
        for node in self.network.Node[1:]:
            for m_id in node.message_id:
                m_vec, m_pro = node.message[m_id], node.message_prob[m_id]
                if node == self.network.Node[des]:
                    self.network.Policy[self.network.Node[des]].map[node][m_id] = node
                else:
                    min_val = inf
                    next_node = -1
                    for state in m_vec:
                        if state.cost + state.head_node.ett < min_val:
                            min_val = state.cost + state.head_node.ett
                            next_node = state.head_node
                    self.network.Policy[self.network.Node[des]].map[node][m_id] = next_node

    def run(self):
        s = pc()
        self.initialize()
        gap = inf
        self.gap_list = []
        iter_times = 0
        while abs(gap) > self.main_gap:
            if iter_times % gap_interval == 0:
                print(f'iter_times{iter_times}: gap={gap}')
            self.all_or_nothing()
            gap = self.convergence()
            self.gap_list.append(gap)
            if iter_times == 0:
                step = 1
                gap = inf
            else:
                step = self.line_search()
            self.update_state_flow(step)
            self.update_state_cost()
            iter_times += 1
        e = pc()
        print(f"\nRunning time is {e - s:.2f}")

    def initialize(self):
        for state in self.network.link_State:
            state.flow = 0
        self.update_state_cost()

    def update_state_cost(self):
        for state in self.network.link_State:
            state.update_cost()

    def all_or_nothing(self):
        for state in self.network.link_State:
            state.aux_flow = 0
        for dest in self.network.Dest:
            self.TD_OSP(dest.node_id)
            vec_b = self.network.generate_b(dest)
            matrix_t = self.network.generate_TM()
            matrix_i = np.identity(self.network.num_link_states)
            matrix_inv_a = np.linalg.inv(matrix_i - matrix_t.T)
            temp = matrix_inv_a @ vec_b
            for i in range(len(self.network.link_State)):
                self.network.link_State[i].aux_flow += temp[i]

    def convergence(self):
        tstt = sum([state.flow * state.cost for state in self.network.link_State])
        sptt = 0
        for des in self.network.Dest:
            self.TD_OSP(des.node_id)
            for od in self.network.OD:
                if od.destination == des:
                    sptt += od.origin.ett * od.demand
        print(f'sptt={sptt}, tstt={tstt}')
        return 1 - (sptt / tstt)
        # numerator = sum([state.cost * state.flow for state in self.network.link_State])
        # denominator = sum([state.cost * state.aux_flow for state in self.network.link_State])
        # return numerator / denominator - 1

    def line_search(self):
        def derivative(step):
            res = 0
            for state in self.network.link_State:
                para = (1 - step) * state.flow + step * state.aux_flow
                res = res + (state.aux_flow - state.flow) * state.obtain_cost(para)
            return res

        left, mid, right = 0, 0.5, 1
        while abs(derivative(mid)) > self.ls_gap:
            if derivative(mid) * derivative(left) > 0:
                left = mid
            if derivative(mid) * derivative(right) > 0:
                right = mid
            mid = (right + left) / 2
        return mid

    def update_state_flow(self, step):
        for state in self.network.link_State:
            state.flow = (1 - step) * state.flow + step * state.aux_flow

    def total_travel_time(self):
        res = 0
        for state in self.network.link_State:
            if len(state.mother_link.state) == 1:
                state_1_cost = state.fft * (
                        1 + state.b * (state.flow * 0.9 / (state.mother_link.capacity * 0.9)) ** state.power)
                state_2_cost = state.fft * (
                        1 + state.b * (state.flow * 0.1 / (state.mother_link.capacity * 0.1 * 0.5)) ** state.power)
                state.cost = state_1_cost * 0.9 + state_2_cost * 0.1
                res += state.cost * state.flow
            else:
                res += state.cost * state.flow
        self.total_time = res
        print(f'\nTotal Travel Time: {self.total_time}\n')

    def state_info(self):
        for state in self.network.link_State:
            print(f'{state}: cost={state.cost}, flow={state.flow}')
        # cur_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # with open(f'data//UER{cur_time}.txt', 'a') as file:
        #     file.write(f'Network Strategy:\n{self.network.strategy}\n')
        #     file.write(f'Total Travel Time: {self.total_time}\n')
        #     for state in self.network.link_State:
        #         file.write(f'{state}: cost={state.cost * state.flow}, flow={state.flow}\n')


class Genetic:
    def __init__(self, dna_size, pop_size, c_rate, m_rate, n_generation):
        self.dna_size: int = dna_size
        self.pop_size: int = pop_size
        self.c_rate: float = c_rate
        self.m_rate: float = m_rate
        self.n_generation: int = n_generation
        self.population: list = []
        self.total_time: list = []
        self.obj_list: list = []
        self.avg_list: list = []
        self.run()

    def run(self):
        for _ in range(self.pop_size):
            ones = np.ones(10, dtype=int)
            zeros = np.zeros(9, dtype=int)
            combined_array = np.concatenate((ones, zeros))
            np.random.shuffle(combined_array)
            self.population.append(combined_array)
        self.population = np.random.randint(2, size=(self.pop_size, self.dna_size))
        iter_time = 1
        best_stra = []
        best_value = inf
        cur_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        for _ in range(self.n_generation):
            print(f'***************Generation {iter_time}***************')
            self.population = np.array(self.crossover_and_mutation())
            self.total_time = []
            fitness = self.get_fitness()
            min_index = np.argmax(fitness)
            print('——————————————————————————————————————————————')
            print(f'the best strategy: stra{min_index+1} {self.population[min_index]}')
            print(f'the best value: {self.total_time[min_index]}')
            print('——————————————————————————————————————————————')
            with open(f'data//Genetic{cur_time}.txt', 'a') as file:
                file.write(f'***************Generation {iter_time}***************\n')
                file.write(f'the best strategy: stra{min_index+1}\n')
                file.write(f'{self.population[min_index]}\n')
                file.write(f'the best value: {self.total_time[min_index]}\n')

            if self.total_time[min_index] < best_value:
                best_value = self.total_time[min_index]
                best_stra = self.population[min_index]
            self.obj_list.append(best_value)
            self.avg_list.append(np.array(self.total_time).mean())
            iter_time += 1

            self.population = self.select(fitness)

        print(f'\n')
        print(f'the result: {best_stra}, {best_value}')
        with open(f'data//Genetic{cur_time}.txt', 'a') as file:
            file.write(f'\n')
            file.write(f'generation:{self.n_generation}  population:{self.pop_size}\n')
            file.write(f'the result:  {best_stra}  {best_value}')

    def get_fitness(self):
        object_value = []
        stra_time = 1
        for stra in self.population:
            print(f'--stra {stra_time}--')
            print(stra)
            net = Network(name=n_name, strategy=stra)
            assign = UE(network=net, main_gap=m_gap, ls_gap=l_gap)
            object_value.append(assign.total_time)
            self.total_time.append(assign.total_time)
            stra_time += 1
        object_value = np.array(object_value)
        return -(object_value - np.max(object_value)) + 1e-3

    def crossover_and_mutation(self):
        new_population = []
        for father in self.population:
            child = father
            if np.random.rand() < self.c_rate:
                mother = self.population[np.random.randint(self.pop_size)]
                cross_point = np.random.randint(0, self.dna_size)
                child[cross_point:] = mother[cross_point:]
            if np.random.rand() < self.m_rate:
                mutate_point = np.random.randint(0, self.dna_size)
                child[mutate_point] = child[mutate_point] ^ 1
            new_population.append(child)
        return new_population

    def select(self, fitness: np.ndarray):
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness/(fitness.sum()))
        return self.population[idx]


class Draw:
    @staticmethod
    def ch2_convergence():
        with open(r'data/ch2_convergence_UE.pkl', 'wb') as file:
            net = Network(name=n_name, strategy=[0 for _ in range(D_size)])
            lower_model = UE(network=net, main_gap=0.0001, ls_gap=0.0001)
            pickle.dump(lower_model.gap_list, file)
        with open(r'data/ch2_convergence_UER.pkl', 'wb') as file:
            net = Network(name=n_name, strategy=[1 for _ in range(D_size)])
            lower_model = UE(network=net, main_gap=0.0001, ls_gap=0.0001)
            pickle.dump(lower_model.gap_list, file)

    @staticmethod
    def plot_ch2_con():
        with open(r'data/ch2/ch2_convergence_UE.pkl', 'rb') as file:
            gap_list_no = pickle.load(file)[1:]
        with open(r'data/ch2/ch2_convergence_UER.pkl', 'rb') as file:
            gap_list_all = pickle.load(file)[1:]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(gap_list_no)), gap_list_no, label='all-information', color='#7EB87B', linestyle='--')
        ax.plot(range(len(gap_list_all)), gap_list_all, label='no-information',color='#2A6398')
        ax.set_yscale("log")
        ax.grid(linestyle='--', linewidth=0.5)
        ax.set_xlabel("Number of Iteration", fontsize=12, fontweight='bold')
        ax.set_ylabel("Relative Gap", fontsize=12, fontweight='bold')
        ax.set_xlim(0, 400)
        ax.set_ylim(1e-4, 1e0)
        plt.legend(frameon=True, edgecolor='black',fontsize='large')
        plt.show()

    @staticmethod
    def plot_cha2_bar():
        flow_pct = [12.4, -16.6, -10.6, 80.7, 7.9]
        cost_pct = [20.9, -17.5, -11.3, 92.2, 16.5]
        labels = ['link(1,3)', 'link(1,4)', 'link(3,2)', 'link(3,4)', 'link(4,2)']
        x = np.arange(len(labels))    # 路段位置
        width = 0.35  # 柱宽
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width / 2, flow_pct, width, label='flow', color='#7EB87B')
        bars2 = ax.bar(x + width / 2, cost_pct, width, label='cost', color='#2A6398')
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.annotate(f'{abs(height):.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -12),  # 上方或下方偏移
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)

        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_ylabel('Comparation Percentage', fontsize=12, fontweight='bold')
        ax.set_xlabel('Network Links', fontsize=12, fontweight='bold')
        ax.set_ylim(min(min(flow_pct), min(cost_pct)) - 10, max(max(flow_pct), max(cost_pct)) + 10)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(frameon=True, edgecolor='black',fontsize='large')
        plt.tight_layout()
        plt.savefig('data/ch2_bar.svg', format='svg')
        plt.show()

    @staticmethod
    def ch3_genetic():
        gen = Genetic(dna_size=D_size, pop_size=P_size, c_rate=C_rate, m_rate=M_rate, n_generation=N_generation)
        with open(r'data/ch3/ch3_genetic_obj.pkl', 'wb') as file:
            pickle.dump(gen.obj_list, file)
        with open(r'data/ch3/ch3_genetic_avg.pkl', 'wb') as file:
            pickle.dump(gen.avg_list, file)

    @staticmethod
    def plot_ch3_genetic():
        with open(r'data/ch3/ch3_genetic_obj.pkl', 'rb') as file:
            obj = pickle.load(file)
        with open(r'data/ch3/ch3_genetic_avg.pkl', 'rb') as file:
            avg = pickle.load(file)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(obj)), obj, label='objection', color='#7EB87B', linestyle='--')
        ax.plot(range(len(avg)), avg, label='average', color='#2A6398')
        ax.grid(linestyle='--', linewidth=0.5)
        ax.set_xlabel("Generaion", fontsize=12, fontweight='bold')
        ax.set_ylim(536, 548)
        ax.set_xlim(1,7)
        ax.set_xticks(range(len(obj)))
        ax.set_xticklabels(range(1, len(obj) + 1))
        plt.legend(frameon=True, edgecolor='black', fontsize='large')
        plt.savefig('data/ch3/ch3_genetic.svg', format='svg')
        plt.show()

    @staticmethod
    def plot_ch3_bar():
        strategies = ['No_Information', 'Optimal_Information', 'All_Information']
        total_costs = [539.9, 537.4, 558.2]
        colors = ['#9DC2D5', '#2A6398', '#7EB87B']
        plt.figure(figsize=(7, 5))
        bar_width = 0.5
        bars = plt.bar(strategies, total_costs, color=colors, width=bar_width)
        # 添加数值标签
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 2, f'{yval:.1f}', ha='center', va='bottom', fontsize=12)
        plt.ylabel('Total Expected Travel Time')
        plt.ylim(520, max(total_costs)+10)
        plt.tight_layout()
        plt.savefig('data/ch3/ch3_bar.svg', format='svg')
        plt.show()

    @staticmethod
    def ch4_convergence():
        with open(r'data/ch4/ch4_convergence_UE.pkl', 'wb') as file:
            net = Network(name=n_name, strategy=[0 for _ in range(D_size)])
            lower_model = UE(network=net, main_gap=0.0001, ls_gap=0.0001)
            pickle.dump(lower_model.gap_list, file)
        with open(r'data/ch4/ch4_convergence_UER.pkl', 'wb') as file:
            net = Network(name=n_name, strategy=[1 for _ in range(D_size)])
            lower_model = UE(network=net, main_gap=0.0001, ls_gap=0.0001)
            pickle.dump(lower_model.gap_list, file)
        with open(r'data/ch4/ch4_convergence_BEST.pkl', 'wb') as file:
            net = Network(name=n_name, strategy=[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1])
            lower_model = UE(network=net, main_gap=0.0001, ls_gap=0.0001)
            pickle.dump(lower_model.gap_list, file)

    @staticmethod
    def ch4_genetic():
        gen = Genetic(dna_size=D_size, pop_size=P_size, c_rate=C_rate, m_rate=M_rate, n_generation=N_generation)
        with open(r'data/ch4_genetic_obj.pkl', 'wb') as file:
            pickle.dump(gen.obj_list, file)
        with open(r'data/ch4_genetic_avg.pkl', 'wb') as file:
            pickle.dump(gen.avg_list, file)

    @staticmethod
    def plot_ch4_con():
        with open(r'data/ch4/ch4_convergence_UE.pkl', 'rb') as file:
            gap_list_no = pickle.load(file)[1:]
        with open(r'data/ch4/ch4_convergence_UER.pkl', 'rb') as file:
            gap_list_all = pickle.load(file)[1:]
        with open(r'data/ch4/ch4_convergence_BEST.pkl', 'rb') as file:
            gap_list_best = pickle.load(file)[1:]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(gap_list_no)), gap_list_no, label='all-information', color='#7EB87B',linewidth=0.5)
        ax.plot(range(len(gap_list_all)), gap_list_all, label='no-information',color='#2A6398',linewidth=0.5)
        ax.plot(range(len(gap_list_best)), gap_list_best, label='optimal-information', color='#9DC2D5',linewidth=0.5)
        ax.set_yscale("log")
        ax.grid(linestyle='--', linewidth=0.5)
        ax.set_xlabel("Number of Iteration", fontsize=12, fontweight='bold')
        ax.set_ylabel("Relative Gap", fontsize=12, fontweight='bold')
        ax.set_ylim(9e-5,1e-1)
        plt.legend(frameon=True, edgecolor='black',fontsize='large')
        plt.savefig('data/ch4/ch4_con.svg', format='svg')
        plt.show()

    @staticmethod
    def plot_ch4_gen():
        with open(r'data/ch4_genetic_obj.pkl', 'rb') as file:
            obj = pickle.load(file)
        with open(r'data/ch4_genetic_avg.pkl', 'rb') as file:
            avg = pickle.load(file)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(obj)), obj, label='objection', color='#7EB87B', linestyle='--')
        ax.plot(range(len(avg)), avg, label='average', color='#2A6398')
        ax.grid(linestyle='--', linewidth=0.5)
        ax.set_xlabel("Generaion", fontsize=12, fontweight='bold')
        plt.legend(frameon=True, edgecolor='black', fontsize='large')
        plt.savefig('data/ch3/ch3_genetic.svg', format='svg')
        plt.savefig('data/ch4/ch4_gen.svg', format='svg')
        plt.show()

    @staticmethod
    def plot_ch4_bar():
        strategies = ['All_Information', 'No_Information', 'Optimal_Information']
        total_costs = [101959.55, 113648.80, 101815.15]
        colors = ['#9DC2D5', '#2A6398', '#7EB87B']
        plt.figure(figsize=(7, 5))
        bar_width = 0.5
        bars = plt.bar(strategies, total_costs, color=colors, width=bar_width)
        # 添加数值标签
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 2, f'{yval:.1f}', ha='center', va='bottom', fontsize=12)
        plt.ylabel('Total Expected Travel Time')
        plt.tight_layout()
        plt.savefig('data/ch4/ch4_bar.svg', format='svg')
        plt.show()

if __name__ == '__main__':
    # draw = Draw()
    # draw.plot_ch3_genetic()
    net = Network(name=n_name, strategy=[1 for _ in range(76)])
    ue = UE(network=net, main_gap=0.0001, ls_gap=0.0001)
    # gen = Genetic(dna_size=D_size, pop_size=P_size, c_rate=C_rate, m_rate=M_rate, n_generation=N_generation)

from collections import defaultdict, deque
import re
from math import inf
from itertools import product
from time import perf_counter as pc
# import numpy as np

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
        self.rho: float = 0

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
        self.state: list[State] = []

    def __repr__(self):
        return f"Link {self.link_id}"

    def update_cost(self):
        self.cost = self.fft * (1 + self.b * (self.flow / self.capacity) ** self.power)


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

    def __repr__(self):
        return f"Link {self.mother_link.link_id} - state {self.state_id}"

    def update_cost(self):
        self.cost = self.fft * (1 + self.b * (self.flow / self.capacity) ** self.power)

    def derivative(self):
        return self.fft * self.b * self.power * self.flow ** (self.power - 1) / self.capacity ** self.power


class Hyperpath:
    def __init__(self, ori, des, included_nodes):
        self.ori: NODE = ori
        self.des: NODE = des
        self.included_nodes: list[NODE] = included_nodes
        self.included_states: list[State] = []
        self.state_rho: dict[State: float] = {}
        self.node_rho:dict[NODE: float] = {}
        self.path_flow: float = 0
        self.path_cost: float = 0

    def __repr__(self):
        return f'{self.ori} - {self.des}: cost= {self.path_cost}, flow= {self.path_flow}'

    def __eq__(self, other):
        return (set(self.included_nodes) == set(other.included_nodes) and
                set(self.included_states) == set(other.included_states) and self.path_cost == other.path_cost)

    def update_path_cost(self):
        self.path_cost = sum(state.cost * (self.state_rho[state] * self.node_rho[state.tail_node]) for state in self.included_states)


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
        self.basic_hyperpaths: list[Hyperpath] = []

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
                        if demand > 0:
                            self.OD.append(ODPair(origin, destination, demand))
                            if destination not in self.Dest:
                                self.Dest.append(destination)
            for des in self.Dest:
                self.Policy[des] = POLICY(des)
            if total_flow != sum([od.demand for od in self.OD]):
                print("demand is wrong!")

    def update_all_state_cost(self):
        for state in self.link_State:
            state.update_cost()

    def update_all_hyperpath_cost(self):
        for od in self.OD:
            for hyperpath in od.basic_hyperpaths:
                hyperpath.update_path_cost()


class GP:
    def __init__(self, net, epsilon):
        self.net: Network = net
        self.epsilon: float = epsilon
        self.run()

    def run(self):
        s = pc()
        iter_times, cur_gap = 1, inf
        self.initialize()
        while cur_gap > self.epsilon:
            print(f'*****************iter_time {iter_times}*********************')
            for od in self.net.OD:
                self.label_correcting(od.destination.node_id)
                self.shift_flow(od)
            iter_times += 1
            cur_gap = self.convergence()
            print(f"cur_gap= {cur_gap}")
        e = pc()
        print(f"\nRunning time is {e - s:.2f}, tstt = {sum([state.flow * state.cost for state in self.net.link_State]):.2f}")

    def label_correcting(self, des: int):
        # def reduce(xi_prime_in, lam_prime_in):
        #     lam_in = list(set(lam_prime_in))
        #     ele_prob = {element: 0 for element in lam_in}
        #     for element, prob in zip(lam_prime_in, xi_prime_in):
        #         ele_prob[element] += prob
        #     xi_in = list(ele_prob.values())
        #     lam_in = list(ele_prob.keys())
        #     return xi_in, lam_in

        # Step 1: initialize
        for node in self.net.Node[1:]:
            node.ett = inf
        self.net.Node[des].ett = 0
        sel = deque()
        for node in self.net.Node[des].up_node:
            sel.append(node)
        # step 2
        while sel:
            # print(sel)
            node_i = sel.popleft()
            # print(f'\n{node_i}')
            message_cost = []
            message_id = 0
            for message in node_i.message:
                # print(message)
                min_cost = min(state.cost + state.head_node.ett for state in message)
                # for state in message:
                #     print(f'state.cost={state.cost}, state.head_node.ett={state.head_node.ett}, message_pro={node_i.message_prob[message_id]}')
                message_cost.append(min_cost * node_i.message_prob[message_id])
                message_id += 1
            # print(message_cost)
            if node_i.ett > sum(message_cost):
                node_i.ett = sum(message_cost)
                sel.extend(node_i.up_node)
        # Step 3: Choose Optimal Policy
        for node in self.net.Node[1:]:
            for m_id in node.message_id:
                m_vec, m_pro = node.message[m_id], node.message_prob[m_id]
                if node == self.net.Node[des]:
                    self.net.Policy[self.net.Node[des]].map[node][m_id] = node
                else:
                    min_val = inf
                    next_node = -1
                    for state in m_vec:
                        if state.cost + state.head_node.ett < min_val:
                            min_val = state.cost + state.head_node.ett
                            next_node = state.head_node
                    self.net.Policy[self.net.Node[des]].map[node][m_id] = next_node

        # Step 2
        # while sel:
        #     node_i = sel.popleft()
        #     xi, lam = [1], [inf]
        #     for node_j in node_i.down_node:
        #         xi_prime, lam_prime = [], []
        #         link = None
        #         for ij in node_i.l_out:
        #             if ij.head_node == node_j:
        #                 link = ij
        #                 break
        #         for l_state in link.state:
        #             for k in range(len(xi)):
        #                 xi_prime.append(xi[k] * l_state.prob)
        #                 if l_state.cost + node_j.ett < lam[k]:
        #                     lam_prime.append(l_state.cost + node_j.ett)
        #                 else:
        #                     lam_prime.append(lam[k])
        #         xi, lam = reduce(xi_prime, lam_prime)
        #     cur = np.dot(xi, lam)
        #     if cur < node_i.ett:
        #         node_i.ett = cur
        #         sel.extend(node_i.up_node)
        # Step 3: Choose Optimal Policy
        # for node in self.net.Node[1:]:
        #     for m_id in node.message_id:
        #         m_vec, m_pro = node.message[m_id], node.message_prob[m_id]
        #         if node == self.net.Node[des]:
        #             self.net.Policy[self.net.Node[des]].map[node][m_id] = node
        #         else:
        #             min_val = inf
        #             next_node = -1
        #             for state in m_vec:
        #                 if state.cost + state.head_node.ett < min_val:
        #                     min_val = state.cost + state.head_node.ett
        #                     next_node = state.head_node
        #             self.net.Policy[self.net.Node[des]].map[node][m_id] = next_node

    def initialize(self):
        for state in self.net.link_State:
            state.flow = 0
        self.net.update_all_state_cost()
        for od in self.net.OD:
            self.label_correcting(od.destination.node_id)
            hyperpath = self.get_hyperpath(od)
            od.basic_hyperpaths.append(hyperpath)
            hyperpath.path_flow = od.demand
            for state in hyperpath.included_states:
                state.flow += hyperpath.path_flow * (hyperpath.state_rho[state]
                               * hyperpath.node_rho[state.tail_node])
        self.net.update_all_state_cost()
        self.net.update_all_hyperpath_cost()

    def get_hyperpath(self, od: ODPair):
        ori, des = od.origin, od.destination
        included_nodes = [ori]
        included_states = []
        included_links = []
        sel = [ori]
        while sel:
            if sel[0] != des:
                current_node = sel[0]
                for m_id in current_node.message_id:
                    next_node = self.net.Policy[des].map[current_node][m_id]
                    if next_node not in included_nodes:
                        included_nodes.append(next_node)
                        sel.append(next_node)
                    for link in next_node.l_in:
                        if link.tail_node == current_node and link not in included_links:
                            included_links.append(link)
                            included_states.extend(link.state)
            sel = sel[1:]
        hyperpath = Hyperpath(ori, des, included_nodes)
        hyperpath.included_states = included_states

        # update state rho
        for link in self.net.Link[1:]:
            for state in link.state:
                state.rho = 0
        for node in hyperpath.included_nodes:
            if node == des:
                continue
            for m_id in node.message_id:
                m_vec, m_prob = node.message, node.message_prob
                next_node = self.net.Policy[des].map[node][m_id]
                for state in m_vec[m_id]:
                    if state.head_node == next_node:
                        state.rho += m_prob[m_id]
                        break
        for state in self.net.link_State:
            hyperpath.state_rho[state] = state.rho

        # update node rho
        for node in self.net.Node[1:]:
            node.rho = 0
        gap = inf
        while gap > 1e-6:
            previous_rho = {node: node.rho for node in hyperpath.included_nodes}
            for node in hyperpath.included_nodes[:]:
                if node.node_id == ori.node_id:
                    node.rho = float(1)
                    for link in node.l_in:
                        node.rho += sum(state.rho * state.tail_node.rho for state in link.state)
                else:
                    node.rho = 0
                    for link in node.l_in:
                        node.rho += sum(state.rho * state.tail_node.rho for state in link.state)
            max_gap = 0
            for node in hyperpath.included_nodes:
                rel_change = abs(node.rho - previous_rho[node])
                max_gap = max(max_gap, rel_change)
            gap = max_gap

        for node in self.net.Node[:]:
            hyperpath.node_rho[node] = node.rho
        hyperpath.update_path_cost()
        return hyperpath

    def shift_flow(self, od: ODPair):
        stt_hyperpath = self.get_hyperpath(od)
        for basic_hyperpath in od.basic_hyperpaths:
            if basic_hyperpath == stt_hyperpath:
                hyperpath = basic_hyperpath
                break
        else:
            hyperpath = stt_hyperpath
            od.basic_hyperpaths.append(hyperpath)

        # shift flow
        for basic_hyperpath in od.basic_hyperpaths:
            if basic_hyperpath == hyperpath:
                continue
            temp = sum([state.derivative() *
                        (basic_hyperpath.state_rho[state] * basic_hyperpath.node_rho[state.tail_node]
                                              - hyperpath.state_rho[state] * hyperpath.node_rho[state.tail_node])**2
                        for state in list(set(hyperpath.included_states) | set(basic_hyperpath.included_states))])
            if basic_hyperpath.path_cost > hyperpath.path_cost:
                shifted_flow = min((basic_hyperpath.path_cost - hyperpath.path_cost) / temp, basic_hyperpath.path_flow)
            else:
                shifted_flow = max((basic_hyperpath.path_cost - hyperpath.path_cost) / temp, -hyperpath.path_flow)
            basic_hyperpath.path_flow -= shifted_flow
            hyperpath.path_flow += shifted_flow
        # update cost
            for state in basic_hyperpath.included_states:
                state.flow -= shifted_flow * basic_hyperpath.state_rho[state] * basic_hyperpath.node_rho[state.tail_node]
            for state in hyperpath.included_states:
                state.flow += shifted_flow * hyperpath.state_rho[state] * hyperpath.node_rho[state.tail_node]
            self.net.update_all_state_cost()
            self.net.update_all_hyperpath_cost()
        od.basic_hyperpaths = [hyperpath for hyperpath in od.basic_hyperpaths if hyperpath.path_flow > 0]

    def convergence(self):
        tstt = sum([state.flow * state.cost for state in self.net.link_State])
        sptt = 0
        for des in self.net.Dest:
            self.label_correcting(des.node_id)
            for od in self.net.OD:
                if od.destination == des:
                    sptt += od.origin.ett * od.demand
                    # print(f'\nori.ett={od.origin.ett} od.demand={od.demand}')
        cur_gap = (tstt / sptt) - 1
        print(f'sptt={sptt}, tstt={tstt}')
        return cur_gap

class Output:
    def __init__(self, net):
        self.net: Network = net

    def print_flow_cost(self):
        res = 0
        print(f'-----------------------------------------------')
        for state in self.net.link_State:
            res += state.cost * state.flow
            print(f"{state}   flow:{state.flow}, cost:{state.cost}")
        print(f'\nTotal Travel Time: {res}')


if __name__ == "__main__":
    network = Network(name="SiouxFalls", strategy=[1 for _ in range(76)])
    assign = GP(network, 1e-4)
    output = Output(network)
    output.print_flow_cost()

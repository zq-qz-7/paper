import re
from math import inf


class NODE:
    def __init__(self, node_id):
        self.node_id: int = node_id
        self.l_in: list[LINK] = []
        self.l_out: list[LINK] = []

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
        self.auxiliary_flow: float = 0
        self.last_flow: float = 0

    def __repr__(self):
        return f"Link {self.link_id}"

    def update_cost(self):
        self.cost = self.fft * (1 + self.b * (self.flow / self.capacity) ** self.power)

    def obtain_cost(self, param):
        return self.fft * (1 + self.b * (param / self.capacity) ** self.power)

    def derivative(self):
        return self.fft * self.b * self.power * self.flow ** (self.power - 1) / self.capacity ** self.power


class PATH:
    def __init__(self, origin, destination, included_links):
        self.origin: NODE = origin
        self.destination: NODE = destination
        self.included_links: list[LINK] = included_links
        self.path_flow: float = 0
        self.path_cost: float = 0

    def __repr__(self):
        nodes = [link.tail_node.node_id for link in self.included_links] + [self.destination.node_id]
        return f"{'-'.join(map(str, nodes))}: cost= {self.path_cost}, flow= {self.path_flow}"

    def update_path_cost(self):
        self.path_cost = sum([link.cost for link in self.included_links])


class ODPair:
    def __init__(self, origin, destination, demand):
        self.origin: NODE = origin
        self.destination: NODE = destination
        self.demand: float = demand
        self.basic_paths: list[PATH] = []

    def __repr__(self):
        return f"{self.origin}-{self.destination}: {self.demand}"


class Network:
    def __init__(self, name):
        self.name: str = name
        self.Link: list[LINK] = []
        self.Node: list[NODE] = []
        self.OD: list[ODPair] = []
        self.num_node: int = 0
        self.num_link: int = 0
        self.total_flow: float = 0
        self.read_net()
        self.read_od()

    def read_net(self):
        with open(f"TransportationNetworks\\{self.name}\\{self.name}_net.txt") as f:
            lines = f.readlines()
            pattern = re.compile(r"[0-9.a-zA-Z]+")
            lines = [pattern.findall(line) for line in lines if pattern.findall(line) != []]
            for i in range(len(lines)):
                line = lines[i]
                if "NUMBER" and "NODES" in line:
                    self.num_node = int(line[-1])
                if "NUMBER" and "LINKS" in line:
                    self.num_link = int(line[-1])
                if "capacity" in line:
                    lines = lines[i+1:]
                    break
            # create NODE and LINK
            self.Node = [NODE(i) for i in range(self.num_node+1)]
            self.Link = [LINK(0)]
            for i in range(len(lines)):
                line = lines[i]
                tail, head = self.Node[int(line[0])], self.Node[int(line[1])]
                link = LINK(i+1, head, tail, float(line[2]), float(line[3]),
                            float(line[4]), float(line[5]), float(line[6]))
                self.Link.append(link)
                tail.l_out.append(link)
                head.l_in.append(link)

    def read_od(self):
        with open(f"TransportationNetworks\\{self.name}\\{self.name}_trips.txt") as f:
            lines = f.readlines()
            pattern = re.compile(r"[0-9.a-zA-Z]+")
            lines = [pattern.findall(line) for line in lines if pattern.findall(line) != []]
            for i in range(len(lines)):
                line = lines[i]
                if "TOTAL" in line:
                    self.total_flow = float(line[-1])
                if "Origin" in line:
                    lines = lines[i:]
                    break
            # create ODPair
            for i in range(len(lines)):
                line = lines[i]
                if "Origin" in line:
                    ori = self.Node[int(line[-1])]
                else:
                    for j in range(len(line) // 2):
                        des = self.Node[int(line[2*j])]
                        demand = float(line[2*j+1])
                        self.OD.append(ODPair(ori, des, demand))

    def update_all_link_cost(self):
        for link in self.Link[1:]:
            link.update_cost()

    def update_all_path_cost(self):
        for od in self.OD:
            for path in od.basic_paths:
                path.update_path_cost()


def dijkstra(o_id: int, d_id: int, network):
    # initialization
    for node in network.Node[1:]:
        node.parent = node
        node.dist = inf
    network.Node[o_id].parent = -1
    network.Node[o_id].dist = 0
    # main loop
    sel = [network.Node[o_id]]
    while sel:
        sel.sort(key=lambda x: x.dist, reverse=True)
        cur = sel.pop()
        if cur == network.Node[d_id]:
            break
        for link in cur.l_out:
            if cur.dist + link.cost < link.head_node.dist:
                link.head_node.dist = cur.dist + link.cost
                link.head_node.parent = cur
                if link.head_node not in sel:
                    sel.append(link.head_node)
    # get path
    shortest_path = []
    cur = network.Node[d_id]
    while cur.parent != -1:
        for link in cur.l_in:
            if cur.parent == link.tail_node:
                shortest_path.append(link)
                cur = link.tail_node
    shortest_path.reverse()
    return shortest_path


class FW:
    def __init__(self, net, main_gap, ls_gap):
        self.net: Network = net
        self.main_gap: float = main_gap
        self.ls_gap: float = ls_gap
        self.run()

    def run(self):
        self.initialize()
        gap = inf
        while gap > self.main_gap:
            self.update_cost()
            self.all_or_nothing()
            self.line_search()
            self.update_flow()
            gap = self.convergence()

    def initialize(self):
        self.update_cost()
        self.all_or_nothing()
        for link in self.net.Link[1:]:
            link.flow = link.auxiliary_flow
            link.auxiliary_flow = 0

    def update_cost(self):
        for link in self.net.Link[1:]:
            link.update_cost()

    def all_or_nothing(self):
        for od in self.net.OD:
            ori = od.origin.node_id
            des = od.destination.node_id
            path = dijkstra(ori, des, self.net)
            for link in path:
                link.auxiliary_flow = link.auxiliary_flow + od.demand

    def line_search(self):
        def derivative(step):
            res = 0
            for link in self.net.Link[1:]:
                res += ((link.auxiliary_flow - link.flow) *
                        link.obtain_cost(link.flow + step * (link.auxiliary_flow - link.flow)))
            return res

        right, mid, left = 0, 0.5, 1
        while abs(derivative(mid)) > self.ls_gap:
            if derivative(mid) * derivative(right) > 0:
                right = mid
            if derivative(mid) * derivative(left) > 0:
                left = mid
            mid = (right + left) / 2
        return mid

    def update_flow(self):
        step = self.line_search()
        for link in self.net.Link[1:]:
            link.last_flow = link.flow
            link.flow = link.flow + step * (link.auxiliary_flow - link.flow)
            link.auxiliary_flow = 0

    def convergence(self):
        numerator = sum([(link.flow - link.last_flow) ** 2 for link in self.net.Link[1:]]) ** 0.5
        denominator = sum([link.flow for link in self.net.Link[1:]])
        return numerator / denominator


class GP:
    def __init__(self, net, epsilon):
        self.net: Network = net
        self.epsilon: float = epsilon
        self.run()

    def run(self):
        iter_times, cur_gap = 0, inf
        self.initialize()
        while cur_gap > self.epsilon:
            print(f"iter_time {iter_times}: cur_gap= {cur_gap}")
            self.shift_flow()
            iter_times += 1
            cur_gap = self.convergence()

    def initialize(self):
        for od in self.net.OD:
            ori, des, demand = od.origin, od.destination, od.demand
            shortest_path = dijkstra(ori.node_id, des.node_id, self.net)
            path = PATH(ori, des, shortest_path)
            od.basic_paths.append(path)
            path.path_flow = demand
            for link in path.included_links:
                link.flow += demand
        self.net.update_all_link_cost()
        self.net.update_all_path_cost()

    def shift_flow(self):
        for od in self.net.OD:
            ori, des, demand = od.origin, od.destination, od.demand
            shortest_path = dijkstra(ori.node_id, des.node_id, self.net)
            min_dist = 0
            for link in shortest_path:
                min_dist += link.cost
            for working_path in od.basic_paths:
                if working_path.included_links == shortest_path:
                    path = working_path
                    break
            else:
                path = PATH(ori, des, shortest_path)
                od.basic_paths.append(path)
            # shift flow
            for basic_path in od.basic_paths:
                if basic_path == path:
                    continue
                temp = sum([link.derivative()
                            for link in list(set(basic_path.included_links) ^ set(path.included_links))])
                shifted_flow = min((basic_path.path_cost - min_dist) / temp, basic_path.path_flow)
                basic_path.path_flow -= shifted_flow
                path.path_flow += shifted_flow
            # update cost
                for link in basic_path.included_links:
                    link.flow -= shifted_flow
                for link in path.included_links:
                    link.flow += shifted_flow
                self.net.update_all_link_cost()
                self.net.update_all_path_cost()
            od.basic_paths = [path for path in od.basic_paths if path.path_flow > 0]

    def convergence(self):
        SPTT = 0
        for od in self.net.OD:
            ori, des, demand = od.origin, od.destination, od.demand
            shortest_path = dijkstra(ori.node_id, des.node_id, self.net)
            path = PATH(ori, des, shortest_path)
            path.update_path_cost()
            min_dist = path.path_cost
            SPTT += min_dist * od.demand
        TSTT = sum([link.flow * link.cost for link in self.net.Link[1:]])
        cur_gap = (TSTT / SPTT) - 1
        return cur_gap


class Output:
    def __init__(self, net):
        self.net: Network = net

    def print_flow_cost(self):
        res = 0
        for link in self.net.Link[1:]:
            res += link.cost * link.flow
            print(f"{link}--flow:{link.flow}, cost:{link.cost}")
        print(f'Total Travel Time:{res}')


if __name__ == "__main__":
    sf = Network("Nguyen-Dupuis")
    assignment = GP(net=sf, epsilon=1e-6)
    output = Output(sf)
    output.print_flow_cost()

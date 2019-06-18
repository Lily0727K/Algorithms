from collections import defaultdict
from heapq import heappop, heappush


class Dijkstra:
    def __init__(self):
        self.edge = defaultdict(list)

    def add(self, u, v, d):
        self.edge[u].append([v, d])
        self.edge[v].append([u, d])

    def search(self, start):
        distance = defaultdict(lambda: float("inf"))
        distance[start] = 0
        queue = []
        heappush(queue, (0, start))
        seen = set()
        while queue:
            k, u = heappop(queue)
            if u in seen:
                continue
            seen.add(u)

            for v, d in self.edge[u]:
                if v in seen:
                    continue
                new_d = k + d
                if distance[v] > new_d:
                    distance[v] = new_d
                    heappush(queue, (new_d, v))
        return distance

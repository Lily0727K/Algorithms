class UnionFind:
    # UnionFind実装。ランク付き、経路短縮付き。
    def __init__(self, num):
        self.rank = [0] * num
        self.par = [i for i in range(num)]
        self.n = num

    def find_root(self, node):
        if self.par[node] == node:
            return node
        else:
            self.par[node] = self.find_root(self.par[node])
            return self.par[node]

    def same_root(self, x, y):
        return self.find_root(x) == self.find_root(y)

    def union(self, x, y):
        x = self.find_root(x)
        y = self.find_root(y)
        if x == y:
            return
        if self.rank[x] > self.rank[y]:
            self.par[y] = x
        else:
            self.par[x] = y
            if self.rank[x] == self.rank[y]:
                self.rank[y] += 1

from collections import deque


class Node:
    def __init__(self, info):
        self.info = info
        self.left = None
        self.right = None
        self.level = None

    def __str__(self):
        return str(self.info)


class BinarySearchTree:
    def __init__(self):
        self.root = None

    def create(self, val):
        if not self.root:
            self.root = Node(val)
        else:
            current = self.root

            while True:
                if val < current.info:
                    if current.left:
                        current = current.left
                    else:
                        current.left = Node(val)
                        break
                elif val > current.info:
                    if current.right:
                        current = current.right
                    else:
                        current.right = Node(val)
                        break
                else:
                    break


def height(root):
    # 木の高さを返す。ノードが1つしかない木は高さ0としている。
    q = deque()
    q.append(root)
    root.level = 0
    max_level = 0
    while q:
        current = q.popleft()
        max_level = max(max_level, current.level)
        if current.left:
            q.append(current.left)
            current.left.level = current.level + 1
        if current.right:
            q.append(current.right)
            current.right.level = current.level + 1
    return max_level


def lca(root, v1, v2):
    # v1, v2は木に含まれるノードの値。lowest common ancestor (LCA)を返す。
    v1, v2 = min(v1, v2), max(v1, v2)
    if v1 <= root.info <= v2:
        return root
    if v2 < root.info:
        return lca(root.left, v1, v2)
    if v1 > root.info:
        return lca(root.right, v1, v2)


def check_bst(root):
    # BSTかどうか判定する。与えられる木のデータはnode.dataで0から10**4まで。
    def check_rec(node, min_value, max_value):
        if not node:
            return True
        if node.data <= min_value or node.data >= max_value:
            return False
        return check_rec(node.left, min_value, node.data) and check_rec(node.right, node.data, max_value)

    return check_rec(root, float("-inf"), float("inf"))


tree = BinarySearchTree()
t = int(input())

arr = list(map(int, input().split()))

for i in range(t):
    tree.create(arr[i])

# print(height(tree.root))

# v = list(map(int, input().split()))
# ans = lca(tree.root, v[0], v[1])
# print (ans.info)

# print(check_bst(tree.root))

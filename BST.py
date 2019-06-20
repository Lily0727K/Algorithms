class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class BST:
    def __init__(self):
        self.root = None

    def insert(self, v):
        x = TreeNode(v)
        if not self.root:
            self.root = x
            return

        r = self.root
        while r is not None:
            parent = r
            if x.val < r.val:
                r = r.left
            else:
                r = r.right

        if x.val < parent.val:
            parent.left = x
        else:
            parent.right = x

    def in_order(self):
        def helper(node):
            if not node:
                return
            helper(node.left)
            res.append(node.val)
            helper(node.right)

        res = []
        helper(self.root)
        return res

    def pre_order(self):
        def helper(node):
            if not node:
                return
            res.append(node.val)
            helper(node.left)
            helper(node.right)

        res = []
        helper(self.root)
        return res

    def find(self, target):
        if not self.root:
            return False
        r = self.root
        while r:
            if target < r.val:
                r = r.left
            elif target == r.val:
                return True
            else:
                r = r.right
        return False

    def delete(self, key):
        def helper(r, p):
            if not r.left and not r.right:
                if r == p.left:
                    p.left = None
                else:
                    p.right = None
                return

            if r.left and not r.right:
                if r == p.left:
                    p.left = r.left
                else:
                    p.right = r.left
                return
            elif not r.left and r.right:
                if r == p.left:
                    p.left = r.right
                else:
                    p.right = r.right
                return

        if not self.find(key):
            return

        p = None
        r = self.root
        while True:
            if key < r.val:
                p = r
                r = r.left
            elif key > r.val:
                p = r
                r = r.right
            else:
                break

        if not p:
            if not r.left and not r.right:
                self.root = None
                return

        if not r.left or not r.right:
            helper(r, p)
            return

        p = r
        curr = r.right
        while curr.left:
            p = curr
            curr = curr.left

        r.val = curr.val
        helper(curr, p)
        return
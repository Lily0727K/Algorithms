class Trie:
    def __init__(self):
        self.end = False
        self.c = {}

    def insert(self, word):
        node = self
        for w in word:
            if w not in node.c:
                node.c[w] = Trie()
            node = node.c[w]
        node.end = True

    def prefixnode(self, word):
        node = self
        for w in word:
            if w not in node.c:
                return None
            node = node.c[w]
        return node

    def search(self, word):
        node = self.prefixnode(word)
        if not node:
            return False
        else:
            return node.end

    def startsWith(self, prefix):
        node = self.prefixnode(prefix)
        return bool(node)


# obj = Trie()
# obj.insert(word)
# obj.search(word)
# obj.startsWith(prefix)
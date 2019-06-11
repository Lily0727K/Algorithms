class TrieNode:
    def __init__(self):
        self.children = {}
        self.end = False
        self.data = None
        self.rank = 0


class AutocompleteSystem:
    def __init__(self, sentences, times):
        self.root = TrieNode()
        self.keyword = ""
        for i, sentence in enumerate(sentences):
            self.add_record(sentence, times[i])

    def add_record(self, sentence, hot):
        p = self.root
        for c in sentence:
            if c not in p.children:
                p.children[c] = TrieNode()
            p = p.children[c]
        p.end = True
        p.data = sentence
        p.rank -= hot

    def dfs(self, root):
        ret = []
        if root:
            if root.end:
                ret.append((root.rank, root.data))
            for child in root.children:
                ret.extend(self.dfs(root.children[child]))
        return ret

    def search(self, sentence):
        p = self.root
        for c in sentence:
            if c not in p.children:
                return []
            p = p.children[c]
        return self.dfs(p)

    def input(self, c):
        results = []
        if c != "#":
            self.keyword += c
            results = self.search(self.keyword)
        else:
            self.add_record(self.keyword, 1)
            self.keyword = ""
        return [item[1] for item in sorted(results)]

#test = AutocompleteSystem(["i love you", "island", "ironman", "i love leetcode"], [5,3,2,2])
#print(test.input("i"))
#test.input("#")
#test.add_record("i love atcoder", 10)
#print(test.input("i"))
#print(test.input("r"))
#test.input("#")

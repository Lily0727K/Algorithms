import sys
import heapq
import random
import numpy as np
from operator import mul, xor
from functools import reduce
from itertools import chain, permutations
from collections import Counter, defaultdict, deque
from bisect import bisect_left


def is_square(n):
    # squareかどうか判定
    return n >= 0 and n ** 0.5 % 1 == 0


def grow(arr):
    # 配列の総積
    # numpy.prodは数が大きくなると桁あふれするので使わない
    return reduce(mul, arr)


def find_it(seq):
    # 唯一、奇数個の要素がある配列 -> 奇数個の要素を探す
    return reduce(xor, seq)


def persistence(n):
    # 再帰関数
    # 各桁の数字をかけていく操作が、1桁になるまでに何回できるか
    return 0 if n < 10 else persistence(reduce(mul, map(int, str(n)))) + 1


def digital_root(n):
    # 各桁の数字を合計していく操作で、1桁になったときの値
    # return n if n < 10 else digital_root(sum(map(int, str(n))))
    # 各桁の総和を9で割った値は変わらないので
    return n % 9 or n and 9


def digit_sum(n):
    # Faster than sum(map(int, str(x)))
    r = 0
    while n:
        r, n = r + n % 10, n // 10
    return r


def tribonacci(signature, n):
    # トリボナッチ数列
    # 最初の3つがリストの形式でsignatureとして与えられる
    # 生成される数列の最初のn個を返す
    for i in range(n - 3):
        signature.append(sum(signature[-3:]))
    return signature[:n]


def make_sum_list(x):
    # 隣り合った要素の和からなる数列
    return list(map(sum, zip(x, x[1:])))


def delete_vowel(string):
    # 母音を削除
    return string.translate(str.maketrans('', '', 'aeiouAEIOU'))


def get_vowel_count(input_str):
    # 母音の数
    return sum(x in "aeiou" for x in input_str)


def dna_strand(dna):
    # DNA塩基配列
    return dna.translate(str.maketrans("ATCG", "TAGC"))


def isprime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


def prime_list(n):
    # n までの素数リスト
    primes = set(range(2, n + 1))
    for i in range(2, int(n ** 0.5 + 1)):
        primes.difference_update(range(i * 2, n + 1, i))
    return list(primes)


def fast_prime_set(n):
    # nまでの素数リスト高速版 setで返す
    # chainの読み込みが必要！
    if n < 4:
        return ({}, {}, {2}, {2, 3})[n]
    n_sqrt = int(n ** 0.5) + 1
    primes = {2, 3} | set(chain(range(5, n + 1, 6), range(7, n + 1, 6)))
    for i in range(5, n_sqrt, 2):
        if i in primes:
            primes.difference_update(range(i * i, n, i * 2))
    return primes


def np_prime(n):
    is_prime = np.ones(n + 1, dtype=np.bool)
    is_prime[:2] = 0
    n_sqrt = int(n ** 0.5) + 1
    for p in range(2, n_sqrt):
        if is_prime[p]:
            is_prime[p * p::p] = 0
    return is_prime.nonzero()[0]


def double_loop(n):
    # 二重ループ　正解を見つけたらループを止める
    for i in range(n):
        for j in range(n):
            if i == j:
                break
        else:
            continue
        break


def frame():
    # 周辺埋め
    h, w = map(int, input().split())
    s = ["." * (w + 2)] + ["." + input() + "." for _ in range(h)] + ["." * (w + 2)]
    return s


def rotation(s):
    s = np.array(s)
    # return s[::-1, :].T  # 90度回転
    return s[::-1, ::-1]  # 180度回転


def sum_pairs(ints, s):
    # 合計がsになるようなペアを検索。ペアの組み合わせの後にある方が、一番先になるペアを返す。
    rest = set()
    for num in ints:
        if num in rest:
            return [s - num, num]
        else:
            rest.add(s - num)


def order_digit_sum(string):
    # スペースで区切られた文字列が与えられ、各桁を合計した値が少ない順に並べ替えて文字列として返す。同値の場合は辞書順。
    data = list(string.split())
    data.sort(key=lambda x: (sum(map(int, x)), x))
    return " ".join(str(x) for x in data)


def lcm(x, y):
    # GCD: greatest common divisor
    # LCM: least common multiple
    return x * y // gcd(x, y)


def smallest_multiple(n):
    # 1からnまでのすべての数字で割れる最小の数
    return reduce(lcm, range(1, n + 1))


def max_sequence(arr):
    # 配列内の連続する要素の和で最大のものを返す
    max_sum, subarray_sum = 0, 0
    for x in arr:
        subarray_sum += x
        subarray_sum = max(0, subarray_sum)
        max_sum = max(max_sum, subarray_sum)
    return max_sum


def product_fib(prod):
    # フィボナッチ数列の隣り合った積が指定された値になればTrue。ならない場合は超えた時点でFalseを返す。
    a, b = 1, 1
    while a * b < prod:
        a, b = b, a + b
    return [a, b, prod == a * b]


class Add(int):
    # SubInt(int)はintのサブクラスであり値をそのまま表示できる
    # クラスオブジェクトを関数化できる特殊メソッド __call__
    # Add(1) -> 1 Add(1)(2) -> 3 Add(1)(2)(3) -> 6
    def __call__(self, n):
        return Add(self + n)


def minimum_swaps(arr):
    # 1からnまでの数字が含まれている配列をスワップで並び替えるのに必要な最小回数
    arr = [0] + arr
    cnt = 0
    changed = True
    while changed:
        i = 1
        changed = False
        while i < len(arr):
            if arr[i] != i:
                cnt += 1
                changed = True
                a = arr[i]
                arr[i], arr[a] = arr[a], arr[i]
            i += 1
    return cnt


def array_sum():
    # n個の0からなる配列にa～bまでの要素にkを足す操作をm回行い、最大値を計算
    # O(n+m)
    n, m = map(int, input().split())
    arr = [0] * (n + 1)
    for _ in range(m):
        a, b, k = map(int, input().split())
        arr[a - 1] += k
        arr[b] -= k
    max_sum, temp = 0, 0
    for item in arr:
        temp += item
        max_sum = max(max_sum, temp)
    return max_sum


def divisible(n):
    # 約数のリスト
    # import sympy
    # return sympy.divisors(n)
    divisors = []
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    # divisors.sort()
    return divisors


def divisible_count(n):
    # 約数の個数
    divisors = 0
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            divisors += 1
            if i != n // i:
                divisors += 1
    return divisors


def divisible_sum(n):
    # 約数の総和
    divisors = 0
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            divisors += i
            if i != n // i:
                divisors += n // i
    return divisors


def amicable(n):
    # 友愛数かどうか
    div_sum = divisible_sum(n) - n
    return n != div_sum and divisible_sum(div_sum) - div_sum == n


def scramble(s1, s2):
    # 文字列s1の一部を並び替えるとs2になるか判定
    c_s2 = Counter(s2)
    c_s1 = Counter(s1)
    # return len(c_s2 - c_s1) == 0
    for item, cnt in c_s2.items():
        if c_s1[item] < cnt:
            return False
    return True


def tree_path():
    # 辺と長さが与えられる
    # 頂点1からの各点の距離を計算する
    # 距離が偶数か奇数かで色分け
    def walk(i, c):
        if d[i] == -1:
            d[i] = c
            for node, dist in path[i]:
                walk(node, c + dist)

    sys.setrecursionlimit(100000)
    n = int(input())
    path = [[] for _ in range(n + 1)]
    for _ in range(n - 1):
        u, v, w = map(int, input().split())
        # 無向グラフ
        # このようにあらわす方法もある edge[u][v] = w
        # edge = defaultdict(lambda: defaultdict(lambda: float("inf")))
        path[u].append([v, w])
        path[v].append([u, w])
    # d[0]は使わない
    d = [-1] * (n + 1)
    walk(1, 0)
    for item in d[1:]:
        print(item % 2)


def make_readable(seconds):
    # 秒数表示を時間表示に変える
    return "{:02}:{:02}:{:02}".format(seconds // 3600, (seconds % 3600) // 60, seconds % 60)


def snail(array):
    # 時計回りにぐるぐると2次元配列を読んでいき、1次元配列で返す
    return list(array[0]) + snail(list(zip(*array[1:]))[::-1]) if array else []


def next_bigger(n):
    # n の各桁を並び替えて、nの次に大きくなる数
    s = list(str(n))
    for i in range(1, len(s))[::-1]:
        if s[i] > s[i - 1]:
            j = i
            while j < len(s) and s[i - 1] < s[j]:
                j += 1
            s[i - 1], s[j - 1] = s[j - 1], s[i - 1]
            return int("".join(s[:i] + sorted(s[i:])))
    return -1


def sudoku_valid(board):
    # sudokuの答え(2次元配列)が正解かどうか判定する
    for row in board:
        if len(set(row)) < 9 or 0 in row:
            return False
    for col in zip(*board):
        if len(set(col)) < 9 or 0 in col:
            return False
    for row in range(3):
        for col in range(3):
            square = [board[y][x] for x in range(col * 3, col * 3 + 3) for y in range(row * 3, row * 3 + 3)]
            if len(set(square)) < 9 or 0 in square:
                return False
    return True


def fib(n):
    # フィボナッチn項目
    mat = np.array([[1, 1], [1, 0]], dtype=object)
    return np.linalg.matrix_power(mat, n)[0, 1]


def fast_fib(n):
    # 高速フィボナッチn項目
    def _fib(i):
        if not i:
            return 0, 1
        else:
            a, b = _fib(i // 2)
            c = a * (b * 2 - a)
            d = a * a + b * b
            if i % 2 == 0:
                return c, d
            else:
                return d, c + d

    return _fib(n)[0]


def last_digit(a, b):
    # a ** bの1桁目
    return pow(a, b, 10)


def queens(n):
    # nクイーン問題の解法が何通りあるか
    cnt = 0
    for perm in permutations(range(n)):
        cnt += len(set(i + x for i, x in enumerate(perm))) == n and \
               len(set(i - x for i, x in enumerate(perm))) == n
    return cnt


def rec_queens(n):
    def is_not_under_attack(row, col):
        return not (rows[col] or hills[row - col] or dales[row + col])

    def place_queen(row, col):
        rows[col] = 1
        hills[row - col] = 1  # "hill" diagonals
        dales[row + col] = 1  # "dale" diagonals

    def remove_queen(row, col):
        rows[col] = 0
        hills[row - col] = 0  # "hill" diagonals
        dales[row + col] = 0  # "dale" diagonals

    def backtrack(row=0, count=0):
        for col in range(n):
            if is_not_under_attack(row, col):
                place_queen(row, col)
                if row + 1 == n:
                    count += 1
                else:
                    count = backtrack(row + 1, count)
                remove_queen(row, col)
        return count

    rows = [0] * n
    hills = [0] * (2 * n - 1)  # "hill" diagonals
    dales = [0] * (2 * n - 1)  # "dale" diagonals
    return backtrack()


def contain(s1, s2):
    # 文章s1の単語を並び替えて文章s2を作ることができるか
    c1 = Counter(s1)
    c2 = Counter(s2)
    return c1 == c2


def common_strings(s1, s2):
    # 文字列s1, s2に共通する文字があるかどうか
    # 集合の積集合が存在すればYES
    return "YES" if set(s1) & set(s2) else "NO"


def coin_sum(n):
    # dp
    # コインを使ってnを作る方法が何通りあるか。
    coins = [1, 2, 5, 10, 20, 50, 100, 200]
    dp = [1] + [0] * n
    for coin in coins:
        for x in range(201):
            if x >= coin:
                dp[x] += dp[x - coin]
        print(dp)
    print(dp[n])


def count_triplets(arr, r):
    # arrの3つ組arr[x] * r = arr[y], arr[y] * r = arr[z], x < y < zの組み合わせ
    v2 = defaultdict(int)
    v3 = defaultdict(int)
    count = 0
    for k in arr:
        count += v3[k]
        v3[k * r] += v2[k]
        v2[k * r] += 1
    return count


def iterative_dfs():
    # 迷路探索、スタックを使った深さ優先探索。入力に壁なし。
    h, w = map(int, input().split())
    field = [["#"] * (w + 2)] + [list("#" + input() + "#") for _ in range(h)] + [["#"] * (w + 2)]

    sr, sc = 0, 0
    for i, row in enumerate(field):
        if "s" in row:
            sr, sc = i, row.index("s")

    drc = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    stack = deque()
    stack.append((sr, sc))

    while stack:
        cr, cc = stack.pop()
        for dr, dc in drc:
            nr, nc = cr + dr, cc + dc
            if field[nr][nc] == "#":
                continue
            if field[nr][nc] == "g":
                print("Yes")
                exit()
            stack.append((nr, nc))
            field[nr][nc] = "#"
    print("No")


def bfs():
    # 迷路探索、キューを使った幅優先探索。入力に壁あり。
    h, w = map(int, input().split())
    sr, sc = map(int, input().split())
    gr, gc = map(int, input().split())
    field = [list(input()) for _ in range(h)]
    drc = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    sr, sc, gr, gc = sr - 1, sc - 1, gr - 1, gc - 1
    q = deque()
    q.append((sr, sc, 0))

    while q:
        cr, cc, ct = q.popleft()
        for dr, dc in drc:
            nr, nc = cr + dr, cc + dc
            if (nr, nc) == (gr, gc):
                print(ct + 1)
                exit()
            if field[nr][nc] == "#":
                continue
            q.append((nr, nc, ct + 1))
            field[nr][nc] = "#"
    print("No")


def union_find():
    # グラフの連結していないグループの数をunion_findで求める
    def get_parent(node):
        if par[node] == -1:
            return node
        else:
            par[node] = get_parent(par[node])
            return par[node]

    def merge(i, j):
        i = get_parent(i)
        j = get_parent(j)
        if i != j:
            par[j] = i
        return

    n, m = map(int, input().split())
    xy = [list(map(int, input().split())) for _ in range(m)]
    # par[0]は使わない
    par = [-1] * (n + 1)
    for x, y in xy:
        merge(x, y)
    print(par)
    print(par[1:].count(-1))


def fft():
    # aとbの畳み込み。c_k = \sum_{i=0}^k a_i b_{k-i}
    n = int(input())
    n0 = 2 ** int(np.ceil(np.log2(2 * n - 1)))
    a = np.zeros(n0)
    b = np.zeros(n0)
    for i in range(n):
        a[i], b[i] = map(int, input().split())
    c = np.fft.ifft(np.fft.fft(a) * np.fft.fft(b))
    print(0)
    for ci in np.real(c[:2 * n - 1] + 0.5):
        print(int(ci))


def heap_sort(arr):
    heapq.heapify(arr)
    return [heapq.heappop(arr) for _ in range(len(arr))]


def bubble_sort(arr):
    n = len(arr)
    flag = True
    i = 0
    while flag:
        flag = False
        for j in range(n - 1, 0, -1):
            if arr[j] < arr[j - 1]:
                arr[j], arr[j - 1] = arr[j - 1], arr[j]
                flag = True
        i += 1
    return arr


def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        minj = i
        for j in range(i, n):
            if arr[j] < arr[minj]:
                minj = j
        arr[i], arr[minj] = arr[minj], arr[i]
    return arr


def insert_sort(arr):
    n = len(arr)
    for i in range(1, n):
        v = arr[i]
        j = i
        while j > 0:
            if arr[j - 1] > v:
                arr[j] = arr[j - 1]
                j -= 1
            else:
                break
        arr[j] = v
    return arr


def merge_sort(arr, left, right):
    # merge_sort(li, 0, n)
    if right - left == 1:
        return
    mid = left + (right - left) // 2
    merge_sort(arr, left, mid)
    merge_sort(arr, mid, right)
    a = [arr[i] for i in range(left, mid)] + [arr[i] for i in range(right - 1, mid - 1, -1)]
    iterator_left = 0
    iterator_right = len(a) - 1
    for i in range(left, right):
        if a[iterator_left] <= a[iterator_right]:
            arr[i] = a[iterator_left]
            iterator_left += 1
        else:
            arr[i] = a[iterator_right]
            iterator_right -= 1


def quick_sort(arr, left, right):
    # quick_sort(li, 0, n)
    if right - left <= 1:
        return
    pivot_index = random.randrange(left, right)
    pivot = arr[pivot_index]
    arr[pivot_index], arr[right - 1] = arr[right - 1], arr[pivot_index]
    i = left
    for j in range(left, right - 1):
        if arr[j] < pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[right - 1] = arr[right - 1], arr[i]
    quick_sort(arr, left, i)
    quick_sort(arr, i + 1, right)


def count_sort(arr):
    max_num = max(arr)
    min_num = min(arr)
    count = [0] * (max_num - min_num + 1)
    for ele in arr:
        count[ele - min_num] += 1
    return [ele for ele, cnt in enumerate(count, start=min_num) for _ in range(cnt)]


def count_inversions(arr):
    # バブルソートの必要回数
    # マージソートを使って計算 O(n log(n))
    n = len(arr)
    if n == 1:
        return 0
    n1 = n // 2
    n2 = n - n1
    arr1 = arr[:n1]
    arr2 = arr[n1:]
    ans = count_inversions(arr1) + count_inversions(arr2)
    i1 = 0
    i2 = 0
    for i in range(n):
        if i1 < n1 and (i2 >= n2 or arr1[i1] <= arr2[i2]):
            arr[i] = arr1[i1]
            ans += i2
            i1 += 1
        elif i2 < n2:
            arr[i] = arr2[i2]
            i2 += 1
    return ans


def factorial_mod(n, mod):
    a = 1
    for i in range(1, n + 1):
        a *= i
        a %= mod
    return a


def comb_mod(n, k, mod):
    if k > n:
        return 0
    fact_n = factorial_mod(n, mod)
    fact_k = factorial_mod(k, mod)
    fact_n_k = factorial_mod(n - k, mod)
    return (fact_n * pow(fact_k, mod - 2, mod) * pow(fact_n_k, mod - 2, mod)) % mod


def prime_factorization(x):
    # 素因数分解
    # 事前に素数のセットが必要
    primes = fast_prime_set(10 ** 7)

    fact = defaultdict(int)
    for prime in primes:
        while x % prime == 0:
            fact[prime] += 1
            x //= prime
        if x == 1:
            break
        if x in primes:
            fact[x] += 1
            break
    return fact


def fast_lcs(a, b):
    arr = []
    for bk in b:
        bgn_idx = 0  # 検索開始位置
        for i, cur_idx in enumerate(arr):
            chr_idx = a.find(bk, bgn_idx) + 1
            if not chr_idx:
                break
            arr[i] = min(cur_idx, chr_idx)
            bgn_idx = cur_idx
        else:
            chr_idx = a.find(bk, bgn_idx) + 1
            if chr_idx:
                arr.append(chr_idx)
    return len(arr)


def lcs(a, b):
    n = len(a)
    m = len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    # return dp[n][m]
    # 文字列を復元する場合は以下
    lcs_str = ''
    i, j = n, m
    while i >= 1 and j >= 1:
        if a[i - 1] == b[j - 1]:
            lcs_str += a[i - 1]  # or b[j - 1]
            i -= 1
            j -= 1
        else:
            if dp[i - 1][j] > dp[i][j - 1]:
                i -= 1
            else:
                j -= 1

    return lcs_str[::-1]


def small_multiple(k):
    # kの倍数で各桁の総数が最も小さくなるような、各桁の総数
    dist = [float("inf")] * k
    q = []
    # kで割った余りが1になるような最小の数は1
    heapq.heappush(q, (1, 1))

    while q:
        n, c = heapq.heappop(q)
        if dist[n] <= c:
            continue
        print(n, c)
        dist[n] = c

        if dist[(n + 1) % k] > c + 1:
            heapq.heappush(q, ((n + 1) % k, c + 1))
        if dist[(n * 10) % k] > c:
            heapq.heappush(q, ((n * 10) % k, c))

    print(dist[0])


def longest_path():
    # メモ化再帰で有向グラフの最長パスの長さを求める

    def rec(node):
        if memo[node]:
            return memo[node]

        res = 0
        for next_node in edge[node]:
            res = max(res, rec(next_node) + 1)

        memo[node] = res
        return res

    n, m = map(int, input().split())
    memo = [0] * (n + 1)
    edge = defaultdict(list)
    for i in range(m):
        x, y = map(int, input().split())
        edge[x].append(y)
    ans = 0
    for i in range(1, n + 1):
        ans = max(ans, rec(i))
    print(ans)


def longest_increasing_subsequence_length(arr):
    # 最長増加列の長さ
    dp = [arr[0]]
    for i in arr[1:]:
        if dp[-1] < i:
            dp.append(i)
        else:
            dp[bisect_left(dp, i)] = i
    return len(dp)


def longest_increasing_subsequence(arr):
    # 最長増加列のリスト
    dp = [arr[0]]
    dp_pos = [0]
    cnt = 0
    for i in arr[1:]:
        if dp[-1] < i:
            cnt += 1
            dp.append(i)
            dp_pos.append(cnt)
        else:
            idx = bisect_left(dp, i)
            dp[idx] = i
            dp_pos.append(idx)
    res = []
    for i in range(len(dp_pos))[::-1]:
        if dp_pos[i] == cnt:
            res.append(arr[i])
            cnt -= 1
    return res[::-1]


def word_break(s, words):
    # wordsの組み合わせでsが作れるかどうか
    ok = [True]
    for i in range(1, len(s)+1):
        ok += any(ok[j] and s[j:i] in words for j in range(i)),
    return ok[-1]


def max_area_of_island(grid):
    # 0と1で構成されるgridで連結している最大の面積
    height = len(grid)
    width = len(grid[0])
    seen = set()
    drc = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def area(r, c):
        if 0 <= r < height and 0 <= c < width and (r, c) not in seen and grid[r][c]:
            seen.add((r, c))
            return 1 + sum(area(r+dr, c+dc) for dr, dc in drc)
        else:
            return 0

    return max(area(r, c) for r in range(height) for c in range(width))


def is_power_of_two(n):
    # 2のべき乗かどうか
    return n > 0 and not (n & n-1)


def coin_change(coins, amount):
    # いくつかの種類のコインを使って、目標数を作る。
    # 最低何枚で作ることができるか。作れない場合は-1を出力。
    dp = [0] + [float('inf')] * amount
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[-1] if dp[-1] != float("inf") else -1


def get_sum(a, b):
    # ビット演算でaとbの和を求める
    max_lim = 2 ** 31 - 1
    mask = 2 ** 32 - 1
    while b != 0:
        a, b = (a ^ b) & mask, ((a & b) << 1) & mask
    return a if a <= max_lim else ~(a ^ mask)


def gcd(a, b):
    # 最大公約数
    while b:
        a, b = b, a % b
    return a


def topological_sort(words):
    # 特定の辞書順に並んでいる単語のリストから文字の順番を返す
    chars = set("".join(words))
    degrees = {x: 0 for x in chars}
    graph = defaultdict(list)
    for pair in zip(words, words[1:]):
        for x, y in zip(*pair):
            if x != y:
                if y not in graph[x]:
                    graph[x].append(y)
                    degrees[y] += 1
                break

    queue = [x for x in degrees.keys() if degrees[x] == 0]
    res = ""
    while queue:
        x = queue.pop()
        res += x
        for n in graph[x]:
            degrees[n] -= 1
            if degrees[n] == 0:
                queue.append(n)

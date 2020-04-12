from functools import reduce


def knapsack(fruits, limit):
    def nextVI(i, values, items):
        return reduce(
            (lambda vis, vi: (vis[0] + [vi[0]], vis[1] + [vi[1]])),
            [(values[w - fruits[i][1]] + fruits[i][2], i)
                if w >= fruits[i][1] and w < limit + 1 and
             values[w - fruits[i][1]] + fruits[i][2] > values[w]
                else (values[w], items[w]) for w in range(len(values))],
            ([], [])
        )

    def iterate(i):
        if i == 0:
            return nextVI(i, [0] * (limit + 1), [0] * (limit + 1))
        else:
            values, items = iterate(i - 1)
            return nextVI(i, values, items)

    def solution(i, items, minWeight):
        return (([fruits[items[i]]] +
                 solution(i - fruits[items[i]][1], items, minWeight))
                if i >= minWeight else [])

    return solution(limit,
                    iterate(len(fruits) - 1)[1], min([f[1] for f in fruits]))


print(knapsack([('李子', 4, 4500), ('蘋果', 5, 5700),
                ('橘子', 2, 2250), ('草莓', 1, 1100),
                ('甜瓜', 6, 6700)], 8))

# https://openhome.cc/Gossip/AlgorithmGossip/KnapsackProblem.htm

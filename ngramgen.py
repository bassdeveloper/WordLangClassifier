

def find_n_grams(input_list, n):
    return ["".join(j) for j in zip(*[input_list[i:] for i in range(n)])]
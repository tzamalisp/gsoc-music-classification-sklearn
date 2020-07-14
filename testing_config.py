from utils import load_yaml, FindCreateDirectory
import numpy as np
import math
from pprint import pprint

C = np.array([[1, 1, 1], [1, 2, 0], [0, 0, 1]])
print(C)

B = C / C.astype(np.float).sum(axis=1)

print(B)

A = C.astype("float") / C.sum(axis=1)[:, np.newaxis]

print(A)

print("2 ** 7 =", 2 ** 7)
print("log(2 ** 7) =", math.log2(2 ** 7))


if __name__ == '__main__':
    config_data = load_yaml("configuration.yaml")
    print(config_data.get("grid_class_weight"))

    print("Seed type:", type(config_data["random_seed"]))
    c = [-5, -3, -1, 1, 3, 5, 7, 9, 11]
    gamma = [3, 1, -1, -3, -5, -7, -9, -11]
    c = [2 ** x for x in c]
    gamma = [2 ** x for x in gamma]
    print(c)
    print(gamma)


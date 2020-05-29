from utils import load_yaml, FindCreateDirectory
import numpy as np
C = np.array([[1, 1, 1], [1, 2, 0], [0, 0, 1]])
print(C)

B = C / C.astype(np.float).sum(axis=1)

print(B)

A = C.astype("float") / C.sum(axis=1)[:, np.newaxis]

print(A)

if __name__ == '__main__':
    config_data = load_yaml()
    print(config_data.get("grid_class_weight"))

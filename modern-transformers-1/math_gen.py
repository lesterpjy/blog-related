import random
import numpy as np
from typing import List
import numexpr as ne
from tqdm import tqdm


ops = ["/", "*", "-", "+"]
nprob = 10000
max_node = 2
pad_len = 10


# calculation node class
class OpNode:
    def __init__(self, left: int, right: int, op: str) -> None:
        self.left = left
        self.right = right
        self.op = op

    def __str__(self) -> str:
        return f"({self.left}{self.op}{self.right})"

    def padded_str(self) -> str:
        padded_left = "0" * (pad_len - len(str(self.left))) + str(self.left)
        padded_right = "0" * (pad_len - len(str(self.right))) + str(self.right)
        return f"({padded_left}{self.op}{padded_right})"


def build_tree(num_nodes: int) -> OpNode:
    """
    build a problem with num_nodes of random ints
    """
    if num_nodes == 1:
        num = random.randint(1, 1000)
        fac = np.random.choice([1, random.random()], p=[0.6, 0.4])
        return num if fac == 1 else round(fac * num, 2)
    left_subtree = build_tree(num_nodes // 2)
    right_subtree = build_tree(num_nodes - (num_nodes // 2))
    op = random.choice(ops)
    return OpNode(left_subtree, right_subtree, op)


def generate_problems(nprob: int, max_node: int) -> List[str]:
    """
    generate nprob problems with at most max_node ints involved
    """
    probs = []
    for _ in tqdm(range(nprob)):
        num_nodes = random.randint(2, max_node)
        prob = build_tree(num_nodes)
        try:
            ans = str(round(ne.evaluate(str(prob)).item(), 2))
        except ZeroDivisionError:
            ans = "nan"
        # pad prob and ans
        ans = "0" * (pad_len - len(ans)) + ans
        probs.append(f"${prob.padded_str()}={ans[::-1]}$")  # reverse ans
    return probs


with open("./test.txt", "w") as f:
    for prob in generate_problems(nprob, max_node):
        f.write(f"{prob}\n")

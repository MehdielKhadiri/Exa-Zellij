def loss(x):
    return sum((value - 0.3) ** 2 for value in x)

from fqo import Config, minimize


def objective(x):
    # Replace this with a simulation, model evaluation or existing library call.
    return sum((value - 0.3) ** 2 for value in x)


for family in ("fbd", "fbo"):
    result = minimize(objective, bounds=[(-5, 5)] * 3,
                      config=Config(family=family, budget=100 if family=="fbo" else None,
                                    components={"decomposition": "trisection"}))
    print(family, result.fun, result.x, result.nfev, result.status)

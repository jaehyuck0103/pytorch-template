from .net1 import Net1


def get_network(name):

    if name == "net1":
        net = Net1()
    else:
        raise ValueError(f"Unexpected Network {name}")

    return net

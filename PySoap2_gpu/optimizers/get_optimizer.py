from PySoap2_gpu import optimizers


def get_optimizer(optimizer_str):
    available_optimizers = ['Adam']
    for existing_optimizer in available_optimizers:
        if existing_optimizer.lower() == optimizer_str.lower():
            return optimizers.__dict__[existing_optimizer]()

    raise ValueError(f"{optimizer_str} is not defined.")

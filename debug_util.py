# デバッグ用...

import torch


def check_requires_grad(model: torch.nn.Module):
    for name, module in list(model.named_modules()):
        if len(list(module.parameters())) > 0:
            print(f"Module: {name}")
            for name, param in list(module.named_parameters()):
                print(f"    Parameter: {name}, Requires Grad: {param.requires_grad}")


def check_training_mode(model: torch.nn.Module):
    for name, module in list(model.named_modules()):
        print(f"Module: {name}, Training Mode: {module.training}")

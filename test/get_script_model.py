import torch


def get_submodule(model: torch.jit.ScriptModule, name: str):
    for n, m in model.named_modules():
        if n == name:
            return m
    raise RuntimeError


model: torch.jit.ScriptModule = torch.jit.load("/home/zhfeing/project/general-KD/run/cifar10/vit/deit_tiny_patch16_224_pretrained-s-1029/jit/iter-187440-jit.pth", map_location="cpu")

# model.get_submodule("model.cls_head")
qkv = get_submodule(model, "model.transformer.layers.11.attention.linear_qkv")

with torch.no_grad():
    print(qkv.bias.norm(), qkv.weight.norm())
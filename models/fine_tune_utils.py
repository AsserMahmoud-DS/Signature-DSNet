import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------
# Freeze utilities
# ---------------------------
def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def freeze_backbone_bn(module: nn.Module):
    """
    Backbone contains BatchNorm2d in patch embedding blocks. Keep them in eval
    so running_mean/var don't change during head-only fine-tune.
    """
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def print_trainable_params(model: nn.Module):
    total, trainable = 0, 0
    for n, p in model.named_parameters():
        num = p.numel()
        total += num
        if p.requires_grad:
            trainable += num
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# ---------------------------
# OPTION A: head-only fine-tune
# ---------------------------
def setup_head_only_finetune(model: nn.Module):
    # 1) Freeze everything
    set_requires_grad(model, False)

    # 2) Unfreeze only head_str (your embedding projection head)  :contentReference[oaicite:4]{index=4}
    set_requires_grad(model.head_str, True)

    # 3) Put model in train, but lock backbone BN stats
    model.train()
    # freeze BN inside backbone (Transformer + patch embeds) :contentReference[oaicite:5]{index=5}
    freeze_backbone_bn(model.model)

    print_trainable_params(model)
    return model

# Example usage:
# model = setup_head_only_finetune(model).to(device)

# ---------------------------
# Optimizer best practice
# ---------------------------
def build_optimizer_for_head_only(model: nn.Module, lr_head=1e-4, weight_decay=1e-4):
    # Only params with requires_grad=True will be optimized
    params = [p for p in model.parameters() if p.requires_grad]
    # AdamW is usually a better default than Adam for fine-tuning transformers
    optimizer = optim.AdamW(params, lr=lr_head, weight_decay=weight_decay)
    return optimizer


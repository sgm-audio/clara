#!/usr/bin/env python3
import torch
ckpt = torch.load("checkpoints_clara/interrupt_045825.pt", map_location="cpu")
print('Keys:', ckpt.keys())
print('Architecture:', ckpt.get('architecture'))
print('Step:', ckpt.get('step'))
print('Loss:', ckpt.get('loss'))
print('Best val loss:', ckpt.get('best_val_loss'))
if 'model_state_dict' in ckpt:
    print('Model keys:')
    for k in list(ckpt["model_state_dict"].keys())[:50]:
        print(f'  {k}: {ckpt["model_state_dict"][k].shape}')
elif 'model' in ckpt:
    print('Model keys (under "model"):')
    for k in list(ckpt["model"].keys())[:50]:
        print(f'  {k}: {ckpt["model"][k].shape}')

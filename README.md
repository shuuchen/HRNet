# HRNet
HRNet implementation in PyTorch

### Training and validation results

### How to use
```python
from hrnet import HRNet

'''
parameters from left to right:
  input channels, first branch channels (hyper parameter), output channels
'''
model = HRNet(3, 16, 8)
```

### Requirements
```
torch==1.5.0
```

### References
- [Deep High-Resolution Representation Learning for Visual Recognition, 2020](https://arxiv.org/abs/1908.07919)
- [Official implementation](https://github.com/HRNet/HRNet-Facial-Landmark-Detection/blob/master/lib/models/hrnet.py)

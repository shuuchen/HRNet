# HRNet
A compact HRNet implementation in PyTorch without YAML configuration

<div align=left>

![](images/hrnet.jpg)

</div>

### Performance

<div align=left>

<img src="https://github.com/shuuchen/HRNet/blob/master/images/loss.png" width="480" height="200" />

</div>

### Usage
```python
from hrnet import HRNet

'''
parameters from left to right:
  input channels, first branch channels (hyper parameter), output channels
  
notice:
  Only the number of first branch channels is necessary, numbers of channels of 
  other branches are calculated according to the paper
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

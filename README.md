# Installation
```bash
pip install git+https://github.com/camilomarino/transforms_1d_torch.git@master
```

# Usage
```python
from transforms_1d_torch import transforms1d

# examples:
transforms1d.RandomCrop(1, 0.8)
transforms1d.Scale(mu=torch.tensor([1, 1, 1]), std=1)

```
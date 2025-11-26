# Signatory 2.0 Migration Guide

## Overview

Signatory 2.0 is a major update that modernizes the library for PyTorch 2.x. This guide explains the changes and how to migrate from Signatory 1.x.

## What's New in 2.0

### PyTorch 2.x Support
- Full compatibility with PyTorch 2.x (tested with PyTorch 2.10)
- Works with NVIDIA PyTorch containers and modern PyTorch installations
- Support for torch.compile (with expected warnings for C++ extensions)

### Python Version Updates
- Now requires Python 3.10 or higher (3.10, 3.11, 3.12)
- Dropped support for Python 3.6, 3.7, 3.8, 3.9

### Critical Bug Fixes
- **Fixed GIL handling bug**: The PyCapsule creation in `make_lyndon_info` was incorrectly releasing the GIL before calling Python C API functions, causing segmentation faults. This is now fixed by properly scoping the GIL release.
- **Deprecated API updates**: Replaced `torch.Tensor()` with `torch.empty(0)` for empty tensor creation

### New Features
- torch.compile support with Signatory operations
- New example: `examples/example_torch_compile.py` demonstrating torch.compile usage
- Updated metadata and installation instructions

## Installation

### From Source (Recommended for 2.0)

```bash
git clone https://github.com/patrick-kidger/signatory.git
cd signatory
git checkout pytorch-2.10-modernization
pip install -e . --no-build-isolation
```

### Requirements

- Python 3.10, 3.11, or 3.12
- PyTorch 2.x
- Linux or Windows
- CUDA (optional, for GPU acceleration)

## API Changes

### No Breaking Changes

The core API remains unchanged! Your existing code using Signatory 1.x will work with Signatory 2.0 without modifications:

```python
import signatory
import torch

# This works exactly the same as in 1.x
path = torch.rand(8, 10, 3)
signature = signatory.signature(path, depth=4)
logsignature = signatory.logsignature(path, depth=4)
```

### New Capabilities

#### torch.compile Support

You can now use torch.compile with models that include Signatory operations:

```python
import signatory
import torch
from torch import nn

class SigNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sig = signatory.Signature(depth=3)
        self.linear = nn.Linear(signatory.signature_channels(3, 3), 10)
    
    def forward(self, x):
        return self.linear(self.sig(x, basepoint=True))

model = SigNet()
compiled_model = torch.compile(model)
# Works! (with warnings about C++ extensions)
```

**Note**: You may see warnings from torch.compile about not being able to trace the C++ extensions. This is expected and does not affect functionality.

## Migration Checklist

If you're upgrading from Signatory 1.x to 2.0:

- [ ] Update Python to 3.10 or higher
- [ ] Update PyTorch to 2.x
- [ ] Reinstall Signatory 2.0
- [ ] Test your code (no API changes required!)
- [ ] Optionally try torch.compile for potential speedups

## Technical Details

### Fixed Issues

1. **GIL Handling in PyCapsule Creation**
   - **Issue**: The `make_lyndon_info` function released the GIL before calling `wrap_capsule`, which internally calls `PyCapsule_New`. This Python C API function requires the GIL to be held.
   - **Fix**: Scoped the GIL release to only cover the Lyndon computation, re-acquiring it before capsule creation.
   - **Impact**: Prevents segmentation faults when using logsignature operations.

2. **Deprecated Tensor Creation**
   - **Issue**: `torch.Tensor()` is deprecated in PyTorch 2.x
   - **Fix**: Replaced with `torch.empty(0)` in `signature_module.py`
   - **Impact**: Eliminates deprecation warnings and ensures future compatibility.

### Compatibility Matrix

| Signatory Version | Python Version | PyTorch Version |
|-------------------|----------------|-----------------|
| 2.0.0             | 3.10-3.12      | 2.x             |
| 1.2.7             | 3.7-3.9        | 1.8.0-1.11.0    |

## Testing

All core functionality has been tested:
- ✅ Signature computation (CPU and CUDA)
- ✅ Logsignature computation (CPU and CUDA)
- ✅ All example scripts
- ✅ torch.compile integration
- ✅ Backwards pass and gradients

## Getting Help

If you encounter issues:
1. Check that you're using Python 3.10+ and PyTorch 2.x
2. Try installing from source if pre-built wheels don't work
3. Open an issue on GitHub with details about your environment

## Acknowledgments

This modernization effort was undertaken to ensure Signatory remains compatible with the latest PyTorch ecosystem and container environments like NVIDIA PyTorch containers.


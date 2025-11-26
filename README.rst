
|Signatory|
###########

.. |Signatory| image:: https://raw.githubusercontent.com/patrick-kidger/signatory/master/docs/_static/signatory.png

Differentiable computations of the signature and logsignature transforms, on both CPU and GPU.




What is the signature transform?
################################
The *signature transform* is roughly analogous to the Fourier transform, in that it operates on a stream of data (often a time series). Whilst the Fourier transform extracts information about frequency, the signature transform extracts information about *order* and *area*. Furthermore (and unlike the Fourier transform), order and area represent all possible nonlinear effects: the signature transform is a *universal nonlinearity*, meaning that every continuous function of the input stream may be approximated arbitrary well by a *linear* function of its signature. If you're doing machine learning then you probably understand why this is such a desirable property!

Besides this, the signature transform has many other nice properties -- robustness to missing or irregularly sampled data; optional translation invariance; optional sampling invariance. Furthermore it can be used to encode certain physical quantities, and may be used for data compression.


Check out `this <https://arxiv.org/abs/1603.03788>`__ for a primer on the use of the signature transform in machine learning, just as a feature transformation, and `this <https://papers.nips.cc/paper/8574-deep-signature-transforms>`__ for a more in-depth look at integrating the signature transform into neural networks.




Installation
############

**Signatory 2.0 - PyTorch 2.x Support**

Signatory 2.0 has been modernized for PyTorch 2.x with support for Python 3.10+.

Installation from source
------------------------

.. code-block:: bash

    git clone https://github.com/patrick-kidger/signatory.git
    cd signatory
    git checkout pytorch-2.10-modernization
    pip install -e . --no-build-isolation

**Requirements:**

* Python 3.10, 3.11, or 3.12
* PyTorch 2.x (tested with PyTorch 2.10)
* Linux or Windows
* CUDA support (optional, for GPU acceleration)

After installation, just ``import signatory`` inside Python.

**Legacy Installation (PyTorch 1.x)**

For PyTorch 1.8.0--1.11.0, use Signatory 1.2.7:

.. code-block:: bash

    pip install signatory==1.2.7.<TORCH_VERSION> --no-cache-dir --force-reinstall

where ``<TORCH_VERSION>`` is your PyTorch version (e.g., ``1.11.0``).

**What's New in 2.0:**

* PyTorch 2.x compatibility
* Python 3.10+ support  
* torch.compile support (with warnings for C++ extensions)
* Fixed GIL handling in PyCapsule creation
* Updated to modern PyTorch APIs

If you have any problems with installation, feel free to `open an issue <https://github.com/patrick-kidger/signatory/issues>`__.



Documentation
#############
The documentation is available `here <https://signatory.readthedocs.io>`__.

Example
#######
Usage is straightforward. As a simple example,

.. code-block:: python

    import signatory
    import torch
    batch, stream, channels = 1, 10, 2
    depth = 4
    path = torch.rand(batch, stream, channels)
    signature = signatory.signature(path, depth)
    # signature is a PyTorch tensor

**PyTorch 2.x torch.compile Support**

Signatory 2.0 works with torch.compile:

.. code-block:: python

    import signatory
    import torch
    from torch import nn

    class SigNet(nn.Module):
        def __init__(self, in_channels, out_dimension, sig_depth):
            super().__init__()
            self.signature = signatory.Signature(depth=sig_depth)
            sig_channels = signatory.signature_channels(in_channels, sig_depth)
            self.linear = nn.Linear(sig_channels, out_dimension)
        
        def forward(self, path):
            return self.linear(self.signature(path, basepoint=True))
    
    model = SigNet(3, 5, 3)
    compiled_model = torch.compile(model)
    # Works! (with expected warnings about C++ extensions)

For further examples, see the `documentation <https://signatory.readthedocs.io/en/latest/pages/examples/examples.html>`__ or ``examples/example_torch_compile.py``.


Citation
########
If you found this library useful in your research, please consider citing `the paper <https://arxiv.org/abs/2001.00706>`__.

.. code-block:: bibtex

    @inproceedings{kidger2021signatory,
      title={{S}ignatory: differentiable computations of the signature and logsignature transforms, on both {CPU} and {GPU}},
      author={Kidger, Patrick and Lyons, Terry},
      booktitle={International Conference on Learning Representations},
      year={2021},
      note={\url{https://github.com/patrick-kidger/signatory}}
    }

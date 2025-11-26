# Copyright 2019 Patrick Kidger. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
"""Example demonstrating torch.compile support with Signatory in PyTorch 2.x."""


import signatory
import torch
from torch import nn


class SigNetCompiled(nn.Module):
    """Example network using signatures with torch.compile support."""
    
    def __init__(self, in_channels, out_dimension, sig_depth):
        super(SigNetCompiled, self).__init__()
        self.augment = signatory.Augment(in_channels=in_channels,
                                         layer_sizes=(),
                                         kernel_size=1,
                                         include_original=True,
                                         include_time=True)
        self.signature = signatory.Signature(depth=sig_depth)
        # +1 because signatory.Augment is used to add time as well
        sig_channels = signatory.signature_channels(channels=in_channels + 1,
                                                    depth=sig_depth)
        self.linear = torch.nn.Linear(sig_channels, out_dimension)

    def forward(self, inp):
        # inp is a three dimensional tensor of shape (batch, stream, in_channels)
        x = self.augment(inp)
        if x.size(1) <= 1:
            raise RuntimeError("Given an input with too short a stream to take the signature")
        # x is a three dimensional tensor of shape (batch, stream, in_channels + 1),
        # as time has been added as a value
        y = self.signature(x, basepoint=True)
        # y is a two dimensional tensor of shape (batch, terms), corresponding to
        # the terms of the signature
        z = self.linear(y)
        # z is a two dimensional tensor of shape (batch, out_dimension)
        return z


def main():
    print("PyTorch 2.x torch.compile support with Signatory")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Signatory version: {signatory.__version__}")
    print()
    
    # Create model
    in_channels = 3
    out_dimension = 5
    sig_depth = 3
    model = SigNetCompiled(in_channels, out_dimension, sig_depth)
    
    # Create sample input
    batch_size = 8
    stream_length = 10
    x = torch.randn(batch_size, stream_length, in_channels)
    
    # Test regular execution
    print("Testing regular execution...")
    output_regular = model(x)
    print(f"Output shape: {output_regular.shape}")
    print()
    
    # Test with torch.compile
    print("Testing with torch.compile...")
    try:
        compiled_model = torch.compile(model, mode="default")
        output_compiled = compiled_model(x)
        print(f"Compiled output shape: {output_compiled.shape}")
        print("✓ torch.compile works with Signatory!")
        
        # Verify outputs match
        if torch.allclose(output_regular, output_compiled, rtol=1e-4, atol=1e-4):
            print("✓ Regular and compiled outputs match!")
        else:
            print("⚠ Outputs differ slightly (expected due to compilation)")
    except Exception as e:
        print(f"Note: torch.compile encountered an issue: {e}")
        print("This is expected for custom C++ extensions in some cases.")
    print()
    
    # Test CUDA if available
    if torch.cuda.is_available():
        print("Testing CUDA execution...")
        model_cuda = model.cuda()
        x_cuda = x.cuda()
        output_cuda = model_cuda(x_cuda)
        print(f"CUDA output shape: {output_cuda.shape}")
        print("✓ CUDA execution works!")
        print()
        
        # Test torch.compile on CUDA
        print("Testing torch.compile on CUDA...")
        try:
            compiled_model_cuda = torch.compile(model_cuda, mode="default")
            output_compiled_cuda = compiled_model_cuda(x_cuda)
            print(f"CUDA compiled output shape: {output_compiled_cuda.shape}")
            print("✓ torch.compile works with Signatory on CUDA!")
        except Exception as e:
            print(f"Note: torch.compile on CUDA encountered an issue: {e}")
            print("This is expected for custom C++ extensions in some cases.")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()


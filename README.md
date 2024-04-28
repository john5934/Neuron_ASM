PURPOSE: To compute the weighted sum of a neuron in a neural net.

These primitives are written in assembler for speed and can be called from C on Linux.

Parameters are passed in registers according to the Linux ABI standard, so any compatible C compiler should work.

The x86_64 version requires the NASM assembler.

The ARM64 version builds with the 'as' assembler available on Raspberry Pi OS.

LICENSE: This code is released to the public domain.
         It is provided "as-is" without warranty of any kind.

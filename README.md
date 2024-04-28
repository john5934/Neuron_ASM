PURPOSE: To compute the weighted sum of a neuron in a neural net.

These primitives are written in assembler for speed and are callable from C on Linux.

Parameters are passed in registers according to the Linux ABI standard, so any compatible C compiler
should be able to call these routines. I've run it with gcc on Debian and Raspberry Pi OS.

The x86_64 version requires the NASM assembler.

The ARM64 version builds with the 'as' assembler available on Raspberry Pi OS.

LICENSE: This code is released to the public domain. 
         It is provided "as-is" without warranty of any kind.

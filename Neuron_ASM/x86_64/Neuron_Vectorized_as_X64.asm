;-------------------------------------------------------------------------------
; Neuron_Vectorized_as_X64.asm
;-------------------------------------------------------------------------------
;
; PURPOSE: To compute the weighted sum of a neuron from its inputs and bias.
;
; DESCRIPTION: Produces a weighted sum ready to be fed into an activation
; function.
;
; Multiply-adds inputs and weights and adds with the neuron bias.
;
; This is a vectorized assembler version of Neuron(). Where there are more than
; two inputs, this routine multiplies inputs and weights two at a time instead
; of one by one.
;
; This routine is a drop-in replacement for Neuron() since the inputs and output
; are identical.
;
; This version does not use fused multiply-add instructions, so it can be run on
; non-Xeon x64 machines. There will be a version of this routine for the Xeon
; that does use fused multiply-add instructions: see
;                                             Neuron_Vectorized_fma_as_X64.asm .
; REQUIRES:
;
;   InputCount > 0
;
;   A CPU that supports x86_64 instructions.
;
;   External code will handle overflow to infinity or NaN's if any.
;
; This C code is functionally equivalent to the NASM assembly code below:
;
;      // OUT: Neuron output value ready for an activation function.
;    f64
;    Neuron(
;        f64* Inputs,
;                // Input values fed to the neuron.
;                //
;        f64* Weights,
;                // Weights corresponding to each input.
;                //
;        u32 InputCount,
;                // Number of inputs and corresponding weights.
;                //
;        f64 Bias )
;                // Bias value of the neuron.
;    {
;        // For each input, multiply it with a weight and accumulate the result.
;        do
;        {
;            // Use Bias as the accumulator to save an addition.
;            Bias += *Inputs++ * *Weights++;
;
;            // Account for having processed one input.
;            InputCount--;
;
;        } while( InputCount );
;
;        // Return the sum of products and the bias.
;        return( Bias );
;    }
;
; For Linux x64, these registers belong to the caller: rbp, rbx and r12 thru r15
; and must be preserved by this routine if they are to be used. The remaining
; registers are available for use by this routine, but must be preserved in its
; local stack frame if they are to be preserved across function calls. From the
; Linux ABI doc: see 'mpx-linux64-abi.pdf' in the x64 folder. xmm registers do
; not need to be preserved for the caller.
;
; REGISTER USAGE:
;
;   Defined inline below.
;
; HISTORY:
;    21Dec2023 From Neuron(), Neuron_as_X64.asm, and Sum_f64s_as.asm.
;    21Dec2023 Passed Neuron_Vectorized_as_Test() OK.
;-------------------------------------------------------------------------------

%include "platform_nasm.inc"

global Neuron_Vectorized_as

;-------------------------------------------------------------------------------
;   // OUT: Neuron output value in xmm0 ready for an activation function.
; f64
; Neuron_Vectorized_as(
;     f64* Inputs,
;              // Input values fed to the neuron.
;              //
;     f64* Weights,
;              // Weights corresponding to each input.
;              //
;     u32 InputCount,
;              // Number of inputs and corresponding weights.
;              //
;     f64 Bias )
;              // Bias value of the neuron.

;///////////////////////
Neuron_Vectorized_as:;//
;///////////////////////

                    ; Inputs:
                    ;
                    ;   rdi  f64* Inputs_rdi, input values fed to the neuron.
                    ;
                    ;   rsi  f64* Weights_rsi, weights corresponding inputs.
                    ;
                    ;   edx  u32 InputCount_edx, input and weight count.
                    ;
                    ;   xmm0  f64 Bias_xmm0, bias value of the neuron.

                    ; Use Bias_xmm0 as the accumulator to save an addition and
                    ; so that the result will be in the return register.

    xorpd xmm1, xmm1
                    ; Zero xmm1 to use it as a pair-wise accumulator.

    mov eax, edx    ; Compute the number of 16-element blocks to multiply-add.
    shr eax, 4      ; nblocks_eax = InputCount_edx >> 4
                    ;
                    ; shr sets ZF flag based on the result in rax.
                    ;
    jz near Tail    ; Skip the block loop if there are no full blocks.
                    ;-----------------------------------------------------------

    and rdx, byte 15
                    ; Mask off all but the low 4 bits of the InputCount_edx.
                    ;
    mov r8, qword 128
                    ; Load 128 into the r8 register for use as an address
                    ; increment. The byte form of the immediate value can't be
                    ; used here because sign-extension would occur in this case.

                    ; xmm0 is holding f64 Bias_xmm0, bias value of the neuron,
                    ; so don't use it till the end.
;////////
ALoop:;// For full 128-byte blocks of data.
;////////
                    ; Fetch 8 pairs of packed f64 values from the Inputs_rdi
                    ; buffer to xmm registers.
    movupd xmm2, [rdi]
    movupd xmm3, [rdi+16]
    movupd xmm4, [rdi+32]
    movupd xmm5, [rdi+48]
    movupd xmm6, [rdi+64]
    movupd xmm7, [rdi+80]
    movupd xmm8, [rdi+96]
    movupd xmm9, [rdi+112]

                    ; Fetch 6 pairs of packed f64 values from the Weights_rsi
                    ; buffer to xmm registers.
    movupd xmm10, [rsi]
    movupd xmm11, [rsi+16]
    movupd xmm12, [rsi+32]
    movupd xmm13, [rsi+48]
    movupd xmm14, [rsi+64]
    movupd xmm15, [rsi+80]

                    ; Multiply 6 pairs of weights with 6 pairs of inputs.
    mulpd xmm2, xmm10
    mulpd xmm3, xmm11
    mulpd xmm4, xmm12
    mulpd xmm5, xmm13
    mulpd xmm6, xmm14
    mulpd xmm7, xmm15

                    ; Fetch 2 pairs of packed f64 values from the Weights_rsi
                    ; buffer to xmm registers.
    movupd xmm10, [rsi+96]
    movupd xmm11, [rsi+112]

                    ; Multiply 2 pairs of weights with 2 pairs of inputs.
    mulpd xmm8, xmm10
    mulpd xmm9, xmm11

                    ; Accumulate 8 pairs of products to xmm1.
    addpd xmm1, xmm2
    addpd xmm1, xmm3
    addpd xmm1, xmm4
    addpd xmm1, xmm5
    addpd xmm1, xmm6
    addpd xmm1, xmm7
    addpd xmm1, xmm8
    addpd xmm1, xmm9

    add rdi, r8     ; Advance the Inputs_rdi address by 128 bytes.
                    ; Inputs_rdi += 128_r8
                    ;
    add rsi, r8     ; Advance the Weights_rsi address by 128 bytes.
                    ; Weights_rsi += 128_r8
                    ;
    dec rax         ; Decrement nblocks_rax.
                    ;
    jnz near ALoop  ; Go add the next block if any.
                    ;-----------------------------------------------------------
;///////
Tail:;// Inputs_rdi and Weights_rsi point to remainder values if any, the tail.
;///////
                    ; xmm1 is the pair-wise accumulator.

                    ; switch( InputCount_edx & 15 )
                    ;
    lea rcx, [rel JumpTable]
                    ; Compute the absolute address of the jump table for the
                    ; switch statement.
                    ;
    add rcx, qword [rcx+rdx*8]
                    ; Lookup and compute the address of the case label derived
                    ; from the low 4 bits of InputCount_edx.
                    ;
    jmp rcx         ; Jump to the computed case label.
    ;---------------------------------------------------------------------------

;//////////
CASE_15:;// 15 input:weights remain to be multiply-accumulated
;//////////
                    ; xmm0 holds bias.
                    ; xmm1 is the pair-wise accumulator.

                    ; Fetch 7 pairs of packed f64 values from the Inputs_rdi
                    ; buffer to xmm registers.
    movupd xmm2, [rdi]
    movupd xmm3, [rdi+16]
    movupd xmm4, [rdi+32]
    movupd xmm5, [rdi+48]
    movupd xmm6, [rdi+64]
    movupd xmm7, [rdi+80]
    movupd xmm8, [rdi+96]

                    ; Fetch 7 pairs of packed f64 values from the Weights_rsi
                    ; buffer to xmm registers.
    movupd xmm9,  [rsi]
    movupd xmm10, [rsi+16]
    movupd xmm11, [rsi+32]
    movupd xmm12, [rsi+48]
    movupd xmm13, [rsi+64]
    movupd xmm14, [rsi+80]
    movupd xmm15, [rsi+96]

                    ; Multiply 7 pairs of weights with 7 pairs of inputs.
    mulpd xmm2, xmm9
    mulpd xmm3, xmm10
    mulpd xmm4, xmm11
    mulpd xmm5, xmm12
    mulpd xmm6, xmm13
    mulpd xmm7, xmm14
    mulpd xmm8, xmm15

                    ; Accumulate 7 pairs of products to xmm1.
    addpd xmm1, xmm2
    addpd xmm1, xmm3
    addpd xmm1, xmm4
    addpd xmm1, xmm5
    addpd xmm1, xmm6
    addpd xmm1, xmm7
    addpd xmm1, xmm8

                    ; 1 input remains to be multiplied by 1 weight.

    movsd xmm2, [rdi+112]
                    ; Get the last f64 input to xmm2.

    mulsd xmm2, [rsi+112]
                    ; Multiply last 8-byte f64 weight to xmm2.

    addsd xmm0, xmm2
                    ; Add xmm2 to the bias in xmm0

    addsd xmm0, xmm1
                    ; Add the low part of xmm1 to xmm0

    shufpd xmm1, xmm1, 01b
                    ; Exchange the quad words in xmm1.
    addsd xmm0, xmm1
                    ; Add what was the high part of xmm1 to the low part of xmm0.

    ret             ; Return xmm0 with the weighted sum.
    ;---------------------------------------------------------------------------

;//////////
CASE_14:;// 14 input:weights remain to be multiply-accumulated
;//////////
                    ; xmm0 holds bias.
                    ; xmm1 is the pair-wise accumulator.

                    ; Fetch 7 pairs of packed f64 values from the Inputs_rdi
                    ; buffer to xmm registers.
    movupd xmm2, [rdi]
    movupd xmm3, [rdi+16]
    movupd xmm4, [rdi+32]
    movupd xmm5, [rdi+48]
    movupd xmm6, [rdi+64]
    movupd xmm7, [rdi+80]
    movupd xmm8, [rdi+96]

                    ; Fetch 7 pairs of packed f64 values from the Weights_rsi
                    ; buffer to xmm registers.
    movupd xmm9,  [rsi]
    movupd xmm10, [rsi+16]
    movupd xmm11, [rsi+32]
    movupd xmm12, [rsi+48]
    movupd xmm13, [rsi+64]
    movupd xmm14, [rsi+80]
    movupd xmm15, [rsi+96]

                    ; Multiply 7 pairs of weights with 7 pairs of inputs.
    mulpd xmm2, xmm9
    mulpd xmm3, xmm10
    mulpd xmm4, xmm11
    mulpd xmm5, xmm12
    mulpd xmm6, xmm13
    mulpd xmm7, xmm14
    mulpd xmm8, xmm15

                    ; Accumulate 7 pairs of products to xmm1.
    addpd xmm1, xmm2
    addpd xmm1, xmm3
    addpd xmm1, xmm4
    addpd xmm1, xmm5
    addpd xmm1, xmm6
    addpd xmm1, xmm7
    addpd xmm1, xmm8

    addsd xmm0, xmm1
                    ; Add the low part of xmm1 to xmm0

    shufpd xmm1, xmm1, 01b
                    ; Exchange the quad words in xmm1.
    addsd xmm0, xmm1
                    ; Add what was the high part of xmm1 to the low part of xmm0.

    ret             ; Return xmm0 with the weighted sum.
    ;---------------------------------------------------------------------------

;//////////
CASE_13:;// 13 input:weights remain to be multiply-accumulated
;//////////
                    ; xmm0 holds bias.
                    ; xmm1 is the pair-wise accumulator.

                    ; Fetch 6 pairs of packed f64 values from the Inputs_rdi
                    ; buffer to xmm registers.
    movupd xmm2, [rdi]
    movupd xmm3, [rdi+16]
    movupd xmm4, [rdi+32]
    movupd xmm5, [rdi+48]
    movupd xmm6, [rdi+64]
    movupd xmm7, [rdi+80]

                    ; Fetch 6 pairs of packed f64 values from the Weights_rsi
                    ; buffer to xmm registers.
    movupd xmm9,  [rsi]
    movupd xmm10, [rsi+16]
    movupd xmm11, [rsi+32]
    movupd xmm12, [rsi+48]
    movupd xmm13, [rsi+64]
    movupd xmm14, [rsi+80]

                    ; Multiply 6 pairs of weights with 6 pairs of inputs.
    mulpd xmm2, xmm9
    mulpd xmm3, xmm10
    mulpd xmm4, xmm11
    mulpd xmm5, xmm12
    mulpd xmm6, xmm13
    mulpd xmm7, xmm14

                    ; Accumulate 6 pairs of products to xmm1.
    addpd xmm1, xmm2
    addpd xmm1, xmm3
    addpd xmm1, xmm4
    addpd xmm1, xmm5
    addpd xmm1, xmm6
    addpd xmm1, xmm7

                    ; 1 input remains to be multiplied by 1 weight.

    movsd xmm2, [rdi+96]
                    ; Get the last f64 input to xmm2.

    mulsd xmm2, [rsi+96]
                    ; Multiply last 8-byte f64 weight to xmm2.

    addsd xmm0, xmm2
                    ; Add xmm2 to the bias in xmm0

    addsd xmm0, xmm1
                    ; Add the low part of xmm1 to xmm0

    shufpd xmm1, xmm1, 01b
                    ; Exchange the quad words in xmm1.
    addsd xmm0, xmm1
                    ; Add what was the high part of xmm1 to the low part of xmm0.

    ret             ; Return xmm0 with the weighted sum.
    ;---------------------------------------------------------------------------

;//////////
CASE_12:;// 12 input:weights remain to be multiply-accumulated
;//////////
                    ; xmm0 holds bias.
                    ; xmm1 is the pair-wise accumulator.

                    ; Fetch 6 pairs of packed f64 values from the Inputs_rdi
                    ; buffer to xmm registers.
    movupd xmm2, [rdi]
    movupd xmm3, [rdi+16]
    movupd xmm4, [rdi+32]
    movupd xmm5, [rdi+48]
    movupd xmm6, [rdi+64]
    movupd xmm7, [rdi+80]

                    ; Fetch 6 pairs of packed f64 values from the Weights_rsi
                    ; buffer to xmm registers.
    movupd xmm9,  [rsi]
    movupd xmm10, [rsi+16]
    movupd xmm11, [rsi+32]
    movupd xmm12, [rsi+48]
    movupd xmm13, [rsi+64]
    movupd xmm14, [rsi+80]

                    ; Multiply 6 pairs of weights with 6 pairs of inputs.
    mulpd xmm2, xmm9
    mulpd xmm3, xmm10
    mulpd xmm4, xmm11
    mulpd xmm5, xmm12
    mulpd xmm6, xmm13
    mulpd xmm7, xmm14

                    ; Accumulate 6 pairs of products to xmm1.
    addpd xmm1, xmm2
    addpd xmm1, xmm3
    addpd xmm1, xmm4
    addpd xmm1, xmm5
    addpd xmm1, xmm6
    addpd xmm1, xmm7

    addsd xmm0, xmm1
                    ; Add the low part of xmm1 to xmm0

    shufpd xmm1, xmm1, 01b
                    ; Exchange the quad words in xmm1.
    addsd xmm0, xmm1
                    ; Add what was the high part of xmm1 to the low part of xmm0.

    ret             ; Return xmm0 with the weighted sum.
    ;---------------------------------------------------------------------------

;//////////
CASE_11:;// 11 input:weights remain to be multiply-accumulated
;//////////
                    ; xmm0 holds bias.
                    ; xmm1 is the pair-wise accumulator.

                    ; Fetch 5 pairs of packed f64 values from the Inputs_rdi
                    ; buffer to xmm registers.
    movupd xmm2, [rdi]
    movupd xmm3, [rdi+16]
    movupd xmm4, [rdi+32]
    movupd xmm5, [rdi+48]
    movupd xmm6, [rdi+64]

                    ; Fetch 5 pairs of packed f64 values from the Weights_rsi
                    ; buffer to xmm registers.
    movupd xmm9,  [rsi]
    movupd xmm10, [rsi+16]
    movupd xmm11, [rsi+32]
    movupd xmm12, [rsi+48]
    movupd xmm13, [rsi+64]

                    ; Multiply 5 pairs of weights with 5 pairs of inputs.
    mulpd xmm2, xmm9
    mulpd xmm3, xmm10
    mulpd xmm4, xmm11
    mulpd xmm5, xmm12
    mulpd xmm6, xmm13

                    ; Accumulate 5 pairs of products to xmm1.
    addpd xmm1, xmm2
    addpd xmm1, xmm3
    addpd xmm1, xmm4
    addpd xmm1, xmm5
    addpd xmm1, xmm6

                    ; 1 input remains to be multiplied by 1 weight.

    movsd xmm2, [rdi+80]
                    ; Get the last f64 input to xmm2.

    mulsd xmm2, [rsi+80]
                    ; Multiply last 8-byte f64 weight to xmm2.

    addsd xmm0, xmm2
                    ; Add xmm2 to the bias in xmm0

    addsd xmm0, xmm1
                    ; Add the low part of xmm1 to xmm0

    shufpd xmm1, xmm1, 01b
                    ; Exchange the quad words in xmm1.
    addsd xmm0, xmm1
                    ; Add what was the high part of xmm1 to the low part of xmm0.

    ret             ; Return xmm0 with the weighted sum.
    ;---------------------------------------------------------------------------

;//////////
CASE_10:;// 10 input:weights remain to be multiply-accumulated
;//////////
                    ; xmm0 holds bias.
                    ; xmm1 is the pair-wise accumulator.

                    ; Fetch 5 pairs of packed f64 values from the Inputs_rdi
                    ; buffer to xmm registers.
    movupd xmm2, [rdi]
    movupd xmm3, [rdi+16]
    movupd xmm4, [rdi+32]
    movupd xmm5, [rdi+48]
    movupd xmm6, [rdi+64]

                    ; Fetch 5 pairs of packed f64 values from the Weights_rsi
                    ; buffer to xmm registers.
    movupd xmm9,  [rsi]
    movupd xmm10, [rsi+16]
    movupd xmm11, [rsi+32]
    movupd xmm12, [rsi+48]
    movupd xmm13, [rsi+64]

                    ; Multiply 5 pairs of weights with 5 pairs of inputs.
    mulpd xmm2, xmm9
    mulpd xmm3, xmm10
    mulpd xmm4, xmm11
    mulpd xmm5, xmm12
    mulpd xmm6, xmm13

                    ; Accumulate 5 pairs of products to xmm1.
    addpd xmm1, xmm2
    addpd xmm1, xmm3
    addpd xmm1, xmm4
    addpd xmm1, xmm5
    addpd xmm1, xmm6

    addsd xmm0, xmm1
                    ; Add the low part of xmm1 to xmm0

    shufpd xmm1, xmm1, 01b
                    ; Exchange the quad words in xmm1.
    addsd xmm0, xmm1
                    ; Add what was the high part of xmm1 to the low part of xmm0.

    ret             ; Return xmm0 with the weighted sum.
    ;---------------------------------------------------------------------------

;/////////
CASE_9:;// 9 input:weights remain to be multiply-accumulated
;/////////
                    ; xmm0 holds bias.
                    ; xmm1 is the pair-wise accumulator.

                    ; Fetch 4 pairs of packed f64 values from the Inputs_rdi
                    ; buffer to xmm registers.
    movupd xmm2, [rdi]
    movupd xmm3, [rdi+16]
    movupd xmm4, [rdi+32]
    movupd xmm5, [rdi+48]

                    ; Fetch 4 pairs of packed f64 values from the Weights_rsi
                    ; buffer to xmm registers.
    movupd xmm9,  [rsi]
    movupd xmm10, [rsi+16]
    movupd xmm11, [rsi+32]
    movupd xmm12, [rsi+48]

                    ; Multiply 4 pairs of weights with 4 pairs of inputs.
    mulpd xmm2, xmm9
    mulpd xmm3, xmm10
    mulpd xmm4, xmm11
    mulpd xmm5, xmm12

                    ; Accumulate 4 pairs of products to xmm1.
    addpd xmm1, xmm2
    addpd xmm1, xmm3
    addpd xmm1, xmm4
    addpd xmm1, xmm5

                    ; 1 input remains to be multiplied by 1 weight.

    movsd xmm2, [rdi+64]
                    ; Get the last f64 input to xmm2.

    mulsd xmm2, [rsi+64]
                    ; Multiply last 8-byte f64 weight to xmm2.

    addsd xmm0, xmm2
                    ; Add xmm2 to the bias in xmm0

    addsd xmm0, xmm1
                    ; Add the low part of xmm1 to xmm0

    shufpd xmm1, xmm1, 01b
                    ; Exchange the quad words in xmm1.
    addsd xmm0, xmm1
                    ; Add what was the high part of xmm1 to the low part of xmm0.

    ret             ; Return xmm0 with the weighted sum.
    ;---------------------------------------------------------------------------

;/////////
CASE_8:;// 8 input:weights remain to be multiply-accumulated
;/////////
                    ; xmm0 holds bias.
                    ; xmm1 is the pair-wise accumulator.

                    ; Fetch 4 pairs of packed f64 values from the Inputs_rdi
                    ; buffer to xmm registers.
    movupd xmm2, [rdi]
    movupd xmm3, [rdi+16]
    movupd xmm4, [rdi+32]
    movupd xmm5, [rdi+48]

                    ; Fetch 4 pairs of packed f64 values from the Weights_rsi
                    ; buffer to xmm registers.
    movupd xmm9,  [rsi]
    movupd xmm10, [rsi+16]
    movupd xmm11, [rsi+32]
    movupd xmm12, [rsi+48]

                    ; Multiply 4 pairs of weights with 4 pairs of inputs.
    mulpd xmm2, xmm9
    mulpd xmm3, xmm10
    mulpd xmm4, xmm11
    mulpd xmm5, xmm12

                    ; Accumulate 4 pairs of products to xmm1.
    addpd xmm1, xmm2
    addpd xmm1, xmm3
    addpd xmm1, xmm4
    addpd xmm1, xmm5

    addsd xmm0, xmm1
                    ; Add the low part of xmm1 to xmm0

    shufpd xmm1, xmm1, 01b
                    ; Exchange the quad words in xmm1.
    addsd xmm0, xmm1
                    ; Add what was the high part of xmm1 to the low part of xmm0.

    ret             ; Return xmm0 with the weighted sum.
    ;---------------------------------------------------------------------------

;/////////
CASE_7:;// 7 input:weights remain to be multiply-accumulated
;/////////
                    ; xmm0 holds bias.
                    ; xmm1 is the pair-wise accumulator.

                    ; Fetch 3 pairs of packed f64 values from the Inputs_rdi
                    ; buffer to xmm registers.
    movupd xmm2, [rdi]
    movupd xmm3, [rdi+16]
    movupd xmm4, [rdi+32]

                    ; Fetch 3 pairs of packed f64 values from the Weights_rsi
                    ; buffer to xmm registers.
    movupd xmm9,  [rsi]
    movupd xmm10, [rsi+16]
    movupd xmm11, [rsi+32]

                    ; Multiply 3 pairs of weights with 3 pairs of inputs.
    mulpd xmm2, xmm9
    mulpd xmm3, xmm10
    mulpd xmm4, xmm11

                    ; Accumulate 3 pairs of products to xmm1.
    addpd xmm1, xmm2
    addpd xmm1, xmm3
    addpd xmm1, xmm4

                    ; 1 input remains to be multiplied by 1 weight.

    movsd xmm2, [rdi+48]
                    ; Get the last f64 input to xmm2.

    mulsd xmm2, [rsi+48]
                    ; Multiply last 8-byte f64 weight to xmm2.

    addsd xmm0, xmm2
                    ; Add xmm2 to the bias in xmm0

    addsd xmm0, xmm1
                    ; Add the low part of xmm1 to xmm0

    shufpd xmm1, xmm1, 01b
                    ; Exchange the quad words in xmm1.
    addsd xmm0, xmm1
                    ; Add what was the high part of xmm1 to the low part of xmm0.

    ret             ; Return xmm0 with the weighted sum.
    ;---------------------------------------------------------------------------

;/////////
CASE_6:;// 6 input:weights remain to be multiply-accumulated
;/////////
                    ; xmm0 holds bias.
                    ; xmm1 is the pair-wise accumulator.

                    ; Fetch 3 pairs of packed f64 values from the Inputs_rdi
                    ; buffer to xmm registers.
    movupd xmm2, [rdi]
    movupd xmm3, [rdi+16]
    movupd xmm4, [rdi+32]

                    ; Fetch 3 pairs of packed f64 values from the Weights_rsi
                    ; buffer to xmm registers.
    movupd xmm9, [rsi]
    movupd xmm10, [rsi+16]
    movupd xmm11, [rsi+32]

                    ; Multiply 3 pairs of weights with 3 pairs of inputs.
    mulpd xmm2, xmm9
    mulpd xmm3, xmm10
    mulpd xmm4, xmm11

                    ; Accumulate 3 pairs of products to xmm1.
    addpd xmm1, xmm2
    addpd xmm1, xmm3
    addpd xmm1, xmm4

    addsd xmm0, xmm1
                    ; Add the low part of xmm1 to xmm0

    shufpd xmm1, xmm1, 01b
                    ; Exchange the quad words in xmm1.
    addsd xmm0, xmm1
                    ; Add what was the high part of xmm1 to the low part of xmm0.

    ret             ; Return xmm0 with the weighted sum.
    ;---------------------------------------------------------------------------

;/////////
CASE_5:;// 5 input:weights remain to be multiply-accumulated
;/////////
                    ; xmm0 holds bias.
                    ; xmm1 is the pair-wise accumulator.

                    ; Fetch 2 pairs of packed f64 values from the Inputs_rdi
                    ; buffer to xmm registers.
    movupd xmm2, [rdi]
    movupd xmm3, [rdi+16]

                    ; Fetch 2 pairs of packed f64 values from the Weights_rsi
                    ; buffer to xmm registers.
    movupd xmm9,  [rsi]
    movupd xmm10, [rsi+16]

                    ; Multiply 2 pairs of weights with 2 pairs of inputs.
    mulpd xmm2, xmm9
    mulpd xmm3, xmm10

                    ; Accumulate 2 pairs of products to xmm1.
    addpd xmm1, xmm2
    addpd xmm1, xmm3

                    ; 1 input remains to be multiplied by 1 weight.

    movsd xmm2, [rdi+32]
                    ; Get the last f64 input to xmm2.

    mulsd xmm2, [rsi+32]
                    ; Multiply last 8-byte f64 weight to xmm2.

    addsd xmm0, xmm2
                    ; Add xmm2 to the bias in xmm0

    addsd xmm0, xmm1
                    ; Add the low part of xmm1 to xmm0

    shufpd xmm1, xmm1, 01b
                    ; Exchange the quad words in xmm1.
    addsd xmm0, xmm1
                    ; Add what was the high part of xmm1 to the low part of xmm0.

    ret             ; Return xmm0 with the weighted sum.
    ;---------------------------------------------------------------------------

;/////////
CASE_4:;// 4 input:weights remain to be multiply-accumulated
;/////////
                    ; xmm0 holds bias.
                    ; xmm1 is the pair-wise accumulator.

                    ; Fetch 2 pairs of packed f64 values from the Inputs_rdi
                    ; buffer to xmm registers.
    movupd xmm2, [rdi]
    movupd xmm3, [rdi+16]

                    ; Fetch 2 pairs of packed f64 values from the Weights_rsi
                    ; buffer to xmm registers.
    movupd xmm9,  [rsi]
    movupd xmm10, [rsi+16]

                    ; Multiply 2 pairs of weights with 2 pairs of inputs.
    mulpd xmm2, xmm9
    mulpd xmm3, xmm10

                    ; Accumulate 2 pairs of products to xmm1.
    addpd xmm1, xmm2
    addpd xmm1, xmm3

    addsd xmm0, xmm1
                    ; Add the low part of xmm1 to xmm0

    shufpd xmm1, xmm1, 01b
                    ; Exchange the quad words in xmm1.
    addsd xmm0, xmm1
                    ; Add what was the high part of xmm1 to the low part of xmm0.

    ret             ; Return xmm0 with the weighted sum.
    ;---------------------------------------------------------------------------

;/////////
CASE_3:;// 3 input:weights remain to be multiply-accumulated
;/////////
                    ; xmm0 holds bias.
                    ; xmm1 is the pair-wise accumulator.

    movupd xmm2, [rdi]
                    ; Fetch 1 pair of packed f64 values from the Inputs_rdi
                    ; buffer to xmm registers.

    movupd xmm9, [rsi]
                    ; Fetch 1 pair of packed f64 values from the Weights_rsi
                    ; buffer to xmm registers.

    mulpd xmm2, xmm9
                    ; Multiply 1 pair of weights with 1 pair of inputs.

    addpd xmm1, xmm2
                    ; Accumulate 1 pair of products to xmm1.

                    ; 1 input remains to be multiplied by 1 weight.

    movsd xmm2, [rdi+16]
                    ; Get the last f64 input to xmm2.

    mulsd xmm2, [rsi+16]
                    ; Multiply last 8-byte f64 weight to xmm2.

    addsd xmm0, xmm2
                    ; Add xmm2 to the bias in xmm0

    addsd xmm0, xmm1
                    ; Add the low part of xmm1 to xmm0

    shufpd xmm1, xmm1, 01b
                    ; Exchange the quad words in xmm1.
    addsd xmm0, xmm1
                    ; Add what was the high part of xmm1 to the low part of xmm0.

    ret             ; Return xmm0 with the weighted sum.
    ;---------------------------------------------------------------------------

;/////////
CASE_2:;// 2 input:weights remain to be multiply-accumulated
;/////////
                    ; xmm0 holds bias.
                    ; xmm1 is the pair-wise accumulator.

    movupd xmm2, [rdi]
                    ; Fetch 1 pair of packed f64 values from the Inputs_rdi
                    ; buffer to xmm registers.

    movupd xmm9, [rsi]
                    ; Fetch 1 pair of packed f64 values from the Weights_rsi
                    ; buffer to xmm registers.

    mulpd xmm2, xmm9
                    ; Multiply 1 pair of weights with 1 pair of inputs.

    addpd xmm1, xmm2
                    ; Accumulate 1 pair of products to xmm1.

    addsd xmm0, xmm1
                    ; Add the low part of xmm1 to xmm0

    shufpd xmm1, xmm1, 01b
                    ; Exchange the quad words in xmm1.
    addsd xmm0, xmm1
                    ; Add what was the high part of xmm1 to the low part of xmm0.

    ret             ; Return xmm0 with the weighted sum.
    ;---------------------------------------------------------------------------

;/////////
CASE_1:;// 1 input:weight remain to be multiply-accumulated
;/////////
                    ; xmm0 holds bias.
                    ; xmm1 is the pair-wise accumulator.

                    ; 1 input remains to be multiplied by 1 weight.

    movsd xmm2, [rdi]
                    ; Get the last f64 input to xmm2.

    mulsd xmm2, [rsi]
                    ; Multiply last 8-byte f64 weight to xmm2.

    addsd xmm0, xmm2
                    ; Add xmm2 to the bias in xmm0

    addsd xmm0, xmm1
                    ; Add the low part of xmm1 to xmm0

    shufpd xmm1, xmm1, 01b
                    ; Exchange the quad words in xmm1.
    addsd xmm0, xmm1
                    ; Add what was the high part of xmm1 to the low part of xmm0.

    ret             ; Return xmm0 with the weighted sum.
    ;---------------------------------------------------------------------------

;/////////
CASE_0:;// No input:weights remain to be multiply-accumulated
;/////////
                    ; xmm0 holds bias.
                    ; xmm1 is the pair-wise accumulator.

    addsd xmm0, xmm1
                    ; Add the low part of xmm1 to xmm0

    shufpd xmm1, xmm1, 01b
                    ; Exchange the quad words in xmm1.
    addsd xmm0, xmm1
                    ; Add what was the high part of xmm1 to the low part of xmm0.

    ret             ; Return xmm0 with the weighted sum.
    ;---------------------------------------------------------------------------

align 16 ; Align to a 16-byte boundary.

;////////////
JumpTable:;// Jump table for the switch statement.
;////////////

    dq  CASE_0  - JumpTable
    dq  CASE_1  - JumpTable
    dq  CASE_2  - JumpTable
    dq  CASE_3  - JumpTable
    dq  CASE_4  - JumpTable
    dq  CASE_5  - JumpTable
    dq  CASE_6  - JumpTable
    dq  CASE_7  - JumpTable
    dq  CASE_8  - JumpTable
    dq  CASE_9  - JumpTable
    dq  CASE_10 - JumpTable
    dq  CASE_11 - JumpTable
    dq  CASE_12 - JumpTable
    dq  CASE_13 - JumpTable
    dq  CASE_14 - JumpTable
    dq  CASE_15 - JumpTable

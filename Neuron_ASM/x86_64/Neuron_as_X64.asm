;-------------------------------------------------------------------------------
; Neuron_as_X64.asm
;-------------------------------------------------------------------------------
;
; PURPOSE: To compute the output of neuron from its inputs.
;
; DESCRIPTION: Produces a value ready to be fed into an activation function.
; Multiply-adds inputs and weights and adds with the neuron bias.
;
; This is an assembler version of Neuron().
;
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
;    19Oct2023 From Neuron() and DotProductOfVectorsFromLINPACK_as_X64.asm.
;    19Oct2023 Passed Neuron_Test() OK.
;    21Dec2023 Use faster Neuron_Vectorized_as() instead of this routine.
;-------------------------------------------------------------------------------

%include "platform_nasm.inc"

global Neuron_as

;-------------------------------------------------------------------------------
;   // OUT: Neuron output value in xmm0 ready for an activation function.
; f64
; Neuron_as(
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

;////////////
Neuron_as:;//
;////////////
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

;////////
ALoop:;// For each pair of (input,weight) elements.
;////////

    movsd xmm1, qword [rdi]
                    ; xmm1 = *Inputs_rdi
                    ;
    mulsd xmm1, [rsi]
                    ; xmm1 *= *Weights_rsi
                    ;
    addsd xmm0, xmm1
                    ; Bias_xmm0 += input * weight
                    ;
                    ; Advance Inputs and Weights to the next element pair.
    add rdi, 8      ; Inputs_rdi += 8
                    ;
    add rsi, 8      ; Weights_rsi += 8
                    ;
    dec edx         ;
    jnz ALoop       ; Decrement InputCount_edx and loop back to ALoop if non-0.
                    ;-----------------------------------------------------------

    ret             ; Return xmm0 with the sum of products and the bias.
    ;---------------------------------------------------------------------------

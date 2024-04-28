;-------------------------------------------------------------------------------
; Neuron_Maskable_as_X64.asm
;-------------------------------------------------------------------------------
;
; PURPOSE: To compute the output of a neuron from its unmasked inputs and bias.
;
; DESCRIPTION: Produces a value ready to be fed into an activation function.
; Multiply-adds inputs and weights and adds with the neuron bias.
;
; This is an assembler version of Neuron_Maskable().
;
; REQUIRES:
;
;   A CPU that supports x86_64 instructions.
;
;   External code will handle overflow to infinity or NaN's if any.
;
; This C code is functionally equivalent to the NASM assembly code below:
;
;      // OUT: Weighted sum of the neuron ready for an activation function.
;    f64
;    Neuron_Maskable(
;        f64* Inputs,
;                // Input values fed to the neuron.
;                //
;        f64* Weights,
;                // Weights corresponding to each input.
;                //
;        u8* Masks,
;                // Input mask set to 1 if the input is present, or 0 if not.
;                //
;        u32 InputCount,
;                // Number of inputs, masks, and weights.
;                //
;        f64 Bias )
;                // Bias value of the neuron.
;    {
;        // The bias is a kind of input that can be thought of as having a
;        // constant weight of 1. The bias term ensures that even when all 
;        // inputs to a neuron are zero, the neuron can still produce an output.
;    
;        // For unmasked inputs, multiply each with a weight and sum the result.
;        while( InputCount-- )
;        {
;            // InputCount is now 1 less than in the while(). On the first pass
;            // it points to the last element of the vectors.
;    
;            // Index using InputCount from the last input to the first.
;    
;            // Use Bias as the accumulator to save an addition.
;    
;            // Accept the current input if it is unmasked. Non-0 means unmasked.
;            Bias +=
;                Masks[InputCount] ? 
;                    Inputs[InputCount] * Weights[InputCount] : 0.0 ;
;                                   // Use a conditional move to avoid a branch.
;        }
;    
;        // Return the sum of unmasked products and the bias.
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
;    14Dec2023 From Neuron_Maskable() and Neuron_as_X64.asm.
;    14Dec2023 Passed Neuron_Maskable_Test() OK.
;-------------------------------------------------------------------------------

%include "platform_nasm.inc"

global Neuron_Maskable_as

;-------------------------------------------------------------------------------
;   // OUT: Neuron output value in xmm0 ready for an activation function.
; f64
; Neuron_Maskable_as(
;     f64* Inputs,
;                // Input values fed to the neuron.
;                //
;     f64* Weights,
;                // Weights corresponding to each input.
;                //
;     u8* Masks,
;                // Input mask set to 1 if the input is present, or 0 if not.
;                //
;     u32 InputCount,
;                // Number of inputs, masks, and weights.
;                //
;     f64 Bias ) // Bias value of the neuron.

;/////////////////////
Neuron_Maskable_as:;//
;/////////////////////

                    ; Inputs:
                    ;
                    ;   rdi  f64* Inputs_rdi, input values fed to the neuron.
                    ;
                    ;   rsi  f64* Weights_rsi, weights corresponding inputs.
                    ;
                    ;   rdx  u8* Masks_rdx, mask set to 1 if an input is present,
                    ;                                               or 0 if not.
                    ;
                    ;   ecx  u32 InputCount_ecx, input and weight count.
                    ;
                    ;   xmm0  f64 Bias_xmm0, bias value of the neuron.

                    ; Use Bias_xmm0 as the accumulator to save an addition and
                    ; so that the result will be in the return register.

                    ; Zero rax to provide 0 if input is masked.
    xor rax, rax    ; Zero_rax = 0.0

;////////
ALoop:;// For each triplet of (input,weight,mask) elements, from last to first.
;////////

    sub rcx, byte 1 ; Decrement InputCount_ecx to enumerate the buffer values
                    ; from last to first. Must use sub because dec doesn't
                    ; update the carry flag.
                    ;
    jb near Done    ; Exit loop if no more elements remain to be processed.
                    ;-----------------------------------------------------------

                    ; rcx is the index of current (input,weight,mask) elements.

    mov r8, qword [rdi+rcx*8]
                    ; Get the current input candidate value to r8.
                    ; Input_r8 = Inputs_rdi[InputCount_rcx]
                    ;
    cmp al, byte [rdx+rcx]
                    ; Compare: 0 - Masks_rdx[InputCount_ecx]
                    ; Test if input mask == 0 setting the status flags.

                    ; Use a conditional move to avoid a branch.

    cmove r8, rax   ; Input_r8 = Zero_rax if mask == 0, or
                    ; Input_r8 = Input_r8 if mask != 0.

    movq xmm1, r8   ; Load xmm1 with input or 0 depending on the mask.

    mulsd xmm1, [rsi+rcx*8]
                    ; xmm1 *= Weights_rsi[InputCount_rcx]
                    ;
    addsd xmm0, xmm1
                    ; Bias_xmm0 += input * weight
                    ;
    jmp ALoop       ; Loop back to sum the next input.
                    ;-----------------------------------------------------------
;///////
Done:;// Jumps here after all inputs have been processed.
;///////

    ret             ; Return xmm0 with the sum of products and the bias.
    ;---------------------------------------------------------------------------

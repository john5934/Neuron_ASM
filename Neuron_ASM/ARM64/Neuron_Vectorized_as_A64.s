/*------------------------------------------------------------------------------
| Neuron_Vectorized_as_A64.s          
|-------------------------------------------------------------------------------
|
| PURPOSE: To compute the weighted sum of a neuron from its inputs and bias.
|
| DESCRIPTION: Produces a weighted sum ready to be fed into an activation
|                                                                      function.
| Multiply-adds inputs and weights and adds with the neuron bias.
|
| This is a vectorized assembler version of Neuron(). Where there are more than
| two inputs, this routine multiplies inputs and weights two at a time instead
| of one by one.
|
| This routine is a drop-in replacement for Neuron() since the inputs and output
| are identical.
|
| This version uses fused multiply-add instructions, so results may differ
| slightly from the C routine Neuron().
|
| REQUIRES:
|
|   InputCount > 0
|
|   A CPU that supports A64 instructions.
|
|   External code will handle overflow to infinity or NaN's if any.
|
| This C code is functionally equivalent to the assembly code below:
|
|      // OUT: Neuron output value ready for an activation function.
|    f64
|    Neuron(
|        f64* Inputs,
|                // Input values fed to the neuron.
|                //
|        f64* Weights,
|                // Weights corresponding to each input.
|                //
|        u32 InputCount,
|                // Number of inputs and corresponding weights.
|                //
|        f64 Bias )
|                // Bias value of the neuron.
|    {
|        // For each input, multiply it with a weight and accumulate the result.
|        do
|        {
|            // Use Bias as the accumulator to save an addition.
|            Bias += *Inputs++ * *Weights++;
|
|            // Account for having processed one input.
|            InputCount--;
|
|        } while( InputCount );
|
|        // Return the sum of products and the bias.
|        return( Bias );
|    }
|
| For Linux A64, registers X19-X30, V8-V15, and LR belong to the caller and must
| be preserved by this routine if they are to be used. The remaining registers
| are available for use by this routine, but must be saved on the stack if they
| are to be preserved across function calls. From "Programming with 64-bit ARM
| Assembly Language.pdf" pages 142 and 274. d8-d15 belong to the caller since
| they are part of V8-V15, but d16-d31 are free for use without being saved.
|
| REGISTER USAGE:
|
|   Defined inline below.
|
| HISTORY:
|    24Dec2023 From Neuron(), Neuron_Vectorized_fma_as_X64.asm,
|              RowTimesMatrix_ColumnMajor_Even_as_A64.s, and Mean_f64s_as_A64.s.
|    24Dec2023 Passed Neuron_Vectorized_as_Test() OK on Pi4GB.
------------------------------------------------------------------------------*/

.global Neuron_Vectorized_as

/*------------------------------------------------------------------------------
|   // OUT: Neuron output value in d0 ready for an activation function.
| f64
| Neuron_Vectorized_as(
|     f64* Inputs,
|              // Input values fed to the neuron.
|              //
|     f64* Weights,
|              // Weights corresponding to each input.
|              //
|     u32 InputCount,
|              // Number of inputs and corresponding weights.
|              //
|     f64 Bias )
|              // Bias value of the neuron.
*/

///////////////////////
Neuron_Vectorized_as://
///////////////////////

                    // Inputs:
                    //
                    //   x0  f64* Inputs_x0, input values fed to the neuron.
                    //
                    //   x1  f64* Weights_x1, weights corresponding inputs.
                    //
                    //   w2  u32 InputCount_w2, input and weight count.
                    //
                    //   d0  f64 Bias_d0, bias value of the neuron.

                    // Use Bias_d0 as the accumulator to save an addition and
                    // so that the result will be in the return register.
                    //
                    // d0 is the lower 64 bits of v0/q0, so use other registers
                    // for incoming data.
 
    eor v1.16b, v1.16b, v1.16b
                    // Zero v1/q1 to use it as a pair-wise accumulator. This
                    // stores 0.0 into each f64 element.

    lsr x3, x2, #4  // Compute the number of 16-element blocks to process.
                    // nblocks_x3 = InputCount_x2 >> 4
                    //
    cbz w3, PartialBlock
                    // Skip the block loop if there are no full blocks.
                    //----------------------------------------------------------

    mov x4, #128    // Load 128 into the x4 register for use as an address
                    // increment for blocks of 16 f64 values, 16 x 8 = 128 bytes.

                    // d0 is holding f64 Bias_d0, bias value of the neuron, so
                    // don't use it till the end.
                    //
                    // d0 is the lower 64 bits of v0/q0, so use other registers
                    // for incoming data.
                    //
                    // v1 is the pair-wise accumulator.
                    //
                    // Avoid using caller owned V8-V15.
////////
ALoop:// For full 128-byte blocks of data.
////////
                    // Fetch 8 pairs of packed f64 values from the Inputs_x0
                    // buffer to v registers, 32 bytes at a time. Each v
                    // register contains two f64 elements.
    ldp q2, q3, [x0]
    ldp q4, q5, [x0, #32]
    ldp q6, q7, [x0, #64]
                    // Avoid using caller owned V8-V15.
    ldp q16, q17, [x0, #96]

                    // Fetch 8 pairs of packed f64 values from the Weights_x1
                    // buffer to v registers, 32 bytes at a time.
    ldp q18, q19, [x1]
    ldp q20, q21, [x1, #32]
    ldp q22, q23, [x1, #64]
    ldp q24, q25, [x1, #96]

                    // Multiply-add 8 pairs of weights with 8 pairs of inputs to
                    // pair-wise accumulator v1.
    fmla v1.2d, v2.2d, v18.2d
    fmla v1.2d, v3.2d, v19.2d
    fmla v1.2d, v4.2d, v20.2d
    fmla v1.2d, v5.2d, v21.2d
    fmla v1.2d, v6.2d, v22.2d
    fmla v1.2d, v7.2d, v23.2d
    fmla v1.2d, v16.2d, v24.2d
    fmla v1.2d, v17.2d, v25.2d

    add x0, x0, x4  // Advance the Inputs_x0 address by 128 bytes.
                    // Inputs_x0 += 128_x4
                    //
    add x1, x1, x4  // Advance the Weights_x1 address by 128 bytes.
                    // Weights_x1 += 128_x4
                    //
    subs x3, x3, #1 // Decrement nblocks_x3, setting the flags.
                    //
    b.ne ALoop      // Go process the next block if there is one.
                    //----------------------------------------------------------
///////
Tail:// Inputs_x0 and Weights_x1 point to remainder values if any, the tail.
///////

    and x5, x2, #15
                    // Mask off all but the low 4 bits of the Count.
                    // CaseNumber_x5 = InputCount_w2 & 15
                    //
    adr x6, JumpTable
                    // Load x6 with the absolute address of the jump table.
                    // JumpTable_x6 = &JumpTable
                    //
    ldr x7, [x6, x5, lsl #3]
                    // Lookup the displacement of the case label derived from
                    // the low 4 bits of the Count.
                    // x7 = JumpTable_x6[CaseNumber_x5]
                    //
    add x8, x6, x7  // Complete the address of the case label by adding the
                    // displacement to the jump table base address.
                    // x8 = &JumpTable + JumpTable_x6[CaseNumber_x5]
                    //
    br x8           // Jump to the computed case label address in x8.
    //--------------------------------------------------------------------------

///////////////
PartialBlock:// Jumps here if InputCount_w2 is less than 16.
///////////////
                    // CaseNumber_x2 = InputCount_w2

    adr x6, JumpTable
                    // Load x6 with the absolute address of the jump table.
                    // JumpTable_x6 = &JumpTable
                    //
    ldr x7, [x6, x2, lsl #3]
                    // Lookup the displacement of the case label derived from
                    // the low 4 bits of the Count.
                    // x7 = JumpTable_x6[CaseNumber_x2]
                    //
    add x8, x6, x7  // Complete the address of the case label by adding the
                    // displacement to the jump table base address.
                    // x8 = &JumpTable + JumpTable_x6[CaseNumber_x2]
                    //
    br x8           // Jump to the computed case label address in x8.
    //--------------------------------------------------------------------------

//////////
CASE_15:// 15 input:weights remain to be multiply-accumulated
//////////
                    // d0 holds bias.
                    // v1 is the pair-wise accumulator.

                    // Fetch 7 pairs of packed f64 values from the Inputs_x0
                    // buffer to v registers, 32 bytes at a time where possible.
    ldp q2, q3, [x0]
    ldp q4, q5, [x0, #32]
    ldp q6, q7, [x0, #64]
                    // Avoid using caller owned V8-V15.
    ldr q16, [x0, #96]
    ldr d17, [x0, #112]
                    // Load the last 8-byte f64 input value to d17.

                    // Fetch 7 pairs of packed f64 values from the Weights_x1
                    // buffer to v registers, 32 bytes at a time where possible.
    ldp q18, q19, [x1]
    ldp q20, q21, [x1, #32]
    ldp q22, q23, [x1, #64]
    ldr q24, [x1, #96]
    ldr d25, [x1, #112]
                    // Load the last 8-byte f64 weight to d25.

                    // Multiply-add 7 pairs of weights with 7 pairs of inputs to
                    // pair-wise accumulator v1.
    fmla v1.2d, v2.2d, v18.2d
    fmla v1.2d, v3.2d, v19.2d
    fmla v1.2d, v4.2d, v20.2d
    fmla v1.2d, v5.2d, v21.2d
    fmla v1.2d, v6.2d, v22.2d
    fmla v1.2d, v7.2d, v23.2d
    fmla v1.2d, v16.2d, v24.2d

    faddp v1.2d, v1.2d, v1.2d
                    // Do pairwise addition to add the high part from v1 to the
                    // low part of v1.

    fadd d0, d0, d1 // Add d1 to the bias in d0.

    fmadd d0, d17, d25, d0
                    // d0 += input_d17 * weight_d25

    ret             // Return d0 with the weighted sum.
    //--------------------------------------------------------------------------

//////////
CASE_14:// 14 input:weights remain to be multiply-accumulated
//////////
                    // d0 holds bias.
                    // v1 is the pair-wise accumulator.

                    // Fetch 7 pairs of packed f64 values from the Inputs_x0
                    // buffer to v registers, 32 bytes at a time where possible.
    ldp q2, q3, [x0]
    ldp q4, q5, [x0, #32]
    ldp q6, q7, [x0, #64]
                    // Avoid using caller owned V8-V15.
    ldr q16, [x0, #96]

                    // Fetch 7 pairs of packed f64 values from the Weights_x1
                    // buffer to v registers, 32 bytes at a time where possible.
    ldp q18, q19, [x1]
    ldp q20, q21, [x1, #32]
    ldp q22, q23, [x1, #64]
    ldr q24, [x1, #96]

                    // Multiply-add 7 pairs of weights with 7 pairs of inputs to
                    // pair-wise accumulator v1.
    fmla v1.2d, v2.2d, v18.2d
    fmla v1.2d, v3.2d, v19.2d
    fmla v1.2d, v4.2d, v20.2d
    fmla v1.2d, v5.2d, v21.2d
    fmla v1.2d, v6.2d, v22.2d
    fmla v1.2d, v7.2d, v23.2d
    fmla v1.2d, v16.2d, v24.2d

    faddp v1.2d, v1.2d, v1.2d
                    // Do pairwise addition to add the high part from v1 to the
                    // low part of v1.

    fadd d0, d0, d1 // Add d1 to the bias in d0.

    ret             // Return d0 with the weighted sum.
    //--------------------------------------------------------------------------

//////////
CASE_13:// 13 input:weights remain to be multiply-accumulated
//////////
                    // d0 holds bias.
                    // v1 is the pair-wise accumulator.

                    // Fetch 6 pairs of packed f64 values from the Inputs_x0
                    // buffer to v registers, 32 bytes at a time.
    ldp q2, q3, [x0]
    ldp q4, q5, [x0, #32]
    ldp q6, q7, [x0, #64]
                    // Avoid using caller owned V8-V15.
    ldr d16, [x0, #96]
                    // Load the last 8-byte f64 input value to d16.

                    // Fetch 6 pairs of packed f64 values from the Weights_x1
                    // buffer to v registers, 32 bytes at a time.
    ldp q18, q19, [x1]
    ldp q20, q21, [x1, #32]
    ldp q22, q23, [x1, #64]
    ldr d24, [x1, #96]
                    // Load the last 8-byte f64 weight to d24.

                    // Multiply-add 6 pairs of weights with 6 pairs of inputs to
                    // pair-wise accumulator v1.
    fmla v1.2d, v2.2d, v18.2d
    fmla v1.2d, v3.2d, v19.2d
    fmla v1.2d, v4.2d, v20.2d
    fmla v1.2d, v5.2d, v21.2d
    fmla v1.2d, v6.2d, v22.2d
    fmla v1.2d, v7.2d, v23.2d

    faddp v1.2d, v1.2d, v1.2d
                    // Do pairwise addition to add the high part from v1 to the
                    // low part of v1.

    fadd d0, d0, d1 // Add d1 to the bias in d0.

    fmadd d0, d16, d24, d0
                    // d0 += input_d16 * weight_d24

    ret             // Return d0 with the weighted sum.
    //--------------------------------------------------------------------------

//////////
CASE_12:// 12 input:weights remain to be multiply-accumulated
//////////
                    // d0 holds bias.
                    // v1 is the pair-wise accumulator.

                    // Fetch 6 pairs of packed f64 values from the Inputs_x0
                    // buffer to v registers, 32 bytes at a time.
    ldp q2, q3, [x0]
    ldp q4, q5, [x0, #32]
    ldp q6, q7, [x0, #64]
                    // Avoid using caller owned V8-V15.

                    // Fetch 6 pairs of packed f64 values from the Weights_x1
                    // buffer to v registers, 32 bytes at a time.
    ldp q18, q19, [x1]
    ldp q20, q21, [x1, #32]
    ldp q22, q23, [x1, #64]

                    // Multiply-add 6 pairs of weights with 6 pairs of inputs to
                    // pair-wise accumulator v1.
    fmla v1.2d, v2.2d, v18.2d
    fmla v1.2d, v3.2d, v19.2d
    fmla v1.2d, v4.2d, v20.2d
    fmla v1.2d, v5.2d, v21.2d
    fmla v1.2d, v6.2d, v22.2d
    fmla v1.2d, v7.2d, v23.2d

    faddp v1.2d, v1.2d, v1.2d
                    // Do pairwise addition to add the high part from v1 to the
                    // low part of v1.

    fadd d0, d0, d1 // Add d1 to the bias in d0.

    ret             // Return d0 with the weighted sum.
    //--------------------------------------------------------------------------

//////////
CASE_11:// 11 input:weights remain to be multiply-accumulated
//////////
                    // d0 holds bias.
                    // v1 is the pair-wise accumulator.

                    // Fetch 5 pairs of packed f64 values from the Inputs_x0
                    // buffer to v registers, 32 bytes at a time where possible.
    ldp q2, q3, [x0]
    ldp q4, q5, [x0, #32]
    ldr q6, [x0, #64]
    ldr d7, [x0, #80]
                    // Load the last 8-byte f64 input value to d7.

                    // Fetch 5 pairs of packed f64 values from the Weights_x1
                    // buffer to v registers, 32 bytes at a time where possible.
    ldp q18, q19, [x1]
    ldp q20, q21, [x1, #32]
    ldr q22, [x1, #64]
    ldr d23, [x1, #80]
                    // Load the last 8-byte f64 weight to d24.

                    // Multiply-add 5 pairs of weights with 5 pairs of inputs to
                    // pair-wise accumulator v1.
    fmla v1.2d, v2.2d, v18.2d
    fmla v1.2d, v3.2d, v19.2d
    fmla v1.2d, v4.2d, v20.2d
    fmla v1.2d, v5.2d, v21.2d
    fmla v1.2d, v6.2d, v22.2d

    faddp v1.2d, v1.2d, v1.2d
                    // Do pairwise addition to add the high part from v1 to the
                    // low part of v1.

    fadd d0, d0, d1 // Add d1 to the bias in d0.

    fmadd d0, d7, d23, d0
                    // d0 += input_d7 * weight_d23

    ret             // Return d0 with the weighted sum.
    //--------------------------------------------------------------------------

//////////
CASE_10:// 10 input:weights remain to be multiply-accumulated
//////////
                    // d0 holds bias.
                    // v1 is the pair-wise accumulator.

                    // Fetch 5 pairs of packed f64 values from the Inputs_x0
                    // buffer to v registers, 32 bytes at a time where possible.
    ldp q2, q3, [x0]
    ldp q4, q5, [x0, #32]
    ldr q6, [x0, #64]

                    // Fetch 5 pairs of packed f64 values from the Weights_x1
                    // buffer to v registers, 32 bytes at a time where possible.
    ldp q18, q19, [x1]
    ldp q20, q21, [x1, #32]
    ldr q22, [x1, #64]

                    // Multiply-add 5 pairs of weights with 5 pairs of inputs to
                    // pair-wise accumulator v1.
    fmla v1.2d, v2.2d, v18.2d
    fmla v1.2d, v3.2d, v19.2d
    fmla v1.2d, v4.2d, v20.2d
    fmla v1.2d, v5.2d, v21.2d
    fmla v1.2d, v6.2d, v22.2d

    faddp v1.2d, v1.2d, v1.2d
                    // Do pairwise addition to add the high part from v1 to the
                    // low part of v1.

    fadd d0, d0, d1 // Add d1 to the bias in d0.

    ret             // Return d0 with the weighted sum.
    //--------------------------------------------------------------------------

/////////
CASE_9:// 9 input:weights remain to be multiply-accumulated
/////////
                    // d0 holds bias.
                    // v1 is the pair-wise accumulator.

                    // Fetch 4 pairs of packed f64 values from the Inputs_x0
                    // buffer to v registers, 32 bytes at a time.
    ldp q2, q3, [x0]
    ldp q4, q5, [x0, #32]
    ldr d6, [x0, #64]
                    // Load the last 8-byte f64 input value to d6.

                    // Fetch 4 pairs of packed f64 values from the Weights_x1
                    // buffer to v registers, 32 bytes at a time.
    ldp q18, q19, [x1]
    ldp q20, q21, [x1, #32]
    ldr d22, [x1, #64]
                    // Load the last 8-byte f64 weight to d22.

                    // Multiply-add 4 pairs of weights with 4 pairs of inputs to
                    // pair-wise accumulator v1.
    fmla v1.2d, v2.2d, v18.2d
    fmla v1.2d, v3.2d, v19.2d
    fmla v1.2d, v4.2d, v20.2d
    fmla v1.2d, v5.2d, v21.2d

    faddp v1.2d, v1.2d, v1.2d
                    // Do pairwise addition to add the high part from v1 to the
                    // low part of v1.

    fadd d0, d0, d1 // Add d1 to the bias in d0.

    fmadd d0, d6, d22, d0
                    // d0 += input_d6 * weight_d22

    ret             // Return d0 with the weighted sum.
    //--------------------------------------------------------------------------

/////////
CASE_8:// 8 input:weights remain to be multiply-accumulated
/////////
                    // d0 holds bias.
                    // v1 is the pair-wise accumulator.

                    // Fetch 4 pairs of packed f64 values from the Inputs_x0
                    // buffer to v registers, 32 bytes at a time.
    ldp q2, q3, [x0]
    ldp q4, q5, [x0, #32]

                    // Fetch 4 pairs of packed f64 values from the Weights_x1
                    // buffer to v registers, 32 bytes at a time.
    ldp q18, q19, [x1]
    ldp q20, q21, [x1, #32]

                    // Multiply-add 4 pairs of weights with 4 pairs of inputs to
                    // pair-wise accumulator v1.
    fmla v1.2d, v2.2d, v18.2d
    fmla v1.2d, v3.2d, v19.2d
    fmla v1.2d, v4.2d, v20.2d
    fmla v1.2d, v5.2d, v21.2d

    faddp v1.2d, v1.2d, v1.2d
                    // Do pairwise addition to add the high part from v1 to the
                    // low part of v1.

    fadd d0, d0, d1 // Add d1 to the bias in d0.

    ret             // Return d0 with the weighted sum.
    //--------------------------------------------------------------------------

/////////
CASE_7:// 7 input:weights remain to be multiply-accumulated
/////////
                    // d0 holds bias.
                    // v1 is the pair-wise accumulator.

                    // Fetch 3 pairs of packed f64 values from the Inputs_x0
                    // buffer to v registers, 32 bytes at a time where possible.
    ldp q2, q3, [x0]
    ldr q4, [x0, #32]
    ldr d5, [x0, #48]
                    // Load the last 8-byte f64 input value to d5.

                    // Fetch 3 pairs of packed f64 values from the Weights_x1
                    // buffer to v registers, 32 bytes at a time where possible.
    ldp q18, q19, [x1]
    ldr q20, [x1, #32]
    ldr d21, [x1, #48]
                    // Load the last 8-byte f64 weight to d21.

                    // Multiply-add 3 pairs of weights with 3 pairs of inputs to
                    // pair-wise accumulator v1.
    fmla v1.2d, v2.2d, v18.2d
    fmla v1.2d, v3.2d, v19.2d
    fmla v1.2d, v4.2d, v20.2d

    faddp v1.2d, v1.2d, v1.2d
                    // Do pairwise addition to add the high part from v1 to the
                    // low part of v1.

    fadd d0, d0, d1 // Add d1 to the bias in d0.

    fmadd d0, d5, d21, d0
                    // d0 += input_d5 * weight_d21

    ret             // Return d0 with the weighted sum.
    //--------------------------------------------------------------------------

/////////
CASE_6:// 6 input:weights remain to be multiply-accumulated
/////////
                    // d0 holds bias.
                    // v1 is the pair-wise accumulator.

                    // Fetch 3 pairs of packed f64 values from the Inputs_x0
                    // buffer to v registers, 32 bytes at a time where possible.
    ldp q2, q3, [x0]
    ldr q4, [x0, #32]

                    // Fetch 3 pairs of packed f64 values from the Weights_x1
                    // buffer to v registers, 32 bytes at a time where possible.
    ldp q18, q19, [x1]
    ldr q20, [x1, #32]

                    // Multiply-add 3 pairs of weights with 3 pairs of inputs to
                    // pair-wise accumulator v1.
    fmla v1.2d, v2.2d, v18.2d
    fmla v1.2d, v3.2d, v19.2d
    fmla v1.2d, v4.2d, v20.2d

    faddp v1.2d, v1.2d, v1.2d
                    // Do pairwise addition to add the high part from v1 to the
                    // low part of v1.

    fadd d0, d0, d1 // Add d1 to the bias in d0.

    ret             // Return d0 with the weighted sum.
    //--------------------------------------------------------------------------

/////////
CASE_5:// 5 input:weights remain to be multiply-accumulated
/////////
                    // d0 holds bias.
                    // v1 is the pair-wise accumulator.

                    // Fetch 2 pairs of packed f64 values from the Inputs_x0
                    // buffer to v registers, 32 bytes at a time.
    ldp q2, q3, [x0]
    ldr d4, [x0, #32]
                    // Load the last 8-byte f64 input value to d4.

                    // Fetch 2 pairs of packed f64 values from the Weights_x1
                    // buffer to v registers, 32 bytes at a time.
    ldp q18, q19, [x1]
    ldr d20, [x1, #32]
                    // Load the last 8-byte f64 weight to d20.

                    // Multiply-add 2 pairs of weights with 2 pairs of inputs to
                    // pair-wise accumulator v1.
    fmla v1.2d, v2.2d, v18.2d
    fmla v1.2d, v3.2d, v19.2d

    faddp v1.2d, v1.2d, v1.2d
                    // Do pairwise addition to add the high part from v1 to the
                    // low part of v1.

    fadd d0, d0, d1 // Add d1 to the bias in d0.

    fmadd d0, d4, d20, d0
                    // d0 += input_d4 * weight_d20

    ret             // Return d0 with the weighted sum.
    //--------------------------------------------------------------------------

/////////
CASE_4:// 4 input:weights remain to be multiply-accumulated
/////////
                    // d0 holds bias.
                    // v1 is the pair-wise accumulator.

    ldp q2, q3, [x0]
                    // Fetch 2 pairs of packed f64 values from the Inputs_x0
                    // buffer to v registers, 32 bytes at a time.

    ldp q18, q19, [x1]
                    // Fetch 2 pairs of packed f64 values from the Weights_x1
                    // buffer to v registers, 32 bytes at a time.

                    // Multiply-add 2 pairs of weights with 2 pairs of inputs to
                    // pair-wise accumulator v1.
    fmla v1.2d, v2.2d, v18.2d
    fmla v1.2d, v3.2d, v19.2d

    faddp v1.2d, v1.2d, v1.2d
                    // Do pairwise addition to add the high part from v1 to the
                    // low part of v1.

    fadd d0, d0, d1 // Add d1 to the bias in d0.

    ret             // Return d0 with the weighted sum.
    //--------------------------------------------------------------------------

/////////
CASE_3:// 3 input:weights remain to be multiply-accumulated
/////////
                    // d0 holds bias.
                    // v1 is the pair-wise accumulator.

    ldr q2, [x0]    // Fetch 1 pair of packed f64 values from the Inputs_x0
                    // buffer to v registers, 16 bytes at a time.

    ldr d3, [x0, #16]
                    // Load the last 8-byte f64 input value to d3.

    ldr q18, [x1]   // Fetch 1 pair of packed f64 values from the Weights_x1
                    // buffer to v registers, 16 bytes at a time.

    ldr d19, [x1, #16]
                    // Load the last 8-byte f64 weight to d19.

                    // Multiply-add 1 pair of weights with 1 pair of inputs to
                    // pair-wise accumulator v1.
    fmla v1.2d, v2.2d, v18.2d

    faddp v1.2d, v1.2d, v1.2d
                    // Do pairwise addition to add the high part from v1 to the
                    // low part of v1.

    fadd d0, d0, d1 // Add d1 to the bias in d0.

    fmadd d0, d3, d19, d0
                    // d0 += input_d3 * weight_d19

    ret             // Return d0 with the weighted sum.
    //--------------------------------------------------------------------------

/////////
CASE_2:// 2 input:weights remain to be multiply-accumulated
/////////
                    // d0 holds bias.
                    // v1 is the pair-wise accumulator.

    ldr q2, [x0]    // Fetch 1 pair of packed f64 values from the Inputs_x0
                    // buffer to v registers, 16 bytes at a time.

    ldr q18, [x1]   // Fetch 1 pair of packed f64 values from the Weights_x1
                    // buffer to v registers, 16 bytes at a time.

    fmla v1.2d, v2.2d, v18.2d
                    // Multiply-add 1 pair of weights with 1 pair of inputs to
                    // pair-wise accumulator v1.

    faddp v1.2d, v1.2d, v1.2d
                    // Do pairwise addition to add the high part from v1 to the
                    // low part of v1.

    fadd d0, d0, d1 // Add d1 to the bias in d0.

    ret             // Return d0 with the weighted sum.
    //--------------------------------------------------------------------------

/////////
CASE_1:// 1 input:weight remain to be multiply-accumulated
/////////
                    // d0 holds bias.
                    // v1 is the pair-wise accumulator.

    faddp v1.2d, v1.2d, v1.2d
                    // Do pairwise addition to add the high part from v1 to the
                    // low part of v1.

    fadd d0, d0, d1 // Add d1 to the bias in d0.

    ldr d2, [x0]    // Load the last 8-byte f64 input value to d3.

    ldr d18, [x1]   // Load the last 8-byte f64 weight to d18.

    fmadd d0, d2, d18, d0
                    // d0 += input_d2 * weight_d18

    ret             // Return d0 with the weighted sum.
    //--------------------------------------------------------------------------

/////////
CASE_0:// No input:weights remain to be multiply-accumulated
/////////
                    // d0 holds bias.
                    // v1 is the pair-wise accumulator.

    faddp v1.2d, v1.2d, v1.2d
                    // Do pairwise addition to add the high part from v1 to the
                    // low part of v1.

    fadd d0, d0, d1 // Add d1 to the bias in d0.

    ret             // Return d0 with the weighted sum.
    //--------------------------------------------------------------------------

.align 4 // Align to a 16-byte boundary.

////////////
JumpTable:// Jump table for the switch statement.
////////////

    .quad  CASE_0  - JumpTable
    .quad  CASE_1  - JumpTable
    .quad  CASE_2  - JumpTable
    .quad  CASE_3  - JumpTable
    .quad  CASE_4  - JumpTable
    .quad  CASE_5  - JumpTable
    .quad  CASE_6  - JumpTable
    .quad  CASE_7  - JumpTable
    .quad  CASE_8  - JumpTable
    .quad  CASE_9  - JumpTable
    .quad  CASE_10 - JumpTable
    .quad  CASE_11 - JumpTable
    .quad  CASE_12 - JumpTable
    .quad  CASE_13 - JumpTable
    .quad  CASE_14 - JumpTable
    .quad  CASE_15 - JumpTable

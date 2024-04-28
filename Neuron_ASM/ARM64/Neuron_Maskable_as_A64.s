/*------------------------------------------------------------------------------
| Neuron_Maskable_as_A64.s
|-------------------------------------------------------------------------------
|
| PURPOSE: To compute the output of a neuron from its unmasked inputs and bias.
|
| DESCRIPTION: Produces a value ready to be fed into an activation function.
| Multiply-adds inputs and weights and adds with the neuron bias.
|
| This is an assembler version of Neuron_Maskable().
|
| REQUIRES:
|
|   A CPU that supports ARM64 instructions.
|
|   External code will handle overflow to infinity or NaN's if any.
|
| This C code is functionally equivalent to the assembly code below:
|
|      // OUT: Weighted sum of the neuron ready for an activation function.
|    f64
|    Neuron_Maskable(
|        f64* Inputs,
|                // Input values fed to the neuron.
|                //
|        f64* Weights,
|                // Weights corresponding to each input.
|                //
|        u8* Masks,
|                // Input mask set to 1 if the input is present, or 0 if not.
|                //
|        u32 InputCount,
|                // Number of inputs, masks, and weights.
|                //
|        f64 Bias )
|                // Bias value of the neuron.
|    {
|        // The bias is a kind of input that can be thought of as having a
|        // constant weight of 1. The bias term ensures that even when all 
|        // inputs to a neuron are zero, the neuron can still produce an output.
|    
|        // For unmasked inputs, multiply each with a weight and sum the result.
|        while( InputCount-- )
|        {
|            // InputCount is now 1 less than in the while(). On the first pass
|            // it points to the last element of the vectors.
|    
|            // Index using InputCount from the last input to the first.
|    
|            // Use Bias as the accumulator to save an addition.
|    
|            // Accept the current input if it is unmasked. Non-0 means unmasked.
|            Bias +=
|                Masks[InputCount] ? 
|                    Inputs[InputCount] * Weights[InputCount] : 0.0 ;
|                                   // Use a conditional move to avoid a branch.
|        }
|    
|        // Return the sum of unmasked products and the bias.
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
|    17Dec2023 From Neuron_Maskable(), Neuron_Maskable_as_X64.asm, and 
|              DotProductOfVectorsFromLINPACK_as_A64.s.
|    17Dec2023 Passed Neuron_Maskable_Test() OK.
------------------------------------------------------------------------------*/

.global Neuron_Maskable_as

/*------------------------------------------------------------------------------
|   // OUT: Neuron output value in d0 ready for an activation function.
| f64
| Neuron_Maskable_as(
|     f64* Inputs,
|                // Input values fed to the neuron.
|                //
|     f64* Weights,
|                // Weights corresponding to each input.
|                //
|     u8* Masks, // Input mask set to 1 if the input is present, or 0 if not.
|                //
|     u32 InputCount,
|                // Number of inputs, masks, and weights.
|                //
|     f64 Bias ) // Bias value of the neuron.
*/

/////////////////////
Neuron_Maskable_as://
/////////////////////
 
                   /* Inputs:
                    |
                    |   x0  f64* Inputs_x0, input values fed to the neuron.
                    |
                    |   x1  f64* Weights_x1, weights corresponding to inputs.
                    |
                    |   x2  u8* Masks_x2, mask set to 1 if an input is present,
                    |                                               or 0 if not.
                    |
                    |   w3  u32 InputCount_w3, input, weight, and mask count.
                    |
                    |   d0  f64 Bias_d0, bias value of the neuron.
                    */

                    // Use Bias_d0 as the accumulator to save an addition and
                    // so that the result will be in the return register.

                    // Zero d1 to provide 0 if input is masked.
    fmov d1, xzr    // Zero_d1 = 0.0

////////
ALoop:// For each triplet of (input,weight,mask) elements, from first to last
//////// since A64 has post-increment load instructions.

    subs w3, w3, #1 // Decrement InputCount_w3, setting the status flags.
                    //
    b.lo Done       // Exit loop if no more elements remain to be processed.
                    //----------------------------------------------------------

    ldr d2, [x0], #8
                    // Get the current input candidate value to d2, advancing
                    // the pointer.
                    // Input_d2 = *Inputs_x0++
                    //
    ldr d3, [x1], #8
                    // Get the current weight value to d3, advancing the pointer.
                    // Weight_d3 = *Weights_x1++
                    //
    ldrb w4, [x2], #1
                    // Get a mask byte, advancing the pointer.
                    // mask_w4 = *Masks_x2++
                    //
    cmp wzr, w4     // Compare: 0 - mask_w4
                    // Test if input mask == 0 setting the status flags.

                    // Use a conditional move to avoid a branch.

                    // Set d4 to Input_d2 or Zero_d1, depending on mask_w4 == 0.
    fcsel d4, d1, d2, eq
                    // in_d4 = Zero_d1  if input mask == 0, or
                    // in_d4 = Input_d2 if input mask != 0.

    fmadd d0, d4, d3, d0
                    // Accumulate the weighted sum using a multiply-add
                    // instruction for increased speed and reduced roundoff
                    // error. This may cause the C version results to differ
                    // from those produced by this routine.
                    // WeightedSum_d0 += in_d4 * Weight_d3
                    //
    b ALoop         // Loop back to sum the next input.
                    //----------------------------------------------------------
///////
Done:// Jumps here after all inputs have been processed.
///////

    ret             // Return d0 with the sum of products and the bias.
    //--------------------------------------------------------------------------

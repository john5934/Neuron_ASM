/*------------------------------------------------------------------------------
| Neuron_as_A64.s
|-------------------------------------------------------------------------------
|
| PURPOSE: To compute the output of a neuron from its inputs and bias.
|
| DESCRIPTION: Produces a value ready to be fed into an activation function.
| Multiply-adds inputs and weights and adds with the neuron bias.
|
| This is an assembler version of Neuron().
|
| REQUIRES:
|
|   InputCount > 0
|
|   A CPU that supports ARM64 instructions.
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
|    17Dec2023 From Neuron(), Neuron_as_X64.asm, and Neuron_Maskable_as_A64.s.
|    17Dec2023 Passed Neuron_Maskable_Test() OK.
------------------------------------------------------------------------------*/

.global Neuron_as

/*------------------------------------------------------------------------------
|   // OUT: Neuron output value in d0 ready for an activation function.
| f64
| Neuron_as(
|     f64* Inputs,
|                // Input values fed to the neuron.
|                //
|     f64* Weights,
|                // Weights corresponding to each input.
|                //
|     u32 InputCount,
|                // Number of inputs, masks, and weights.
|                //
|     f64 Bias ) // Bias value of the neuron.
*/

//////////// 
Neuron_as://
//////////// 
                   /* Inputs:
                    |
                    |   x0  f64* Inputs_x0, input values fed to the neuron.
                    |
                    |   x1  f64* Weights_x1, weights corresponding to inputs.
                    |
                    |   w2  u32 InputCount_w2, input, weight, and mask count.
                    |
                    |   d0  f64 Bias_d0, bias value of the neuron.
                    */
 
                    // Use Bias_d0 as the accumulator to save an addition and so
                    // that the result will be in the return register.
////////
ALoop:// For each pair of (input,weight) elements, from first to last.
////////

    ldr d1, [x0], #8
                    // Get the current input value to d1, advancing the pointer.
                    // Input_d1 = *Inputs_x0++
                    //
    ldr d2, [x1], #8
                    // Get the current weight value to d2, advancing the pointer.
                    // Weight_d2 = *Weights_x1++
                    //
    fmadd d0, d1, d2, d0
                    // Accumulate the weighted sum using a multiply-add
                    // instruction for increased speed and reduced roundoff
                    // error. This may cause the C version results to differ
                    // from those produced by this routine.
                    // WeightedSum_d0 += Input_d1 * Weight_d2
                    //
    subs w2, w2, #1 // Decrement InputCount_w2, setting the status flags.
                    //
    b.ne ALoop      // Loop back if elements remain to be processed.
                    //----------------------------------------------------------

    ret             // Return d0 with the sum of products and the bias.
    //--------------------------------------------------------------------------

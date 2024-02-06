/**
  ******************************************************************************
  * @file    network_custom_layers.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Wed Dec  6 17:46:55 2023
  * @brief   AI Tool Automatic Code Generator for Custom Layers Implementation
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2020 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#include "ai_layer_custom_interface.h"

#define AI_LAYER_CUSTOM_POSITIONAL_ENCODING_WEIGHTS_ID           (0)


/*USER CODE BEGINS HERE*/

/*USER CODE ENDS HERE*/


/* Layer Init Function #0 */
void custom_init_PositionalEncoding(ai_layer* layer)
{
  /* ai_layer_custom* l = ai_layer_custom_get(layer); */
  /* ai_tensor* t_in0 = ai_layer_get_tensor_in(l, 0); */
  /* ai_tensor* t_out0 = ai_layer_get_tensor_out(l, 0); */

  /*USER CODE BEGINS HERE*/
  /*USER CODE ENDS HERE*/

  ai_layer_custom_release(layer);
}

/* Layer Forward Function #0 */
void custom_forward_PositionalEncoding(ai_layer* layer)
{
  ai_layer_custom* l = ai_layer_custom_get(layer);
  ai_tensor* t_in0 = ai_layer_get_tensor_in(l, 0);
  ai_tensor* t_out0 = ai_layer_get_tensor_out(l, 0);
  

  /*USER CODE BEGINS HERE*/
  const int t_in_size = ai_tensor_get_data_size(t_in0);

  ai_tensor *t_weights = ai_layer_get_tensor_weights(l, 0);

  ai_float *d_weights = ai_tensor_get_data(t_weights).float32;
  ai_float *d_in = ai_tensor_get_data(t_in0).float32; // Data of the input tensor
  ai_float *d_out = ai_tensor_get_data(t_out0).float32; // Data of the output tensor
  
  for (int i=0; i<t_in_size; i++)
  {
    d_out[i] = d_in[i] + d_weights[i];
  }

  /*USER CODE ENDS HERE*/
  ai_layer_custom_release(layer);
}

#undef AI_LAYER_CUSTOM_POSITIONAL_ENCODING_ID




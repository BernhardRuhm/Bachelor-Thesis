#include <stdio.h>
#include <stdlib.h>

#include "network.h"
#include "network_data.h"
#include "data.h"



static ai_handle network = AI_HANDLE_NULL;

static ai_u8 activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];
static ai_float in_data[AI_NETWORK_IN_1_SIZE];
static ai_float out_data[AI_NETWORK_OUT_1_SIZE];

static ai_buffer *ai_input;
static ai_buffer *ai_output;


void get_input()
{
    for(int i=0; i<AI_NETWORK_IN_1_SIZE; i++)
    {
        in_data[i] = data[i]; 
    }
}

void ai_log_error(const ai_error err)
{
  printf("E: AI error - type=%d code=%d\r\n", err.type, err.code);
}


int ai_init()
{
    ai_error error;
    const ai_handle acts[] = {activations};

    error = ai_network_create_and_init(&network, acts, NULL);
    if (error.type != AI_ERROR_NONE) 
    {
        ai_log_error(error); 
        return -1;
    }

    ai_input = ai_network_inputs_get(network, NULL);
    ai_output = ai_network_outputs_get(network, NULL);

    return 0;
}

int ai_run(const void* in_data, const void* out_data)
{
    ai_i32 n_batch;
    ai_error error;

    ai_input[0].data = AI_HANDLE_PTR(in_data);
    ai_output[0].data = AI_HANDLE_PTR(out_data);

    n_batch = ai_network_run(network, &ai_input[0], &ai_output[0]);
    if (n_batch != 1) 
    {
        error = ai_network_get_error(network);
        ai_log_error(error);
        return -1;
    }

    return 0;
}

int main(int argc, char const *argv[])
{
    ai_init();
    get_input();
    ai_run(in_data, out_data);

    for (int i=0; i<AI_NETWORK_OUT_1_SIZE; i++) 
    {
        printf("%f ,", out_data[i]);
    }
    return 0;
}

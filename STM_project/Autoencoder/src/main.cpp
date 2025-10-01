#include "board_setup.h"

#define PI                  2.f * 3.14159265359f

#define TENSOR_ARENA_SIZE   60 * 1024

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// The necessary tensorflowlite-micro library 
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

// To get the model 
#include "neural_filter.h"
char c_str[100];

// generate data libraries
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* generate Gaussian random number using Box-Muller transform */
float rand_normal()
{
    float u1 = (rand() + 1.0) / (RAND_MAX + 1.0);
    float u2 = (rand() + 1.0) / (RAND_MAX + 1.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* generate noisy sine wave */
void generate_data(uint32_t num_samples, float noise_factor, float *x, float *s)
{
    // srand(42); // fixed seed for reproducibility
    for (int i = 0; i < num_samples; i++)
    {
        float t = (float)i / (num_samples - 1); // linspace 0..1
        s[i] = sin(2.0 * M_PI * t);               // clean signal
        float n = noise_factor * rand_normal();  // noise
        x[i] = s[i] + n;                          // noisy signal
    }
}



int main(void) {

    HAL_Init();
    init_GPIO_pins();
    init_UART2();
    init_TIM2();

    const tflite::Model* model = tflite::GetModel(neural_filter);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        UART_printf("The model version of %d does not match the version of the schema of version %d", model->version(), TFLITE_SCHEMA_VERSION);
    }
    
    // tflite::AllOpsResolver resolver;

    tflite::MicroMutableOpResolver<2> resolver;
    if (resolver.AddFullyConnected() != kTfLiteOk || resolver.AddRelu() != kTfLiteOk) {
        UART_printf("Failed to add all the ops.");
        return -1;
    }

    // Keep aligned to 16 bytes for CMSIS
    alignas(16) uint8_t tensor_arena[TENSOR_ARENA_SIZE];

    tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    tflite::MicroInterpreter* interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        UART_printf("Failed to allocate tensors.\n");
        return -1;
    }

    TfLiteTensor* input = interpreter->input(0);
    TfLiteTensor* output = interpreter->output(0);

    int n_samples = 1000;
    float noise_factor = 0.5;

    while (1) {

            // wrong inputs
            float *x = (float*)malloc(n_samples * sizeof(float));
            float *s = (float*)malloc(n_samples * sizeof(float));

            generate_data(n_samples, noise_factor, x, s);

            uint32_t start_ms = HAL_GetTick();

            input->data.f[0] = *x;

            if (interpreter->Invoke() != kTfLiteOk) {
                UART_printf("Failed to invoke for (%d)", x);
                continue;
            }
            float y = output->data.f[0];
            uint32_t inf_time = (uint32_t)HAL_GetTick() - start_ms;
            UART_printf("Restored value:", y, inf_time);
            free(x);
            free(s);
    }
}
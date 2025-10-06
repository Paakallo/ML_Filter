#include "board_setup.h"

#define PI                  2.f * 3.14159265359f

#define TENSOR_ARENA_SIZE   120 * 1024

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
#include "conv_downsize.h"
#include "neural_filter.h"
#include "sine_model.h"


int main(void) {

    HAL_Init();
    init_GPIO_pins();
    init_UART2();
    init_TIM2();

    // const tflite::Model* model = tflite::GetModel(adaptive_filter);
    // const tflite::Model* model = tflite::GetModel(neural_filter);
    const tflite::Model* model = tflite::GetModel(conv_downsize);
    // const tflite::Model* model = tflite::GetModel(sine_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        UART_printf("The model version of %d does not match the version of the schema of version %d", model->version(), TFLITE_SCHEMA_VERSION);
    }
    
    tflite::MicroMutableOpResolver<4> resolver;
       
    resolver.AddFullyConnected();
    resolver.AddExpandDims();
    resolver.AddReshape();
    resolver.AddConv2D();
    
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

    uint32_t n_samples = 500;
    float noise_factor = 0.5;

    float x[n_samples];
    float s[n_samples];
    while (1) {
            uint32_t win_start = HAL_GetTick(); 
            // generate data window
            for (int i = 0; i < n_samples; i++){
                float u1 = (rand() + 1.0) / (RAND_MAX + 1.0);
                float u2 = (rand() + 1.0) / (RAND_MAX + 1.0);
                float g_num = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
                
                float t = (float)i / (n_samples - 1);
                s[i] = sin(2.0 * M_PI * t);
                float n = noise_factor * g_num;
                x[i] = s[i] + n; 
            

                uint32_t start_ms = HAL_GetTick();

                input->data.f[0] = x[i];

                if (interpreter->Invoke() != kTfLiteOk) {
                    UART_printf("Failed to invoke for (%d)", x);
                    continue;
                }
                float y = output->data.f[0];
                uint32_t inf_time = (uint32_t)HAL_GetTick() - start_ms;
                UART_printf("Restored value:", y, inf_time);
            }
            uint32_t win_stop = (uint32_t)HAL_GetTick() - win_start;
            UART_printf("Window inference time: ", win_stop);
        }
}
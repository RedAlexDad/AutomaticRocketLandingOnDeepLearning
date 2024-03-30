#include <iostream>
// Определим конкретную версию
#define CL_HPP_TARGET_OPENCL_VERSION 210
#include <CL/cl2.hpp>

int main() {
    // Шаг 1: Инициализация OpenCL
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    cl_int err;

    // Инициализация платформы и устройства
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // Создание контекста и командной очереди
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);

    // Шаг 2: Создание и сборка программы OpenCL
    const char *source = R"""(
        __kernel void vectorAdd(__global const float* a, __global const float* b, __global float* result, const unsigned int size) {
            int i = get_global_id(0);
            if (i < size) {
                result[i] = a[i] + b[i];
            }
        }
    )""";
    program = clCreateProgramWithSource(context, 1, &source, NULL, &err);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // Шаг 3: Создание ядра
    kernel = clCreateKernel(program, "vectorAdd", &err);

    // Шаг 4: Подготовка данных
    int size = 100;
    float a[size], b[size], result[size];
    // Инициализация массивов a и b

    // Шаг 5: Создание буферов OpenCL
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * size, a, &err);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * size, b, &err);
    cl_mem bufferResult = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                         sizeof(float) * size, NULL, &err);

    // Шаг 6: Установка аргументов ядра
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferResult);
    clSetKernelArg(kernel, 3, sizeof(int), &size);

    // Шаг 7: Запуск ядра
    size_t globalSize = size;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

    // Шаг 8: Чтение данных обратно
    clEnqueueReadBuffer(queue, bufferResult, CL_TRUE, 0, sizeof(float) * size, result, 0, NULL, NULL);

    // Вывод результата
    for (int i = 0; i < size; ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    // Шаг 9: Очистка ресурсов
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferResult);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}

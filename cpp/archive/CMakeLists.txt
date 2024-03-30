cmake_minimum_required(VERSION 3.26)
project(AutomaticRocketLandingOnDeepLearning)

set(CMAKE_CXX_STANDARD 20)

# Добавляем исполняемый файл, если это необходимо
add_executable(
        AutomaticRocketLandingOnDeepLearning
        main.cpp
)

# Создаем shared library
add_library(
        AutomaticRocketLandingOnDeepLearningLib SHARED
        main.cpp
)

# Устанавливаем флаги компиляции для создания позиционно-независимого кода
set_property(TARGET AutomaticRocketLandingOnDeepLearningLib PROPERTY POSITION_INDEPENDENT_CODE ON)

# Указываем имя выходного файла .so
# Если нет файла, то компилируем вручную
# cmake --build cmake-build-debug
set_target_properties(AutomaticRocketLandingOnDeepLearningLib PROPERTIES OUTPUT_NAME "AutomaticRocketLandingOnDeepLearning")

message(STATUS "CMake System Name: ${CMAKE_SYSTEM_NAME}")

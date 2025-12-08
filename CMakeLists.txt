cmake_minimum_required(VERSION 3.16)
project(climate_lib C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(HDF5 REQUIRED COMPONENTS C)

add_library(climate STATIC
    climate.c
)

target_include_directories(climate
    PUBLIC
        ${HDF5_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(climate
    PUBLIC
        HDF5::HDF5
        m
)

target_compile_definitions(climate PRIVATE $<$<BOOL:${MSVC}>:_CRT_SECURE_NO_WARNINGS>)

add_executable(test_climate
    test_climate.c
)

target_link_libraries(test_climate
    PRIVATE
        climate
)

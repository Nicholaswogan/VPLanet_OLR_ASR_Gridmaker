cmake_minimum_required(VERSION 3.12)
project(ClimateC LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Sources for the static library
set(CLIMATE_LIB_SOURCES
    gridutils_c.c
    climate_c.c
)

add_library(climatec STATIC ${CLIMATE_LIB_SOURCES})
target_include_directories(climatec PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Find HDF5 (C)
find_package(HDF5 REQUIRED COMPONENTS C)
target_include_directories(climatec PUBLIC ${HDF5_INCLUDE_DIRS})
target_link_libraries(climatec PUBLIC ${HDF5_C_LIBRARIES} m)

# Test executable
add_executable(test_climate_c test_climate_c.c)
target_link_libraries(test_climate_c PRIVATE climatec)

add_executable(test_carbonate_silicate_c test_carbonate_silicate_c.c)
target_link_libraries(test_carbonate_silicate_c PRIVATE climatec)

# Convenience install targets (optional)
install(TARGETS climatec test_climate_c
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib)

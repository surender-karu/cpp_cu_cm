cmake_minimum_required(VERSION 3.20.0)

# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++")
# set(CMAKE_CUDA_COMPILER clang++)
# set(CMAKE_CUDA_ARCHITECTURES 86)

project(cpp_cu_cm VERSION 0.2.0
               DESCRIPTION "CPP Fin"
               LANGUAGES C CXX CUDA)

#set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(LLVM REQUIRED CONFIG)
find_package(OpenMP 5.0 REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

if (OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP ${OpenMP_CXX_VERSION}")
    message(STATUS "OpenMP CXX include dir: ${OpenMP_CXX_INCLUDE_DIRS}")
    message(STATUS "OpenMP CXX lib dir: ${OpenMP_CXX_LIBRARIES}")
endif()

if (CUDAToolkit_FOUND)
    message(STATUS "Found CUDA ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA include dir: ${CUDAToolkit_INCLUDE_DIRS}")
endif()

# add_link_options("-fuse-ld=lld")

include_directories(include)
file(GLOB SOURCES "src/*.cpp" "src/*.cu")

add_executable(cpp_cu_cm ${SOURCES})

if(OpenMP_CXX_FOUND)
    target_link_libraries(cpp_cu_cm PUBLIC OpenMP::OpenMP_CXX)
endif()

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

add_custom_command(
    TARGET cpp_cu_cm
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:cpp_cu_cm> ../bin/
)
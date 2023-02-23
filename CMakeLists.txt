cmake_minimum_required(VERSION 3.16)
project(MPI_SLAE_solution C)

set(CMAKE_C_STANDARD 99)

include_directories(/usr/lib/x86_64-linux-gnu/mpich/include)
include_directories(/usr/local/include/mpe.h)
set(CMAKE_CXX_COMPILER "/usr/bin/mpicxx")

add_executable(MPI_SLAE_solution main.c)

cmake_minimum_required(VERSION 3.16)
project(MPI_SLAE_solution C)

set(CMAKE_C_STANDARD 99)

include_directories(/usr/lib/x86_64-linux-gnu/mpich/include)
set(CMAKE_CXX_COMPILER "/usr/bin/mpicc")

add_executable(MPI_SLAE_solution main.c)

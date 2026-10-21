find_package(MPI COMPONENTS C)
if (NOT ${MPI_C_FOUND})
    message(FATAL_ERROR "MPI C not found")
endif()

find_package(BLAS)

add_executable(serial serial.c)

add_executable(mpi mpi.c)
target_link_libraries(mpi PRIVATE MPI::MPI_C)

add_executable(mpiblas mpiblas.c)
target_link_libraries(mpiblas PRIVATE MPI::MPI_C PRIVATE BLAS::BLAS)

find_package(MPI COMPONENTS C)
if (NOT ${MPI_C_FOUND})
    message(FATAL_ERROR "MPI C not found")
endif()

# OpenBLAS's pkg-config file gives both the include directory (cblas.h) and
# the libraries, wherever it is installed
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENBLAS REQUIRED IMPORTED_TARGET openblas)
# OpenBLAS built with OpenMP needs the OpenMP runtime (libgomp), which its
# pkg-config file does not always list
find_package(OpenMP REQUIRED COMPONENTS C)

add_executable(serial serial.c matrix.c)

add_executable(mpi mpi.c matrix.c)
target_link_libraries(mpi PRIVATE m PRIVATE MPI::MPI_C)

add_executable(mpiblas mpiblas.c matrix.c)
target_link_libraries(mpiblas PRIVATE m PRIVATE MPI::MPI_C PRIVATE PkgConfig::OPENBLAS PRIVATE OpenMP::OpenMP_C)

gfortran -fopenmp -c data_mod.f90
gfortran -fopenmp -c bessel_mod.f90
gfortran -fopenmp -c solver_mod.f90
gfortran -fopenmp -c assembly_mod.f90
gfortran -fopenmp -c io_mod.f90
gfortran -fopenmp main.f90 data_mod.o bessel_mod.o solver_mod.o io_mod.o assembly_mod.o -llapack -lblas -o quadpleh_modern.exe

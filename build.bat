gfortran -fopenmp -fdefault-real-8 -c data_mod.f90
gfortran -fopenmp -fdefault-real-8 -c bessel_mod.f90
gfortran -fopenmp -fdefault-real-8 -c solver_mod.f90
gfortran -fopenmp -fdefault-real-8 -c assembly_mod.f90
gfortran -fopenmp -fdefault-real-8 -c io_mod.f90
gfortran -fopenmp -fdefault-real-8 main.f90 data_mod.o bessel_mod.o solver_mod.o io_mod.o assembly_mod.o -llapack -lblas -o quadpleh_modern.exe

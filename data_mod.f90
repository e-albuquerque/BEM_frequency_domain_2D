MODULE data_mod
  IMPLICIT NONE
  INTEGER :: N, L, INP, IPR, NE, NFR
  COMPLEX :: CP, CS, GE
  REAL :: RO, XNU, FR

  COMPLEX, ALLOCATABLE :: FI(:), DFI(:), BC(:), G(:,:), H(:,:), DSOL(:)
  REAL, ALLOCATABLE :: X(:), Y(:), CX(:), CY(:), AFR(:)
  INTEGER, ALLOCATABLE :: KODE(:)
END MODULE data_mod

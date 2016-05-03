CFLAGS	         =

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

quantum_ss: quantum_ss.o  chkopts
	-${CLINKER} -o quantum_ss quantum_ss.o  ${PETSC_KSP_LIB}
	${RM} quantum_ss.o


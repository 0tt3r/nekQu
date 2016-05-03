/*T
   Concepts: KSP^solving a Helmholtz equation
   Concepts: complex numbers;
   Concepts: Helmholtz equation
   Processors: n
T*/

/*
   Description: Solves a complex linear system in parallel with KSP.

   The model problem:
      Solve Helmholtz equation on the unit square: (0,1) x (0,1)
          -delta u - sigma1*u + i*sigma2*u = f,
           where delta = Laplace operator
      Dirichlet b.c.'s on all sides
      Use the 2-D, five-point finite difference stencil.

   Compiling the code:
      This code uses the complex numbers version of PETSc, so configure
      must be run to enable this
*/

/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/
#include <petscksp.h>
#include <stdio.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Vec            x,b;      /* approx solution, RHS, exact solution */
  Mat            A;            /* linear system matrix */
  KSP            ksp;         /* linear solver context */
  PetscReal      norm;         /* norm of solution error */
  PetscInt       dim,i,Istart,Iend,n = 6,its;
  PetscInt       ham_nnz,lin_nnz;
  PetscErrorCode ierr;
  PetscScalar    mat_tmp,one = 1.0,*xa;
  PetscRandom    rctx;
  FILE           *fp_ham,*fp_lin;
  double         *ham,*lin;
  int            *ham_row,*ham_col,nid;
  int            *lin_row,*lin_col,initial_guess,initial_guess_index;

  PetscInitialize(&argc,&args,(char*)0,NULL);
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,1,"This example requires complex numbers");
#endif
  initial_guess = 0;
  ierr=MPI_Comm_rank(PETSC_COMM_WORLD,&nid);

  fp_ham = fopen("ham_tilde","r");
  fscanf(fp_ham,"%d %d",&n,&ham_nnz);
  printf("nstate: %d %d\n",n,ham_nnz);

  ham     = malloc(ham_nnz*sizeof(double));
  ham_row = malloc(ham_nnz*sizeof(int));
  ham_col = malloc(ham_nnz*sizeof(int));

  for(i=0;i<ham_nnz;i++){
    fscanf(fp_ham,"%lf %d %d",&ham[i],&ham_col[i],&ham_row[i]);
  }

  fp_lin = fopen("l_tilde","r");
  fscanf(fp_lin,"%d",&lin_nnz);
  lin     = malloc(lin_nnz*sizeof(double));
  lin_row = malloc(lin_nnz*sizeof(int));
  lin_col = malloc(lin_nnz*sizeof(int));

  for(i=0;i<lin_nnz;i++){
    fscanf(fp_lin,"%lf %d %d",&lin[i],&lin_col[i],&lin_row[i]);
  }

  dim  = n*n;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Create parallel matrix, specifying only its global dimensions.
     When using MatCreate(), the matrix format can be specified at
     runtime. Also, the parallel partitioning of the matrix is
     determined by PETSc at runtime.
  */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,dim,dim);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  /*
     Currently, all PETSc parallel matrix formats are partitioned by
     contiguous chunks of rows across the processors.  Determine which
     rows of the matrix are locally owned.
  */
  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);

  /*
     Set matrix elements in parallel.
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly).
      - Always specify global rows and columns of matrix entries.
  */

  /*
    Set Hamiltonian matrix elements
    loop through ham; if I own the row, then add it to the matrix
   */
  for (i=0;i<ham_nnz;i++){
    if(ham_row[i]<Iend&&ham_row[i]>Istart) {
      mat_tmp = 0. + ham[i]*PETSC_i;
      ierr = MatSetValues(A,1,&ham_row[i],1,&ham_col[i],&mat_tmp,ADD_VALUES);CHKERRQ(ierr);
    }
  }

  /*
    Set Lindblad matrix elements
    loop through ham; if I own the row, then add it to the matrix
   */

  for (i=0;i<lin_nnz;i++){
    if(lin_row[i]<Iend&&lin_row[i]>Istart) {
      mat_tmp = lin[i] + 0.*PETSC_i;
      ierr = MatSetValues(A,1,&lin_row[i],1,&lin_col[i],&mat_tmp,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  /*
    Set diagonal to 0
   */
  for(i=Istart;i<Iend;i++){
      mat_tmp = 0 + 0.*PETSC_i;
      ierr = MatSetValues(A,1,&i,1,&i,&mat_tmp,ADD_VALUES);CHKERRQ(ierr);
  }    
  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
  */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //  MatView(A,PETSC_VIEWER_STDOUT_WORLD);
  /*
     Create parallel vectors.
      - When using VecCreate(), VecSetSizes() and VecSetFromOptions(),
      we specify only the vector's global
        dimension; the parallel partitioning is determined at runtime.
      - Note: We form 1 vector from scratch and then duplicate as needed.
  */
  ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
  ierr = VecSetSizes(b,PETSC_DECIDE,dim);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);

  /*
     Set rhs, b, to 0.0
  */
  
  ierr = VecSet(b,0.0);

  /*
     Set inital guess, x, to 1.0 in a diagonal spot, 0 elsewhere
  */

  ierr = VecSet(x,0.0);
  initial_guess_index = initial_guess*n + initial_guess;
  initial_guess_index = 0;
  VecSetValues(x,1,&initial_guess_index,&one,ADD_VALUES);
  VecSetValues(b,1,&initial_guess_index,&one,ADD_VALUES);
  VecAssemblyBegin(x);
  VecAssemblyEnd(x);

  VecAssemblyBegin(b);
  VecAssemblyEnd(b);

  ierr = VecGetArray(b,&xa);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"The first three entries of x are:\n");CHKERRQ(ierr);
  for (i=0; i<3; i++) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"x[%D] = %g + %g i\n",i,(double)PetscRealPart(xa[i]),(double)PetscImaginaryPart(xa[i]));CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(b,&xa);CHKERRQ(ierr);


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create linear solver context
  */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);

  /*
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);

  /*
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
  */
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
      Print the first 3 entries of x; this demonstrates extraction of the
      real and imaginary components of the complex vector, x.
  */
  ierr = VecGetArray(x,&xa);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"The first three entries of x are:\n");CHKERRQ(ierr);
  for (i=0; i<3; i++) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"x[%D] = %g + %g i\n",i,(double)PetscRealPart(xa[i]),(double)PetscImaginaryPart(xa[i]));CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(x,&xa);CHKERRQ(ierr);


  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  if (norm < 1.e-12) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error < 1.e-12 iterations %D\n",its);CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g iterations %D\n",(double)norm,its);CHKERRQ(ierr);
  }

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr); ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

/*T
   Concepts: KSP^solving a Helmholtz equation
   Concepts: complex numbers;
   Concepts: Helmholtz equation
   Processors: n
T*/

/*
   Description: Solves a complex linear system in parallel with KSP.

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
#include <petsctime.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Vec            x,b;      /* approx solution, RHS, exact solution */
  Mat            A;            /* linear system matrix */
  KSP            ksp;         /* linear solver context */
  PetscInt       dim,i,j,Istart,Iend,n = 6,its,num_state;
  PetscInt       ham_nnz,lin_nnz,x_low,x_high;
  PetscErrorCode ierr;
  PetscScalar    mat_tmp,one = 1.0,*xa;
  PetscReal      tmp_real;
  FILE           *fp_ham,*fp_lin,*fp_state;
  double         *ham,*lin,sum,*pops;
  int            *ham_row,*ham_col,nid,row,col,**cur_state;
  int            *lin_row,*lin_col,initial_guess,initial_guess_index;
  PetscLogDouble  t1,t2;

  PetscInitialize(&argc,&args,(char*)0,NULL);
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,1,"This example requires complex numbers");
#endif

  initial_guess = 0;
  ierr=MPI_Comm_rank(PETSC_COMM_WORLD,&nid);
  PetscGetCPUTime(&t1);
  fp_ham = fopen("ham_tilde","r");
  fscanf(fp_ham,"%d %d",&n,&ham_nnz);
  if(nid==0) printf("nstate: %d %d\n",n,ham_nnz);

  ham     = malloc(ham_nnz*sizeof(double));
  ham_row = malloc(ham_nnz*sizeof(int));
  ham_col = malloc(ham_nnz*sizeof(int));

  for(i=0;i<ham_nnz;i++){
    fscanf(fp_ham,"%lf %d %d",&ham[i],&ham_col[i],&ham_row[i]);
  }
  if(nid==0) printf("read ham\n");
  fp_lin = fopen("l_tilde","r");
  fscanf(fp_lin,"%d",&lin_nnz);
  lin     = malloc(lin_nnz*sizeof(double));
  lin_row = malloc(lin_nnz*sizeof(int));
  lin_col = malloc(lin_nnz*sizeof(int));

  for(i=0;i<lin_nnz;i++){
    fscanf(fp_lin,"%lf %d %d",&lin[i],&lin_col[i],&lin_row[i]);
  }
  if(nid==0) printf("read lin\n");
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
  ierr = MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,dim,dim,
                      5000,NULL,5000,NULL,&A);CHKERRQ(ierr);
  /* ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,dim,dim, */
  /*                     1000,NULL,100,NULL,&A);CHKERRQ(ierr); */
  //  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,dim,dim);CHKERRQ(ierr);
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
    if(ham_row[i]<Iend&&ham_row[i]>=Istart) {
      mat_tmp = 0. + ham[i]*PETSC_i;
      ierr = MatSetValue(A,ham_row[i],ham_col[i],mat_tmp,ADD_VALUES);CHKERRQ(ierr);
    }
  }

  if(nid==0) printf("set Ham\n");
  /*
    Set Lindblad matrix elements
    loop through ham; if I own the row, then add it to the matrix
   */
  
  for (i=0;i<lin_nnz;i++){
    if(lin_row[i]<Iend&&lin_row[i]>=Istart) {
      mat_tmp = lin[i] + 0.*PETSC_i;
      ierr = MatSetValues(A,1,&lin_row[i],1,&lin_col[i],&mat_tmp,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  //      ierr = MatSetValues(A,lin_nnz,lin_row,lin_nnz,lin_col,lin,ADD_VALUES);CHKERRQ(ierr);

  if(nid==0) printf("set lin\n");
  /*
    Add elements to the matrix to make the normalization work
    I have no idea why this works, I am copying it from qutip
    We add 1.0 in the 0th spot and every n+1 after
  */
  if(nid==0) {
    row = 0;
    for(i=0;i<n;i++){
      col = i*(n+1);
      mat_tmp = 1.0 + 0.*PETSC_i;
      ierr = MatSetValues(A,1,&row,1,&col,&mat_tmp,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  if(nid==0) printf("set struc\n");
  /*
    Set diagonal to 0
   */
  for(i=Istart;i<Iend;i++){
      mat_tmp = 0 + 0.*PETSC_i;
      ierr = MatSetValue(A,i,i,mat_tmp,ADD_VALUES);CHKERRQ(ierr);
  }
  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
  */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if(nid==0) printf("assembly done\n");
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

  if(nid==0) {
    VecSetValues(x,1,&initial_guess_index,&one,INSERT_VALUES);
    initial_guess_index = 0;  
    VecSetValues(b,1,&initial_guess_index,&one,INSERT_VALUES);
  }
  VecAssemblyBegin(x);
  VecAssemblyEnd(x);

  VecAssemblyBegin(b);
  VecAssemblyEnd(b);
  PetscGetCPUTime(&t2);
  if(nid==0) printf("setup time: %f\n",t2-t1);
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
  //  VecView(x,PETSC_VIEWER_STDOUT_WORLD);

  // Normalize (in the quantum sense, trace(rho) = 1.0)
  /* ierr = VecGetArray(x,&xa);CHKERRQ(ierr); */
  /* sum = 0.0; */
  /* for(i=0;i<n;i++) { */
  /*   sum += (double)PetscRealPart(xa[i*n+i]); */
  /* } */
  /* //  VecScale(x,1/sum); */
  /* //   VecView(x,PETSC_VIEWER_STDOUT_WORLD); */
  /* printf("sum: %f\n",sum); */

  /* 
     Get populations by multiplying by curState_vec
  */

  fp_state = fopen("cur_state","r");
  fscanf(fp_state,"%d",&num_state);

  int* block = (int*)malloc(n*num_state*sizeof(int));
  cur_state = (int**)malloc(n*sizeof(int*));
  for (i=0;i<n;i++) {
    cur_state[i] = block + num_state * i;
  }

  for(i=0;i<n;i++){
    for(j=0;j<num_state;j++){
      fscanf(fp_state,"%d",&cur_state[i][j]);
    }
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"The first three entries of x are:\n");CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(x,&x_low,&x_high);
  ierr = VecGetArray(x,&xa);CHKERRQ(ierr); 

  pops = (double*)malloc(num_state*sizeof(double));
  for(i=0;i<num_state;i++){
    pops[i] = 0.0;
  }

  for(i=0;i<n;i++){
    if((i*n+i)>=x_low&&(i*n+i)<x_high) {
      tmp_real = (double)PetscRealPart(xa[i*(n)+i-x_low]);
      for(j=0;j<num_state;j++){
        pops[j] += cur_state[i][j]*tmp_real;
      }
    }
  } 
  if(nid==0) {
    MPI_Reduce(MPI_IN_PLACE,pops,num_state,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  } else {
    MPI_Reduce(pops,pops,num_state,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  }
  if(nid==0) {
    printf("Populations: ");
    for(i=0;i<num_state;i++){
      printf(" %e ",pops[i]);
    }
    printf("\n");
  }
    
  for (i=0; i<3; i++) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"x[%D] = %g + %g i\n",i,(double)PetscRealPart(xa[i]),(double)PetscImaginaryPart(xa[i]));CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(x,&xa);CHKERRQ(ierr);


  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"iterations %D\n",its);CHKERRQ(ierr);

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr); ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();

  free (cur_state[0]);
  free (cur_state);
  return 0;
}

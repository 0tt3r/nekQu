c-------------------------------------------
c csr mxm
c Based off of SparseKit, http://people.sc.fsu.edu/~jburkardt/f77_src/sparsekit/sparsekit.f
c I don't know the licensing information
c and Mills Vectorized sparse mxm for csr
c-------------------------------------------



c*********************************************************************72
      subroutine aplb1(nrow,ncol,job,a,ja,ia,b,jb,ib,c,jc,ic,nzmax,ierr)

c*********************************************************************72
c
cc APLB1 performs the sum C = A + B for sorted CSR matrices.
c
c the difference with aplb  is that the resulting matrix is such that
c the elements of each row are sorted with increasing column indices in
c each row, provided the original matrices are sorted in the same way.
c
c on entry:
c
c nrow      = integer. The row dimension of A and B
c ncol  = integer. The column dimension of A and B.
c job   = integer. Job indicator. When job = 0, only the structure
c                  (i.e. the arrays jc, ic) is computed and the
c                  real values are ignored.
c
c a,
c ja,
c ia   = Matrix A in compressed sparse row format with entries sorted
c
c b,
c jb,
c ib      =  Matrix B in compressed sparse row format with entries sorted
c        ascendly in each row
c
c nzmax      = integer. The  length of the arrays c and jc.
c         amub will stop if the result matrix C  has a number
c         of elements that exceeds exceeds nzmax. See ierr.
c
c on return:
c
c c,
c jc,
c ic      = resulting matrix C in compressed sparse row sparse format
c         with entries sorted ascendly in each row.
c
c ierr      = integer. serving as error message.
c         ierr = 0 means normal return,
c         ierr .gt. 0 means that amub stopped while computing the
c         i-th row  of C with i=ierr, because the number
c         of elements in C exceeds nzmax.
c
c Notes:
c
c     this will not work if any of the two input matrices is not sorted
c
      double precision a(*), b(*), c(*)
      integer ja(*),jb(*),jc(*),ia(nrow+1),ib(nrow+1),ic(nrow+1)
      logical values

      values = (job .ne. 0)
      ierr = 0
      kc = 1
      ic(1) = kc

      do 6 i=1, nrow
         ka = ia(i)
         kb = ib(i)
         kamax = ia(i+1)-1
         kbmax = ib(i+1)-1
 5       continue
         if (ka .le. kamax) then
            j1 = ja(ka)
         else
            j1 = ncol+1
         end if
         if (kb .le. kbmax) then
            j2 = jb(kb)
         else
            j2 = ncol+1
         end if
c
c     three cases
c
         if (j1 .eq. j2) then
            if (values) c(kc) = a(ka)+b(kb)
            jc(kc) = j1
            ka = ka+1
            kb = kb+1
            kc = kc+1
         else if (j1 .lt. j2) then
            jc(kc) = j1
            if (values) c(kc) = a(ka)
            ka = ka+1
            kc = kc+1
         else if (j1 .gt. j2) then
            jc(kc) = j2
            if (values) c(kc) = b(kb)
            kb = kb+1
            kc = kc+1
         end if
         if (kc .gt. nzmax) goto 999
         if (ka .le. kamax .or. kb .le. kbmax) goto 5
         ic(i+1) = kc
 6    continue
      return
 999  ierr = i
      return
      end
       
      
c*********************************************************************72
      subroutine aplb (nrow,ncol,job,a,ja,ia,b,jb,ib,
     $     c,jc,ic,nzmax,iw,ierr)

c*********************************************************************72
c
cc APLB performs the CSR matrix sum C = A + B.
c
c on entry:
c
c nrow      = integer. The row dimension of A and B
c ncol  = integer. The column dimension of A and B.
c job   = integer. Job indicator. When job = 0, only the structure
c                  (i.e. the arrays jc, ic) is computed and the
c                  real values are ignored.
c
c a,
c ja,
c ia   = Matrix A in compressed sparse row format.
c
c b,
c jb,
c ib      =  Matrix B in compressed sparse row format.
c
c nzmax      = integer. The  length of the arrays c and jc.
c         amub will stop if the result matrix C  has a number
c         of elements that exceeds exceeds nzmax. See ierr.
c
c on return:
c
c c,
c jc,
c ic      = resulting matrix C in compressed sparse row sparse format.
c
c ierr      = integer. serving as error message.
c         ierr = 0 means normal return,
c         ierr .gt. 0 means that amub stopped while computing the
c         i-th row  of C with i=ierr, because the number
c         of elements in C exceeds nzmax.
c
c work arrays:
c
c iw      = integer work array of length equal to the number of
c   columns in A.
c
      double precision a(*), b(*), c(*)
      integer ja(*),jb(*),jc(*),ia(nrow+1),ib(nrow+1),ic(nrow+1),
     $     iw(ncol)
      logical values

      values = (job .ne. 0)
      ierr = 0
      len = 0
      ic(1) = 1
      do 1 j=1, ncol
         iw(j) = 0
 1    continue

      do 500 ii=1, nrow
c     row i
         do 200 ka=ia(ii), ia(ii+1)-1
            len = len+1
            jcol    = ja(ka)
            if (len .gt. nzmax) goto 999
            jc(len) = jcol
            if (values) c(len)  = a(ka)
            iw(jcol)= len
 200     continue

         do 300 kb=ib(ii),ib(ii+1)-1
            jcol = jb(kb)
            jpos = iw(jcol)
            if (jpos .eq. 0) then
               len = len+1
               if (len .gt. nzmax) goto 999
               jc(len) = jcol
               if (values) c(len)  = b(kb)
               iw(jcol)= len
            else
               if (values) c(jpos) = c(jpos) + b(kb)
            end if
 300     continue
         do 301 k=ic(ii), len
            iw(jc(k)) = 0
 301     continue
         ic(ii+1) = len+1
 500  continue
      return
 999  ierr = ii
      return
      end
       

c*********************************************************************72
      subroutine coicsr (n,nnz,job,a,ja,ia,iwk)

c*********************************************************************72
c
cc COICSR converts COO to CSR in place.
c
c IN-PLACE coo-csr conversion routine.
c
c COICSR converts a matrix stored in coordinate format into
c the csr format. The conversion is done in place in that the arrays
c a,ja,ia of the result are overwritten onto the original arrays.
c
c on entry:
c
c n      = integer. row dimension of A.
c nnz      = integer. number of nonzero elements in A.
c job   = integer. Job indicator. when job=1, the real values in a are
c         filled. Otherwise a is not touched and the structure of the
c         array only (i.e. ja, ia)  is obtained.
c a      = double precision array of size nnz (number of nonzero elements in A)
c         containing the nonzero elements
c ja      = integer array of length nnz containing the column positions
c         of the corresponding elements in a.
c ia      = integer array of length nnz containing the row positions
c         of the corresponding elements in a.
c iwk      = integer work array of length n.
c on return:
c
c a
c ja
c ia      = contains the compressed sparse row data structure for the
c         resulting matrix.
c Note:
c
c         the entries of the output matrix are not sorted (the column
c         indices in each are not in increasing order) use coocsr
c         if you want them sorted.
c
c  Coded by Y. Saad, Sep. 26 1989
c
      integer ia(nnz),ja(nnz),iwk(n)
      double precision a(*)
      double precision t,tnext
      logical values

      values = (job .eq. 1)
c
c find pointer array for resulting matrix.
c
      do 35 i=1,n+1
         iwk(i) = 0
 35   continue
      do 4 k=1,nnz
         i = ia(k)
         iwk(i+1) = iwk(i+1)+1
 4    continue

      iwk(1) = 1
      do 44 i=2,n
         iwk(i) = iwk(i-1) + iwk(i)
 44   continue
c
c  loop for a cycle in chasing process.
c
      init = 1
      k = 0
 5    if (values) t = a(init)
      i = ia(init)
      j = ja(init)
      ia(init) = -1

 6    k = k+1
c
c  current row number is i.  determine  where to go.
c
      ipos = iwk(i)
c
c  save the chased element.
c
      if (values) tnext = a(ipos)
      inext = ia(ipos)
      jnext = ja(ipos)
c
c  then occupy its location.
c
      if (values) a(ipos)  = t
      ja(ipos) = j
c
c  update pointer information for next element to come in row i.
c
      iwk(i) = ipos+1
c
c  determine  next element to be chased,
c
      if (ia(ipos) .lt. 0) goto 65
      t = tnext
      i = inext
      j = jnext
      ia(ipos) = -1
      if (k .lt. nnz) goto 6
      goto 70
 65   init = init+1
      if (init .gt. nnz) goto 70
      if (ia(init) .lt. 0) goto 65
c
c  restart chasing.
c
      goto 5
 70   do 80 i=1,n
         ia(i+1) = iwk(i)
 80   continue
      ia(1) = 1
      return
      end

c*********************************************************************72
      subroutine coocsr(nrow,nnz,a,ir,jc,ao,jao,iao)

c*********************************************************************72
c
cc COOCSR converts COO to CSR.
c
c converts a matrix that is stored in coordinate format
c  a, ir, jc into a row general sparse ao, jao, iao format.
c
c on entry:
c
c nrow      = dimension of the matrix
c nnz      = number of nonzero elements in matrix
c a,
c ir,
c jc    = matrix in coordinate format. a(k), ir(k), jc(k) store the nnz
c         nonzero elements of the matrix with a(k) = actual real value of
c         the elements, ir(k) = its row number and jc(k) = its column
c        number. The order of the elements is arbitrary.
c
c on return:
c
c ir       is destroyed
c
c ao, jao, iao = matrix in general sparse matrix format with ao
c       continung the real values, jao containing the column indices,
c      and iao being the pointer to the beginning of the row,
c      in arrays ao, jao.
c
c Notes:
c This routine is NOT in place.  See coicsr
c
      complex a(*),ao(*),x
      integer ir(*),jc(*),jao(*),iao(*)

      do 1 k=1,nrow+1
         iao(k) = 0
 1           continue
c determine row-lengths.
      do 2 k=1, nnz
         iao(ir(k)) = iao(ir(k))+1
 2           continue
c starting position of each row.
      k = 1
      do 3 j=1,nrow+1
         k0 = iao(j)
         iao(j) = k
         k = k+k0
 3           continue
c go through the structure  once more. Fill in output matrix.
      do 4 k=1, nnz
         i = ir(k)
         j = jc(k)
         x = a(k)
         iad = iao(i)
         ao(iad) =  x
         jao(iad) = j
         iao(i) = iad+1
 4           continue
c shift back iao
      do 5 j=nrow,1,-1
         iao(j+1) = iao(j)
 5           continue
      iao(1) = 1
      return
      end

c*********************************************************************72
c
cc quantum_mult_csr performs the matrix product C = A * B.
c
c on entry:
c
c nrow  = integer. The row dimension of A
c ncol  = integer. The column dimension of A
c job   = integer. Job indicator. When job = 0, only the structure
c                  (i.e. the arrays jc, ic) is computed and the
c                  real values are ignored.
c
c a,
c ja,
c ia   = Matrix A in compressed sparse row format.
c
c b,
c jb,
c ib    =  Matrix B in compressed sparse row format.
c
c nzmax = integer. The  length of the arrays c and jc.
c         amub will stop if the result matrix C  has a number
c         of elements that exceeds exceeds nzmax. See ierr.
c
c on return:
c
c c,
c jc,
c ic    = resulting matrix C in compressed sparse row sparse format.
c
c ierr      = integer. serving as error message.
c         ierr = 0 means normal return,
c         ierr .gt. 0 means that amub stopped while computing the
c         i-th row  of C with i=ierr, because the number
c         of elements in C exceeds nzmax.
c
c work arrays:
c
c iw      = integer work array of length equal to the number of
c         columns in A.
c Notes:
c
c         The column dimension of B is not needed.
c
      subroutine quantum_mult_csr (nrow,ncol,job,a,ja,ia,b,jb,ib,
     *     c,jc,ic,nzmax)
      double precision a(*), b(*), c(*)
      integer ja(*),jb(*),jc(*),ia(nrow+1),ib(ncol+1),
     *     ic(ncol+1),iw(ncol+1)

      double precision scal
      logical values
      values = (job .ne. 0)
      len = 0
      ic(1) = 1
      ierr = 0
c
c  Initialize array iw.
c
      do j=1, ncol
         iw(j) = 0
      end do

      do ii=1,nrow
c     row i
         do 200 ka=ia(ii), ia(ii+1)-1
          if (values) scal = a(ka)
          jj   = ja(ka)
          do 100 kb=ib(jj),ib(jj+1)-1
               jcol = jb(kb)
                  jpos = iw(jcol)
                  if (jpos .eq. 0) then
                     len = len+1
                     if (len .gt. nzmax) then
                        print*,'Error in MULT CSR'
                        print*,'Len greater than nzmax!'
                        stop
                        return
                     end if
                     jc(len) = jcol
                     iw(jcol)= len
                     if (values) c(len)  = scal*b(kb)

                  else
                     if (values) c(jpos) = c(jpos) + scal*b(kb)
                  end if

 100                     continue

 200                          continue
         do k=ic(ii), len
          iw(jc(k)) = 0
       enddo
         ic(ii+1) = len+1
      enddo
      return
      end

!************************************************************
!     Wrapper for general tensor product, using handles of operators
!************************************************************

      subroutine quantum_kron(handle1,handle2,c,jc,ic)
      implicit none
      include 'SIZE'
      include 'QUANT'

      integer handle1,handle2,nnz_c
      integer jc(*),ic(*)
      real    c(*)
      call quantum_csr_kron(op_csr_a(1,handle1),op_csr_ja(1,handle1),
     $     op_csr_ia(1,handle1),nstate,nstate,op_csr_a(1,handle2),
     $     op_csr_ja(1,handle2),op_csr_ia(1,handle2),nstate,nstate,
     $     c,jc,ic,nnz_c)


      end

!************************************************************
!     Calculate I cross H - H cross I
!************************************************************

      subroutine quantum_kron_ham()
      implicit none
      include 'SIZE'
      include 'QUANT'

      integer ja(nstate2*nstate),ia(nstate2*nstate),nnz_a
      real a(nstate2*nstate)
      integer jb(nstate2*nstate),ib(nstate2*nstate),nnz_b
      real b(nstate2*nstate)
      integer jc(nstate2*nstate),ic(nstate2*nstate),nnz_c
      real c(nstate2*nstate)
      integer i_eye(nstate+1),j_eye(nstate),iw(nstate2)
      real eye(nstate),op(nstate2,nstate2)
      integer ii,jj,ierr,kk
!     Construct identity matrix in csr format
      do ii=1,nstate
         eye(ii) = 1.0
         i_eye(ii) = ii
         j_eye(ii) = ii
      enddo
      i_eye(nstate+1) = nstate+1

      !Do a = I cross H
      call quantum_csr_kron(eye,j_eye,i_eye,nstate,nstate,
     $     ham_i,ham_ja,ham_ia,nstate,nstate,a,ja,ia)

      !Do b = H cross I
      call quantum_csr_kron(ham_i,ham_ja,ham_ia,nstate,nstate,
     $     eye,j_eye,i_eye,nstate,nstate,b,jb,ib)

!Make all elements in b negative, because we want I cross H - H cross I
      do ii=1,nstate2
         do kk=ib(ii),ib(ii+1)-1
            b(kk) = -b(kk)
         enddo
      enddo


!Add a and b
      call aplb(nstate2,nstate2,1,a,ja,ia,b,jb,ib,c,jc,ic,
     $     nstate2*nstate,iw,ierr)

      call csrcoo(nstate2,1,nstate2*nstate,c,jc,ic,nnz_c,c,ic,jc,ierr)

!     Now write the coo matrix to file
      if(nid.eq.0) then
         !count real nnz
         nnz_b = 0
         do ii=1,nnz_c
            if(c(ii).ne.0.0) then
               nnz_b = nnz_b + 1
            endif
         enddo

         open(unit=20,file='ham_tilde')
         !Write nstate, nnz_c
         write(20,*) nstate, nnz_b
         do ii=1,nnz_c
            !We subtract 1 to count from 0 for C code
            !Also, it's -i * [H,p], so add the -1
            if(c(ii).ne.0.0) then
               write(20,'(es20.13,2i7)') -c(ii),jc(ii)-1,ic(ii)-1
            endif
         enddo
         close(unit=20)
      endif
      
      end


!************************************************************
!     Calculate L_tilde, the explicit n2 x n2 matrix representation
!     of the lindblad
!
!     L = xt x rho + rho xt x - 2 x rho xt
!     L_tilde = I cross xt x + xt x cross I - 2*(x cross transpose(xt))
!             = ............. ... ..        - 2*(x cross x), since transpose(xt) = x
!     
!************************************************************

      subroutine quantum_kron_lin()
      implicit none
      include 'SIZE'
      include 'QUANT'

      integer ja(nstate2*nstate),ia(nstate2*nstate),nnz_a
      real a(nstate2*nstate)
      integer jb(nstate2*nstate),ib(nstate2*nstate),nnz_b
      real b(nstate2*nstate)
      integer jc(nstate2*nstate),ic(nstate2*nstate),nnz_c
      real c(nstate2*nstate)
      integer jL(nstate2*nstate),iL(nstate2*nstate)
      real L(nstate2*nstate)
      integer i_eye(nstate+1),j_eye(nstate),iw(nstate2)
      real eye(nstate),op(nstate2,nstate2)
      integer ii,jj,ierr,kk,iOp,handle
!     And initialize Ltilde to zeros on the diagonal in CSR format
!     So that we have a csr matrix to add to
      do ii=1,nstate2
         L(ii)     = 0.0
         iL(ii)    = ii
         jL(ii)    = ii
      enddo
      iL(nstate2+1)   = nstate2+1

      do ii=1,nstate
         eye(ii) = 1.0
         i_eye(ii) = ii
         j_eye(ii) = ii
      enddo
      i_eye(nstate+1) = nstate+1
      
!     During setup, we previously stored the lindblad handles as:
!     (x)_i = lin_mxm_op(1,i)
!     (xt x)_i = lin_mxm_op(2,i)

!     Loop over lindblad operators
      do iOp=1,numberOfLinOps
        handle = lin_mxm_op(2,iOp)
!     Do A = I cross at a
         call quantum_csr_kron(eye,j_eye,i_eye,nstate,nstate,
     $       op_csr_a(1,handle),op_csr_ja(1,handle),
     $       op_csr_ia(1,handle),nstate,nstate,a,ja,ia)

!     Do B = at a cross I
         call quantum_csr_kron(op_csr_a(1,handle),op_csr_ja(1,handle),
     $        op_csr_ia(1,handle),nstate,nstate,
     $        eye,j_eye,i_eye,nstate,nstate,b,jb,ib)

!     Add a and b into c
         call aplb(nstate2,nstate2,1,a,ja,ia,b,jb,ib,c,jc,ic,
     $        nstate2*nstate,iw,ierr)

!     Add c and L_tilde, store in b (just tmp storage, sense aplb overwrites)
         call aplb(nstate2,nstate2,1,c,jc,ic,L,jL,iL,b,jb,ib,
     $        nstate2*nstate,iw,ierr)

!     Copy b to L_tilde
         do ii=1,nstate2
            iL(ii) = ib(ii)
            do kk=ib(ii),ib(ii+1)
               jL(kk) = jb(kk)
               L(kk)  = b(kk)
            enddo
         enddo
         iL(nstate2+1) = ib(nstate2+1)
         
         handle = lin_mxm_op(1,iOp)
!     Now, do x cross x
         call quantum_csr_kron(op_csr_a(1,handle),op_csr_ja(1,handle),
     $        op_csr_ia(1,handle),nstate,nstate,
     $        op_csr_a(1,handle),op_csr_ja(1,handle),
     $        op_csr_ia(1,handle),nstate,nstate,c,jc,ic)

!     Multiply c = x cross x by -2
         do ii=1,nstate2
            do kk=ic(ii),ic(ii+1)
               c(kk)  = -2*c(kk)
            enddo
         enddo
         
!     Add c and L_tilde 
         call aplb(nstate2,nstate2,1,c,jc,ic,L,jL,iL,b,jb,ib,
     $        nstate2*nstate,iw,ierr)
        
!     Copy b to L_tilde and multiply by the scalar
         do ii=1,nstate2
            iL(ii) = ib(ii)
            do kk=ib(ii),ib(ii+1)
               jL(kk) = jb(kk)
               L(kk)  = lin_mxm_a(iOp)*b(kk)
            enddo
         enddo
         iL(nstate2+1) = ib(nstate2+1)
      enddo
      
c$$$      call csrdns(nstate2,nstate2,L,
c$$$     $     jL,iL,op,nstate2,ierr)
c$$$
c$$$      do ii=1,nstate2
c$$$         write(20,'(300g15.5)') (op(ii,jj), jj=1,nstate2)
c$$$      enddo
      

      
!     Convert to COO format, in place
      call csrcoo(nstate2,1,nstate2*nstate,L,jL,iL,nnz_c,L,iL,jL,ierr)
!     Now write the coo matrix to file
      if(nid.eq.0) then
         open(unit=20,file='l_tilde')
         !count real nnz
         nnz_b = 0
         do ii=1,nnz_c
            if(L(ii).ne.0.0) then
               nnz_b = nnz_b + 1
            endif
         enddo
         write(20,*)nnz_b
         do ii=1,nnz_c
            !We subtract 1 to count from 0 for C code
            if(L(ii).ne.0.0) then
               write(20,'(g35.15,2i9)') L(ii),jL(ii)-1,iL(ii)-1
            endif
         enddo
         close(unit=20)
      endif
      
      end
!************************************************************
!     General tensor product of 2 CSR matrices
!     C = kron(A,B)
!     Input:
!            a, ja, ia, nrowa, ncola - A matrix in CSR format
!            b, jb, ib, nrowb, ncolb - B matrix in CSR format
!     Output:
!            c, ic, jc - C matrix in CSR format
!            nnz_c     - nonzeros in C
!************************************************************
      subroutine quantum_csr_kron(a,ja,ia,nrowa,ncola,b,jb,ib,
     $     nrowb,ncolb,c,jc,ic)

      implicit none
      integer ia(*),ja(*),ib(*),jb(*),ic(*),jc(*),iwk(nrowa*nrowb)
      integer nrowa,ncola,nrowb,ncolb
      real    a(*),b(*),c(*)
      integer ii,jj,ka,ll,mm,kb,nnz_c
      real atmp
      
      nnz_c = 0
!Loop over the A matrix
      do ii=1,nrowa
         do ka=ia(ii),ia(ii+1)-1
            jj = ja(ka)
            atmp = a(ka)
!     The i,j element of A is nonzero - multiply it by the block of B
!     Loop over B matrix
            do ll=1,nrowb
               do kb=ib(ll),ib(ll+1)-1
                  mm = jb(kb)
!     We now take the ii,jj element of A and multiply by
!     the ll,mm element of B (both are nonzero, by design)
!     The location in C is:
!                 nrowb*(ii-1)+1+(ll-1) , ncolb*(jj-1)+1+(mm-1)
                  nnz_c = nnz_c + 1
                  c(nnz_c) = atmp * b(kb)
                  ic(nnz_c) = nrowb*(ii-1) + ll
                  jc(nnz_c) = ncolb*(jj-1) +  mm
               enddo
            enddo
         enddo
      enddo

!Convert to CSR matrix
      call coicsr(nrowa*nrowb,nnz_c,1,c,jc,ic,iwk)

      

      end

      



!************************************************************
      subroutine quantum_csr_getdia (nrow,ncol,a,ja,ia,diag)

c*********************************************************************72
c
cc GETDIA extracts a given diagonal from a matrix stored in CSR format.
c on entry:
c
c
c nrow      = integer. The row dimension of the matrix A.
c ncol      = integer. The column dimension of the matrix A.
c a,ja,
c    ia = matrix stored in compressed sparse row a,ja,ia,format
c on return:
c
c len   = number of nonzero elements found in diag.
c         (len .le. min(nrow,ncol-ioff)-max(1,1-ioff) + 1 )
c
c diag  = double precision array of length nrow containing the wanted diagonal.
c        diag contains the diagonal (a(i,j),j-i = ioff ) as defined
c         above.
c
c           Y. Saad, Sep. 21 1989 - modified and tested May 9, 1990.
c
      implicit double precision (a-h,o-z)
      double precision diag(*),a(*)
      integer nrow, ncol, job, ia(*), ja(*)
      integer istart, max, iend, i, k

      istart = max(0,-ioff)
      iend = min0(nrow,ncol-ioff)

      do i=1,nrow
         diag(i) = 0.0
      enddo
c
c     extract  diagonal elements
c
      do i=istart+1, iend
         do k= ia(i),ia(i+1) -1
            if (ja(k)-i .eq. ioff) then
               diag(i)= a(k)
               exit
            end if
         enddo
      enddo


      return
      end

c*********************************************************************72
c
cc quantum_add_csr performs the CSR matrix sum C = A + B.
c
c on entry:
c
c nrow      = integer. The row dimension of A and B
c ncol  = integer. The column dimension of A and B.
c job   = integer. Job indicator. When job = 0, matrices A and B
c                  will be changed on return to match jc,ic
c
c a,
c ja,
c ia   = Matrix A in compressed sparse row format.
c
c b,
c jb,
c ib      =  Matrix B in compressed sparse row format.
c
c nzmax      = integer. The  length of the arrays c and jc.
c         amub will stop if the result matrix C  has a number
c         of elements that exceeds exceeds nzmax. See ierr.
c
c on return:
c
c c,
c jc,
c ic      = resulting matrix C in compressed sparse row sparse format.
c
c ierr      = integer. serving as error message.
c         ierr = 0 means normal return,
c         ierr .gt. 0 means that amub stopped while computing the
c         i-th row  of C with i=ierr, because the number
c         of elements in C exceeds nzmax.
c
c work arrays:
c
c iw      = integer work array of length equal to the number of
c         columns in A.
c
      subroutine quantum_add_csr (nrow,ncol,job,a,ja,ia,b,jb,ib,
     *     c,jc,ic,nzmax)

      double precision a(*), b(*), c(*),d(nzmax),e(nzmax)
      integer ja(*),jb(*),jc(*),ia(nrow+1),ib(nrow+1),ic(nrow+1),
     *     iw(nrow+1),jd(nzmax),id(nrow+1)
      logical values

      values = (job .ne. 0)
      ierr = 0
      len = 0
      id(1) = 1
      do 1 j=1, ncol
         iw(j) = 0
 1           continue

      do 500 ii=1, nrow
c     row i
         do 200 ka=ia(ii), ia(ii+1)-1
            len = len+1
            jcol    = ja(ka)
            if (len .gt. nzmax) then
               print*,'len ',len,' nzmax ',nzmax
               print*,'ERROR in ADD CSR - LEN GT NZMAX'
               stop
            endif
            jd(len) = jcol
            d(len)  = a(ka)
            e(len)  = 0.0
            
            iw(jcol)= len
 200             continue

         do 300 kb=ib(ii),ib(ii+1)-1
            jcol = jb(kb)
            jpos = iw(jcol)

            if (jpos .eq. 0) then
               len = len+1
               if (len .gt.nzmax) then
                  print*,'ERROR in ADD CSR - LEN GT NZMAX2'
                  stop
               endif
               jd(len) = jcol
               if (values) then
                  d(len)  = b(kb)
               else
                  d(len)  = 0.0
                  e(len) = b(kb)
               endif
               iw(jcol)= len
            else
               if (values) then
                  d(jpos) = d(jpos) + b(kb)
               else
                  e(jpos) = b(kb)
               endif
            end if
 300             continue
         do 301 k=id(ii), len
          iw(jd(k)) = 0
 301           continue
         id(ii+1) = len+1
 500       continue
           !copy results into output vector
           do ii=1,len
              if(values) then 
                 c(ii) = d(ii)
              else
                 a(ii) = d(ii)
                 b(ii) = e(ii)
              endif
              jc(ii) = jd(ii)
           enddo
           do ii = 1,nrow+1
              ic(ii) = id(ii)
           enddo
           nzmax = len
      return

      end

c*********************************************************************72
c    quantum_add_csr_ham does the final ham addition - allowing 
c    ham_i and ham_d to be padded with 0's 
c
      subroutine quantum_add_csr_ham(nrow,ncol,job,a,ja,ia,b,jb,ib,
     *     c,jc,ic,nzmax)

      double precision a(*), b(*), c(*),d(nzmax)
      integer ja(*),jb(*),jc(*),ia(nrow+1),ib(nrow+1),ic(nrow+1),
     *     iw(nrow+1),jd(nzmax),id(nrow+1)
      logical values

      values = (job .ne. 0)
      ierr = 0
      len = 0
      id(1) = 1
      do 1 j=1, ncol
         iw(j) = 0
 1           continue

      do 500 ii=1, nrow
c     row i
         do 200 ka=ia(ii), ia(ii+1)-1
            len = len+1
            jcol    = ja(ka)
            if (len .gt. nzmax) then
               print*,'ERROR in ADD CSR_HAM - LEN GT NZMAX'
               stop
            endif
            jd(len) = jcol
            if (values) d(len)  = a(ka)
            iw(jcol)= len
 200             continue

         do 300 kb=ib(ii),ib(ii+1)-1
            jcol = jb(kb)
            jpos = iw(jcol)

            if (jpos .eq. 0) then
               len = len+1
               if (len .gt.nzmax) then
                  print*,'ERROR in ADD CSR_HAM2 - LEN GT NZMAX2'
                  stop
               endif
               jd(len) = jcol
               if (values) d(len)  = b(kb)
               iw(jcol)= len
            else
               if (values) d(jpos) = d(jpos) + b(kb)
            end if
 300             continue
         do 301 k=id(ii), len
          iw(jd(k)) = 0
 301           continue
         id(ii+1) = len+1
 500       continue
           !copy results into output vector
           do ii=1,len
              c(ii) = d(ii)
              jc(ii) = jd(ii)
           enddo
           do ii = 1,nrow+1
              ic(ii) = id(ii)
           enddo
           nzmax = len
      return

      end

c*********************************************************************72
      subroutine csrcoo (nrow,job,nzmax,a,ja,ia,nnz,ao,ir,jc,ierr)

c*********************************************************************72
c
cc CSRCOO converts Compressed Sparse Row to Coordinate format.
c
c converts a matrix that is stored in coordinate format
c  a, ir, jc into a row general sparse ao, jao, iao format.
c
c on entry:
c
c nrow      = dimension of the matrix.
c job   = integer serving as a job indicator.
c         if job = 1 fill in only the array ir, ignore jc, and ao.
c         if job = 2 fill in ir, and jc but not ao
c         if job = 3 fill in everything.
c         The reason why these options are provided is that on return
c         ao and jc are the same as a, ja. So when job = 3, a and ja are
c         simply copied into ao, jc.  When job=2, only jc and ir are
c         returned. With job=1 only the array ir is returned. Moreover,
c         the algorithm is in place:
c     call csrcoo (nrow,1,nzmax,a,ja,ia,nnz,a,ia,ja,ierr)
c         will write the output matrix in coordinate format on a, ja,ia.
c         (Important: note the order in the output arrays a, ja, ia. )
c         i.e., ao can be the same as a, ir can be the same as ia
c         and jc can be the same as ja.
c
c a,
c ja,
c ia    = matrix in compressed sparse row format.
c nzmax = length of space available in ao, ir, jc.
c         the code will stop immediatly if the number of
c         nonzero elements found in input matrix exceeds nzmax.
c
c on return:
c
c ao, ir, jc = matrix in coordinate format.
c
c nnz        = number of nonzero elements in matrix.
c ierr       = integer error indicator.
c         ierr .eq. 0 means normal retur
c         ierr .eq. 1 means that the the code stopped
c         because there was no space in ao, ir, jc
c         (according to the value of  nzmax).
c
      double precision a(*),ao(*)
      integer ir(*),jc(*),ja(*),ia(*)

      ierr = 0
      nnz = ia(nrow+1)-1
      if (nnz .gt. nzmax) then
         ierr = 1
         return
      end if

      goto (3,2,1) job
 1    do 10 k=1,nnz
         ao(k) = a(k)
 10   continue
 2    do 11 k=1,nnz
         jc(k) = ja(k)
 11   continue
c copy backward to allow
 3    do 13 i=nrow,1,-1
         k1 = ia(i+1)-1
         k2 = ia(i)
         do 12 k=k1,k2,-1
            ir(k) = i
 12      continue
 13   continue
      return
      end
        
c*********************************************************************72
c
cc CSRCSC converts Compressed Sparse Row to Compressed Sparse Column.
c
c (transposition operation)   Not in place.
c
c on entry:
c
c n      = dimension of A.
c job      = integer to indicate whether or not to fill the values of the
c         matrix ao or only the pattern (ia, and ja). Enter 1 for yes.
c ipos  = starting position in ao, jao of the transposed matrix.
c         the iao array takes this into account (thus iao(1) is set to ipos.)
c         Note: this may be useful if one needs to append the data structure
c         of the transpose to that of A. In this case use
c                call csrcsc (n,1,n+2,a,ja,ia,a,ja,ia(n+2))
c        for any other normal usage, enter ipos=1.
c a      = double precision array of length nnz (nnz=number of nonzero elements in input
c         matrix) containing the nonzero elements.
c ja      = integer array of length nnz containing the column positions
c         of the corresponding elements in a.
c ia      = integer of size n+1. ia(k) contains the position in a, ja of
c        the beginning of the k-th row.
c
c on return:
c
c output arguments:
c ao      = double precision array of size nzz containing the "a" part of the transpose
c jao      = integer array of size nnz containing the column indices.
c iao      = integer array of size n+1 containing the "ia" index array of
c        the transpose.
c
      subroutine csrcsc(n,job,ipos,a,ja,ia,ao,jao,iao)
      integer ia(n+1),iao(n+1),ja(*),jao(*)
      double precision  a(*),ao(*)
c
c  compute lengths of rows of transp(A)
      do 1 i=1,n+1
         iao(i) = 0
 1           continue
      do 3 i=1, n
         do 2 k=ia(i), ia(i+1)-1
            j = ja(k)+1
            iao(j) = iao(j)+1
 2                 continue
 3                     continue
c compute pointers from lengths
      iao(1) = ipos
      do 4 i=1,n
         iao(i+1) = iao(i) + iao(i+1)
 4           continue
c  now do the actual copying
      do 6 i=1,n
         do 62 k=ia(i),ia(i+1)-1
            j = ja(k)
            next = iao(j)
            if (job .eq. 1)  ao(next) = a(k)
            jao(next) = i
            iao(j) = next+1
 62               continue
 6                    continue
c  reshift iao and leave
      do 7 i=n,1,-1
         iao(i+1) = iao(i)
 7           continue
      iao(1) = ipos
      end


!**********************************************************************
      subroutine cem_quantum3_op_mxm(rrho,tmprr,handle)
      implicit none
      include 'SIZE'
      include 'QUANT'

      integer handle,ii,jj,ierr
      real*8  rrho (maxN,maxN/lp)
      real*8  tmprr(maxN,maxN/lp)

      if(handle.lt.0.or.handle.gt.newHandleNumber) then
         print*,handle,' is not a valid handle for operator.'
         return
      endif

      call cem_quantum3_csr_mxm(rrho,tmprr,op_csr_a(1,handle),
     $     op_csr_ia(1,handle),op_csr_ja(1,handle))

      return
      end

c---------------------------------------------------------------------
      subroutine cem_quantum3_csr_mxm(rrho,tmprr,mat_a,mat_ia,mat_ja)
c---------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'QUANT'


      !Maybe give these definite sizeps?
      integer ii,jj,kk,istart,iend,mat_ia(*),mat_ja(*)
      real    sum
      real    mat_a(*)
      real*8  rrho (maxN,maxN/lp)
      real*8  tmprr(maxN,maxN/lp)

!$acc data present(ham_ia,ham_ja,ham_a,rrho,tmprr)
!$acc parallel loop private(istart,iend)
      do jj=1,nstate_np
!$acc loop vector private(sum)
      do ii=1,nstate
         istart = mat_ia(ii)
         iend   = mat_ia(ii+1)-1
         sum = 0
         do kk=istart,iend
            sum = sum + mat_a(kk)*rrho(mat_ja(kk),jj)
         enddo
         tmprr(ii,jj) = sum
      enddo
      enddo
!$acc end data

      return
      end

c---------------------------------------------------------------------
      subroutine cem_quantum3_csr_mxm_complex(rrho,tmprr,
     $     mat_a,mat_ia,mat_ja)
c---------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'QUANT'


      !Maybe give these definite sizeps?
      integer ii,jj,kk,istart,iend,mat_ia(*),mat_ja(*)
      complex sum,mult,ci
      complex mat_a(*)      
      complex rrho (maxN,maxN/lp)
      complex tmprr(maxN,maxN/lp)

!     $acc data present(ham_ia,ham_ja,ham_a,rrho,tmprr)
!     $acc parallel loop private(istart,iend)
         do jj=1,nstate_np
!     $acc loop vector private(sum)
            do ii=1,nstate
               istart = mat_ia(ii)
               iend   = mat_ia(ii+1)-1
               sum = 0
               do kk=istart,iend
                  sum = sum + mat_a(kk)*rrho(mat_ja(kk),jj)
               enddo
               tmprr(ii,jj) = sum
            enddo
         enddo
!$acc end data


      return
      end


!************************************************************************

      subroutine quant_dnscsr(dns_i,dns_d,a_i,a_d,ja,ia,nnz)

c*********************************************************************
c
cc DNSCSR converts Dense to Compressed Row Sparse format.
c
c converts a densely stored matrix into a row orientied
c compactly sparse matrix. ( reverse of csrdns )
c Note: this routine does not check whether an element
c is small. It considers that a(i,j) is zero if it is exactly
c equal to zero: see test below.
c
c on entry:
c
c
c nrow      = row-dimension of a
c ncol      = column dimension of a
c nzmax = maximum number of nonzero elements allowed. This
c         should be set to be the lengths of the arrays a and ja.
c dns   = input nrow x ncol (dense) matrix.
c ndns      = first dimension of dns.
c
c on return:
c
c
c a, ja, ia = value, column, pointer  arrays for output matrix
c
c ierr      = integer error indicator:
c         ierr .eq. 0 means normal retur
c         ierr .eq. i means that the the code stopped while
c         processing row number i, because there was no space left in
c         a, and ja (as defined by parameter nzmax).
c     
      implicit none
      include 'SIZE'
      include 'QUANT'
      real dns_i(maxN,maxN),dns_d(maxN,maxN),a_i(*),a_d(*)
      integer ia(*),ja(*),nnz,next,ii,jj

      next = 1
      nnz  = 0
      ia(1) = 1
      do 4 ii=1,nstate
         do 3 jj=1,nstate
            if (dns_i(ii,jj)+dns_d(ii,jj) .eq. 0.0) goto 3
            if (next .gt. maxN*maxN/lp) then
               write(6,*) 'not enough space in ham!'
               print*,ii
               stop
            end if
            ja(next) = jj
            a_i(next) = dns_i(ii,jj)
            a_d(next) = dns_d(ii,jj)
            nnz = nnz+1
            next = next+1
 3       continue
         ia(ii+1) = next
 4    continue
      
      return
      end


!*********************************************************
      subroutine p_dns_to_csr(col_i,col_d,a_i,a_d,ja,ia,nnz)
      implicit none
      include 'SIZE'
      include 'QUANT'
#ifdef MPI
      include 'mpif.h'
#endif
      real col_i(maxN,maxN),col_d(maxN,maxN),a_i(*),a_d(*)
      real col_i_l,col_d_l
      complex sndBufC(maxN,maxN),rcvBufC,ci
      integer ja(*),ia(*),nnz,ierr,next,targetCore,kk,jj,ii
      integer jGlo,srequest,tag,rrequest,status
      ci=(0.0,1.0)

#ifdef MPI
      if(nid.ne.0) then
         kk = 0
         do jj=1,nstate/np
            do ii=1,nstate
               kk   = kk + 1
               jGlo = jj+c_offset
               sndBufC(ii,jj) = col_i(ii,jj) + ci * col_d(ii,jj)
               tag      = nstate*(jGlo-1)+ii
               !Complex numbers allow me to send 2 real messages with the same tag
               !I have to do the matrix because MPI only looks at the LAST thing in memory when
               !IRECV is called, not what was in memory when ISEND was posted
               call MPI_ISEND(sndBufC(ii,jj),1,MPI_COMPLEX16,0,tag,
     $              MPI_COMM_WORLD,srequest,IERR)
               call MPI_REQUEST_FREE(srequest,ierr)
            enddo
         enddo
      else
#endif
         ierr = 0
         nnz  = 0
         next = 1
         ia(1) = 1
         do 4 ii=1,nstate
            do 3 jj=1,nstate
#ifdef MPI
               targetCore=(jj-1)/(nstate/np)
               if(targetCore.ne.0) then 
                  tag=nstate*(jj-1)+ii
!Complex numbers allow me to receive 2 real messages with the same tag
!                  call MPI_RECV(rcvBufC,1,MPI_COMPLEX16,targetCore,
!     $                 tag,MPI_COMM_WORLD,IERR)
                  call MPI_IRECV(rcvBufC,1,MPI_COMPLEX16,targetCore,
     $                 tag,MPI_COMM_WORLD,rrequest,IERR)
                  call MPI_WAIT(rrequest,MPI_STATUS_IGNORE)
                  col_i_l = real(rcvBufC)
                  col_d_l = imag(rcvBufC)
               else
#endif
                  col_i_l=col_i(ii,jj)
                  col_d_l=col_d(ii,jj)
#ifdef MPI
               endif
#endif
               if (col_i_l+col_d_l.eq.0.0) goto 3
               if (next.gt.4000) then !Temporary: level=nzmax
                  ierr = ii
                  write(6,*) 'not enough space in ham!'
                  stop
                  return
               end if
               ja(next) = jj
               a_i(next) = col_i_l
               a_d(next) = col_d_l
               nnz  = nnz+1
               next = next+1
 3          continue
            ia(ii+1) = next
 4       continue
#ifdef MPI
      endif
#endif
      end

      subroutine csrdns(nrow,ncol,a,ja,ia,dns,ndns,ierr)

c*********************************************************************72
c
cc CSRDNS converts Compressed Sparse Row to Dense format.
c
c converts a row-stored sparse matrix into a densely stored one
c
c On entry:
c
c
c nrow      = row-dimension of a
c ncol      = column dimension of a
c a,
c ja,
c ia    = input matrix in compressed sparse row format.
c         (a=value array, ja=column array, ia=pointer array)
c dns   = array where to store dense matrix
c ndns      = first dimension of array dns
c
c on return:
c
c dns   = the sparse matrix a, ja, ia has been stored in dns(ndns,*)
c
c ierr  = integer error indicator.
c         ierr .eq. 0  means normal return
c         ierr .eq. i  means that the code has stopped when processing
c         row number i, because it found a column number .gt. ncol.
c
      double precision dns(ndns,*),a(*)
      integer ja(*),ia(*)
      
      ierr = 0
      do 1 i=1, nrow
         do 2 j=1,ncol
          dns(i,j) = 0.0
 2     continue
 1    continue
      
      do 4 i=1,nrow
         do 3 k=ia(i),ia(i+1)-1
            j = ja(k)
            if (j .gt. ncol) then
               ierr = i
               print*,'ERROR IN CSRDNS'
               print*,'j greater than ncol!'
               stop
               return
            end if
            dns(i,j) = a(k)
 3       continue
 4    continue

      return
      end
      

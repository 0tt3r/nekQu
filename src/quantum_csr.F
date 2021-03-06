c-------------------------------------------
c csr mxm
c Based off of SparseKit, http://people.sc.fsu.edu/~jburkardt/f77_src/sparsekit/sparsekit.f
c I don't know the licensing information
c and Mills Vectorized sparse mxm for csr
c-------------------------------------------


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
               print*,'ERROR in ADD CSR'
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
                  print*,'ERROR in ADD CSR'
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
               print*,'ERROR in ADD CSR'
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
                  print*,'ERROR in ADD CSR'
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

c---------------------------------------------------------------------
      subroutine cem_quantum3_csr_mxm(rrho,tmprr,mat_a,mat_ia,mat_ja)
c---------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'QUANT'


      !Maybe give these definite sizes?
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
               stop
               return
            end if
            dns(i,j) = a(k)
 3       continue
 4    continue

      return
      end
      

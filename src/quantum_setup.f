
      subroutine quantum_setup_initialize()
      implicit none
      include 'SIZEQ'
      include 'QUANT'


      newHandleNumber = 1
      quantN          = 1
      numberOfOps     = 0
      nTimeDep        = 0
      numberOfLinOps  = 0

      end

!**************************************************************

      subroutine quantum_create_op(qlevel,handle,handle_t,handle_m)
      implicit none
      include 'SIZEQ'
      include 'QUANT'

      integer qlevel
      integer handle,handle_t
      integer ii,jj
      
!     Be sure to initialize newHandlenumber,N
      handle          = newHandleNumber
      newHandleNumber = newHandleNumber+1
      handle_t        = newHandleNumber
      newHandleNumber = newHandleNumber+1
      handle_m        = newHandleNumber
      newHandleNumber = newHandleNumber+1
      numberOfOps = numberOps+1
      
!     oldN is how many times we need to repeat this ops levels
!     It represents the previous total number of levels
      oldN(handle)   = quantN
      oldN(handle_t) = quantN
      quantN = quantN*qlevel
      do ii = 1,qlevel
         do jj = 1,oldN
            kk = jj + ii - 1
            state(numberOfOps,kk) = ii-1
         enddo
      enddo

      end


!*******************************************************************
      subroutine quantum_finalize_op()
      implicit none
      include 'SIZEQ'
      include 'QUANT'


      integer ii,jj,tilingNumber,maxLevel,curState,lBound,uBound
      real    value

      nstate   = quantN
      nstate_p = nstate*nstate/np
      c_offset = nid*nstate/np



      !Now that we have sizes, 0 out operators
      !Zero out operator - maybe move to init?
      do ii=1,numberOfOps
         do jj=1,nstate/np
            do kk=1,nstate
               operator(ii,kk,jj) = 0.0
            enddo
         enddo
      enddo

      do jj=1,nstate/np
         do ii=1,nstate
            lin_comb_op(ii,jj) = 0.0
            ham_indep(ii,jj)   = 0.0
         enddo
      enddo

      do kk=1,max_time_dep
         do jj=1,nstate/np
            do ii=1,nstate
               ham_dep(kk,ii,jj) = 0.0
            enddo
         enddo
      enddo

      lBound   = nstate/np*nid+1
      uBound   = nstate/np*(nid+1)
         
      !Construct our matrices
      oldN(newHandleNumber) = nstate
      do ii=1,numberOfOps
         maxLevel     = levels(ii)
         tilingNumber = oldN(ii+1)
         do jj=1,nstate
            curState = state(ii,mod(jj,tilingNumber)
            !Every core goes through the whole list, but a core only
            !saves a value if that value belongs to it
            if((jj.ge.lBound).and.(jj.le.uBound)) then
               if(jj.eq.ii) then
                  operator(ii+2,jj,jj) = value
               endif
            endif

            if(curState.lt.maxLevel-1) then
               value = sqrt(real(curState+1))

               if((jj.ge.lBound).and.(jj.le.uBound)) then
                  operator(ii+1,jj+oldN(ii),jj) = value
                  if(jj.eq.ii) then
                     operator(ii+2,jj,jj) = value
                  endif
               endif

               if((jj+oldN(ii).ge.lBound).and.(jj+oldN(ii).le.uBound)) 
     $              then
                  operator(ii,jj,jj+oldN(ii)) = value
               endif
            endif
         enddo
      enddo

      end

!*********************************************************************
      subroutine quantum_combine_op(handle1,handle2,operation,handle3)
      implicit none
      include 'SIZEQ'
      include 'QUANT'

      integer operation,handle1,handle2,handle3,ii,jj

      handle3 = newHandleNumber
      newHandleNumber = newHandleNumber+1
!     matAdd would be a local operation
!     matMult involves communication
      if(operation.eq.0) then 
         do jj=1,nstate/np
            do ii=1,nstate
               operator(handle3,ii,jj) = operator(handle1,ii,jj) + 
     $              operator(handle2,ii,jj)
            enddo
         enddo
      endif

      if(operation.eq.1) call matMult(handle1,handle2,handle3)
      
      end

!******************************************************************
      subroutine matMult(handle1,handle2,handle3)
      implicit none
      include 'SIZEQ'
      include 'QUANT'
      include 'mpif.h'

      integer ii,jj,jGlo,target_core,lBound,uBound
      integer srequest,ierr,rrequest
      !May need to change srequest
      real    recvBuf

      !First, post isends
      do jj=1,nstate/np
         do ii=1,nstate
            jGlo = jj+c_offset
            target_core = (ii-1)/(nstate/np)
          
            call MPI_ISEND(operator(handle1,ii,jj),1,MPI_REAL8,
     $           target_core,nstate*(jGlo-1)+ii,MPI_COMM_WORLD,srequest)
         enddo
      enddo
     
      lBound = nstate/np*nid+1
      uBound = nstate/np*(nid+1)

      !Do mxm
      do jj=1,nstate/np
         do ii=1,nstate
            do kk=1,nstate
               !local data
               if((kk.ge.lBound).and.(kk.le.uBound)) then
                  operator(handle3,ii,jj) = operator(handle3,ii,jj) +
     $                 operator(handle1,ii,kk)*operator(handle2,kk,jj)
               else
               !Nonlocal Data
                  target_core = (kk-1)/(nstate/np)
                  CALL MPI_IRECV(recvBuf,1,MPI_REAL8,target_core,
     $                 nstate*(kk-1)+ii,MPI_COMM_WORLD,rrequest)
                  CALL MPI_WAIT(rrequest,MPI_STATUS_IGNORE)
                  operator(handle3,ii,jj) = operator(handle3,ii,jj) +
     $                 recvBuf * operator(handle2,kk,jj)
               endif
            enddo
         enddo
      enddo                  
      
      end

!**********************************************************************
      subroutine quantum_add_to_ham(handle,scalar,timeDep)
      implicit none
      include 'SIZEQ'
      include 'QUANT'
      
      integer ii,jj,timeDep,handle
      real    scalar
            
      do jj=1,nstate/np
         do ii=1,nstate
            if(timeDependence.eq.0) then
               ham_indep(ii,jj) = ham_indep(ii,jj) + 
     $              scalar*operator(handle,ii,jj)
            else
               ham_dep(timeDep,ii,jj) = ham_dep(timeDep,ii,jj)+
     $              scalar*operator(handle,ii,jj)
               nTimeDep = nTimeDep+1
            endif
         enddo
      enddo
            
      end

!**********************************************************************
      subroutine quantum_ham_finalize()
      implicit none
      include 'SIZEQ'
      include 'QUANT'
      include 'mpif.h'

      integer ii,jj,IERR
      real    col_i(lstate,lstate/lp),col_j(lstate,lstate/lp)
      
      !Post ISENDS
      do jj=1,nstate/np
         do ii=1,nstate
            col_i(ii,jj) = ham_indep(ii,jj)
            col_d(ii,jj) = 0
            do kk = 1,nTimeDep
               col_d(ii,jj) = col_d(ii,jj)+ ham_dep(kk,ii,jj)
            enddo
         enddo
      enddo

      call p_dns_to_csr(col_i,col_d,ham_i,ham_d,ham_ja,ham_ia,ham_nnz)
      
      !BCAST to everyone
      call MPI_BCAST(ham_i,mat_a_max,MPI_REAL8,0,MPI_COMM_WORLD,IERR)
      call MPI_BCAST(ham_d,mat_a_max,MPI_REAL8,0,MPI_COMM_WORLD,IERR)
      call MPI_BCAST(ham_ja,mat_ja_max,MPI_INTEGER,0,
     $     MPI_COMM_WORLD,IERR)
      call MPI_BCAST(ham_ia,mat_ia_max,MPI_INTEGER,0,
     $     MPI_COMM_WORLD,IERR)
      call MPI_BCAST(ham_nnz,1,MPI_INTEGER,0,MPI_COMM_WORLD,IERR)

      end



!*********************************************************
      subroutine p_dns_to_csr(col_i,col_d,a_i,a_d,ja,ia,nnz)
      implicit none
      include 'SIZEQ'
      include 'QUANT'
      include 'mpif.h'

      real col_i(*,*),col_d(*,*),a_i(*),a_d(*)
      real sndBuf(2),rcvBuf(2),col_i_l,col_d_l
      integer ja(*),ia(*),nnz,ierr,next,currentCore,kk,jj,ii
      integer jGlo,srequest,tag,rrequest
      
      if(nid.ne.0) then
         do jj=1,nstate/np
            do ii=1,nstate
               jGlo = jj+c_offset
               sndBuf(1) = col_i(ii,jj)
               sndBuf(2) = col_d(ii,jj)
               tag      = nstate*(jGlo-1)+ii
               call MPI_ISEND(sndBuf,2,MPI_REAL8,0,tag,
     $              MPI_COMM_WORLD,srequest,IERR)
            enddo
         enddo
      else
         ierr = 0
         next = 1
         ia(1) = 1
         currentCore = 0
         do 4 ii=1,nstate
            do 3 jj=1,nstate
               targetCore=(jj-1)/(nstate/np)
               if(targetCore.ne.0) then 
                  tag=nstate*(jj-1)+ii
                  call MPI_IRECV(rcvBuf,2,MPI_REAL8,targetCore,
     $                 tag,MPI_COMM_WORLD,rrequest,IERR)
                  call MPI_WAIT(rrequest,MPI_STATUS_IGNORE,IERR)
                  col_i_l = rcvBuf(1)
                  col_d_l = rcvBuf(2)
               else
                  col_i_l=col_i(ii,jj)
                  col_d_l=col_d(ii,jj)
               endif
               if (col_i_l+col_d_l.eq.0.0) goto 3
               if (next.gt.level) then !Temporary: level=nzmax
                  ierr = ii
                  return
               end if
               ja(next) = jGlo
               a_i(next) = col_i_l
               a_d(next) = col_d_l
               next = next+1
 3          continue
            ia(ii+1) = next
 4       continue

      endif
      
      !Is this barrier necessary?
      call MPI_BARRIER(MPI_COMM_WORLD,IERR)

      end


!***********************************************************
      subroutine quantum_lin_finalize()
      implicit none
      include 'SIZEQ'
      include 'QUANT'
      include 'mpif.h'

      integer ii,jj,kk,nnz
      real columns(lstate,lstate/lp),columns_0(lstate,lstate/lp),ierr

      do jj=1,nstate/np
         do ii=1,nstate
            columns(ii,jj) = lin_comb_op(ii,jj)
            columns_0(ii,jj) = 0
         enddo
      enddo
      
      call p_dns_to_csr(columns,columns_0,lin_comb_a,lin_comb_0,
     $     lin_comb_ja,lin_comb_ia,nnz)

!     BCAST to everyone
      call MPI_BCAST(lin_comb_a,mat_a_max,MPI_REAL8,0,
     $     MPI_COMM_WORLD,IERR)
      call MPI_BCAST(lin_comb_ja,mat_ja_max,MPI_INTEGER,0,
     $     MPI_COMM_WORLD,IERR)
      call MPI_BCAST(lin_comb_ia,mat_ia_max,MPI_INTEGER,0,
     $     MPI_COMM_WORLD,IERR)
      
      do kk=1,numberOfLinOps
         do jj=1,nstate/np
            do ii=1,nstate
               columns(ii,jj) = lin_one_op(kk,ii,jj)
            enddo
         enddo
         
         call p_dns_to_csr(columns,columns_0,lin_tmp_a,lin_comb_0,
     $        lin_tmp_ja,lin_tmp_ia,nnz)
         
         if(nid.eq.0) then
            do jj=1,nnz
               lin_one_ia(kk,jj) = lin_tmp_ia(jj)
            enddo
            
            do jj=1,nnz
               lin_one_ja(kk,jj) = lin_tmp_ja(jj)
            enddo

            do jj=1,nnz
               lin_one_a(kk,jj) = lin_tmp_a(jj)
            enddo
            lin_one_nnz(kk) = nnz
         endif
      enddo

      call MPI_BCAST(lin_one_a,numberOfLinOps*nnz,
     $     MPI_REAL8,0,MPI_COMM_WORLD,IERR)
      call MPI_BCAST(lin_one_ja,numberOfLinOps*nnz,MPI_INTEGER,0,
     $     MPI_COMM_WORLD,IERR)
      call MPI_BCAST(lin_one_ia,numberOfLinOps*nnz,MPI_INTEGER,0,
     $     MPI_COMM_WORLD,IERR)
      call MPI_BCAST(lin_one_nnz,numberOfLinOps,MPI_INTEGER,0,
     $     MPI_COMM_WORLD,IERR)
         
      end


c-------------------------------------------
c csr mxm
c Based off of SparseKit, http://people.sc.fsu.edu/~jburkardt/f77_src/sparsekit/sparsekit.f
c and Mills Vectorized sparse mxm for csr
c-------------------------------------------
c---------------------------------------------------------------------
      subroutine cem_quantum3_csr_mxm(rrho,tmprr,mat_a,mat_ia,mat_ja)
c---------------------------------------------------------------------
      implicit none
      include 'SIZEQ'
      include 'QUANT'


      !Maybe give these definite sizes?
      integer ii,jj,kk,istart,iend,mat_ia(*),mat_ja(*)
      real    sum
      real    mat_a(*)
      real*8       rrho (lstate,lstate/lp)
      real*8       tmprr(lstate,lstate/lp),tmp_acc

!$acc data present(ham_ia,ham_ja,ham_a,rrho,tmprr)
!$acc parallel loop private(istart,iend)
      do jj=1,nstate/np
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

!************************************************************
      subroutine quantum_lin_add(handle1,handle2,scalar)
      implicit none
      include 'SIZEQ'
      include 'QUANT'

      integer jj,ii

      !be sure to set this to 1 initially!
      numberOfLinOps = numberOfLinOps + 1 
      do jj=1,nstate/np
         do ii=1,nstate
            lin_comb_op(ii,jj) = lin_comb_op(ii,jj) + 
     $           scalar*operator(handle1,ii,jj)*operator(handle2,ii,jj)
            lin_one_op(numberOfLinOps,ii,jj) = scalar*
     $           operator(handle1,ii,jj)
         enddo
      enddo

      end


!**************************************************************
      subroutine quantum_lin_gs_op()
      implicit none
      include 'SIZEQ'
      include 'QUANT'

      complex ci
      integer ii,jj,kk,j0

      ci = (0.0,1.0)

!     Calculate opdag*op * rho
      call cem_quantum3_csr_mxm(ii,rho_r,tmp_r1,lin_comb_a,lin_comb_ia
     $     ,lin_comb_ja)
      call cem_quantum3_csr_mxm(ii,rho_i,tmp_i1,lin_comb_a,lin_comb_ia
     $     ,lin_comb_ja)
      
      do jj = 1,nstate/np
         do kk = 1,nstate
            j0 = (jj-1)*nstate+kk
            tmprv(j0) = tmp_r1(kk,jj)
            tmpiv(j0) = tmp_i1(kk,jj)
         enddo
      enddo
      
      do ii=1,numberOfLinOps              
         do jj=1,lin_one_nnz(ii)
            mat_a(jj) = lin_one_a(ii,jj)
         enddo
         
         do jj=1,lin_one_n_ia(ii)
            mat_ia(jj) = lin_one_ia(ii,jj)
         enddo
         
         do jj=1,lin_one_n_ja(ii)
            mat_ja(jj) = lin_one_ja(ii,jj)
         enddo

!     calculate op*rho
         call cem_quantum3_csr_mxm(rho_r,tmp_r2,mat_a,mat_ia
     $        ,mat_ja)
         call cem_quantum3_csr_mxm(rho_i,tmp_i2,mat_a,mat_ia
     $        ,mat_ja)

         do jj = 1,nstate/np
            do kk = 1,nstate
               j0 = (jj-1)*nstate+kk+nstate_p*ii
               tmprv(j0) = tmp_r2(kk,jj)
               tmpiv(j0) = tmp_i2(kk,jj)
            enddo
         enddo

      enddo

!     Do all communication
!     Get opdag*op * rho + (opdag*op*rho)^dag
!     Get (op*rho)^dag
      call gs_op_fields(gs_handle_q_H,tmprv2,nstate_p,
     $     numberOfLinOps+1,1,1,0)
      call gs_op_fields(gs_handle_q_H,tmpiv2,nstate_p,
     $     numberOfLinOps+1,1,1,0)

!     sum opdag*op * rho + (opdag*op*rho)^dag parts
      do jj = 1,nstate/np
         do kk = 1,nstate
            j0 = (jj-1)*nstate+kk
!     opdag*op * rho + (opdag*op*rho)^dag parts                 
            if(kk.eq.jj+c_offset) then
               tmprr = 2*tmp_r1(kk,jj)
               tmpii = 0
            else
               tmpii = 2*tmp_i1(kk,jj)-tmpiv(j0)
               tmprr = tmprv(j0)
            endif
            drho(kk,jj)= drho(kk,jj) + 
     $           (tmprr + ci*tmpii)
         enddo
      enddo

      do ii = 1,numberOfLinOps
!     Unpack lin_one_op buffers
         do jj= 1,nstate/np
            do kk = 1,nstate
               j0 = (jj-1)*nstate+kk+nstate_p*ii
               !We want b, so we take a+b-a
               tmp_r1(kk,jj) = tmprv2(j0) - tmp_r2(kk,jj)
               !We want -b, so we tak a - (a+b)
               tmp_ii(kk,jj) = tmp_i1(kk,jj) - tmpiv2(j0)
            enddo
         enddo         

         do jj=1,lin_one_nnz(ii)
            mat_a(jj) = lin_one_a(ii,jj)
         enddo
         
         do jj=1,lin_one_n_ia(ii)
            mat_ia(jj) = lin_one_ia(ii,jj)
         enddo
         
         do jj=1,lin_one_n_ja(ii)
            mat_ja(jj) = lin_one_ja(ii,jj)
         enddo

!     calculate op*(op*rho)^dag
         call cem_quantum3_csr_mxm(tmp_r1,tmp_r2,mat_a,mat_ia
     $        ,mat_ja)
         call cem_quantum3_csr_mxm(tmp_i1,tmp_i2,mat_a,mat_ia
     $        ,mat_ja)

         !sum all parts
         do jj = 1,nstate/np
            do kk = 1,nstate
              !-2 * op * rho * opdag part
              tmprr = -2*tmp_r3(kk,jj)
              tmpii = -2*tmp_i3(kk,jj)
              
              drho(kk,jj)= drho(kk,jj) + 
     $             (tmprr + ci*tmpii)
           enddo
         enddo

      enddo

      end
      
      

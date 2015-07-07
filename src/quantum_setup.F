
      subroutine quantum_setup_initialize()
      implicit none
      include 'SIZE'
      include 'QUANT'

      integer ii,jj,kk
      current_gate     = 1
      number_of_gates  = 0
      generated_gates  = 0
      numObservables   = 0
      newHandleNumber  = 1
      quantN           = 1
      numberOfOps      = 0
      nTimeDep         = 0
      numberOfLinOps   = 0
      num_nl_lin_ops   = 0
      max_time_dep     = 2
      ifPulse          = .false.
      rwa              = .true.
      calc_concurrence = .false.
      pi               = 4.*ATAN(1.0)

      !Zero out the gates_table
      do ii=1,6
         do jj = 1,maxOp
            gates_table(ii,jj) = 0
         enddo
      enddo

      !zero out the cnot_table and crk_table
      do ii=1,maxOp
         do jj=1,maxOp
            cnot_table(ii,jj) = 0
            do kk=1,maxOp
               crk_table(ii,jj,kk)  = 0
            enddo
         enddo
      enddo

      end

!**************************************************************

      subroutine quantum_create_op(qlevel,handle,handle_t,handle_m)
      implicit none
      include 'SIZE'
      include 'QUANT'

      integer qlevel
      integer handle,handle_t,handle_m
      integer ii,jj,kk
      numberOfOps = numberOfOps+1      
!     Be sure to initialize newHandlenumber,N
      handle          = newHandleNumber
      newHandleNumber = newHandleNumber+1
      handle_t        = newHandleNumber
      newHandleNumber = newHandleNumber+1
      handle_m        = newHandleNumber
      newHandleNumber = newHandleNumber+1

      if(newHandleNumber.gt.maxOp) then
         if(nid.eq.0) then
            print*,'ERROR!'
            print*,'MAXOP TOO LOW'
            print*,'Please recompile with a larger maxOp'
         endif
         call exitt()
      endif

      
!     oldN is how many times we need to repeat this ops levels
!     It represents the previous total number of levels

      levels(numberOfOps) = qlevel

      kk=1
      do ii = 1,qlevel
         do jj = 1,quantN
            state(numberOfOps,kk) = ii-1
            kk = kk + 1
         enddo
      enddo

      lin_tiling_N(numberOfOps) = quantN
      quantN = quantN*qlevel

      oldN(numberOfOps)   = quantN


      return
      end


!*******************************************************************
      subroutine quantum_finalize_op_csr()
      implicit none
      include 'SIZE'
      include 'QUANT'


      integer ii,jj,kk,tilingNumber,maxLevel,curState
      integer mat_offset,op_num,mat_bound,tmp_sum1,tmp_sum2
      real    value,dEh,eV
      logical foundState,foundState2

      nstate    = quantN
      nstate_p  = nstate*nstate/np
      nstate2   = nstate*nstate
      nstate_np = nstate/np
      c_offset  = nid*nstate/np
      
      if(nid.eq.0) write(6,*) 'nstate: ',nstate,' np: ',np
!     Checks for array size consistency
      if(nstate/np.gt.maxN/lp) then
         if(nid.eq.0) then
            print*,'ERROR!'
            print*,'COMPILED WITH TOO SMALL MAXN/LP'
            print*,'Run with more cores or recompile with smaller lp'
         endif
         call exitt()
      endif

      if(mod(nstate,np).ne.0) then
         if(nid.eq.0) then
            print*,'ERROR!'
            print*,'nstate/np is not a whole number!'
            print*,'Run with np s.t. columns can be evenly distributed!'
         endif
         call exitt()
      endif
c      print*,'after check two',nid
     
      !Now that we have sizes, 0 out operators

      do jj=1,nstate_np
         do ii=1,nstate
            lin_comb_op(ii,jj) = 0.0
         enddo
      enddo

      do ii=1,numberOfOps
         maxLevel     = levels(ii)
         tilingNumber = oldN(ii)
         op_num       = (ii-1)*3
         !!Possible change this to nstate/npx
         do jj=1,nstate
            curState = state(ii,mod(jj-1,tilingNumber)+1)
            value = sqrt(real(curState+1))
            
!     op_num+1,op_num+2, refer to the different basic operators
!     op_num+1 = op, op_num+2 = op^dag, op_num+3 = op^dag*op
            
            op_csr_a(jj,op_num+3) = curState
            op_csr_ja(jj,op_num+3) = jj
            op_csr_ia(jj,op_num+3) = jj

            if(curState.lt.maxLevel-1) then
               mat_offset = oldN(ii)/maxLevel
               mat_bound  = jj+mat_offset
               op_csr_a(jj,op_num+1) = value
               op_csr_ia(jj,op_num+1) = jj
               op_csr_ja(jj,op_num+1) = mat_bound
            else
               op_csr_a(jj,op_num+1) = 0.0
               op_csr_ia(jj,op_num+1) = jj
               op_csr_ja(jj,op_num+1) = jj
            endif
         enddo
         op_csr_ia(nstate+1,op_num+3) = nstate+1
         op_csr_ia(nstate+1,op_num+1) = nstate+1
!     Get transpose of op_num+1 (to get op^dag)
         call csrcsc(nstate,1,1,op_csr_a(1,op_num+1),
     $        op_csr_ja(1,op_num+1),op_csr_ia(1,op_num+1),
     $        op_csr_a(1,op_num+2),
     $        op_csr_ja(1,op_num+2),op_csr_ia(1,op_num+2))

      enddo

      eV       = 27.21140                ! energy Eh  
      dEh  = (Emax-Emin)/(nEh+1)     
      do ii=1,nEh
         omega_quant(ii) = (Emin + ii*dEh)/eV
      enddo

      !Define the state_sums array for RWA
      if(rwa.eqv..true.) then
         do jj=1,nstate_np
            tmp_sum1 = 0
            do ii=1,numberOfOps
               tilingNumber = oldN(ii)
               tmp_sum1 = tmp_sum1 + 
     $              state(ii,mod(jj+c_offset-1,tilingNumber)+1)
            enddo
            do kk=1,nstate
               tmp_sum2 = 0
               do ii=1,numberOfOps
                  tilingNumber = oldN(ii)
                  tmp_sum2 = tmp_sum2 +
     $                 state(ii,mod(kk-1,tilingNumber)+1)
               enddo
               state_sums(kk,jj) = tmp_sum2-tmp_sum1
            enddo
         enddo
      else
         do jj=1,nstate_np
            do kk=1,nstate
               state_sums(kk,jj) = 0
            enddo
         enddo
      endif
      ! Initialize a hamiltonian array
      do jj=1,nstate
         ham_d(jj) = 0
         ham_d_ja(jj) = jj
         ham_d_ia(jj) = jj
         ham_i(jj) = 0
         ham_i_ja(jj) = jj
         ham_i_ia(jj) = jj
      enddo

      return
      end

!**********************************************************************
      subroutine quantum_combine_op_csr(handle1,handle2,op,handle3)
      implicit none
      include 'SIZE'
      include 'QUANT'

      integer op,handle1,handle2,handle3,ii,jj,nzmax

      nzmax   = maxN*10
      handle3 = newHandleNumber
      if(newHandleNumber.gt.maxOp) then
         if(nid.eq.0) then
            print*,'ERROR!'
            print*,'MAXOP TOO LOW'
            print*,'Please recompile with a larger maxOp'
         endif
         call exitt()
      endif
      newHandleNumber = newHandleNumber+1


      if(op.eq.0) call quantum_add_csr(nstate,nstate,1,
     $     op_csr_a(1,handle1),op_csr_ja(1,handle1),op_csr_ia(1,handle1)
     $     ,op_csr_a(1,handle2),op_csr_ja(1,handle2)
     $     ,op_csr_ia(1,handle2),op_csr_a(1,handle3)
     $     ,op_csr_ja(1,handle3),op_csr_ia(1,handle3),nzmax)

      if(op.eq.1) call quantum_mult_csr(nstate,nstate,1,
     $     op_csr_a(1,handle1),op_csr_ja(1,handle1),op_csr_ia(1,handle1)
     $     ,op_csr_a(1,handle2),op_csr_ja(1,handle2)
     $     ,op_csr_ia(1,handle2),op_csr_a(1,handle3)
     $     ,op_csr_ja(1,handle3),op_csr_ia(1,handle3),nzmax)

      
      end


!*****************************************************************
      subroutine quantum_observable(handle1,scalar)
      implicit none
      include 'SIZE'
      include 'QUANT'

      integer ii,jj,handle1,ubound,lbound,next,kk,istart,iend
      real    scalar
!      integer glo_num_loc(nstate,nstate/np)

      !gs_op way: setup a glo_num_loc with 1s in the appropriate places
      !and zeroes elsewhere. Use obs_local_mult
      !to premultiply each entry, then gs_op_fields to reduce.

c$$$      do jj=1,nstate/np
c$$$         do ii=1,nstate
c$$$            if(operator(handle1,ii,jj).ne.0) then
c$$$               !glo_num_loc(ii,jj) = 1
c$$$             else
c$$$               !glo_num_loc(ii,jj) = 0
c$$$             endif
c$$$          enddo
c$$$       enddo
      

!      call gs_setup(gs_handle_q_H,glo_num_q_Hv,nstate_p,
!     $     MPI_COMM_WORLD,np)


      !Pure MPI (dense): Use operator to premultiply each entry, then
      !MPI_REDUCE to reduce results: loop through whole matrix O(n^2)

      lBound   = nstate/np*nid+1
      uBound   = nstate/np*(nid+1)

      numObservables             = numObservables+1
     
      if(numObservables.gt.maxOp/4) then
         if(nid.eq.0) then
            print*,'ERROR!'
            print*,'TOO MANY OBSERVABLES FOR THIS MAXOP'
         endif
         call exitt()
      endif
      obs_handle(numObservables) = handle1
      obs_scalar(numObservables) = scalar
      next = 1
      obs_csr_ia(1,numObservables) = 1
      do jj=1,nstate
         istart = op_csr_ia(jj,handle1)
         iend = op_csr_ia(jj+1,handle1)-1
         do kk=istart,iend
            if((op_csr_ja(kk,handle1).ge.lBound).and
     $           .(op_csr_ja(kk,handle1).le.uBound)) then
               obs_csr_ja(next,numObservables) = op_csr_ja(kk,handle1)-
     $              c_offset
               obs_csr_a(next,numObservables) =  
     $              scalar* op_csr_a(kk,handle1)
               next = next + 1
            endif
         enddo
         obs_csr_ia(jj+1,numObservables) = next
      enddo
           


      !Pure MPI (sparse): have something also hold map data so that
      !0s are not summed. O(~2n)
      

      end

!******************************************************************
      subroutine sparseMatMult(handle1,handle2,handle3)
      implicit none
      include 'SIZE'
      include 'QUANT'
      integer ii,jj,kk,ll,mm
      integer handle1,handle2,handle3,kkp
      !Updated to non-parallel setup. Parallel setup available in r2029
      !Removed operator - this routine is now deprecated. Last available
      !version in r2040
      
      do jj = 1,nstate
         do ii = 1,nstate
!            operator(handle3,ii,jj) = 0.0
         enddo
      enddo

      ii  = initIndex(handle1,1)
      kk  = initIndex(handle1,2)
      kkp = initIndex(handle2,1) 
      jj  = initIndex(handle2,2)

      if(kk.lt.kkp) then
         do while(kk.lt.kkp)
            ii = ii+1
            kk = kk+1
         enddo
      else if(kkp.lt.kk) then
         do while(kkp.lt.kk)
            jj = jj+1
            kkp = kkp+1
         enddo
      endif

      do while((ii.le.nstate).and.(kk.le.nstate).and.(jj.le.nstate))
!         operator(handle3,ii,jj) = operator(handle1,ii,kk)*
!     $        operator(handle2,kk,jj)

         ii = ii + 1
         kk = kk + 1
         jj = jj + 1
      enddo

      end


!**********************************************************************
      subroutine quantum_add_to_ham_csr(handle,scalar,timeDep)
      implicit none
      include 'SIZE'
      include 'QUANT'
      
      integer ii,jj,timeDep,handle,nzmax
      real    scalar
            

      nzmax = maxN*maxN/lp
      
      do ii=1,maxN*10 !Consider changing this size - but to what?
         ham_a(ii) = op_csr_a(ii,handle)*scalar
      enddo
      if(timeDep.eq.0) then
         call quantum_add_csr(nstate,nstate,1,ham_i,ham_i_ja,ham_i_ia,
     $        ham_a,op_csr_ja(1,handle),op_csr_ia(1,handle)
     $        ,ham_i,ham_i_ja,ham_i_ia,nzmax)
      else
         call quantum_add_csr(nstate,nstate,1,ham_d,ham_d_ja,ham_d_ia,
     $        ham_a,op_csr_ja(1,handle),op_csr_ia(1,handle)
     $        ,ham_d,ham_d_ja,ham_d_ia,nzmax)
         nTimeDep = 1
      endif


      end


!**********************************************************************
      subroutine quantum_ham_finalize_csr()
      implicit none
      include 'SIZE'
      include 'QUANT'
      integer ii,jj,kk,IERR,nzmax
      real op(32,32)
      ham_nnz = nstate2
      !Add together independent and dependent parts to get indices
      !0 means do the add such that ham_i and ham_d are promoted to
      !csr, padded with 0s
      call quantum_add_csr(nstate,nstate,0,ham_i,ham_i_ja,ham_i_ia,
     $     ham_d,ham_d_ja,ham_d_ia,ham_a,ham_ja,ham_ia,ham_nnz)

      return
      end



!***********************************************************
      subroutine quantum_lin_finalize()
      implicit none
      include 'SIZE'
      include 'QUANT'
#ifdef MPI
      include 'mpif.h'
#endif

      integer ii,jj,kk,nnz,ierr
      integer*8 size_type !Has to be integer*8 for MPI

!     Setup window information
      size_type = maxN*maxN/lp
#ifdef MPI
      call MPI_WIN_CREATE(rho,size_type,16,MPI_INFO_NULL,
     $     MPI_COMM_WORLD,mpi_win_c,IERR)
#endif
      return
      end


!************************************************************
      subroutine quantum_lin_add(handle1,handle2,scalar)
      implicit none
      include 'SIZE'
      include 'QUANT'

      integer jj,ii,kk,handle1,handle2,tilingNum,state1,state2
      integer ll,mm,maxState,target_core,iOp,kkOld
      integer*8 target_disp
      integer tilingNum2
      real scalar

      !be sure to set this initially!

      numberOfLinOps = numberOfLinOps + 1 
      if(numberOfLinOps.gt.maxOp/4) then
         if(nid.eq.0) then
            print*,'ERROR!'
            print*,'TOO MANY LINOPS FOR THIS MAXOP'
         endif
         call exitt()
      endif
      iOp = numberOfLinOps
      !FIXME: handle2/3 is dangerous - put a check or think of a better way!
      tilingNum = oldN(handle2/3)
      do jj=1,nstate_np
         state1 = state(handle2/3,mod(jj+c_offset-1,tilingNum)+1)
         do ii=1,nstate
            state2 = state(handle2/3,mod(ii-1,tilingNum)+1)
            lin_comb_op(ii,jj) = lin_comb_op(ii,jj) + 
     $           scalar*(state1+state2)
            if(handle1.eq.handle2) then !FIXME handle1.eq.handle2 is dangerous
               lin_comb_op(ii,jj) = lin_comb_op(ii,jj) -
     $              2*scalar*state1*state2
            endif
         enddo
      enddo

!     Lin_gs_op setup 
      if(handle1.ne.handle2) then
         num_nl_lin_ops = num_nl_lin_ops+1
         iOp            = num_nl_lin_ops
         if(num_nl_lin_ops.gt.maxOp/4) then
            if(nid.eq.0) then
               print*,'ERROR!'
               print*,'TOO MANY LINOPS FOR THIS MAXOP'
            endif
            call exitt()
         endif

!     lin_get setup
         maxState = levels(handle2/3)
         tilingNum = lin_tiling_N(handle2/3)
         tilingNum2 = oldN(handle2/3)
         kk = 0
         kkOld = 0
         ll = 0


         lin_offset(iOp) = tilingNum
         do jj=1,nstate_np
            target_core = (jj+c_offset+tilingNum-1)*np/nstate
            state1 = state(handle2/3,mod(jj+c_offset-1,tilingNum2)+1)
            if(target_core.ne.nid.and.state1+1.lt.maxState.and.
     $           target_core.lt.np.and.jj+c_offset+tilingNum.le.nstate) 
     $           then
               kk = kk + 1
               target_disp = (jj+c_offset+tilingNum-target_core*nstate
     $              /np-1)*maxN
               lin_target(kk,iOp) = target_core
               lin_disp(kk,iOp)   = target_disp
            endif

            mm = 0
            do ii=1,nstate
               state2 = state(handle2/3,mod(ii-1,tilingNum2)+1)
               if(state1+1.lt.maxState.and.state2+1.lt.
     $              maxState.and.jj+c_offset+tilingNum.le.nstate.and.
     $              ii+tilingNum.le.nstate) then
                  if(target_core.eq.nid) then
                     ll = ll+1
                     lin_l_index(1,ll,iOp) = ii !i0
                     lin_l_index(2,ll,iOp) = jj !j0
                     lin_l_a(ll,iOp) = -2*scalar*sqrt(real((state1+1)*
     $                    (state2+1)))
                  else
                     mm = mm+1
                     lin_nl_index(1,mm,kk,iOp) = ii !i0
                     lin_nl_index(2,mm,kk,iOp) = jj !j0
                     lin_nl_a(mm,kk,iOp) = -2*scalar*sqrt(real(
     $                    (state1+1)*(state2+1)))
                  endif
               endif
            enddo
            if(kk.ne.kkOld) lin_nl_length(kk,iOp) = mm
         enddo
         lin_l_length(iOp)      = ll
         lin_nl_col_length(iOp) = kk
      endif

      end

!**************************************************************
      subroutine quantum_lin_mpi_get()
      implicit none
      include 'SIZE'
      include 'QUANT'
      include 'mpif.h'

      complex ci,tmpcc,tmpGet(nstate)
      integer ii,jj,kk,i0,j0,ierr,iOp
      real tmprr,tmpii
      real op(12,12)
      integer*8 target_disp !has to be integer*8 for MPI!
      !lnumqd = lin_max_ops

      ci = (0.0,1.0)
!      call MPI_BARRIER(MPI_COMM_WORLD,IERR)!Is this necessary?
      !Do opdag*op * rho + rho * opdag*op by utilizing the fact that
      !opdag * op is just the state - the sum of states, with gammas
      !stored in lin_comb_op
      do jj = 1,nstate_np
         do kk = 1,nstate
      !Rho was copied over in hamiltonian - we needn't copy it here.
           drho(kk,jj) = drho(kk,jj) + lin_comb_op(kk,jj) *rho(kk,jj)
         enddo
      enddo
      
      do iOp=1,num_nl_lin_ops
!     local opdag*rho*op parts
         do jj=1,lin_l_length(iOp)
            i0 = lin_l_index(1,jj,iOp)
            j0 = lin_l_index(2,jj,iOp)
            drho(i0,j0) = drho(i0,j0) + lin_l_a(jj,iOp)*
     $           rho(i0+lin_offset(iOp),j0+lin_offset(iOp))
         enddo

!     nonlocal opdag*rho*op parts
         do jj=1,lin_nl_col_length(iOp)
            target_disp = lin_disp(jj,iOp)
!     get columns from other cores
#ifdef MPI
            call MPI_WIN_LOCK(MPI_LOCK_SHARED,lin_target(jj,iOp),0,
     $           mpi_win_c,IERR)
            call MPI_GET(tmpGet,nstate,MPI_DOUBLE_COMPLEX
     $           ,lin_target(jj,iOp),target_disp,nstate,
     $           MPI_DOUBLE_COMPLEX,mpi_win_c,IERR)
            call MPI_WIN_UNLOCK(lin_target(jj,iOp),mpi_win_c,IERR)
#endif
!     sum the appropriate parts of the columns
            do kk=1,lin_nl_length(jj,iOp)
               i0 = lin_nl_index(1,kk,jj,iOp)
               j0 = lin_nl_index(2,kk,jj,iOp)
               drho(i0,j0) = drho(i0,j0) + lin_nl_a(kk,jj,iOp)*
     $           tmpGet(i0+lin_offset(iOp))
            enddo

         enddo
      enddo

      !Store in resrho
      do jj=1,nstate/np
         do ii=1,nstate
            resrho_r(ii,jj)= real(drho(ii,jj))
            resrho_i(ii,jj)= imag(drho(ii,jj))
         enddo
      enddo


      return

      end

!*******************************************************************
!*
!* Returns the binary rep for a specific bit, given a epcific number
!* useful for representing the heading as the 2 qubit states
!* 
!*******************************************************************


      integer function binary_rep(bit,number,numberOfBits)
      implicit none

      integer bit,number,currentBit,maxNumber,localNumber,ii
      integer numberOfBits
      localNumber = number-1
      currentBit = 0
      if(number-2**(bit-1).lt.0) then 
         binary_rep=0
         return
      endif
      do ii=1,numberOfBits+1
         currentBit = currentBit+1
         binary_rep = mod(localNumber,2)
         if(currentBit.eq.bit) then
            return
         endif
         localNumber = int(real(localNumber/2))
      enddo
      
      end


!*******************************************************************
!*
!* Adds a CNOT gate to the quantum gate hamiltonian
!*
!* inputs:
!*      control - identifies the control qubit for the CNOT gate
!*      target  - identifies the target of the qubit
!*      totalQubits - total number of qubits in the system
!*      time - the time at which the gate should be applied
!*
!* outputs:
!*      none, but stores generated gate in quantum_gate_csr_*,
!*      the times to apply the gates in quantum_gate_times,
!*      and the gates to apply in quantum_gate (a pointer into
!*      the quantum_gate_csr* arrays, allowing us to reuse a 
!*      previously generated gate).
!*      
!*******************************************************************
      subroutine quantum_add_CNOT(control,target,totalQubits,gatetime)
      implicit none
      include 'SIZE'
      include 'QUANT'

      integer control,target,totalQubits,nzmax
      real    gatetime,dns(2**totalQubits,2**totalQubits)
      integer n,ierr,ii,jj,m,kk,binary_rep,ll
      integer control_ii,control_jj,target_ii,target_jj
      logical otherSame
      nzmax   = maxN*10
      n       = totalQubits

      !Input checking
      if ((control.lt.1).or.(control.gt.totalQubits)) then
         if(nid.eq.0) then
            print*,'ERROR!'
            print*,'control qubit number doesn\'t make sense!'
            print*,'Please use a number greater than 1 and less than the
     $total number of qubits'
         endif
         call exitt()
      endif

      !Input checking
      if ((target.lt.1).or.(target.gt.totalQubits)) then
         if(nid.eq.0) then
            print*,'ERROR!'
            print*,'target qubit number doesn\'t make sense!'
            print*,'Please use a number greater than 1 and less than the
     $total number of qubits'
         endif
         call exitt()
      endif

      !Input checking
      if (control.eq.target) then
         if(nid.eq.0) then
            print*,'ERROR!'
            print*,'control and target must be different!'
         endif
         call exitt()
      endif

      !Input checking
      if(totalQubits.gt.numberOfOps) then
         if(nid.eq.0) then
            print*,'ERROR!'
            print*,'Having more qubits than the number of operators
     $doesn\'t make any sense!'
         endif
         call exitt()
      endif

      !Input checking
      if(number_of_gates.gt.0) then
         if(gatetime.lt.quantum_gate_times(number_of_gates)) then
            if(nid.eq.0) then
               print*,'ERROR!'
               print*,'Please enter gates in order of the time that 
     $that they will be applied!'
            endif
            call exitt()
         endif
      endif

      number_of_gates = number_of_gates + 1
      !Save the time to apply the gate
      quantum_gate_times(number_of_gates) = gatetime

      !Only regenerate the gate if we haven't seen it yet
      if(cnot_table(target,control).eq.0) then
!Construct the full gate
         generated_gates = generated_gates + 1
         kk = 0
!     FIXME Figure out a better way to do this!
         do jj=1,2**n
            quantum_gate_coo_a(kk+1) = 0
            control_jj = binary_rep(control,jj,n)
            target_jj  = binary_rep(target,jj,n)
            do ii=1,2**n
               otherSame = .true.
               do ll=1,n
                  if((ll.ne.control).and.(ll.ne.target)) then
                     if(binary_rep(ll,jj,n).ne.binary_rep(ll,ii,n)) then
                        otherSame = .false.
                     endif
                  endif
               enddo

               if(otherSame.eqv..true.) then
                  control_ii = binary_rep(control,ii,n)
                  target_ii  = binary_rep(target,ii,n)
                  if((control_ii.eq.0).and.(control_jj.eq.0).and.
     $                 (target_ii.eq.0).and.(target_jj.eq.0)) then
                     kk = kk + 1
                     quantum_gate_coo_ia(kk) = ii
                     quantum_gate_coo_ja(kk) = jj
                     quantum_gate_coo_a(kk) = 1.0
                  else if((control_ii.eq.0).and.(control_jj.eq.0).and.
     $                    (target_ii.eq.1).and.(target_jj.eq.1)) then
                     kk = kk + 1
                     quantum_gate_coo_ia(kk) = ii
                     quantum_gate_coo_ja(kk) = jj
                     quantum_gate_coo_a(kk) = 1.0
                  else if((control_ii.eq.1).and.(control_jj.eq.1).and.
     $                    (target_ii.eq.1).and.(target_jj.eq.0)) then
                     kk = kk + 1
                     quantum_gate_coo_ia(kk) = ii
                     quantum_gate_coo_ja(kk) = jj
                     quantum_gate_coo_a(kk) = 1.0
                  else if((control_ii.eq.1).and.(control_jj.eq.1).and.
     $                    (target_ii.eq.0).and.(target_jj.eq.1)) then
                     kk = kk + 1
                     quantum_gate_coo_ia(kk) = ii
                     quantum_gate_coo_ja(kk) = jj
                     quantum_gate_coo_a(kk) = 1.0
                  endif
               endif
            enddo
         enddo

!subroutine coocsr(nrow,nnz,a,ir,jc,ao,jao,iao)
         call coocsr(2**(n),kk,quantum_gate_coo_a,quantum_gate_coo_ia,
     $        quantum_gate_coo_ja,quantum_gate_csr_a(1,generated_gates),
     $        quantum_gate_csr_ja(1,generated_gates),
     $        quantum_gate_csr_ia(1,generated_gates))

         quantum_gate(number_of_gates) = generated_gates

         cnot_table(target,control) = generated_gates
c$$$         do ii=1,5*nstate
c$$$            tmprv(ii) = real(quantum_gate_csr_a(ii,generated_gates))
c$$$         enddo
c$$$
c$$$!debug statements
c$$$         call csrdns(2**n,2**n,tmprv,
c$$$     $        quantum_gate_csr_ja(1,generated_gates),
c$$$     $        quantum_gate_csr_ia(1,generated_gates),dns,2**n,ierr)
c$$$         
c$$$         do ii=1,2**n
c$$$            write(11,'(100g25.15)') (dns(ii,jj), jj=1,2**n)
c$$$         enddo
c$$$

      else
         quantum_gate(number_of_gates) = 
     $        cnot_table(target,control)
      endif

      end


!*******************************************************************
!*
!* Adds a CRk gate to the quantum gate hamiltonian
!*
!* inputs:
!*      control - identifies the control qubit for the CNOT gate
!*      target  - identifies the target of the qubit
!*      totalQubits - total number of qubits in the system
!*      k           - Rk = 1 0
!*                         0 e^(i*2*pi/2^k)
!*      time - the time at which the gate should be applied
!*
!* outputs:
!*      none, but stores generated gate in quantum_gate_csr_*,
!*      the times to apply the gates in quantum_gate_times,
!*      and the gates to apply in quantum_gate (a pointer into
!*      the quantum_gate_csr* arrays, allowing us to reuse a 
!*      previously generated gate).
!*      
!*******************************************************************
      subroutine quantum_add_CRk(control,target,totalQubits,k,gatetime)
      implicit none
      include 'SIZE'
      include 'QUANT'

      integer control,target,totalQubits,nzmax
      real    gatetime,dns(2**totalQubits,2**totalQubits)
      integer n,ierr,ii,jj,m,kk,binary_rep,ll,k
      integer control_ii,control_jj,target_ii,target_jj
      logical otherSame
      complex ci
      nzmax   = maxN*10
      n       = totalQubits
      ci      = (0,1)
      !Input checking
      if ((control.lt.1).or.(control.gt.totalQubits)) then
         if(nid.eq.0) then
            print*,'ERROR!'
            print*,'control qubit number doesn\'t make sense!'
            print*,'Please use a number greater than 1 and less than the
     $total number of qubits'
         endif
         call exitt()
      endif

      !Input checking
      if ((target.lt.1).or.(target.gt.totalQubits)) then
         if(nid.eq.0) then
            print*,'ERROR!'
            print*,'target qubit number doesn\'t make sense!'
            print*,'Please use a number greater than 1 and less than the
     $total number of qubits'
         endif
         call exitt()
      endif

      !Input checking
      if (control.eq.target) then
         if(nid.eq.0) then
            print*,'ERROR!'
            print*,'control and target must be different!'
         endif
         call exitt()
      endif

      !Input checking
      if(totalQubits.gt.numberOfOps) then
         if(nid.eq.0) then
            print*,'ERROR!'
            print*,'Having more qubits than the number of operators
     $doesn\'t make any sense!'
         endif
         call exitt()
      endif

      !Input checking
      if(number_of_gates.gt.0) then
         if(gatetime.lt.quantum_gate_times(number_of_gates)) then
            if(nid.eq.0) then
               print*,'ERROR!'
               print*,'Please enter gates in order of the time that 
     $that they will be applied!'
            endif
            call exitt()
         endif
      endif

      number_of_gates = number_of_gates + 1
      !Save the time to apply the gate
      quantum_gate_times(number_of_gates) = gatetime

      !Only regenerate the gate if we haven't seen it yet
      if(crk_table(target,control,k).eq.0) then
!Construct the full gate
         generated_gates = generated_gates + 1
         kk = 0
!     FIXME Figure out a better way to do this!
         do jj=1,2**n
            quantum_gate_coo_a(kk+1) = 0
            control_jj = binary_rep(control,jj,n)
            target_jj  = binary_rep(target,jj,n)
            do ii=1,2**n
               otherSame = .true.
               do ll=1,n
                  if((ll.ne.control).and.(ll.ne.target)) then
                     if(binary_rep(ll,jj,n).ne.binary_rep(ll,ii,n)) then
                        otherSame = .false.
                     endif
                  endif
               enddo

               if(otherSame.eqv..true.) then
                  control_ii = binary_rep(control,ii,n)
                  target_ii  = binary_rep(target,ii,n)
                  if((control_ii.eq.0).and.(control_jj.eq.0).and.
     $                 (target_ii.eq.0).and.(target_jj.eq.0)) then
                     kk = kk + 1
                     quantum_gate_coo_ia(kk) = ii
                     quantum_gate_coo_ja(kk) = jj
                     quantum_gate_coo_a(kk) = 1.0
                  else if((control_ii.eq.0).and.(control_jj.eq.0).and.
     $                    (target_ii.eq.1).and.(target_jj.eq.1)) then
                     kk = kk + 1
                     quantum_gate_coo_ia(kk) = ii
                     quantum_gate_coo_ja(kk) = jj
                     quantum_gate_coo_a(kk) = 1.0
                  else if((control_ii.eq.1).and.(control_jj.eq.1).and.
     $                    (target_ii.eq.0).and.(target_jj.eq.0)) then
                     kk = kk + 1
                     quantum_gate_coo_ia(kk) = ii
                     quantum_gate_coo_ja(kk) = jj
                     quantum_gate_coo_a(kk) = 1.0
                  else if((control_ii.eq.1).and.(control_jj.eq.1).and.
     $                    (target_ii.eq.1).and.(target_jj.eq.1)) then
                     kk = kk + 1
                     quantum_gate_coo_ia(kk) = ii
                     quantum_gate_coo_ja(kk) = jj
                     quantum_gate_coo_a(kk) = exp(ci*2*pi/(2**k))
                  endif
               endif
            enddo
         enddo

!subroutine coocsr(nrow,nnz,a,ir,jc,ao,jao,iao)
         call coocsr(2**(n),kk,quantum_gate_coo_a,quantum_gate_coo_ia,
     $        quantum_gate_coo_ja,quantum_gate_csr_a(1,generated_gates),
     $        quantum_gate_csr_ja(1,generated_gates),
     $        quantum_gate_csr_ia(1,generated_gates))

         quantum_gate(number_of_gates) = generated_gates
         crk_table(target,control,k)   = generated_gates
c$$$         do ii=1,5*nstate
c$$$            tmprv(ii) = real(quantum_gate_csr_a(ii,generated_gates))
c$$$         enddo
c$$$
c$$$!debug statements
c$$$         call csrdns(2**n,2**n,tmprv,
c$$$     $        quantum_gate_csr_ja(1,generated_gates),
c$$$     $        quantum_gate_csr_ia(1,generated_gates),dns,2**n,ierr)
c$$$         
c$$$         do ii=1,2**n
c$$$            write(11,'(100g25.15)') (dns(ii,jj), jj=1,2**n)
c$$$         enddo
c$$$

      else
         quantum_gate(number_of_gates) = 
     $        crk_table(target,control,k)
      endif

      end

!*******************************************************************
!*
!* Adds a gate to the quantum gate hamiltonian
!*
!* inputs:
!*      qubit - identifies the qubit the gate should be applied to
!*      totalQubits - total number of qubits in the system
!*      gateIdentifier - integer between 1 and 6 identifying the basic
!*                       gate to be applied
!*                       1: Px = 0  1     2: Py = 0 -i 
!*                               1  0             i  0
!*                       3: Pz = 1  0    
!*                               0 -1    
!*                       4: Phase = 1 0    
!*                                  0 i 
!*                       5: pi/8 = 1     0
!*                                 0 exp(i*pi/4)  
!*                       6: hadamard = 1/sqrt(2) * 1  1
!*                                                 1 -1
!*      time - the time at which the gate should be applied
!*
!* outputs:
!*      none, but stores generated gate in quantum_gate_csr_*,
!*      the times to apply the gates in quantum_gate_times,
!*      and the gates to apply in quantum_gate (a pointer into
!*      the quantum_gate_csr* arrays, allowing us to reuse a 
!*      previously generated gate).
!*      
!*******************************************************************
      subroutine quantum_add_gate(qubit,totalQubits,gateIdentifier,
     $     gatetime)
      implicit none
      include 'SIZE'
      include 'QUANT'

      integer qubit,totalQubits,gateIdentifier,gate,nzmax
      real    gatetime,dns(2**totalQubits,2**totalQubits)
      integer n,ierr,ii,jj,m
      complex ci,um(2,2)
      n = totalQubits
      nzmax   = maxN*10
      ci = (0,1)

      !Input checking
      if ((qubit.lt.1).or.(qubit.gt.totalQubits)) then
         if(nid.eq.0) then
            print*,'ERROR!'
            print*,'qubit number doesn\'t make sense!'
            print*,'Please use a number greater than 1 and less than the
     $total number of qubits'
         endif
         call exitt()
      endif

      !Input checking
      if(totalQubits.gt.numberOfOps) then
         if(nid.eq.0) then
            print*,'ERROR!'
            print*,'Having more qubits than the number of operators
     $doesn\'t make any sense!'
         endif
         call exitt()
      endif

      !Input checking
      if(number_of_gates.gt.0) then
         if(gatetime.lt.quantum_gate_times(number_of_gates)) then
            if(nid.eq.0) then
               print*,'ERROR!'
               print*,'Please enter gates in order of the time that 
     $that they will be applied!'
            endif
            call exitt()
         endif
      endif


      !Construct one of the six basic 2x2 gates
      ! FIXME: Consider storing these somewhere?
      if(gateIdentifier.eq.1) then
         um(1,1) = 0
         um(1,2) = 1
         um(2,1) = 1
         um(2,2) = 0
      else if(gateIdentifier.eq.2) then
         um(1,1) = 0
         um(1,2) = -ci
         um(2,1) = ci
         um(2,2) = 0
      else if(gateIdentifier.eq.3) then
         um(1,1) = 1
         um(1,2) = 0
         um(2,1) = 0
         um(2,2) = -1
      else if(gateIdentifier.eq.4) then
         um(1,1) = 1
         um(1,2) = 0
         um(2,1) = 0
         um(2,2) = ci
      else if(gateIdentifier.eq.5) then
         um(1,1) = 1
         um(1,2) = 0
         um(2,1) = 0
         um(2,2) = exp(ci*pi/4)
      else if(gateIdentifier.eq.6) then
         um(1,1) = 2**(-0.5)
         um(1,2) = 2**(-0.5)
         um(2,1) = 2**(-0.5)
         um(2,2) = -2**(-0.5)
      else
         !Input checking
         if(nid.eq.0) then
            print*,'ERROR!'
            print*,'gateIdentifier out of range'
            print*,'Please use a number between 1-6'
         endif
         call exitt()
      endif

      number_of_gates = number_of_gates + 1
      m = totalQubits - qubit + 1
!      qubit = totalQubits - qubit +1 
!      qubit = totalQubits - qubit + 1

      !Save the time to apply the gate
      quantum_gate_times(number_of_gates) = gatetime

      !Only regenerate the gate if we haven't seen it yet
      if(gates_table(gateIdentifier,qubit).eq.0) then
         !Construct the full gate
         call quantum_tensor_product_um(um,m,totalQubits)
         quantum_gate(number_of_gates) = generated_gates
      else
         quantum_gate(number_of_gates) = 
     $        gates_table(gateIdentifier,m)
      endif

c$$$      do ii=1,5*nstate
c$$$         tmprv(ii) = real(quantum_gate_csr_a(ii,gate))
c$$$      enddo

c$$$      !debug statements
c$$$      call csrdns(2**n,2**n,tmprv,
c$$$     $     quantum_gate_csr_ja(1,gate),
c$$$     $     quantum_gate_csr_ia(1,gate),dns,2**n,ierr)
c$$$      
c$$$      do ii=1,2**n
c$$$         write(6,'(100g25.15)') (dns(ii,jj), jj=1,2**n)
c$$$      enddo



      end

!********************************************************************
!*
!* tensor product of a 2x2 matrix, Um, for qubit m (out of n qubits)
!*
!* U = 1 cross 1 cross 1 ... cross Um cross 1 ... cross 1
!*
!* inputs: 
!*    um = matrix representing gate acting on one qubit
!*    m  = number of qubit
!*    n  = total number of qubits
!*
!* outputs:
!*    none, but stores the generated gate into quantum_gate_csr
!* Note: Works only on systems of qubits; other degrees of freedom
!*       need to be traced out
!*
!********************************************************************

      subroutine quantum_tensor_product_um(um,m,n)
      implicit none
      include 'SIZE'
      include 'QUANT'
      
      complex um(2,2)
      integer n
      integer m,ii,i_coo_a,jj

      generated_gates = generated_gates + 1

      !First, construct as COO format then convert to CSR
      
      !Construct the subblock, representing
      ! block = um cross 1 ... cross 1
      ! This happens n-m times, where n is the total number of qubits
      ! The size of 1 is defined by the levels of all operators


      i_coo_a = 0
      ! Construct all of the blocks
      ! We want 2**(m-1) blocks
      ! each block is just a copy of the subblock, offset by the 
      ! subblocks size
      ! I.E.
      ! u = 1 2    becomes u = 1 2 0 0 
      !     3 4                3 4 0 0
      !                        0 0 1 2
      !                        0 0 3 4
      do jj=1,2**(m-1)
         ! Construct the subblock
         ! Each subblock consists of taking the normal matrix and 'expanding' it
         ! by making each entry a small 'diagonal' like element.
         ! I.E.
         ! u = 1 2    becomes  u = 1 0 2 0
         !     3 4                 0 1 0 2
         !                         3 0 4 0
         !                         0 3 0 4 
         ! Additional kronecker products with 1 just continue to expand the matrix
         ! in this way
         do ii=1,2**(n-m)
            ! um(1,1) part
            if((real(um(1,1)).ne.0).or.imag(um(1,1)).ne.0) then
               i_coo_a = i_coo_a + 1
               quantum_gate_coo_a(i_coo_a) = um(1,1)
               quantum_gate_coo_ia(i_coo_a) = ii+(jj-1)*2**(n-m+1)
               quantum_gate_coo_ja(i_coo_a) = ii+(jj-1)*2**(n-m+1)
            endif
! um(2,2) part
            if((real(um(2,2)).ne.0).or.imag(um(2,2)).ne.0) then
               i_coo_a = i_coo_a + 1
               quantum_gate_coo_a(i_coo_a) = um(2,2)
               quantum_gate_coo_ia(i_coo_a) = ii+2**(n-m)+
     $              (jj-1)*2**(n-m+1)
               quantum_gate_coo_ja(i_coo_a) = ii+2**(n-m)+
     $              (jj-1)*2**(n-m+1)
            endif
! um(1,2) part
            if((real(um(1,2)).ne.0).or.imag(um(1,2)).ne.0) then
               i_coo_a = i_coo_a + 1
               quantum_gate_coo_a(i_coo_a) = um(1,2)
               quantum_gate_coo_ia(i_coo_a) = ii+(jj-1)*2**(n-m+1)
               quantum_gate_coo_ja(i_coo_a) = ii+2**(n-m)+
     $              (jj-1)*2**(n-m+1)
            endif
! um(2,1) 
            if((real(um(2,1)).ne.0).or.imag(um(2,1)).ne.0) then
               i_coo_a = i_coo_a + 1
               quantum_gate_coo_a(i_coo_a) = um(2,1)
               quantum_gate_coo_ia(i_coo_a) = ii+2**(n-m)+
     $              (jj-1)*2**(n-m+1)
               quantum_gate_coo_ja(i_coo_a) = ii+(jj-1)*2**(n-m+1)
            endif
         enddo
      enddo

      ! Convert the COO matrix to CSR

      !subroutine coocsr(nrow,nnz,a,ir,jc,ao,jao,iao)
      call coocsr(2**(n),i_coo_a,quantum_gate_coo_a,quantum_gate_coo_ia,
     $     quantum_gate_coo_ja,quantum_gate_csr_a(1,generated_gates),
     $     quantum_gate_csr_ja(1,generated_gates),
     $     quantum_gate_csr_ia(1,generated_gates))

      end


!     quantum_lin_gs_op has been deprecated as of r2040
c$$$
c$$$
c$$$!**************************************************************
c$$$      subroutine quantum_lin_gs_op()
c$$$      implicit none
c$$$      include 'SIZE'
c$$$      include 'QUANT'
c$$$#ifdef MPI
c$$$      include 'mpif.h'
c$$$#endif
c$$$      complex ci,tmpcc
c$$$      integer ii,jj,kk,j0,ierr
c$$$      integer mat_ia(maxN+1),mat_ja(maxN*maxN/lp)!Change to smaller(maxnnz) size
c$$$      real tmprr,tmpii
c$$$      real mat_a(maxN*maxN/lp)!Change to smaller(maxnnz) size
c$$$      real op(12,12)
c$$$      !lnumqd = lin_max_ops
c$$$
c$$$      ci = (0.0,1.0)
c$$$
c$$$      !Do opdag*op * rho + rho * opdag*op by utilizing the fact that
c$$$      !opdag * op is just the state - the sum of states, with gammas
c$$$      !stored in lin_comb_op
c$$$      do jj = 1,nstate_np
c$$$         do kk = 1,nstate
c$$$            tmpcc = (rho_r(kk,jj)+ci*rho_i(kk,jj)) * lin_comb_op(kk,jj)
c$$$            drho(kk,jj) = drho(kk,jj) + tmpcc
c$$$         enddo
c$$$      enddo
c$$$
c$$$      do ii=1,num_nl_lin_ops
c$$$         do jj=1,lin_one_nnz(ii)
c$$$            mat_a(jj) = lin_one_a(ii,jj)
c$$$         enddo
c$$$         
c$$$         do jj=1,nstate+1
c$$$            mat_ia(jj) = lin_one_ia(ii,jj)
c$$$         enddo
c$$$         
c$$$         do jj=1,lin_one_nnz(ii)
c$$$            mat_ja(jj) = lin_one_ja(ii,jj)
c$$$         enddo
c$$$
c$$$
c$$$         call cem_quantum3_csr_mxm(rho_r,tmp_r2,mat_a,mat_ia
c$$$     $        ,mat_ja)
c$$$         call cem_quantum3_csr_mxm(rho_i,tmp_i2,mat_a,mat_ia
c$$$     $        ,mat_ja)
c$$$
c$$$
c$$$         do jj = 1,nstate/np
c$$$            do kk = 1,nstate
c$$$               j0 = (jj-1)*nstate+kk
c$$$               tmprv(j0) = tmp_r2(kk,jj)
c$$$               tmpiv(j0) = tmp_i2(kk,jj)
c$$$            enddo
c$$$         enddo
c$$$
c$$$!     Do communication
c$$$!     Consider doing all communication at once (with numberOfLinOps fields)
c$$$!     This involves a lot of extra computation, though
c$$$!        - Recalculating mat_a, etc, for each lin_op
c$$$!        - Storing tmp_r2 for every lin_op
c$$$!     Get (op*rho)^dag
c$$$         call gs_op_fields(gs_handle_q_H,tmprv,nstate_p,
c$$$     $        1,1,1,0)
c$$$         call gs_op_fields(gs_handle_q_H,tmpiv,nstate_p,
c$$$     $        1,1,1,0)
c$$$
c$$$         do jj= 1,nstate/np
c$$$            do kk = 1,nstate
c$$$               if(jj+c_offset.eq.kk) then
c$$$                  tmp_r1(kk,jj) = tmp_r2(kk,jj)
c$$$                  tmp_i1(kk,jj) = -tmp_i2(kk,jj)
c$$$               else
c$$$                  j0 = (jj-1)*nstate+kk
c$$$                  !We want b, so we take a+b-a
c$$$                  tmp_r1(kk,jj) = tmprv(j0) - tmp_r2(kk,jj)
c$$$                  !We want -b, so we take a - (a+b)
c$$$                  tmp_i1(kk,jj) = tmp_i2(kk,jj) - tmpiv(j0)
c$$$               endif
c$$$            enddo
c$$$         enddo        
c$$$
c$$$
c$$$!     calculate op*(op*rho)^dag
c$$$         call cem_quantum3_csr_mxm(tmp_r1,tmp_r2,mat_a,mat_ia
c$$$     $        ,mat_ja)
c$$$         call cem_quantum3_csr_mxm(tmp_i1,tmp_i2,mat_a,mat_ia
c$$$     $        ,mat_ja)
c$$$
c$$$         !sum all parts
c$$$         do jj = 1,nstate/np
c$$$            do kk = 1,nstate
c$$$              !-2 * op * rho * opdag part
c$$$              tmprr = -2*lin_scalar(ii)*tmp_r2(kk,jj)
c$$$              tmpii = -2*lin_scalar(ii)*tmp_i2(kk,jj)
c$$$!              print*,'tmprr,tmpii2',tmp_r2(kk,jj),tmp_i2(kk,jj)
c$$$              drho(kk,jj)= drho(kk,jj) + 
c$$$     $             (tmprr + ci*tmpii)
c$$$           enddo
c$$$         enddo
c$$$         if(debug.eq.0) then
c$$$         print*,'tmpr2',ii,lin_one_nnz(ii)
c$$$         do kk=1,nstate
c$$$            write(6,'(100g25.15)') (tmp_r2(kk,jj), jj=1,nstate)
c$$$         enddo
c$$$
c$$$         print*,'tmpi2',ii
c$$$         do kk=1,nstate
c$$$            write(6,'(100g25.15)') (tmp_i2(kk,jj), jj=1,nstate)
c$$$         enddo
c$$$
c$$$!         call csrdns(nstate,nstate,mat_a,mat_ja,mat_ia,op,nstate,kk)
c$$$
c$$$         print*,'mat_a'
c$$$         do kk=1,nstate
c$$$            write(6,'(100g25.15)') (op(kk,jj), jj=1,nstate)
c$$$         enddo
c$$$         endif
c$$$      enddo
c$$$
c$$$
c$$$      !Store in resrho
c$$$      do jj=1,nstate/np
c$$$         do ii=1,nstate
c$$$            resrho_r(ii,jj)= real(drho(ii,jj))
c$$$            resrho_i(ii,jj)= imag(drho(ii,jj))
c$$$         enddo
c$$$      enddo
c$$$      return
c$$$      end
c$$$      
c$$$      

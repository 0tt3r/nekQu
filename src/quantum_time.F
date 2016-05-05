c--------------------------------------------------------------------- 
      subroutine cem_quantum3_initialize    
c--------------------------------------------------------------------- 
      implicit none
      include 'SIZE'
      include 'QUANT'
#ifdef MPI
      include 'mpif.h'
#endif MPI
      integer  IERR,ii,jj
      real     one

      maxPartialTrace = 8

c$$$      call czero(s_mu_frequency,nEh)
c$$$      call czero(q_mu_frequency,nEh)
c$$$      call czero(energy_t,      nEh)

      do jj=1,nstate_np
         do ii=1,nstate
            rho_r(ii,jj) = 0.0
            rho_i(ii,jj) = 0.0
            k_rho_r(ii,jj) = 0.0
            k_rho_i(ii,jj) = 0.0
            resrho_r(ii,jj) = 0.0
            resrho_i(ii,jj) = 0.0
         enddo
      enddo
c$$$      call rzero(   rho_r,nstate2)
c$$$      call rzero(   rho_i,nstate2)
c$$$      call rzero( k_rho_r,nstate2)
c$$$      call rzero( k_rho_i,nstate2)
c$$$      call rzero(resrho_r,nstate2)
c$$$      call rzero(resrho_i,nstate2)

c$$$      do jj=1,nstate
c$$$         do ii=1,nstate
c$$$            rho_r(ii,jj) = 1./nstate
c$$$c$$$            if(ii.gt.jj) then
c$$$c$$$               rho_i(ii,jj) = -1/nstate
c$$$c$$$            else if(ii.eq.jj) then
c$$$c$$$               rho_i(ii,jj) = 0
c$$$c$$$            else
c$$$c$$$               rho_i(ii,jj) = 1/nstate
c$$$c$$$            endif
c$$$         enddo
c$$$      enddo
c!$acc parallel
c$$$      rho_r(2,2) = 0.5
c$$$      rho_r(3,3) = 0.5
c$$$      rho_r(2,3) = -0.5
c$$$      rho_r(3,2) = -0.5

!FIXME: Move density matrix initialization to user routine!
      if(levels(2).eq.2) then
      do ii=1,nstate_np
!         if(ii+c_offset.eq.16+6*32) then
!            rho_r(ii+c_offset,ii) = 1.0!/3.0
!            rho_r(ii+c_offset,ii) = 1.0
!         endif
         if(ii+c_offset.eq.4) then
            rho_r(ii+c_offset,ii) = 1.0
         endif
c$$$         if(ii+c_offset.eq.3) then
c$$$            rho_r(ii+c_offset,ii) = 1.0!/3.0
c$$$!            rho_r(ii+c_offset,ii) = 1.0
c$$$         endif
c$$$         if(ii+c_offset.eq.16) then
c$$$            rho_r(ii+c_offset,ii) = 1.0!/3.0
c$$$!            rho_r(ii+c_offset,ii) = 1.0
c$$$         endif

c$$$         if(ii+c_offset.eq.12+16) then
c$$$            rho_r(ii+c_offset,ii) = 1.0/3.0
c$$$         endif
c$$$         if(ii+c_offset.eq.12+2*16) then
c$$$            rho_r(ii+c_offset,ii) = 1.0/3.0
c$$$         endif
      enddo
      endif
!$acc end parallel

!Initial guesss for 7state model
      if(levels(2).eq.7) then
      do ii=1,nstate_np
         if(ii+c_offset.eq.8+0*levels(1)) then
            rho_r(ii+c_offset,ii) = 2.68506263e-02
         endif
         if(ii+c_offset.eq.8+1*levels(1)) then
            rho_r(ii+c_offset,ii) = 4.77526837e-02
         endif
         if(ii+c_offset.eq.8+2*levels(1)) then
            rho_r(ii+c_offset,ii) = 4.5296189e-02
         endif
         if(ii+c_offset.eq.8+3*levels(1)) then
            rho_r(ii+c_offset,ii) = 2.41502947e-02
         endif
         if(ii+c_offset.eq.8+4*levels(1)) then
            rho_r(ii+c_offset,ii) = 8.22440709e-02
         endif
         if(ii+c_offset.eq.8+5*levels(1)) then
            rho_r(ii+c_offset,ii) = 4.04902601e-02
         endif
         if(ii+c_offset.eq.8+6*levels(1)) then
            rho_r(ii+c_offset,ii) = 0.73321738 - .000001504700000
         endif
      enddo
      endif

!      if(nid.eq.0) rho_r(2,2) = 1.0
!$acc update device(rho_r)
c!$acc update device(rho_i)
c!$acc update device(k_rho_r)
c!$acc update device(k_rho_i)
c!$acc update device(resrho_r)
c!$acc update device(resrho_i)

      qctime1= 0.0
      qctime2= 0.0
      qctime3= 0.0
      qctime4= 0.0

      qrtime1= 0.0
      qrtime2= 0.0
      qrtime3= 0.0
      qrtime4= 0.0

      dt           = timemax/(nqstep)
      if(nqstep_print.gt.0) then
         printdt      = timemax/(nqstep_print)
      else
         printdt = dt
      endif
      obsPrintSkip = 1
      time    = 0
      timeOld = -1

      return
      end


c--------------------------------------------------------------------- 
      subroutine cem_quantum3_gs
c--------------------------------------------------------------------- 
      implicit none
      include 'SIZE'
      include 'QUANT'
#ifdef MPI
      include 'mpif.h'
#endif
      common /nekmpi/ nekcomm
      integer ii,jj,kk,ll,IERR,gloJ,gloI,locJ,nekcomm
      integer*8 size_type

!      nqstep  = 10000
!      dt      = timemax/(nqstep-1)
!      printdt = timemax/(999)             !We only need to print 1000 points
      time    = 0
      timeOld = -1

!     I need to update the concurrence gs_op_fields connectivity!

ccc This doesn't need to be done on the GPU, because it is for host to communicate
      ll = 1
      do jj=1,nstate_np
         do ii=1,nstate
            j0_array(ll) = (jj-1)*nstate+ii
            kk = jj+c_offset
            if(kk.ge.ii) then
               glo_num_q_Hv(ii+(jj-1)*nstate) = (ii-1)*nstate - 
     $              (ii)*(ii-1)/2 + kk
            else
               glo_num_q_Hv(ii+(jj-1)*nstate) = (kk-1)*nstate - 
     $              (kk-1)*(kk)/2 + ii
            endif
            ll = ll + 1
         enddo
      enddo


!     Skip concurrence for now
      if(calc_concurrence.eqv..true.) then
         do ii=1,nstate/ltonq
            do jj=1,ltonq
               do kk=1,ltonq
                  gloJ = ltonq*ii-(ltonq-jj)
                  gloI = ltonq*ii-(ltonq-kk)

                  if((gloJ.gt.nid*nstate_np).and.
     $                 (gloJ.le.(nid+1)*nstate_np)) then
                     locJ = gloJ - c_offset
                     glo_num_q_concV(gloI+(locJ-1)*nstate) =
     $                    kk+(jj-1)*maxPartialTrace
                  endif

c$$$  red_rho(kk,jj) = red_rho(kk,jj) + 
c$$$  $              rho(ltonq*ii-(ltonq-kk),ltonq*ii-(ltonq-jj))

               enddo
            enddo
         enddo
      endif
c$$$      open(unit=123,file='matrix')
c$$$      do ii=1,nstate*nstate
c$$$         gloJ = ii/nstate
c$$$         gloI = mod(ii,nstate)
c$$$         write(123,*)ii,gloI,gloJ,glo_num_q_concV(ii)
c$$$      enddo
c$$$      call outmat(k_rho_r,nstate,nstate,'glo_num ',nid)
#ifdef MPI
      call gs_setup(gs_handle_q_H,glo_num_q_Hv,nstate_p,
     $     MPI_COMM_WORLD,np)
      if(calc_concurrence.eqv..true.) then
         call gs_setup(gs_handle_q_conc,glo_num_q_concV,nstate_p,
     $        MPI_COMM_WORLD,np)
      endif
#else
      call gs_setup(gs_handle_q_H,glo_num_q_Hv,nstate_p,
     $     nekcomm,np)
      if(calc_concurrence.eqv..true.) then
         call gs_setup(gs_handle_q_conc,glo_num_q_concV,nstate_p,
     $        nekcomm,np)
      endif
#endif
      return
      end


c--------------------------------------------------------------------- 
      subroutine cem_quantum3_rk_storage
      implicit none
      include 'SIZE'
      include 'RK5'


      rk4a(1) =   0.0
      rk4a(2) =  -567301805773.0/1357537059087.0
      rk4a(3) =  -2404267990393.0/2016746695238.0
      rk4a(4) =  -3550918686646.0/2091501179385.0
      rk4a(5) =  -1275806237668.0/842570457699.0

      rk4b(1) =   1432997174477.0/9575080441755.0
      rk4b(2) =   5161836677717.0/13612068292357.0
      rk4b(3) =   1720146321549.0/2090206949498.0
      rk4b(4) =   3134564353537.0/4481467310338.0
      rk4b(5) =   2277821191437.0/14882151754819.0

      rk4c(1) =   0.0
      rk4c(2) =   1432997174477.0/9575080441755.0
      rk4c(3) =   2526269341429.0/6820363962896.0
      rk4c(4) =   2006345519317.0/3224310063776.0
      rk4c(5) =   2802321613138.0/2924317926251.0
      rk4c(6) =   1.

      return
      end

c-----------------------------------------------------------------------
      subroutine rk4_quantum3_upd(h,kh,resh,cb,ca)
c-----------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'QUANT'
      real h(maxN,maxN/lp),kh(maxN,maxN/lp)
      real resh(maxN,maxN/lp),cb,ca
      integer i,j
c
      do j=1,nstate_np 
         do i=1,nstate
            kh(i,j) = ca*kh(i,j) + dt*resh(i,j)
            h (i,j) = h(i,j) + cb*kh(i,j)
         enddo
      enddo

      return
      end


c-----------------------------------------------------------------------
      subroutine rk_quantum3_c(i)
c-----------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'QUANT'
      include 'RK5'

      integer  i

      RKtime= time+dt*rk4c(i)
!      writE(6,6) dt, time, RKtime
!  6   format('dt/time/RKtime:: ',1p3e20.9)

      return
      end

c-----------------------------------------------------------------------
      subroutine rk_quantum3_ab(ii)
c-----------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'QUANT'
      include 'RK5'
      include 'mpif.h'
      real*8     ca,cb
      integer  ii2,jl,kl,jj,ii

      ca = rk4a(ii)
      cb = rk4b(ii)

c$$$      do jl=1,nstate_np
c$$$         do kl=1,nstate
c$$$            k_rho_r(kl,jl) = ca*k_rho_r(kl,jl)+dt*resrho_r(kl,jl)
c$$$            rho_r(kl,jl)   = rho_r(kl,jl) + cb*k_rho_r(kl,jl)
c$$$            k_rho_i(kl,jl) = ca*k_rho_i(kl,jl)+dt*resrho_i(kl,jl)
c$$$            rho_i(kl,jl)   = rho_i(kl,jl) + cb*k_rho_i(kl,jl)
c$$$         enddo
c$$$      enddo




      call rk4_quantum3_upd(rho_r,k_rho_r,resrho_r,cb,ca)
      call rk4_quantum3_upd(rho_i,k_rho_i,resrho_i,cb,ca)

      return
      end

c--------------------------------------------------------------------- 
      subroutine cem_quantum3_propagation_exp
c--------------------------------------------------------------------- 
      implicit none
      include 'SIZE'
      include 'QUANT'

      integer  iEh, ii
      real*8   dclock
      real*8    stime0,stime1,stime2,stime3,stime4,stime5,stime6,stime7
      complex  ci

   
      ci=(0.0,1.0)
      stime0= 0.0          
      stime1= 0.0          
      stime2= 0.0          
      stime3= 0.0          
      stime4= 0.0      
      stime5= 0.0      
      stime6= 0.0      
      stime7= 0.0      
      stime0= dclock()     
      stime1= dclock()

      call cem_quantum3_propagation_init

      iqstep=1
      iiqstep = 1
      do while(time.lt.timemax)
         stime1= dclock()
         call cem_quantum3_exp
         time= time+dt

         if(useGates.eqv..true.) then
            call quantum_apply_gates()
         endif

         call cem_quantum3_observables
         if(time.gt.timeOld) then
            call cem_quantum3_pop()
            timeOld = timeOld+printdt
            iiqstep = iiqstep + 1
            stime7= dclock()-stime7
         endif
         stime1= dclock()-stime1
         qrtime1= qrtime1+stime1
         qrtime2= stime4
         qrtime3= stime5
!         qrtime4= stime7
         call userchk
         iqstep=iqstep+1

         stime2= dclock()-stime2
      enddo
      stime1= dclock()-stime1

      stime0 = dclock()-stime0
      if(nid.eq.0) write(6,*)'stime0:',stime0 

      call cem_quantum3_propagation_finalize

      return
      end

c--------------------------------------------------------------------- 
      subroutine cem_quantum3_exp
c--------------------------------------------------------------------- 
      implicit none
      include 'SIZE'
      include 'QUANT'
      integer  icalld
      save     icalld
      data     icalld/0/
      real*8     glsc2,beta,tmp,s1
      integer  m,info,i,j,j0
      integer  n2
      save     n2


      qmarnol= larnol 
      m      = 5 !qmarnol

      if (icalld.eq.0) then

         n2     = 2*nstate2 
         call rzero(qu ,n2)
         call rzero(qv ,n2*m)
         call rzero(qh ,m*m)
         call rzero(qw ,m)
         call rzero(qws,2*m*(m+2))
         icalld = 1

      endif

      !consider rolling below into quantum3_arnoldi_vec
      do j=1,nstate_np
         do i=1,nstate
            j0 = (j-1)*nstate+i
            tmprv(j0)        = rho_r(i,j)
            tmpiv(j0)        = rho_i(i,j)
         enddo
      enddo

      beta = glsc2(tmprv,tmprv,nstate2) + glsc2(tmpiv,tmpiv,nstate2)
      s1   = beta
      beta= sqrt (beta)

      call quantum3_arnoldi_vec (qv,qh,n2,m,s1)

      call copy_r2z   (qzh,qh,m*m)
      
      call zgeev      ('N','V',m,qzh,m,qeval,qzri,m,qzr,m,qawork,
     $                  2*m,qrwork,info)

      call copy_z2z   (qzri,qzr,m*m)

      call zgetrf     (m,m,qzri,m,iqpiv,info)        !vri= LU factorization                

      call zgetri     (m,qzri,m,iqpiv,qawork,m,info) !vri= X^(-1)

      call z_exp      (qeval,dt,m)                   !eval=exp(dt*eval)
      call z_rowscale (qzri,qeval,m)                 !vri =eval*X^(-1); 
      call c_mxm      (qzr,m,qzri,m,qzh,m)           !zh =X*vri  

      call copy_z2r   (qw,qzh(1,1),m)                 !zh(i,1) (i=1:m)
      call mxm        (qv,n2,qw,m,qu,1)
   
      do j=1,nstate_np
         do i=1,nstate
            j0 = (j-1)*nstate+i
            rho_r(i,j) = beta*qu(0*nstate2+j0)
            rho_i(i,j) = beta*qu(1*nstate2+j0)
         enddo
      enddo

      return
      end

c--------------------------------------------------------------------- 
      subroutine quantum3_arnoldi_vec(v,h,n,m,s1)
c--------------------------------------------------------------------- 
      implicit none
      include 'SIZE'
      include 'QUANT'
      
      integer  i,j,m,n
      real*8     v(n,m)
      real*8     h(m,m)
      real*8     eps, s1, si, glsc2, cnt

      eps = 1.e-13

      si = 1./sqrt(s1)                     ! should add a check and a comment here

      do i=1,nstate2
         v(i,1) = tmprv(i)*si
         v(nstate2+i,1) = tmpiv(i)*si
      enddo

!     We split the final iteration out so that we can save a few flops
!     but mostly so we can have only one array of size (n,m) instead of
!     needing an array of size (n,m+1)
      do j=1,m-1
         call quantum3_mult (v(1,j+1),v(1,j),n)
         do i=1,j
            h(i,j) = glsc2 (v(1,j+1),v(1,i),n)
            cnt    =       -h(i,j  )
            call add2s2    (v(1,j+1),v(1,i),cnt,n)
         enddo
         s1 = glsc2(v(1,j+1),v(1,j+1),n)
         s1 = sqrt(s1)
         h(j+1,j) = s1
         si = 1.0/s1
         do i=1,n
            v(i,j+1) = si*v(i,j+1)
         enddo
      enddo

!     Final iteration: j=m
      call quantum3_mult (qu,v(1,m),n)
      do i=1,m
         h(i,m) = glsc2 (qu,v(1,i),n)
         cnt    =       -h(i,m  )
         call add2s2    (qu,v(1,i),cnt,n)
      enddo

      return
      end

c-----------------------------------------------------------------------
      subroutine quantum3_mult(w,v,n2)
c-----------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'QUANT' 
      include 'mpif.h'
      integer  n2,i,j,j0
      real*8     w(n2),v(n2)

      do j=1,nstate_np
         do i=1,nstate
            j0 = (j-1)*nstate+i
            rho_r(i,j) = v(0*nstate2+j0)
            rho_i(i,j) = v(1*nstate2+j0)
         enddo
      enddo

      call cem_quantum3_hamiltonian
      call quantum_lin_mxm
!      call quantum_lin_mpi_get

      do j=1,nstate_np
         do i=1,nstate
            j0 = (j-1)*nstate+i
            w(0*nstate2+j0) = resrho_r(i,j)
            w(1*nstate2+j0) = resrho_i(i,j)
         enddo
      enddo

      return
      end



c------------------------------------------------------------------
      subroutine cem_quantum3_propagation_init
c------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'QUANT'
      integer ii,jj
      complex  sig

      if(nqstep_print.gt.maxTimeSteps) then
         print*, 'ERROR!'
         print*, 'nqstep_print TOO LARGE FOR THIS maxTimeSteps!'
         print*, 'maxTimeSteps = ',maxTimeSteps
         call exitt()
      endif
      if(ham_finalized.eqv..false.) then
         print*, 'ERROR!'
         print*, 'NO HAMILTONIAN!'
         print*, 'Did you call quantum_ham_finalize_csr?'
         call exitt()
      endif

      if(nid.eq.0) then
c     Write file headers 
         if(ifPulse.eqv..true.) then
            write(13,*)'#Time,Pulse,Observables'
         else
            write(13,*)'#Time,Observables'
         endif
         write(12,*)'#Time,Populations'
         if(calc_concurrence.eqv..true.) then
            if(ltonq.eq.8) then
               write(14,*)'#Time,3:2,3:1,2:1'
               write(15,*)'#Time,3:2 - popS,popA,popS1,
     $              popA1, 3:1 - , 2:1'
            else if(ltonq.eq.4) then
               write(14,*)'#Time,1:2'
               write(15,*)'#Time,popS,popA,popS1,popA1'
            endif
         endif
      endif

      steady_state = .false.

      call cem_quantum3_initialize      
      call cem_quantum3_gs


      if(ifPulse.eqv..true.) then
         do ii=1,nEh
            en_t(ii) = (0.0,0.0)
            do jj=1,numObservables
               obs_freq(jj,ii) = (0.0,0.0)
            enddo
         enddo
      endif



      return
      end


c--------------------------------------------------------------------- 
      subroutine cem_quantum3_propagation_pece
c--------------------------------------------------------------------- 
      implicit none
      include 'SIZE'
      include 'QUANT'
      include 'RK5'
      integer  iEh, ii, IERR, jj,kk,j0,iflag,neqn
      real*8     stime0,stime1,stime2,stime3,stime4,stime5,stime6,stime7
      real*8   dclock,old_dt,relerr,abserr,told,tnew
      complex  sig,tmpc1,ci
      external quantum3_mult_pece
      ci=(0.0,1.0)
      stime0= 0.0          
      stime1= 0.0          
      stime2= 0.0          
      stime3= 0.0          
      stime4= 0.0      
      stime5= 0.0      
      stime6= 0.0      
      stime7= 0.0      
      stime0= dclock()     
      stime1= dclock()
      relerr = 1e-10
      abserr = 1e-10
      call cem_quantum3_propagation_init
      
ccc Setup acc stuff

!$acc data copyin(hamilt_I_m_p,hamilt_0_m_p,hamilt_m_p)
!$acc&     copyin(ham_a,ham_ia,ham_ja,ham_0_m_p_a,ham_m_p_a)
!$acc&     pcreate(rho_i,rho_r,rho,drho,resrho_r,resrho_i)
!$acc&     copyin(isstate,iqstate_m,state_sums,nq_minimal)
!$acc&     pcreate(tmp_r1,tmp_i1,tmp_c1,k_rho_r,k_rho_i)
!$acc&     copyin(rk4a,rk4b,rk4c)
ccc parameters for the GPU to use - scalars automatically imported
c!$acc&     copyin(numqd,numsp,nstate,n_in_row,nEh,nqstep,timemax)
c!$acc&     copyin(Emin,Emax,dEh,relerr,abserr)
c!$acc&     copyin(omega_s,gamma_s,s_mu)
!$acc&     copyin(omega_q_m,gamma_q_m,gamma_d_m,q_mu_m,g_couple_m)
c!$acc&     copyin(pulse_omega_0,pulse_t_0,pulse_duration,energy0_pls)
c!$acc&     copyin(nstate_p,c_offset,dt,nid,ltonq,IFSMALLMEMORY)

      neqn = 2*nstate_np*nstate
      iflag = 1
       iqstep=1
       iiqstep = 1
!       old_dt = dt

       do iqstep=0,nqstep-1
          stime1= dclock()
          !Store in vector
          do jj=1,nstate_np
             do ii=1,nstate
                j0 = (jj-1)*nstate+ii
                r_cg(0*nstate2+j0) = rho_r(ii,jj)
                r_cg(1*nstate2+j0) = rho_i(ii,jj)
             enddo
          enddo

          
          told = iqstep*dt
          tnew = (iqstep+1)*dt
          iflag=1
          call ode(quantum3_mult_pece,neqn,r_cg,told,tnew,
     $         relerr,abserr,iflag,work_ode,iwork)

! Check that integration went okay
          if(iflag.ne.2) then
             print*,"ODE integrator gave warning flag", iflag, 
     $            "at time step", ii
             call exitt()
          endif
             
          stime5 = stime5 + dclock()-stime7
          stime7 = dclock()

          do jj=1,nstate_np
             do ii=1,nstate
                j0 = (jj-1)*nstate+ii
                rho_r(ii,jj) = r_cg(0*nstate2+j0)
                rho_i(ii,jj) = r_cg(1*nstate2+j0)
             enddo
          enddo
          

          qrtime4 = qrtime4 + dclock() - stime7

c         if(abs(time-pulse_t_0).lt.3*pulse_duration) then
c             dt=dt_small
c         else
c             dt=dt_large
c         endif
c$$$         if(time.gt.timemax/2) then
c$$$            dt = old_dt
c$$$         endif
         time= time+dt
         stime7= dclock()
         if(useGates.eqv..true.) then
            call quantum_apply_gates()
         endif
         call cem_quantum3_observables
         if(time.gt.timeOld) then
            call cem_quantum3_pop()
            timeOld = timeOld+printdt
            iiqstep = iiqstep + 1
            stime7= dclock()-stime7
         endif
         stime1= dclock()-stime1
         qrtime1= qrtime1+stime1
         qrtime2= stime4
         qrtime3= stime5
!         qrtime4= stime7
         call userchk
!         if(windowing) call quantum_update_window()
c$$$         if(iqstep.gt.100) then
c$$$            stime0 = dclock()-stime0
c$$$            if(nid.eq.0) write(6,*)'stime0:',stime0 
c$$$            call exitt()
c$$$         endif
      enddo

      stime0 = dclock()-stime0
      if(nid.eq.0) write(6,*)'stime0:',stime0 

c$$$      print*,"Rho_r final"
c$$$      do ii=1,nstate
c$$$         write(6,'(100g15.5)') (rho_r(ii,jj), jj=1,nstate)
c$$$      enddo

      call cem_quantum3_propagation_finalize

!$acc end data

      return
      end



c--------------------------------------------------------------------- 
      subroutine cem_quantum3_propagation_rk
c--------------------------------------------------------------------- 
      implicit none
      include 'SIZE'
      include 'QUANT'
      include 'RK5'
      integer  iEh, ii, IERR, jj,kk,ii2
      real*8     stime0,stime1,stime2,stime3,stime4,stime5,stime6,stime7
      real*8   dclock,old_dt
      complex  sig,tmpc1,ci

      ci=(0.0,1.0)
      stime0= 0.0          
      stime1= 0.0          
      stime2= 0.0          
      stime3= 0.0          
      stime4= 0.0      
      stime5= 0.0      
      stime6= 0.0      
      stime7= 0.0      
      stime0= dclock()     
      stime1= dclock()

      call cem_quantum3_propagation_init
      call cem_quantum3_rk_storage

ccc Setup acc stuff

!$acc data copyin(hamilt_I_m_p,hamilt_0_m_p,hamilt_m_p)
!$acc&     copyin(ham_a,ham_ia,ham_ja,ham_0_m_p_a,ham_m_p_a)
!$acc&     pcreate(rho_i,rho_r,rho,drho,resrho_r,resrho_i)
!$acc&     copyin(isstate,iqstate_m,state_sums,nq_minimal)
!$acc&     pcreate(tmp_r1,tmp_i1,tmp_c1,k_rho_r,k_rho_i)
!$acc&     copyin(rk4a,rk4b,rk4c)
ccc parameters for the GPU to use - scalars automatically imported
c!$acc&     copyin(numqd,numsp,nstate,n_in_row,nEh,nqstep,timemax)
c!$acc&     copyin(Emin,Emax,dEh,relerr,abserr)
c!$acc&     copyin(omega_s,gamma_s,s_mu)
!$acc&     copyin(omega_q_m,gamma_q_m,gamma_d_m,q_mu_m,g_couple_m)
c!$acc&     copyin(pulse_omega_0,pulse_t_0,pulse_duration,energy0_pls)
c!$acc&     copyin(nstate_p,c_offset,dt,nid,ltonq,IFSMALLMEMORY)


       iqstep=1
       iiqstep = 1
!       old_dt = dt

       do while(time.lt.timemax)
          stime1= dclock()
         do ii= 1,5
            stime7 = dclock()


            call cem_quantum3_hamiltonian()
            stime4 = stime4 + dclock()-stime7
            stime7 = dclock()



            call quantum_lin_mxm()
!            call quantum_lin_mpi_get()


            stime5 = stime5 + dclock()-stime7
            stime7 = dclock()
            call rk_quantum3_ab(ii)
            qrtime4 = qrtime4 + dclock() - stime7
         enddo
c         if(abs(time-pulse_t_0).lt.3*pulse_duration) then
c             dt=dt_small
c         else
c             dt=dt_large
c         endif
c$$$         if(time.gt.timemax/2) then
c$$$            dt = old_dt
c$$$         endif
         time= time+dt
         stime7= dclock()
         if(useGates.eqv..true.) then
            call quantum_apply_gates()
         endif
         call cem_quantum3_observables
         if(time.gt.timeOld) then
            call cem_quantum3_pop()
            timeOld = timeOld+printdt
            iiqstep = iiqstep + 1
            stime7= dclock()-stime7
         endif
         stime1= dclock()-stime1
         qrtime1= qrtime1+stime1
         qrtime2= stime4
         qrtime3= stime5
!         qrtime4= stime7
         call userchk
!         if(windowing) call quantum_update_window()
         iqstep=iqstep+1
c$$$         if(iqstep.gt.100) then
c$$$            stime0 = dclock()-stime0
c$$$            if(nid.eq.0) write(6,*)'stime0:',stime0 
c$$$            call exitt()
c$$$         endif
      enddo

      stime0 = dclock()-stime0
      if(nid.eq.0) write(6,*)'stime0:',stime0 

c$$$      print*,"Rho_r final"
c$$$      do ii=1,nstate
c$$$         write(6,'(100g15.5)') (rho_r(ii,jj), jj=1,nstate)
c$$$      enddo

      call cem_quantum3_propagation_finalize

!$acc end data

      return
      end


! Do RK45 without initialization
c--------------------------------------------------------------------- 
      subroutine cem_quantum3_propagation2_rk
c--------------------------------------------------------------------- 
      implicit none
      include 'SIZE'
      include 'QUANT'
      include 'RK5'
      integer  iEh, ii, IERR, jj,kk,ii2
      real*8     stime0,stime1,stime2,stime3,stime4,stime5,stime6,stime7
      real*8   dclock,old_dt
      complex  sig,tmpc1,ci

      ci=(0.0,1.0)
      stime0= 0.0          
      stime1= 0.0          
      stime2= 0.0          
      stime3= 0.0          
      stime4= 0.0      
      stime5= 0.0      
      stime6= 0.0      
      stime7= 0.0      
      stime0= dclock()     
      stime1= dclock()

ccc Setup acc stuff

!$acc data copyin(hamilt_I_m_p,hamilt_0_m_p,hamilt_m_p)
!$acc&     copyin(ham_a,ham_ia,ham_ja,ham_0_m_p_a,ham_m_p_a)
!$acc&     pcreate(rho_i,rho_r,rho,drho,resrho_r,resrho_i)
!$acc&     copyin(isstate,iqstate_m,state_sums,nq_minimal)
!$acc&     pcreate(tmp_r1,tmp_i1,tmp_c1,k_rho_r,k_rho_i)
!$acc&     copyin(rk4a,rk4b,rk4c)
ccc parameters for the GPU to use - scalars automatically imported
c!$acc&     copyin(numqd,numsp,nstate,n_in_row,nEh,nqstep,timemax)
c!$acc&     copyin(Emin,Emax,dEh,relerr,abserr)
c!$acc&     copyin(omega_s,gamma_s,s_mu)
!$acc&     copyin(omega_q_m,gamma_q_m,gamma_d_m,q_mu_m,g_couple_m)
c!$acc&     copyin(pulse_omega_0,pulse_t_0,pulse_duration,energy0_pls)
c!$acc&     copyin(nstate_p,c_offset,dt,nid,ltonq,IFSMALLMEMORY)


       iqstep=1
       iiqstep = 1
!       old_dt = dt

       do while(time.lt.timemax)
          stime1= dclock()
         do ii= 1,5
            stime7 = dclock()


            call cem_quantum3_hamiltonian()
            stime4 = stime4 + dclock()-stime7
            stime7 = dclock()



            call quantum_lin_mxm()
!            call quantum_lin_mpi_get()


            stime5 = stime5 + dclock()-stime7
            stime7 = dclock()
            call rk_quantum3_ab(ii)
            qrtime4 = qrtime4 + dclock() - stime7
         enddo
c         if(abs(time-pulse_t_0).lt.3*pulse_duration) then
c             dt=dt_small
c         else
c             dt=dt_large
c         endif
c$$$         if(time.gt.timemax/2) then
c$$$            dt = old_dt
c$$$         endif
         time= time+dt
         stime7= dclock()
         if(useGates.eqv..true.) then
            call quantum_apply_gates()
         endif
         call cem_quantum3_observables
         if(time.gt.timeOld) then
            call cem_quantum3_pop()
            timeOld = timeOld+printdt
            iiqstep = iiqstep + 1
            stime7= dclock()-stime7
         endif
         stime1= dclock()-stime1
         qrtime1= qrtime1+stime1
         qrtime2= stime4
         qrtime3= stime5
!         qrtime4= stime7
         call userchk
!         if(windowing) call quantum_update_window()
         iqstep=iqstep+1
c$$$         if(iqstep.gt.100) then
c$$$            stime0 = dclock()-stime0
c$$$            if(nid.eq.0) write(6,*)'stime0:',stime0 
c$$$            call exitt()
c$$$         endif
      enddo

      stime0 = dclock()-stime0
      if(nid.eq.0) write(6,*)'stime0:',stime0 

c$$$      print*,"Rho_r final"
c$$$      do ii=1,nstate
c$$$         write(6,'(100g15.5)') (rho_r(ii,jj), jj=1,nstate)
c$$$      enddo


!$acc end data

      return
      end

C   ROUTINE BELOW UNTESTED
c---------------------------------------------------------------------
      subroutine quantum_update_window
c---------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'QUANT'
#ifdef MPI
      include 'mpif.h'
#endif
      integer ii,jj,move_window,ierr,req_sen,req_rec,j0
      integer status,buf_size
      real pop,minimum_pop

      !Check if highest energy state is depopulated
      !Highest energy state is assumed to be last state, last core
      !Assumes oscillator is first operator
      !Assumes atleast 2*ltonq is on one core
      if(nid.eq.(np-1)) then

         pop = 0
      !Find population in last oscillator state
         do ii=1,ltonq !ltonq is everything but windowed operator
            ! Don't multiply by state because we only care about the 
            ! value of the elements
            pop = pop + rho_r(nstate_np-(ii-1),nstate-(ii-1))
         enddo

         !If the pop is small enough, we need to 
         !move the window
         if(pop.lt.minimum_pop) then
            !Add the pop from the exiting columns to the columns
            !1 state down - probably not correct, but if
            !pop is low enough, should not matter
            do ii=1,ltonq
               rho_r(nstate_np-(ltonq + ii-1),nstate-(ltonq + ii-1)) = 
     $              rho_r(nstate_np-(ltonq+ii-1),nstate-(ltonq+ii-1)) +
     $              rho_r(nstate_np-(ii-1),nstate-(ii-1))
            enddo

            !Send '1' to other cores to signify that we need
            !to move the window
            move_window = 1
#ifdef MPI            
            call MPI_BCAST(move_window,1,MPI_INTEGER,np,
     $              MPI_COMM_WORLD,ierr)
#endif
         else 
            !Send '0' to other cores to signify that we do not need
            !to move the window

            move_window = 0
#ifdef MPI            
            call MPI_BCAST(move_window,1,MPI_INTEGER
     $              ,np,MPI_COMM_WORLD,ierr)
#endif

         endif
      else
         !Receive instructions about moving window
#ifdef MPI
         call MPI_BCAST(move_window,1,MPI_INTEGER
     $        ,np,MPI_COMM_WORLD,ierr)
#endif
      endif

         
      if(move_window.eq.1) then
         buf_size = ltonq*(nstate-ltonq)
         
         !all cores but final
         if(nid.ne.(np-1)) then

            if(nid.ne.0) then
!     Post MPI_IRECV on our incoming columns - core 0 has no upper neighbor
               call MPI_IRECV(tmpiv,2*buf_size,MPI_REAL,
     $              nid-1,MPI_ANY_TAG,MPI_COMM_WORLD,req_rec,ierr)
            endif

            !Send off last ltonq columns
            !Pack buffers
            do jj=nstate_np-ltonq+1,nstate_np
               do ii=1,nstate-ltonq 
!     We don't care about the last rows, they are being windowed away
                  !We want to fill the first part of 
                  !tmprv, so we offset j0 to start at 1
                  j0 = (jj-(nstate_np-ltonq)-1)*nstate+ii
                  tmprv(j0)          = rho_r(ii,jj)
                  tmprv(j0+buf_size) = rho_i(ii,jj)
               enddo
            enddo

            !MPI_ISEND to next core
            call MPI_ISEND(tmprv,2*buf_size,MPI_REAL,nid+1,
     $           MPI_ANY_TAG,MPI_COMM_WORLD,req_sen,ierr)
!     Now shuffle the columns down the diagonal while we wait for the message to come
            do jj=1,nstate_np-ltonq
               do ii=1,nstate-ltonq
!     We want to go in reverse order, hence the nstate- ...
                  rho_r(nstate_np-jj+1,nstate-ii+1) = 
     $                 rho_r(nstate_np-ltonq-jj+1,nstate-ltonq-ii+1)
               enddo
            enddo

            if(nid.ne.0) then
               call MPI_WAIT(req_rec,status,ierr)
               
!     Shuffle the new columns into their proper places
               do jj=1,ltonq
                  do ii=ltonq+1,nstate
                     !We want to store these new columns in the lower
                     !parts of the columns
                     j0=(jj-1)*nstate+ii-ltonq
                     rho_r(ii,jj) = tmpiv(j0)
                     rho_i(ii,jj) = tmpiv(j0+buf_size)
                  enddo
               enddo
               
            else 
!     First core needs to 0 out the whole first ltonq columns,
!     in addition to the firt ltonq rows
               do jj=1,ltonq
                  do ii=1,nstate
                     rho_r(ii,jj) = 0.0
                     rho_i(ii,jj) = 0.0
                  enddo
               enddo
            endif
         else                   !Final core, who has nothing to send

!     Post MPI_IRECV on our incoming columns - core 0 has no upper neighbor
            if(np.gt.1) then
               call MPI_IRECV(tmpiv,2*buf_size,MPI_REAL,nid-1,
     $              MPI_ANY_TAG,MPI_COMM_WORLD,req_rec,ierr)
            endif

!     Now shuffle the columns down the diagonal while we wait for the message to come
            do jj=1,nstate_np-ltonq
               do ii=1,nstate-ltonq
!     We want to go in reverse order, hence the nstate- ...
                  rho_r(nstate_np-jj+1,nstate-ii+1) = 
     $                 rho_r(nstate_np-ltonq-jj+1,nstate-ltonq-ii+1)
               enddo
            enddo
            !Wait for our new data
            if(np.gt.1) then
               call MPI_WAIT(req_rec,status,ierr)

!     Shuffle the new columns into their proper places
               do jj=1,ltonq
                  do ii=ltonq+1,nstate
!     We want to store these new columns in the lower
!     parts of the columns
                     j0=(jj-1)*nstate+ii-ltonq
                     rho_r(ii,jj) = tmpiv(j0)
                     rho_i(ii,jj) = tmpiv(j0+buf_size)
                  enddo
               enddo
            else 
!     First core needs to 0 out the whole first ltonq columns,
!     in addition to the first ltonq rows
               do jj=1,ltonq
                  do ii=1,nstate
                     rho_r(ii,jj) = 0.0
                     rho_i(ii,jj) = 0.0
                  enddo
               enddo
            endif

         endif
!     Everyone does the below
!     The first ltonq rows of rho are 0, since they represent the newly introduced state
         do jj=1,nstate_np
            do ii=1,ltonq
               rho_r(ii,jj) = 0.0
               rho_i(ii,jj) = 0.0
            enddo
         enddo

      endif

      return
      end

c---------------------------------------------------------------------
      subroutine cem_quantum3_propagation_finalize
c---------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'QUANT'
#ifdef MPI
      include 'mpif.h'
#endif

      integer  iEh, ii, IERR, jj,kk,number_pops
      real*8     sigr,sigi,tmpr,tmpi,rinc,sigcon,pulseFactor,fullPulse
      real*8   dclock,c_speed,d_per_au,eesu_per_au,eps_med,eV,GHz
      complex  sig,tmpc1,ci,tmpc2

      tmpr = 0
!Reduce the trace of rho as a sanity check - should always be 1!
      do ii=1,nstate_np
         tmpr = tmpr + rho_r(ii+c_offset,ii)
      enddo
      if(nid.eq.0) then
         call MPI_REDUCE(MPI_IN_PLACE,tmpr,1,MPI_REAL8,MPI_SUM,0,
     $        MPI_COMM_WORLD,IERR)
      else
         call MPI_REDUCE(tmpr,tmpr,1,MPI_REAL8,MPI_SUM,0,
     $        MPI_COMM_WORLD,IERR)
      endif
      if(nid.eq.0) print*,"rho_r trace",tmpr

      iiqstep = iiqstep - 1
      iqstep = iqstep -1
!     Reduce time vectors
#ifdef MPI
      kk=1
      if(print_during_run.eqv..false.) then
         do ii=1,numberOfOps
            if(is_vec(ii).eqv..true.) then
               do jj=1,levels(ii)
                  if(nid.eq.0) then
                     call MPI_REDUCE(MPI_IN_PLACE,pop_time(1,kk),
     $                    iiqstep,MPI_REAL8,MPI_SUM,0,
     $                    MPI_COMM_WORLD,IERR)
                  else
                     call MPI_REDUCE(pop_time(1,kk),pop_time(1,kk),
     $                    iiqstep,MPI_REAL8,MPI_SUM,0,
     $                    MPI_COMM_WORLD,IERR)
                  endif
                  kk = kk+1
               enddo
            else
               if(nid.eq.0) then
                  call MPI_REDUCE(MPI_IN_PLACE,pop_time(1,kk),iiqstep,
     $                 MPI_REAL8,MPI_SUM,0,MPI_COMM_WORLD,IERR)
               else
                  call MPI_REDUCE(pop_time(1,kk),pop_time(1,kk),iiqstep,
     $                 MPI_REAL8,MPI_SUM,0,MPI_COMM_WORLD,IERR)
               endif
               kk = kk+1
            endif
         enddo

      
         number_pops = kk - 1
         do ii=1,numObservables
            if(nid.eq.0) then
               call MPI_REDUCE(MPI_IN_PLACE,obs_time(1,ii),iqstep,
     $              MPI_REAL8,MPI_SUM,0,MPI_COMM_WORLD,IERR)
            else
               call MPI_REDUCE(obs_time(1,ii),obs_time(1,ii),iqstep,
     $              MPI_REAL8,MPI_SUM,0,MPI_COMM_WORLD,IERR)
            endif
         enddo
      endif
#endif

      ci = (0.0,1.0)
      if(nid.eq.0) then
         time = dt
         kk = 1
         if(ifPulse.eqv..true.) then
            
            do ii=1,iqstep
               
               call usertimedep(pulseFactor)

               if(rwa.eqv..true.) then
                  fullPulse = 2*pulseFactor*
     $                 cos(pulse_omega_0*(time-pulse_t_0))
               else
                  fullPulse = pulseFactor
               endif
c     Now, do the fourier transform to save the value
!     Envelope*oscillations --- consider putting oscillations in user control?
               
               do kk=1,nEh
                  en_t(kk) = en_t(kk)+fullPulse*
     $                 exp(ci*omega_quant(kk)*(time-pulse_t_0))
                  do jj=1,numObservables
                     obs_freq(jj,kk) = obs_freq(jj,kk) +  
     $                    (exp(ci*omega_quant(kk)*(time-pulse_t_0))*
     $                    obs_time(ii,jj))
                  enddo

               enddo
               time = time + dt
            enddo
         endif

         kk = 1
         time = dt
         if(print_during_run.eqv..false.) then
            do while(time.lt.timemax)
               if(ifPulse.eqv..true.) then
                  call usertimedep(pulseFactor)
                  if(rwa.eqv..true.) then
                     fullPulse = 2*pulseFactor*
     $                    cos(pulse_omega_0*(time-pulse_t_0))
                  else
                     fullPulse = pulseFactor
                  endif
                  if(abs(fullPulse).lt.1e-20) fullPulse = 0
                  
                  write(13,12)time*timeunit,fullPulse
     $                 ,(obs_time(kk,jj),jj=1,numObservables)
               else
                  write(13,12)time*timeunit
     $                 ,(obs_time(kk,jj),jj=1,numObservables)
               endif
               kk = kk + obsPrintSkip
               time = time + obsPrintSkip*dt
            enddo

            time = dt

            do ii=1,iiqstep
!Print populations
               write(12,12)time*timeunit
     $              ,(pop_time(ii,jj),jj=1,number_pops)
               
               time = time+printdt
            enddo
 12         format(1p20g20.8)
         endif
c     Print out results
         if(ifPulse.eqv..true.) then
            write(11,*)'#eV,sig_abs,sig_scatt,sig_tot'

!     Update this stuff!
!     I'm not sure how to do this is a really user friendly manner
!     FIXME! Hardcoded
            c_speed = 2.99792458d10 !cm/s
            d_per_au = 1.0d0 / 3.93430307d-1
            eesu_per_au = 5.14220652d17 / c_speed
            eps_med     = 2.25
            eV       = 27.21140 ! energy Eh  
            GHz = 1!.519827e-7   !In AU
            do ii=1,nEh
               rinc = 4.0*pi*(omega_quant(ii)/timeunit)/
     $              (c_speed*eps_med**0.5)*
     $              d_per_au*1.0d-18/eesu_per_au
               sig = (0,0)
               do jj=1,numObservables
                  sig = sig + obs_freq(jj,ii)
               enddo
               tmpc2 = sig

               sig  = rinc*sig/en_t(ii)
               rinc = 8.0d0*pi*(omega_quant(ii)/timeunit)**4/
     $              (3.0d0*c_speed**4)*
     $              d_per_au*1.0d-18/eesu_per_au
               tmpc1 = (0,0)
               do jj = 1,numObservables
                  tmpc1 = tmpc1 + obs_freq(jj,ii)
               enddo
               sigr = rinc * cabs(tmpc1/en_t(ii))**2
               sigi  = imag(sig)
!     write(11,13) omega_quant(ii),en_t(ii),
!     $        obs_freq(1,ii),obs_freq(2,ii)+
!     $        obs_freq(3,ii)
               write(11,13) omega_quant(ii),sigi,sigr,sigr+sigi,
     $              real(tmpc2),imag(tmpc2),real(en_t(ii)),
     $              imag(en_t(ii))
            enddo
            
 13         format(1p9e20.10)
         endif
         close(12)
         close(11)
      endif


      return
      end
c--------------------------------------------------------------------- 
      subroutine cem_quantum3_hamiltonian()
c--------------------------------------------------------------------- 
      implicit none
      include 'SIZE'
      include 'QUANT'
#ifdef MPI
      include 'mpif.h'
#endif
      integer    ii,jj,iEh,ierr,i_one,i_zero,i0,j0,kk
      real*8     tmprr,tmpii,pulseFactor_loc,sqrtns
      real*8     alpha,pulseFactor
      complex*16 tmpfq,tmpcc
      complex*16 ci
      real*8     dclock,ttmp

      i_one = 1
      i_zero = 0
      ci=(0.0,1.0)


!$acc data pcreate(tmprv,tmpiv)
!$acc&     present(rho_r,rho_i,tmp_r1,tmp_i1,resrho_i,resrho_r,drho)
!$acc&     present (hamilt_0_m_p,hamilt_I_m_p,hamilt_m_p,rho,state_sums)
!$acc&     present(ham_a,ham_0_m_p_a,ham_m_p_a)

c     Ensure Hermiticity of rho
!      kk = 1
!$acc parallel loop collapse(2) private(j0)
c!$omp parallel do collapse(2)
      do jj = 1,nstate_np
      do ii = 1,nstate
!         j0 = j0_array(kk)!(jj-1)*nstate+ii
         j0 = (jj-1)*nstate+ii
         tmprv(j0) = rho_r(ii,jj)
         tmpiv(j0) = rho_i(ii,jj)
!         kk = kk + 1
      enddo
      enddo
         
c!$omp end parallel do
!$acc end parallel loop

      ttmp = dclock()

      
!      call nekgsync()!#FIXME WHY IS THIS NECESSARY?
 !     print*,'before gs_op_fields',nid
!      call gs_op_fields(gs_handle_q_H,tmppp,
!     $     nstate_p,2,1,1,0) 
      call gs_op_fields(gs_handle_q_H,tmprv,
     $     nstate_p,1,1,1,0) 
!      print*,'before gs_op_fields2'
      call gs_op_fields(gs_handle_q_H,tmpiv,nstate_p,1,1,1,0)
!      qctime1= qctime1+dclock()-ttmp     
!      print*,'after gs_op2'
!      kk = 1
!$acc parallel loop collapse(2) private(j0)
c!$omp parallel do collapse(1)
      do jj=1,nstate_np
      do ii=1,nstate
!         j0 = j0_array(kk)!(jj-1)*nstate+ii
         j0 = (jj-1)*nstate+ii
         !if(ii.ne.jj+c_offset) rho_r(ii,jj) = tmprv(j0)/2
         rho_r(ii,jj) = tmprv(j0)/2
         rho_i(ii,jj) = (2*rho_i(ii,jj) - 
     $        tmpiv(j0))/2
!         kk = kk + 1
!         if(jj+c_offset.eq.ii) rho_i(ii,jj) = 0
      enddo
      enddo

!     The diagonal parts need special treatment
      do jj=1,nstate_np
         rho_r(jj+c_offset,jj) = rho_r(jj+c_offset,jj)*2
         rho_i(jj+c_offset,jj) = 0.0
      enddo

c!$omp end parallel do
!$acc end parallel loop


!$acc parallel loop collapse(2)
c!$omp parallel do collapse(1) 
      do jj = 1,nstate_np
      do ii = 1,nstate
         rho(ii,jj)= rho_r(ii,jj)+ci*rho_i(ii,jj)      
      enddo
      enddo

      !Time dependent stuff will need to change!
      if(ifPulse.eqv..true.) then
         call usertimedep(pulseFactor_loc)
      endif
!$acc parallel loop

      do ii=1,ham_nnz
         ham_a(ii) = ham_i(ii)
         if(ifPulse.eqv..true.) then
            ham_a(ii) = ham_a(ii) + pulseFactor_loc*ham_d(ii)
         endif
         !Currently only supports a single time dependence
c$$$         do jj=1,nTimeDep
c$$$            ham_a(ii) = ham_a(ii) + pulseFactor_loc*ham_d(ii)
c$$$         enddo
         !if(nid.eq.0) print*,ham_a(ii),ham_i(ii),ham_d(ii)
      enddo

!$acc end parallel loop
      
c$$$      print*,'pre'
c$$$      print*,'rho_r'
c$$$      do ii=1,nstate
c$$$         write(6,'(100g15.5)') (rho_r(ii,jj), jj=1,nstate)
c$$$      enddo
c$$$      print*,'rho_i'
c$$$      do ii=1,nstate
c$$$         write(6,'(100g15.5)') (rho_i(ii,jj), jj=1,nstate)
c$$$      enddo
      ttmp = dclock()
      call cem_quantum3_csr_mxm(rho_r,tmp_r1,ham_a,ham_ia,ham_ja)
      call cem_quantum3_csr_mxm(rho_i,tmp_i1,ham_a,ham_ia,ham_ja)
      qctime3 = qctime3 + dclock() - ttmp
c$$$      print*,'post'
c$$$      print*,'tmp_r1'
c$$$      do ii=1,nstate
c$$$         write(6,'(100g15.5)') (tmp_r1(ii,jj), jj=1,nstate)
c$$$      enddo
c$$$      print*,'tmp_i1'
c$$$      do ii=1,nstate
c$$$         write(6,'(100g15.5)') (tmp_i1(ii,jj), jj=1,nstate)
c$$$      enddo

!      kk = 1
!$acc parallel loop collapse(2)
c!$omp parallel do collapse(1)
      do jj = 1,nstate_np
      do ii = 1,nstate
!         j0 = j0_array(kk)!(jj-1)*nstate+ii
         j0 = (jj-1)*nstate+ii
         tmprv(j0) = tmp_r1(ii,jj)
         tmpiv(j0) = tmp_i1(ii,jj)
!         kk = kk + 1
      enddo
      enddo
c!$omp end parallel do
!$acc end parallel loop


      ttmp= dclock()
      call gs_op_fields(gs_handle_q_H,tmprv,
     $     nstate_p,1,1,1,0)
      call gs_op_fields(gs_handle_q_H,tmpiv,nstate_p,1,1,1,0)

      qctime1= qctime1+dclock()-ttmp     

!$acc parallel loop collapse(2) private(tmpfq,tmprr,tmpii,tmpcc)         
c!$omp parallel do collapse(1)
      do jj = 1,nstate_np
      do ii = 1,nstate
         j0 = (jj-1)*nstate+ii
         tmpfq= state_sums(ii,jj)*pulse_omega_0*rho(ii,jj)
         tmprr= 2*tmp_r1(ii,jj)-tmprv(j0)
!         if(ii.eq.jj+c_offset) tmprr = 0 !Diagonal parts are treated below
         tmpii = tmpiv(j0)
!         if(ii.eq.jj+c_offset) tmpii = 2*tmp_i1(ii,jj)
         tmpcc= tmprr + ci*tmpii
         drho(ii,jj)= - ci*(tmpcc - tmpfq) 

         resrho_r(ii,jj)=real(drho(ii,jj) )    
         resrho_i(ii,jj)=imag(drho(ii,jj) )    
      enddo
      enddo

!     The diagonal parts need special treatment
      do jj=1,nstate_np
         tmpfq= state_sums(jj+c_offset,jj)*pulse_omega_0
     $        *rho(jj+c_offset,jj)
         tmpcc = ci*2*tmp_i1(jj+c_offset,jj)
         drho(jj+c_offset,jj) = -ci*(tmpcc-tmpfq)
         resrho_r(jj+c_offset,jj) = real(drho(jj+c_offset,jj))
         resrho_i(jj+c_offset,jj) = imag(drho(jj+c_offset,jj))
      enddo

c$$$      print*,'post2'
c$$$      print*,'resrho_r'
c$$$      do ii=1,nstate
c$$$         write(6,'(100g15.5)') (resrho_r(ii,jj), jj=1,nstate)
c$$$      enddo
c$$$      print*,'resrho_i'
c$$$      do ii=1,nstate
c$$$         write(6,'(100g15.5)') (resrho_i(ii,jj), jj=1,nstate)
c$$$      enddo

c!$omp end parallel do
!$acc end parallel loop

!$acc end data

      return
      end


!This subroutine and quantum3_mult should be folded together
c-----------------------------------------------------------------------
      subroutine quantum3_mult_pece(t,y,yp)
c-----------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'QUANT' 
      integer  n2,i,j,j0
      real*8   y(nstate2_2),yp(nstate2_2),t

      do j=1,nstate_np
         do i=1,nstate
            j0 = (j-1)*nstate+i
            rho_r(i,j) = y(0*nstate2+j0)
            rho_i(i,j) = y(1*nstate2+j0)
c$$$            if(i.eq.j) then
c$$$               print*,'before',y(0*nstate2+j0)
c$$$            endif
         enddo
      enddo


      call cem_quantum3_hamiltonian
      call quantum_lin_mxm
!      call quantum_lin_mpi_get

      do j=1,nstate_np
         do i=1,nstate
            j0 = (j-1)*nstate+i
            yp(0*nstate2+j0) = resrho_r(i,j)
            yp(1*nstate2+j0) = resrho_i(i,j)
c$$$            if(i.eq.j) then
c$$$               print*,'after',yp(0*nstate2+j0)
c$$$            endif

         enddo
      enddo

      return
      end



!******************************************************************
!*
!* quantum_apply_gates applies the user defined gate sequence in time
!* 
!* inputs: none, depends on setup rouints
!*
!* outputs: none, but does the U rho U^dagger operation
!*
!******************************************************************

      subroutine quantum_apply_gates
      implicit none
      include 'SIZE'
      include 'QUANT'
#ifdef MPI
      include 'mpif.h'
#endif

      integer nzmax,gate,ii,jj,kk,j0
      real    dclock,ttmp
      complex ci
      
      ci = (0,1)

      nzmax = 10*maxN

      if(time.gt.quantum_gate_times(current_gate).and.
     $     (current_gate.le.number_of_gates)) then
c$$$         print*,"Rho before"
c$$$         do ii=1,nstate
c$$$            write(6,'(100g25.15)') (rho_r(ii,jj), jj=1,nstate)
c$$$         enddo
c$$$

         do jj=1,nstate_np
            do ii=1,nstate
               rho(ii,jj) = rho_r(ii,jj) + ci*rho_i(ii,jj)
            enddo
         enddo
!     Do Um*rho
         gate = quantum_gate(current_gate)

         call cem_quantum3_csr_mxm_complex(rho,tmp_c1,
     $        quantum_gate_csr_a(1,gate),
     $        quantum_gate_csr_ia(1,gate),
     $        quantum_gate_csr_ja(1,gate))

         do jj = 1,nstate_np
            do ii = 1,nstate
               j0 = (jj-1)*nstate+ii
               tmprv(j0) = real(tmp_c1(ii,jj))
               tmpiv(j0) = imag(tmp_c1(ii,jj))
            enddo
         enddo
c$$$         print*,"after mxm 1"
c$$$         do ii=1,nstate
c$$$            write(6,'(100g25.15)') (tmp_c1(ii,jj), jj=1,nstate)
c$$$         enddo
 

!     Do (Um*rho)^dagger
         ttmp= dclock()
         call gs_op_fields(gs_handle_q_H,tmprv,
     $        nstate_p,1,1,1,0)
         call gs_op_fields(gs_handle_q_H,tmpiv,nstate_p,1,1,1,0)

         qctime1= qctime1+dclock()-ttmp     


         do jj= 1,nstate/np
            do kk = 1,nstate
               if(jj+c_offset.ne.kk) then
                  j0 = (jj-1)*nstate+kk
!     We want b, so we take a+b-a
!     We want -b, so we take a - (a+b)
                  tmp_c1(kk,jj) = tmprv(j0) - real(tmp_c1(kk,jj))
     $                 + ci*(imag(tmp_c1(kk,jj)) - tmpiv(j0))
               else
                  tmp_c1(kk,jj) = conjg(tmp_c1(kk,jj))
               endif
            enddo
         enddo        
c$$$         print*,"after conj transpose"
c$$$         do ii=1,nstate
c$$$            write(6,'(100g25.15)') (tmp_c1(ii,jj), jj=1,nstate)
c$$$         enddo


!     calculate Um*(Um*rho)^dag
         call cem_quantum3_csr_mxm_complex(tmp_c1,rho,
     $        quantum_gate_csr_a(1,gate),
     $        quantum_gate_csr_ia(1,gate),
     $        quantum_gate_csr_ja(1,gate))

         do jj=1,nstate_np
            do ii=1,nstate
               rho_r(ii,jj) = real(rho(ii,jj))
               rho_i(ii,jj) = imag(rho(ii,jj))
            enddo
         enddo

c$$$         print*,"Rho after"
c$$$         do ii=1,nstate
c$$$            write(6,'(100g25.15)') (rho(ii,jj), jj=1,nstate)
c$$$         enddo

!     Increment the current_gate
         current_gate = current_gate + 1
      endif
      
      end

c-------------------------------------------------------------------
      subroutine cem_quantum3_pop()
c-------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'QUANT'
#ifdef MPI
      include 'mpif.h'
#endif
      integer ii,jj,iq,tilingNumber,curState,ierr,ll,kk
      complex*16 ci,tmp_cs
      real*8     pop,currentPop


      ci = (0.0,1.0)

!$acc data present(state_sums,rho_r,rho_i,rho)
!$acc parallel loop collapse(2)
      do jj=1,nstate_np
      do ii=1,nstate
         tmp_c1(ii,jj)= (rho_r(ii,jj)+ci*rho_i(ii,jj))
         if(ifPulse.eqv..true.) then
            tmp_c1(ii,jj) = tmp_c1(ii,jj)*
     $           exp(-ci*state_sums(ii,jj)*pulse_omega_0
     $           *(time-pulse_t_0))
         endif
      enddo
      enddo
!$acc end parallel loop
!$acc update host(rho)
!$acc end data

      ll=1
c     Find populations 
      do ii=1,numberOfOps
         tilingNumber = oldN(ii)
         if(is_vec(ii).eqv..true.) then
            !if the operator is a vec operator, then we loop through each
            !level of the vec operator, which is treated as a series of 
            !individual populations. If the curState is equal to the
            !operator we are looking at, then we add the population
            !This could be sped up, since we could go through jj
            !once, with more if statement

            !FIXME - pop_time size may need a firmer check now
            do kk=1,levels(ii)
               pop = 0.0
               do jj=1,nstate_np
                  curState = state(ii,mod(jj+c_offset-1,tilingNumber)+1)
                  currentPop = real(tmp_c1(jj+c_offset,jj))
!                  currentPop = real(rho_r(jj+c_offset,jj))
                  if(curState.eq.kk-1) then
!     print*,curState,currentPop
                     pop = pop + currentPop
                  endif
               enddo
               if(abs(pop).le.1e-13) pop = 0.0
               pop_time(iiqstep,ll) = pop
               tmprv(ll) = pop
               ll=ll+1
            enddo
         else
            pop = 0.0
            do jj=1,nstate_np
               curState = state(ii,mod(jj+c_offset-1,tilingNumber)+1)
               currentPop = real(tmp_c1(jj+c_offset,jj))
!     print*,curState,currentPop
               pop = pop + 
     $              curState*currentPop
            enddo


!     Save results
            if(abs(pop).le.1e-13) pop = 0.0
            pop_time(iiqstep,ll) = pop
            tmprv(ll) = pop
            ll=ll+1
         endif

      enddo


      if(print_during_run.eqv..true.) then
         ll = ll-1

         if(nid.eq.0) then
            call MPI_REDUCE(MPI_IN_PLACE,tmprv,
     $           ll,MPI_REAL8,MPI_SUM,0,MPI_COMM_WORLD,IERR)
            if(steady_state.eqv..true.) then
               write(6,*) "Steady state populations: "
               write(6,12) (tmprv(ii),ii=1,ll)
            else
               write(12,12)time*timeunit,(tmprv(ii),ii=1,ll)
               call flush(12)
            endif
 12         format(1p20g20.8)
         else
            call MPI_REDUCE(tmprv,tmprv,
     $           ll,MPI_REAL8,MPI_SUM,0,MPI_COMM_WORLD,IERR)
         endif
         
      endif


      

c     Check for radically wrong results
c     tr(rho) = 1
c     Local only check (just for exploding values)
      pop = 0.0
      do ii=1,nstate_np
         pop = pop+rho_r(ii+c_offset,ii)
      enddo
      if(pop.gt.1.01) then
         print*,'LOCAL TRACE GREATER THAN ONE ON'
         print*,'CORE ',nid
         stop
         call exitt()
      endif

      if(pop.ne.pop) then
         print*,'LOCAL TRACE GREATER NAN'
         print*,'CORE ',nid
         stop
         call exitt()
      endif

c     global check
      call gop(pop,currentPop,'+  ',1)

      if(nid.eq.0.and.abs(currentPop-1).gt.1e-10) then 
         print*,'GLOBAL TRACE DIFFERENT THAN ONE'
         print*,'TRACE',currentPop
c$$$         stop
c$$$         call exitt()
      endif
c     Reduce results
c$$$#ifdef MPI
c$$$         if(nid.eq.0) then
c$$$            call MPI_REDUCE(MPI_IN_PLACE,observable(ii),1,MPI_REAL8,
c$$$     $           MPI_SUM,0,MPI_COMM_WORLD,IERR)
c$$$            if(abs(observable(ii)).le.1e-14) observable(ii) = 0.0
c$$$         else
c$$$            call MPI_REDUCE(observable(ii),observable(ii),1,MPI_REAL8,
c$$$     $           MPI_SUM,0,MPI_COMM_WORLD,IERR)
c$$$         endif
c$$$#endif

c$$$
c$$$#ifdef MPI
c$$$      if(nid.eq.0) then
c$$$         call MPI_REDUCE(MPI_IN_PLACE,pop,numberOfOps,MPI_REAL8,
c$$$     $        MPI_SUM,0,MPI_COMM_WORLD,IERR)
c$$$      else
c$$$         call MPI_REDUCE(pop,pop,numberOfOps,MPI_REAL8,
c$$$     $        MPI_SUM,0,MPI_COMM_WORLD,IERR)
c$$$      endif
c$$$  #endif
      if(calc_concurrence.eqv..true.) then
         call quantum3_concurrence()
      endif

      return
      end




c--------------------------------------------------------------------
      subroutine cem_quantum3_observables
c--------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'QUANT'
#ifdef MPI
      include 'mpif.h'
#endif
      integer  iEh,ii,jj,iq,kk,handle,istart,iend
      complex*16 ci
      real*8     pulseFactor,fullPulse
      real     ttmp

      ci = (0.0,1.0)

!$acc data present(state_sums,rho_r,rho_i,rho)
!$acc parallel loop collapse(2)
      do jj=1,nstate_np
      do ii=1,nstate
            tmp_c1(ii,jj)= (rho_r(ii,jj)+ci*rho_i(ii,jj))
            tmp_c1(ii,jj) = tmp_c1(ii,jj)
     $           *exp(-ci*state_sums(ii,jj)*pulse_omega_0
     $           *(time-pulse_t_0))
      enddo
      enddo
!$acc end parallel loop
!$acc update host(rho)
!$acc end data
c$$$
      do ii=1,numObservables
         observable(ii) = 0.0
         handle = obs_handle(ii)
         do jj=1,nstate
            istart = obs_csr_ia(jj,ii)
            iend   = obs_csr_ia(jj+1,ii)-1
            do kk=istart,iend
               observable(ii) = observable(ii) + obs_csr_a(kk,ii) * 
     $              real(tmp_c1(jj,obs_csr_ja(kk,ii)))
            enddo
         enddo
      enddo


c$$$      call usertimedep(pulseFactor)
c$$$      if(rwa.eqv..true.) then
c$$$         fullPulse = 2*pulseFactor*
c$$$     $        cos(pulse_omega_0*(time-pulse_t_0))
c$$$      else
c$$$         fullPulse = pulseFactor
c$$$      endif
c$$$      
c$$$      do kk=1,nEh
c$$$         en_t(kk) = en_t(kk)+fullPulse*
c$$$     $        exp(ci*omega_quant(kk)*(time-pulse_t_0))
c$$$         do jj=1,numObservables
c$$$            obs_freq(jj,kk) = obs_freq(jj,kk) +  
c$$$     $           (exp(ci*omega_quant(kk)*(time-pulse_t_0))*
c$$$     $           observable(jj))
c$$$         enddo
c$$$         
c$$$      enddo

      do ii=1,numObservables
         obs_time(iqstep,ii) = observable(ii)
      enddo

      if(print_during_run.eqv..true.) then
         write(13,12)time*timeunit,(observable(jj),jj=1,numObservables)
      endif
 12   format(1p20g20.8)


!      if(abs(pulseFactor).lt.1e-10) pulseFactor=0
c$$$      if(nid.eq.0) then
c$$$         write(12,12)time*timeunit,pulseFactor,pop
c$$$!     $        concurrence
c$$$         write(13,12)time*timeunit,observable(1),observable(2)
c$$$ 12      format(1p20g20.8)
c$$$      endif

      return
      end


c**************************
c
c subroutine quantum3_concurrence finds and prints the concurrence
c using wooters formula. 
c
c
c**************************

c-----------------------------------------------------------------------
      subroutine quantum3_concurrence()
c-----------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'QUANT'
#ifdef MPI
      include 'mpif.h'
#endif
      complex red_rho2(4,4),red_rho3(8,8),ci
      complex red_rho(maxPartialTrace,maxPartialTrace)
      real*8  conc1,conc2,conc3,conc4,conc5,conc6
      real*8  popSymm1(4),popSymm2(4),popSymm3(4),popSymm4(4)
      real*8  popSymm5(4),popSymm6(4)
      integer ii,jj,kk,j0

      ci=(0.0,1.0)

c     Partial trace to maxPartialTrace size

c     Prepare communcation vector
!      kk = 1
c!$acc parallel loop collapse(2) private(j0)
c!$omp parallel do collapse(2)
      do jj = 1,nstate_np
      do ii = 1,nstate
!         j0 = j0_array(kk)!(jj-1)*nstate+ii
         j0 = (jj-1)*nstate+ii
         tmprv(j0) = rho_r(ii,jj)
         tmpiv(j0) = rho_i(ii,jj)
!         kk = kk + 1
      enddo
      enddo
c!$omp end parallel do
c!$acc end parallel loop
      
      call gs_op_fields(gs_handle_q_conc,tmprv,
     $     nstate_p,1,1,1,0) 
      call gs_op_fields(gs_handle_q_conc,tmpiv,
     $     nstate_p,1,1,1,0) 

!      kk = 1
c     store result in red_rho
      do jj=1,maxPartialTrace
         do ii=1,maxPartialTrace
!            j0 = j0_array(kk)!(jj-1)*nstate+ii
            j0 = (jj-1)*nstate+ii
            red_rho(ii,jj) = (tmprv(j0) + ci*tmpiv(j0))
            red_rho(ii,jj) = red_rho(ii,jj)
     $           *exp(-ci*state_sums(ii,jj)*pulse_omega_0
     $           *(time-pulse_t_0))
!            kk = kk+1
         enddo
      enddo

c$$$      print*,"Rho"
c$$$      do ii=1,nstate
c$$$         write(6,'(100g25.15)') (rho_r(ii,jj), jj=1,nstate)
c$$$      enddo
c$$$
c$$$      print*,"red_rho"
c$$$      do ii=1,nstate
c$$$         write(6,'(100g25.15)') (real(red_rho(ii,jj)), jj=1,nstate)
c$$$      enddo

c$$$      call exitt()
      if((maxPartialTrace.eq.4).and.(nid.eq.0)) then
         call cem_quantum3_bipartite_concurrence(red_rho,conc1,popSymm1)
         write(14,13)time*timeunit,conc1
 13      format(1p20g20.8)
         write(15,13)time*timeunit,popSymm1         
      endif



      if((maxPartialTrace.eq.8).and.(nid.eq.0)) then
         call cem_quantum3_partial_trace_3_to_2(red_rho,red_rho2,1)
c     concurrence(3:2)
         call cem_quantum3_bipartite_concurrence(red_rho2,conc1
     $        ,popSymm1)
         call cem_quantum3_partial_trace_3_to_2(red_rho,red_rho2,2)
c     concurrence(3:1)
         call cem_quantum3_bipartite_concurrence(red_rho2,conc2
     $        ,popSymm2)
         call cem_quantum3_partial_trace_3_to_2(red_rho,red_rho2,3)
c     concurrence(2:1)
         call cem_quantum3_bipartite_concurrence(red_rho2,conc3
     $        ,popSymm3)
         
         write(14,13)time*timeunit,conc1,conc2,conc3
         write(15,13)time*timeunit,popSymm1,popSymm2
     $        ,popSymm3

      endif

      if((maxPartialTrace.eq.16).and.(nid.eq.0)) then
c     First, trace out qd 1 with 4->3 quantum dots
         call cem_quantum3_partial_trace_4_to_3(red_rho,red_rho3,1)
c     Now, trace 3->2 3 times, and find bipartite concurrence for each
c     First, concurrence(4:3) (trace out 2)
         call cem_quantum3_partial_trace_3_to_2(red_rho3,red_rho2,1)
         call cem_quantum3_bipartite_concurrence(red_rho2,conc1
     $        ,popSymm1)
c     concurrence(4:2) (trace out 3)
         call cem_quantum3_partial_trace_3_to_2(red_rho3,red_rho2,2)
         call cem_quantum3_bipartite_concurrence(red_rho2,conc2
     $        ,popSymm2)
c     concurrence(3:2) (trace out 4)
         call cem_quantum3_partial_trace_3_to_2(red_rho3,red_rho2,3)
         call cem_quantum3_bipartite_concurrence(red_rho2,conc3
     $        ,popSymm3)
c     trace out qd 3 from full
         call cem_quantum3_partial_trace_4_to_3(red_rho,red_rho3,3)
c     concurrence(4:1)
         call cem_quantum3_partial_trace_3_to_2(red_rho3,red_rho2,2)
         call cem_quantum3_bipartite_concurrence(red_rho2,conc4
     $        ,popSymm4)
c     concurrence(2:1)
         call cem_quantum3_partial_trace_3_to_2(red_rho3,red_rho2,3)
         call cem_quantum3_bipartite_concurrence(red_rho2,conc5
     $        ,popSymm5)
c     trace out qd 4 from full
         call cem_quantum3_partial_trace_4_to_3(red_rho,red_rho3,4)
c     concurrence(3:1)
         call cem_quantum3_partial_trace_3_to_2(red_rho3,red_rho2,3)
         call cem_quantum3_bipartite_concurrence(red_rho2,conc6
     $        ,popSymm6)
         write(14,13)time*timeunit,conc1,conc2,conc3,
     $        conc4,conc5,conc6
         write(15,13)time*timeunit,popSymm1,popSymm2
     $        ,popSymm3,popSymm4,popSymm5,popSymm6
      endif
         
      end

c-------------------------------------------------------------------
      subroutine cem_quantum3_bipartite_concurrence(rho_conc,bi_conc,
     $     popSymm)
c-------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'QUANT'
      complex    rho_conc(4,4),rho_svd2(4,4),rho_work2(8),rhoVL2(4,4)
      complex    spinFlip2(4,4),rhoFinal2(4,4),rhoVR2(4,4),rhoEval2(4)
      real       rho_workr2(2*4),bi_conc,rhoOrdering,popSymm(4)
      integer    ii,jj,info

      do ii=1,4
         do jj=1,4
            spinFlip2(jj,ii) = (0.0,0.0)
         enddo
      enddo

      spinFlip2(1,4) = -1
      spinFlip2(4,1) = -1
      spinFlip2(2,3) = 1
      spinFlip2(3,2) = 1
      
      do ii=1,4
         do jj=1,4
            rhoVL2(ii,jj) = conjg(rho_conc(ii,jj))
         enddo
      enddo
      
      call zgemm('N','N',4,4,4,(1.0,0.0),
     $     spinFlip2,4,rhoVL2,4,(0.0,0.0),rhoVR2,4)
      call zgemm('N','N',4,4,4,(1.0,0.0),rhoVR2,4,
     $     spinFlip2,4,(0.0,0.0),rho_svd2,4)
      call zgemm('N','N',4,4,4,(1.0,0.0),rho_conc,4,
     $     rho_svd2,4,(0.0,0.0),rhoFinal2,4)
      call zgeev('N','N',4,rhoFinal2,4,rhoEval2,rhoVL2,4,
     $     rhoVR2,4,rho_work2,2*4,rho_workR2,info)

      
      do ii=2,4
         rhoOrdering = rhoEval2(ii)
         jj=ii-1
         do while(jj.ge.1.and.real(rhoEval2(jj)).gt.real(rhoOrdering))
            rhoEval2(jj+1) = rhoEval2(jj)
            jj = jj - 1
         enddo
         rhoEval2(jj+1) = rhoOrdering
      enddo
      do ii=1,4
         if(abs(real(rhoEval2(ii))).lt.1e-14) rhoEval2(ii)=0
      enddo

      popSymm(1) = 0.5*(rho_conc(2,2) + rho_conc(3,3) + 
     $     rho_conc(2,3) + rho_conc(3,2))
      popSymm(2) = 0.5*(rho_conc(2,2) + rho_conc(3,3) - 
     $     rho_conc(2,3) - rho_conc(3,2))
      popSymm(3) = 0.5*(rho_conc(1,1) + rho_conc(4,4) + 
     $     rho_conc(1,4) + rho_conc(4,1))
      popSymm(4) = 0.5*(rho_conc(1,1) + rho_conc(4,4) - 
     $     rho_conc(1,4) - rho_conc(4,1))
      


      bi_conc=0
      do ii=1,4
         bi_conc = bi_conc + sqrt(real(rhoEval2(ii)))
      enddo
      bi_conc = 2*sqrt(real(rhoEval2(4)))-bi_conc
c      print*,bi_conc


      return
      end

c--------------------------------------------------------------------
      subroutine cem_quantum3_partial_trace_4_to_3(rho_16by16,rho_8by8,
     $     trace_over)
c         Reduce rho even further
c         HARD CODED IN: CONSIDER REVISING
      implicit none
      complex*16 rho_16by16(16,16),rho_8by8(8,8)
      integer trace_over,twotoiq,jjplus,kkplus,jj,kk
      
      if(trace_over.eq.1) then
         twotoiq = 1
         do jj=1,8
            do kk=1,8
c     2*jj-1 is to get the stride right.                                                          
               rho_8by8(jj,kk) =
     $               rho_16by16(2*jj-1,2*kk-1)+
     $               rho_16by16(2*jj-1+twotoiq,2*kk-1+twotoiq)
c     $               +rho_16by16(2*jj-1+twotoiq,2*kk-1)
c     $               +rho_16by16(2*jj-1,2*kk-1+twotoiq)
             enddo
          enddo
       else if(trace_over.eq.3) then
          twotoiq = 4
c        If statements to get the stride correct
          do jj=1,8
             jjplus=4
             if(jj.le.4) jjplus=0
             do kk=1,8
                kkplus=4
                if(kk.le.4) kkplus=0
                rho_8by8(jj,kk) = 
     $               rho_16by16(jj+jjplus,kk+kkplus)+
     $               rho_16by16(jj+twotoiq+jjplus,kk+twotoiq+kkplus)
c     $               +rho_16by16(jj+jjplus,kk+twotoiq+kkplus)
c     $               +rho_16by16(jj+twotoiq+jjplus,kk+kkplus)
             enddo
          enddo
       else if(trace_over.eq.4) then
c        Trace out the highest bit
          twotoiq=8
          do jj=1,8
             do kk=1,8
                rho_8by8(jj,kk) = 
     $               rho_16by16(jj,kk)+
     $               rho_16by16(jj+twotoiq,kk+twotoiq)
c     $               +rho_16by16(jj,kk+twotoiq)
c     $               +rho_16by16(jj+twotoiq,kk)
             enddo
          enddo
       endif
       
      


      return
      end

c--------------------------------------------------------------------
      subroutine cem_quantum3_partial_trace_3_to_2(rho_8by8,rho_4by4,
     $     trace_over)
c--------------------------------------------------------------------
c         Reduce rho even further
c         HARD CODED IN: CONSIDER REVISING
c         NEED LOOP STRUCTURE: 11110000, 110011001100,1010101010


      implicit none
      complex*16 rho_8by8(8,8),rho_4by4(4,4)
      integer trace_over,twotoiq,jjplus,kkplus,jj,kk
c     Trace out lowest bit                                                                        
      if(trace_over.eq.1) then
         twotoiq = 1
         do jj=1,4
            do kk=1,4
c     2*jj-1 is to get the stride right.                                                          
               rho_4by4(jj,kk) =
     $               rho_8by8(2*jj-1,2*kk-1)+
     $               rho_8by8(2*jj-1+twotoiq,2*kk-1+twotoiq)
c     $               +rho_8by8(2*jj-1+twotoiq,2*kk-1)
c     $               +rho_8by8(2*jj-1,2*kk-1+twotoiq)
             enddo
          enddo
       else if(trace_over.eq.2) then
c        Trace out the second lowest bit
          twotoiq = 2
c        If statements to get the stride correct
          do jj=1,4
             jjplus=2
             if(jj.le.2) jjplus=0
             do kk=1,4
                kkplus=2
                if(kk.le.2) kkplus=0
                rho_4by4(jj,kk) = 
     $               rho_8by8(jj+jjplus,kk+kkplus)+
     $               rho_8by8(jj+twotoiq+jjplus,kk+twotoiq+kkplus)
c     $               +rho_8by8(jj+jjplus,kk+twotoiq+kkplus)
c     $               +rho_8by8(jj+twotoiq+jjplus,kk+kkplus)
 
            enddo
          enddo
       else if(trace_over.eq.3) then
c        Trace out the third lowest bit
          twotoiq=4
          do jj=1,4
             do kk=1,4
                rho_4by4(jj,kk) = 
     $               rho_8by8(jj,kk)+
     $               rho_8by8(jj+twotoiq,kk+twotoiq)
c     $               +rho_8by8(jj+twotoiq,kk)
c     $               +rho_8by8(jj,kk+twotoiq)
             enddo
          enddo
       endif



      return
      end

      !Use CG to solve for the steady state
c=======================================================================
      subroutine cem_quantum3_steadystate_gmres()
c=======================================================================      
      implicit none
      include 'SIZE'
      include 'QUANT'
#ifdef MPI
      include 'mpif.h'
#endif

      real*8 alpha,rtr1,rtr2,beta,pAp,tol,multiplier
      integer iter,niter,ii,jj,j0,ierr
C          Test example
c$$$      integer n
c$$$      parameter (n=4)
c$$$      real*8 a(n,n),b(n)
c$$$      data (a(1,ii),ii=1,n) /420.,210.,140.,105./
c$$$      data (a(2,ii),ii=1,n) /210.,140.,105.,84./
c$$$      data (a(3,ii),ii=1,n) /140.,105.,84.,70./
c$$$      data (a(4,ii),ii=1,n) /105.,84.,70.,60./
c$$$      data (b(ii),ii=1,n) /875.,539.,399.,319./
c$$$      nqstep_print = 990
c$$$      call cem_quantum3_propagation_init    
c$$$      call cem_quantum3_rk_storage

      !do a quick thermalization
      
      print_during_run = .true. !This makes it print in quantum_pop
c$$$      call cem_quantum3_pop
c$$$      call quantum_gmres !Get steady state using gmres
c$$$      call cem_quantum3_pop
c$$$      
      timemax = 0.01e-6/timeunit
      nqstep  = 5000

      dt = timemax/(nqstep)
!      call cem_quantum3_propagation2_rk
c$$$      steady_state = .true.
c$$$      call cem_quantum3_pop
c$$$      call quantum_gmres !Get steady state using gmres
c$$$      call cem_quantum3_pop
c$$$      steady_state = .false.
c$$$      timemax = 0.2e-6/timeunit
c$$$
c$$$      dt = (timemax-time)/(nqstep)
c$$$      call cem_quantum3_propagation2_rk
c$$$      steady_state = .true.
c$$$      call cem_quantum3_pop
c$$$      call quantum_gmres !Get steady state using gmres
c$$$      call cem_quantum3_pop
c$$$      steady_state = .false.
c$$$      timemax = 0.06e-6/timeunit
c$$$      dt = (timemax-time)/(nqstep)
c$$$      call cem_quantum3_propagation2_rk
c$$$      steady_state = .true.
c$$$      call cem_quantum3_pop
c$$$      call quantum_gmres !Get steady state using gmres
c$$$      call cem_quantum3_pop

      call cem_quantum3_initialize
      call cem_quantum3_gs
     
      steady_state = .true.
      call cem_quantum3_pop
      call quantum_gmres !Get steady state using gmres
      call cem_quantum3_pop

      end


      !Use CG to solve for the steady state - won't work with Lindblad
c=======================================================================
      subroutine cem_quantum3_steadystate_cg()
c=======================================================================      
      implicit none
      include 'SIZE'
      include 'QUANT'
#ifdef MPI
      include 'mpif.h'
#endif

      real*8 alpha,rtr1,rtr2,beta,pAp,tol
      integer iter,niter,ii,jj,j0,ierr
C          Test example
c$$$      integer n
c$$$      parameter (n=4)
c$$$      real*8 a(n,n),b(n)
c$$$      data (a(1,ii),ii=1,n) /420.,210.,140.,105./
c$$$      data (a(2,ii),ii=1,n) /210.,140.,105.,84./
c$$$      data (a(3,ii),ii=1,n) /140.,105.,84.,70./
c$$$      data (a(4,ii),ii=1,n) /105.,84.,70.,60./
c$$$      data (b(ii),ii=1,n) /875.,539.,399.,319./

      niter = 100 !move to user controlled with default
      tol   = 1.0e-13 !move to user controlled with default
      steady_state = .true.
      print_during_run = .true. !This makes it print in quantum_pop
      call cem_quantum3_initialize      
      call cem_quantum3_gs
      

      do jj=1,nstate_np
         do ii=1,nstate
            j0 = (jj-1)*nstate+ii
            p_cg(0*nstate2+j0) = rho_r(ii,jj)
            p_cg(1*nstate2+j0) = rho_i(ii,jj)
         enddo
      enddo

      !Calculate -i * [H,p] - L(p) ==> A*x
      !r = b - A*x, but b=0 for steady state, so r=-A*x = -r_cg
      call quantum3_mult(r_cg,p_cg,nstate2_2)      
      do ii=1,nstate2_2
         r_cg(ii) = -r_cg(ii)
      enddo


!      call residual(n,a,x_cg,b,r_cg)
!      write(6,*) (r_cg(ii),ii=1,n)
      do ii=1,nstate2_2
         p_cg(ii) = 0.0
      enddo

      rtr1 = 1.0
      do iter=1,niter
         rtr2 = rtr1
         !rtr1 = r_k^T r_k
         rtr1 = 0
         do ii=1,nstate2_2
            rtr1 = rtr1 + r_cg(ii)*r_cg(ii)
         enddo

         call MPI_ALLREDUCE(MPI_IN_PLACE,rtr1,1,MPI_REAL8,MPI_SUM,
     $        MPI_COMM_WORLD,IERR)


         print*,'sqrt(rtr1) ',sqrt(rtr1)
         if(sqrt(rtr1).le.tol) then
            go to 1000 !EXIT loop
         endif

         !beta = r_k+1^T r_k+1 / r_k^T r_k
         beta = rtr1/rtr2
         !if(iter.eq.1) beta = 0.0 !Since initial p is 0, this is not necessary
         
         do ii=1,nstate2_2
            p_cg(ii) = r_cg(ii) + beta*p_cg(ii)
          enddo

         !Calculate Ap
         call quantum3_mult(Ap_cg,p_cg,nstate2_2)

!          call mult(n,a,p_cg,Ap_cg)

         pAp = 0
         do ii=1,nstate2_2
            pAp = pAp + p_cg(ii) * Ap_cg(ii)
         enddo

         call MPI_ALLREDUCE(MPI_IN_PLACE,pAp,1,MPI_REAL8,MPI_SUM,
     $        MPI_COMM_WORLD,IERR)
         
         !alpha = r_k^T r_k/p_k^T A p_k
         alpha = rtr1/pAp
         !r_k+1 = r_k - alpha_k*Ap_k
         !x_k = x_k + alpha_k*p_k
         do ii=1,nstate2_2
            r_cg(ii) = r_cg(ii) - alpha*Ap_cg(ii)
            x_cg(ii) = x_cg(ii) + alpha*p_cg(ii)
         enddo
      enddo
 1000 continue

      !Get the final, steady state populations
      do jj=1,nstate_np
         do ii=1,nstate
            j0 = (jj-1)*nstate+ii
            rho_r(ii,jj) = x_cg(0*nstate2+j0)
            rho_i(ii,jj) = x_cg(1*nstate2+j0)
         enddo
      enddo
      call cem_quantum3_pop

      end
c===========================================================================

c$$$      subroutine residual(n,a,x,b,r)
c$$$c
c$$$c     compute the residual r = b-Ax
c$$$c
c$$$      dimension a(n,n),x(n),b(n),r(n)
c$$$      call mult(n,a,x,r)
c$$$c
c$$$      do 2 i=1,n
c$$$         r(i) = b(i) - r(i)
c$$$ 2           continue
c$$$c
c$$$      return
c$$$      end
c$$$
c$$$      subroutine mult(n,a,x,y)
c$$$c
c$$$c     compute the matrix-vector product
c$$$c
c$$$      dimension a(n,n),x(n),y(n)
c$$$c
c$$$      do 3 i=1,n
c$$$         sum = 0.0
c$$$         do 2 j=1,n
c$$$            sum = sum + a(i,j)*x(j)
c$$$ 2                 continue
c$$$         y(i) = sum
c$$$ 3           continue
c$$$c
c$$$      return
c$$$      end

c=======================================================================
      subroutine ode(f,neqn,y,t,tout,relerr,abserr,iflag,work,iwork)
      implicit real*8(a-h,o-z)
c Shampine's differential equations integrator
c   subroutine ode integrates a system of neqn
c   first order ordinary differential equations of the form:
c             dy(i)/dt = f(t,y(1),y(2),...,y(neqn))
c             y(i) given at  t .
c   the subroutine integrates from  t  to  tout .  on return the
c   parameters in the call list are set for continuing the integration.
c   the user has only to define a new value  tout  and call  ode  again.
c
c   the differential equations are actually solved by a suite of codes
c   de ,  step , and  intrp .  ode  allocates virtual storage in the
c   arrays  work  and  iwork  and calls  de .  de  is a supervisor which
c   directs the solution.  it calls on the routines  step  and  intrp
c   to advance the integration and to interpolate at output points.
c   step  uses a modified divided difference form of the adams pece
c   formulas and local extrapolation.  it adjusts the order and step
c   size to control the local error per unit step in a generalized
c   sense.  normally each call to  step  advances the solution one step
c   in the direction of  tout .  for reasons of efficiency  de
c   integrates beyond  tout  internally, though never beyond
c   t+10*(tout-t), and calls  intrp  to interpolate the solution at
c   tout .  an option is provided to stop the integration at  tout  but
c   it should be used only if it is impossible to continue the
c   integration beyond  tout .
c
c   this code is completely explained and documented in the text,
c   computer solution of ordinary differential equations:  the initial
c   value problem  by l. f. shampine and m. k. gordon.
c
c   the parameters represent:
c      f -- real subroutine f(t,y,yp) to evaluate
c                derivatives yp(i)=dy(i)/dt
c      neqn -- number of equations to be integrated (integer)
c      y(*) -- solution vector at t                 (real)
c      t -- independent variable                    (real)
c      tout -- point at which solution is desired   (real)
c      relerr,abserr -- relative and absolute error tolerances for local
c           error test (real).  at each step the code requires
c             abs(local error) .le. abs(y)*relerr + abserr
c           for each component of the local error and solution vectors
c      iflag -- indicates status of integration     (integer)
c      work(*)  (real)  -- arrays to hold information internal to
c      iwork(*) (integer)    which is necessary for subsequent calls
c
c   first call to ode --
c
c   the user must provide storage in his calling program for the arrays
c   in the call list,
c      y(neqn), work(100+21*neqn), iwork(5),
c   declare  f  in an external statement, supply the real
c   subroutine f(t,y,yp)  to evaluate
c      dy(i)/dt = yp(i) = f(t,y(1),y(2),...,y(neqn))
c   and initialize the parameters:
c      neqn -- number of equations to be integrated
c      y(*) -- vector of initial conditions
c      t -- starting point of integration
c      tout -- point at which solution is desired
c      relerr,abserr -- relative and absolute local error tolerances
c      iflag -- +1,-1.  indicator to initialize the code.  normal input
c           is +1.  the user should set iflag=-1 only if it is
c           impossible to continue the integration beyond  tout .
c   all parameters except  f ,  neqn  and  tout  may be altered by the
c   code on output so must be variables in the calling program.
c
c   output from  ode  --
c
c      neqn -- unchanged
c      y(*) -- solution at  t
c      t -- last point reached in integration.  normal return has
c           t = tout .
c      tout -- unchanged
c      relerr,abserr -- normal return has tolerances unchanged.  iflag=3
c           signals tolerances increased
c      iflag = 2 -- normal return.  integration reached  tout
c            = 3 -- integration did not reach  tout  because error
c                   tolerances too small.  relerr ,  abserr  increased
c                   appropriately for continuing
c            = 4 -- integration did not reach  tout  because more than
c                   500 steps needed
c            = 5 -- integration did not reach  tout  because equations
c                   appear to be stiff
c            = 6 -- invalid input parameters (fatal error)
c           the value of  iflag  is returned negative when the input
c           value is negative and the integration does not reach  tout ,
c           i.e., -3, -4, -5.
c      work(*),iwork(*) -- information generally of no interest to the
c           user but necessary for subsequent calls.
c
c   subsequent calls to  ode --
c
c   subroutine  ode  returns with all information needed to continue
c   the integration.  if the integration reached  tout , the user need
c   only define a new  tout  and call again.  if the integration did not
c   reach  tout  and the user wants to continue, he just calls again.
c   the output value of  iflag  is the appropriate input value for
c   subsequent calls.  the only situation in which it should be altered
c   is to stop the integration internally at the new  tout , i.e.,
c   change output  iflag=2  to input  iflag=-2 .  error tolerances may
c   be changed by the user before continuing.  all other parameters must
c   remain unchanged.
c
c***********************************************************************
c*  subroutines  de  and  step  contain machine dependent constants.   *
c*  be sure they are set before using  ode .                           *
c***********************************************************************
      logical start,phase1,nornd
      dimension y(neqn),work(1),iwork(5)
      external f
      data ialpha,ibeta,isig,iv,iw,ig,iphase,ipsi,ix,ih,ihold,istart,
     1  itold,idelsn/1,13,25,38,50,62,75,76,88,89,90,91,92,93/
      iyy = 100
      iwt = iyy + neqn
      ip = iwt + neqn
      iyp = ip + neqn
      iypout = iyp + neqn
      iphi = iypout + neqn
      if(abs(iflag) .eq. 1) go to 1
      start = work(istart) .gt. 0.0d0
      phase1 = work(iphase) .gt. 0.0d0
      nornd = iwork(2) .ne. -1
    1 call de(f,neqn,y,t,tout,relerr,abserr,iflag,work(iyy),
     1  work(iwt),work(ip),work(iyp),work(iypout),work(iphi),
     2  work(ialpha),work(ibeta),work(isig),work(iv),work(iw),work(ig),
     3  phase1,work(ipsi),work(ix),work(ih),work(ihold),start,
     4  work(itold),work(idelsn),iwork(1),nornd,iwork(3),iwork(4),
     5  iwork(5))
      work(istart) = -1.0d0
      if(start) work(istart) = 1.0d0
      work(iphase) = -1.0d0
      if(phase1) work(iphase) = 1.0d0
      iwork(2) = -1
      if(nornd) iwork(2) = 1
      return
      end
c
c
      subroutine de(f,neqn,y,t,tout,relerr,abserr,iflag,
     1  yy,wt,p,yp,ypout,phi,alpha,beta,sig,v,w,g,phase1,psi,x,h,hold,
     2  start,told,delsgn,ns,nornd,k,kold,isnold)
      implicit real*8(a-h,o-z)
      logical stiff,crash,start,phase1,nornd
      dimension y(neqn),yy(neqn),wt(neqn),phi(neqn,16),p(neqn),yp(neqn),
     1  ypout(neqn),psi(12),alpha(12),beta(12),sig(13),v(12),w(12),g(13)
      external f
c
c***********************************************************************
c*  the only machine dependent constant is based on the machine unit   *
c*  roundoff err  u  which is the smallest positive number such that *
c*  1.0+u .gt. 1.0 .  u  must be calculated and  fouru=4.0*u  inserted *
c*  in the following data statement before using  de .  the routine    *
c*  machin  calculates  u .  fouru  and  twou=2.0*u  must also be      *
c*  inserted in subroutine  step1  before calling  de .
      data fouru/.888e-15/
c***********************************************************************
c
c   the constant  maxnum  is the maximum number of steps allowed in one
c   call to  de .  the user may change this limit by altering the
c   following statement
      data maxnum/10000/
c
c            ***            ***            ***
c   test for improper parameters
c
      if(neqn .lt. 1) go to 10
      if(t .eq. tout) go to 10
      if(relerr .lt. 0.0d0  .or.  abserr .lt. 0.0d0) go to 10
      eps = max(relerr,abserr)
      if(eps .le. 0.0d0) go to 10
      if(iflag .eq. 0) go to 10
      isn = sign(1,iflag)
      iflag = abs(iflag)
      if(iflag .eq. 1) go to 20
      if(t .ne. told) go to 10
      if(iflag .ge. 2  .and.  iflag .le. 5) go to 20
   10 iflag = 6
      return
c
c   on each call set interval of integration and counter for number of
c   steps.  adjust input err tolerances to define weight vector for
c   subroutine  step1
c
   20 del = tout - t
      absdel = abs(del)
      tend = t + 10.0d0*del
      if(isn .lt. 0) tend = tout
      nostep = 0
      kle4 = 0
      stiff = .false.
      releps = relerr/eps
      abseps = abserr/eps
      if(iflag .eq. 1) go to 30
      if(isnold .lt. 0) go to 30
      if(delsgn*del .gt. 0.0d0) go to 50
c
c   on start and restart also set work variables x and yy(*), store the
c   direction of integration and initialize the step size
c
   30 start = .true.
      x = t
      do 40 l = 1,neqn
   40   yy(l) = y(l)
      delsgn = sign(1.0d0,del)
      h = sign(max(abs(tout-x),fouru*abs(x)),tout-x)
c
c   if already past output point, interpolate and return
c
   50 if(abs(x-t) .lt. absdel) go to 60
      call intrp(x,yy,tout,y,ypout,neqn,kold,phi,psi)
      iflag = 2
      t = tout
      told = t
      isnold = isn
      return
c
c   if cannot go past output point and sufficiently close,
c   extrapolate and return
c
   60 if(isn .gt. 0  .or.  abs(tout-x) .ge. fouru*abs(x)) go to 80
      h = tout - x
      call f(x,yy,yp)
      do 70 l = 1,neqn
   70   y(l) = yy(l) + h*yp(l)
      iflag = 2
      t = tout
      told = t
      isnold = isn
      return
c
c   test for too many steps
c
   80 if(nostep .lt. maxnum) go to 100
      iflag = isn*4
      if(stiff) iflag = isn*5
      do 90 l = 1,neqn
   90   y(l) = yy(l)
      t = x
      told = t
      isnold = 1
      return
c
c   limit step size, set weight vector and take a step
c
  100 h = sign(min(abs(h),abs(tend-x)),h)
      do 110 l = 1,neqn
  110   wt(l) = releps*abs(yy(l)) + abseps
      call step1(x,yy,f,neqn,h,eps,wt,start,
     1  hold,k,kold,crash,phi,p,yp,psi,
     2  alpha,beta,sig,v,w,g,phase1,ns,nornd)
c
c   test for tolerances too small
c
      if(.not.crash) go to 130
      iflag = isn*3
      relerr = eps*releps
      abserr = eps*abseps
      do 120 l = 1,neqn
  120   y(l) = yy(l)
      t = x
      told = t
      isnold = 1
      return
c
c   augment counter on number of steps and test for stiffness
c
  130 nostep = nostep + 1
      kle4 = kle4 + 1
      if(kold .gt. 4) kle4 = 0
      if(kle4 .ge. 50) stiff = .true.
      go to 50
      end
c
      subroutine step1(x,y,f,neqn,h,eps,wt,start,
     1  hold,k,kold,crash,phi,p,yp,psi,
     2  alpha,beta,sig,v,w,g,phase1,ns,nornd)
      implicit real*8(a-h,o-z)
c
      logical start,crash,phase1,nornd
      dimension y(neqn),wt(neqn),phi(neqn,16),p(neqn),yp(neqn),psi(12)
      dimension alpha(12),beta(12),sig(13),w(12),v(12),g(13),
     1  gstr(13),two(13)
      external f
c***********************************************************************
c*  the only machine dependent constants are based on the machine unit *
c*  roundoff err  u  which is the smallest positive number such that *
c*  1.0+u .gt. 1.0  .  the user must calculate  u  and insert          *
c*  twou=2.0*u  and  fouru=4.0*u  in the data statement before calling *
c*  the code.  the routine  machin  calculates  u .                    *
      data twou,fouru/.444e-15,.888e-15/
c***********************************************************************
      data two/2.0d0,4.0d0,8.0d0,16.0d0,32.0d0,64.0d0,128.0d0,256.0d0,
     1  512.0d0,1024.0d0,2048.0d0,4096.0d0,8192.0d0/
      data gstr/0.500d0,0.0833d0,0.0417d0,0.0264d0,0.0188d0,0.0143d0,
     1  0.0114d0,0.00936d0,0.00789d0,0.00679d0,0.00592d0,0.00524d0,
     2  0.00468d0/
c
c
c       ***     begin block 0     ***
c   check if step size or err tolerance is too small for machine
c   precision.  if first step, initialize phi array and estimate a
c   starting step size.
c                   ***
c
c   if step size is too small, determine an acceptable one
c
      crash = .true.
      if(abs(h) .ge. fouru*abs(x)) go to 5
      h = sign(fouru*abs(x),h)
      return
    5 p5eps = 0.5d0*eps
c
c   if err tolerance is too small, increase it to an acceptable value
c
      round = 0.0d0
      do 10 l = 1,neqn
   10   round = round + (y(l)/wt(l))**2
      round = twou*sqrt(round)
      if(p5eps .ge. round) go to 15
      eps = 2.0*round*(1.0d0 + fouru)
      return
   15 crash = .false.
      g(1)=1.0d0
      g(2)=0.5d0
      sig(1)=1.0d0
      if(.not.start) go to 99
c
c   initialize.  compute appropriate step size for first step
c
      call f(x,y,yp)
      sum = 0.0d0
      do 20 l = 1,neqn
        phi(l,1) = yp(l)
        phi(l,2) = 0.0d0
   20   sum = sum + (yp(l)/wt(l))**2
      sum = sqrt(sum)
      absh = abs(h)
      if(eps .lt. 16.0d0*sum*h*h) absh = 0.25d0*sqrt(eps/sum)
      h = sign(max(absh,fouru*abs(x)),h)
      hold = 0.0d0
      k = 1
      kold = 0
      start = .false.
      phase1 = .true.
      nornd = .true.
      if(p5eps .gt. 100.0d0*round) go to 99
      nornd = .false.
      do 25 l = 1,neqn
   25   phi(l,15) = 0.0d0
   99 ifail = 0
c       ***     end block 0     ***
c
c       ***     begin block 1     ***
c   compute coefficients of formulas for this step.  avoid computing
c   those quantities not changed when step size is not changed.
c                   ***
c
  100 kp1 = k+1
      kp2 = k+2
      km1 = k-1
      km2 = k-2
c
c   ns is the number of steps taken with size h, including the current
c   one.  when k.lt.ns, no coefficients change
c
      if(h .ne. hold) ns = 0
      if(ns.le.kold)   ns=ns+1
      nsp1 = ns+1
      if (k .lt. ns) go to 199
c
c   compute those components of alpha(*),beta(*),psi(*),sig(*) which
c   are changed
c
      beta(ns) = 1.0d0
      realns = ns
      alpha(ns) = 1.0d0/realns
      temp1 = h*realns
      sig(nsp1) = 1.0d0
      if(k .lt. nsp1) go to 110
      do 105 i = nsp1,k
        im1 = i-1
        temp2 = psi(im1)
        psi(im1) = temp1
        beta(i) = beta(im1)*psi(im1)/temp2
        temp1 = temp2 + h
        alpha(i) = h/temp1
        reali = i
  105   sig(i+1) = reali*alpha(i)*sig(i)
  110 psi(k) = temp1
c
c   compute coefficients g(*)
c
c   initialize v(*) and set w(*).  g(2) is set in data statement
c
      if(ns .gt. 1) go to 120
      do 115 iq = 1,k
        temp3 = iq*(iq+1)
        v(iq) = 1.0d0/temp3
  115   w(iq) = v(iq)
      go to 140
c
c   if order was raised, update diagonal part of v(*)
c
  120 if(k .le. kold) go to 130
      temp4 = k*kp1
      v(k) = 1.0d0/temp4
      nsm2 = ns-2
      if(nsm2 .lt. 1) go to 130
      do 125 j = 1,nsm2
        i = k-j
  125   v(i) = v(i) - alpha(j+1)*v(i+1)
c
c   update v(*) and set w(*)
c
  130 limit1 = kp1 - ns
      temp5 = alpha(ns)
      do 135 iq = 1,limit1
        v(iq) = v(iq) - temp5*v(iq+1)
  135   w(iq) = v(iq)
      g(nsp1) = w(1)
c
c   compute the g(*) in the work vector w(*)
c
  140 nsp2 = ns + 2
      if(kp1 .lt. nsp2) go to 199
      do 150 i = nsp2,kp1
        limit2 = kp2 - i
        temp6 = alpha(i-1)
        do 145 iq = 1,limit2
  145     w(iq) = w(iq) - temp6*w(iq+1)
  150   g(i) = w(1)
  199   continue
c       ***     end block 1     ***
c
c       ***     begin block 2     ***
c   predict a solution p(*), evaluate derivatives using predicted
c   solution, estimate local err at order k and errs at orders k,
c   k-1, k-2 as if constant step size were used.
c                   ***
c
c   change phi to phi star
c
      if(k .lt. nsp1) go to 215
      do 210 i = nsp1,k
        temp1 = beta(i)
        do 205 l = 1,neqn
  205     phi(l,i) = temp1*phi(l,i)
  210   continue
c
c   predict solution and differences
c
  215 do 220 l = 1,neqn
        phi(l,kp2) = phi(l,kp1)
        phi(l,kp1) = 0.0d0
  220   p(l) = 0.0d0
      do 230 j = 1,k
        i = kp1 - j
        ip1 = i+1
        temp2 = g(i)
        do 225 l = 1,neqn
          p(l) = p(l) + temp2*phi(l,i)
  225     phi(l,i) = phi(l,i) + phi(l,ip1)
  230   continue
      if(nornd) go to 240
      do 235 l = 1,neqn
        tau = h*p(l) - phi(l,15)
        p(l) = y(l) + tau
  235   phi(l,16) = (p(l) - y(l)) - tau
      go to 250
  240 do 245 l = 1,neqn
  245   p(l) = y(l) + h*p(l)
  250 xold = x
      x = x + h
      absh = abs(h)
      call f(x,p,yp)
c
c   estimate errs at orders k,k-1,k-2
c
      erkm2 = 0.0d0
      erkm1 = 0.0d0
      erk = 0.0d0
      do 265 l = 1,neqn
        temp3 = 1.0d0/wt(l)
        temp4 = yp(l) - phi(l,1)
        if(km2)265,260,255
  255   erkm2 = erkm2 + ((phi(l,km1)+temp4)*temp3)**2
  260   erkm1 = erkm1 + ((phi(l,k)+temp4)*temp3)**2
  265   erk = erk + (temp4*temp3)**2
      if(km2)280,275,270
  270 erkm2 = absh*sig(km1)*gstr(km2)*sqrt(erkm2)
  275 erkm1 = absh*sig(k)*gstr(km1)*sqrt(erkm1)
  280 temp5 = absh*sqrt(erk)
      err = temp5*(g(k)-g(kp1))
      erk = temp5*sig(kp1)*gstr(k)
      knew = k
c
c   test if order should be lowered
c
      if(km2)299,290,285
  285 if(max(erkm1,erkm2) .le. erk) knew = km1
      go to 299
  290 if(erkm1 .le. 0.5d0*erk) knew = km1
c
c   test if step successful
c
  299 if(err .le. eps) go to 400
c       ***     end block 2     ***
c
c       ***     begin block 3     ***
c   the step is unsuccessful.  restore  x, phi(*,*), psi(*) .
c   if third consecutive failure, set order to one.  if step fails more
c   than three times, consider an optimal step size.  double err
c   tolerance and return if estimated step size is too small for machine
c   precision.
c                   ***
c
c   restore x, phi(*,*) and psi(*)
c
      phase1 = .false.
      x = xold
      do 310 i = 1,k
        temp1 = 1.0d0/beta(i)
        ip1 = i+1
        do 305 l = 1,neqn
  305     phi(l,i) = temp1*(phi(l,i) - phi(l,ip1))
  310   continue
      if(k .lt. 2) go to 320
      do 315 i = 2,k
  315   psi(i-1) = psi(i) - h
c
c   on third failure, set order to one.  thereafter, use optimal step
c   size
c
  320 ifail = ifail + 1
      temp2 = 0.5d0
      if(ifail - 3) 335,330,325
  325 if(p5eps .lt. 0.25d0*erk) temp2 = sqrt(p5eps/erk)
  330 knew = 1
  335 h = temp2*h
      k = knew
      if(abs(h) .ge. fouru*abs(x)) go to 340
      crash = .true.
      h = sign(fouru*abs(x),h)
      eps = eps + eps
      return
  340 go to 100
c       ***     end block 3     ***
c
c       ***     begin block 4     ***
c   the step is successful.  correct the predicted solution, evaluate
c   the derivatives using the corrected solution and update the
c   differences.  determine best order and step size for next step.
c                   ***
  400 kold = k
      hold = h
c
c   correct and evaluate
c
      temp1 = h*g(kp1)
      if(nornd) go to 410
      do 405 l = 1,neqn
        rho = temp1*(yp(l) - phi(l,1)) - phi(l,16)
        y(l) = p(l) + rho
  405   phi(l,15) = (y(l) - p(l)) - rho
      go to 420
  410 do 415 l = 1,neqn
  415   y(l) = p(l) + temp1*(yp(l) - phi(l,1))
  420 call f(x,y,yp)
c
c   update differences for next step
c
      do 425 l = 1,neqn
        phi(l,kp1) = yp(l) - phi(l,1)
  425   phi(l,kp2) = phi(l,kp1) - phi(l,kp2)
      do 435 i = 1,k
        do 430 l = 1,neqn
  430     phi(l,i) = phi(l,i) + phi(l,kp1)
  435   continue
c
c   estimate err at order k+1 unless:
c     in first phase when always raise order,
c     already decided to lower order,
c     step size not constant so estimate unreliable
c
      erkp1 = 0.0d0
      if(knew .eq. km1  .or.  k .eq. 12) phase1 = .false.
      if(phase1) go to 450
      if(knew .eq. km1) go to 455
      if(kp1 .gt. ns) go to 460
      do 440 l = 1,neqn
  440   erkp1 = erkp1 + (phi(l,kp2)/wt(l))**2
      erkp1 = absh*gstr(kp1)*sqrt(erkp1)
c
c   using estimated err at order k+1, determine appropriate order
c   for next step
c
      if(k .gt. 1) go to 445
      if(erkp1 .ge. 0.5d0*erk) go to 460
      go to 450
  445 if(erkm1 .le. min(erk,erkp1)) go to 455
      if(erkp1 .ge. erk  .or.  k .eq. 12) go to 460
c
c   here erkp1 .lt. erk .lt. max(erkm1,erkm2) else order would have
c   been lowered in block 2.  thus order is to be raised
c
c   raise order
c
  450 k = kp1
      erk = erkp1
      go to 460
c
c   lower order
c
  455 k = km1
      erk = erkm1
c
c   with new order determine appropriate step size for next step
c
  460 hnew = h + h
      if(phase1) go to 465
      if(p5eps .ge. erk*two(k+1)) go to 465
      hnew = h
      if(p5eps .ge. erk) go to 465
      temp2 = k+1
      r = (p5eps/erk)**(1.0d0/temp2)
      hnew = absh*max(0.5d0,min(0.9d0,r))
      hnew = sign(max(hnew,fouru*abs(x)),h)
  465 h = hnew
      return
c       ***     end block 4     ***
      end
c
      subroutine intrp(x,y,xout,yout,ypout,neqn,kold,phi,psi)
      implicit real*8(a-h,o-z)
       dimension y(neqn),yout(neqn),ypout(neqn),phi(neqn,16),psi(12)
      dimension g(13),w(13),rho(13)
      data g(1)/1.0d0/,rho(1)/1.0d0/
c
      hi = xout - x
      ki = kold + 1
      kip1 = ki + 1
c
c   initialize w(*) for computing g(*)
c
      do 5 i = 1,ki
        temp1 = i
    5   w(i) = 1.0d0/temp1
      term = 0.0d0
c
c   compute g(*)
c
      do 15 j = 2,ki
        jm1 = j - 1
        psijm1 = psi(jm1)
        gamma = (hi + term)/psijm1
        eta = hi/psijm1
        limit1 = kip1 - j
        do 10 i = 1,limit1
   10     w(i) = gamma*w(i) - eta*w(i+1)
        g(j) = w(1)
        rho(j) = gamma*rho(jm1)
   15   term = psijm1
c
c   interpolate
c
      do 20 l = 1,neqn
        ypout(l) = 0.0d0
   20   yout(l) = 0.0d0
      do 30 j = 1,ki
        i = kip1 - j
        temp2 = g(i)
        temp3 = rho(i)
        do 25 l = 1,neqn
          yout(l) = yout(l) + temp2*phi(l,i)
   25     ypout(l) = ypout(l) + temp3*phi(l,i)
   30   continue
      do 35 l = 1,neqn
   35   yout(l) = y(l) + hi*yout(l)
      return
      end

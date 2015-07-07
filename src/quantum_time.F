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

c$$$      do jj=1,nstate/8
c$$$         do ii=1,nstate/8
c$$$            rho_r(ii,jj) = 1
c$$$            if(ii.gt.jj) then
c$$$               rho_i(ii,jj) = -1
c$$$            else if(ii.eq.jj) then
c$$$               rho_i(ii,jj) = 0
c$$$            else
c$$$               rho_i(ii,jj) = 1
c$$$            endif
c$$$         enddo
c$$$      enddo
c!$acc parallel
c$$$      rho_r(2,2) = 0.5
c$$$      rho_r(3,3) = 0.5
c$$$      rho_r(2,3) = -0.5
c$$$      rho_r(3,2) = -0.5
      if(nid.eq.0) rho_r(2,2) = 1

c!$acc end parallel


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
      printdt      = dt
      obsPrintSkip =1 
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
      do ii=1,numsp
         do jj=1,ltonq
            do kk=1,ltonq
               gloJ = ltonq*ii-(ltonq-jj)
               gloI = ltonq*ii-(ltonq-kk)

               if((gloJ.gt.nid*nstate_np).and.
     $              (gloJ.le.(nid+1)*nstate_np)) then
                  locJ = gloJ - c_offset
                  glo_num_q_concV(gloI+(locJ-1)*nstate) =
     $                 kk+(jj-1)*maxPartialTrace
               endif

c$$$               red_rho(kk,jj) = red_rho(kk,jj) + 
c$$$     $              rho(ltonq*ii-(ltonq-kk),ltonq*ii-(ltonq-jj))

            enddo
         enddo
      enddo
      
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
      call gs_setup(gs_handle_q_conc,glo_num_q_concV,nstate_p,
     $     MPI_COMM_WORLD,np)
#else
      call gs_setup(gs_handle_q_H,glo_num_q_Hv,nstate_p,
     $     nekcomm,np)
      call gs_setup(gs_handle_q_conc,glo_num_q_concV,nstate_p,
     $     nekcomm,np)
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
      integer  ii,jl,kl

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
      subroutine cem_quantum3_propagation_rk
c--------------------------------------------------------------------- 
      implicit none
      include 'SIZE'
      include 'QUANT'
      include 'RK5'
#ifdef MPI
      include 'mpif.h'
#endif
      integer  iEh, ii, IERR, jj,kk
      real*8     stime0,stime1,stime2,stime3,stime4,stime5,stime6,stime7
      real*8     sigr,sigi,tmpr,tmpi,rinc,sigcon,pulseFactor,fullPulse
      real*8   dclock,c_speed,d_per_au,eesu_per_au,eps_med,eV
      complex  sig,en_t(nEh),obs_freq(maxOp,nEh),tmpc1,ci

      ci=(0.0,1.0)
      if(nqstep.gt.maxTimeSteps) then
         print*, 'ERROR!'
         print*, 'nqstep TOO LARGE FOR THIS maxTimeSteps!'
         call exitt()
      endif

      if(nid.eq.0) then
c     Write file headers 
         write(12,*)'#Time,populations'
         write(13,*)'#Time,observables'
         if(calc_concurrence.eqv..true.) then
            write(14,*)'#Time,3:2,3:1,2:1'
         endif
      endif

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
      call cem_quantum3_initialize      
      call cem_quantum3_gs
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

      if(ifPulse.eqv..true.) then
         do ii=1,nEh
            en_t(ii) = (0.0,0.0)
            do jj=1,numObservables
               obs_freq(jj,ii) = (0.0,0.0)
            enddo
         enddo
      endif

c     Pulsed case
c      do iqstep= 1,nqstep
       iqstep=1
       iiqstep = 1
       do while(time.lt.timemax)
!         print *,nid,time
          stime1= dclock()
         do ii= 1,5
!            call rk_quantum3_c(ii)
            stime7 = dclock()
            call cem_quantum3_hamiltonian()
!            print*,'after ham',nid,iqstep,ii
            stime4 = stime4 + dclock()-stime7
            stime7 = dclock()
            call quantum_lin_mpi_get()
!            print*,'after get',nid,iqstep,ii
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
         time= time+dt
         stime7= dclock()
         if(useGates.eqv..true.) then
            call quantum_apply_gates()
         endif
         call cem_quantum3_observables(obs_freq,en_t)
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
c$$$         if(iqstep.gt.100) then
c$$$            stime0 = dclock()-stime0
c$$$            if(nid.eq.0) write(6,*)'stime0:',stime0 
c$$$            call exitt()
c$$$         endif
      enddo
      
      stime0 = dclock()-stime0
      if(nid.eq.0) write(6,*)'stime0:',stime0 
      iiqstep = iiqstep - 1
      iqstep = iqstep -1
!     Reduce time vectors
#ifdef MPI
      do ii=1,numberOfOps
         if(nid.eq.0) then
            call MPI_REDUCE(MPI_IN_PLACE,pop_time(1,ii),iiqstep,
     $           MPI_REAL8,MPI_SUM,0,MPI_COMM_WORLD,IERR)
         else
            call MPI_REDUCE(pop_time(1,ii),pop_time(1,ii),iiqstep,
     $           MPI_REAL8,MPI_SUM,0,MPI_COMM_WORLD,IERR)
         endif
      enddo

      do ii=1,numObservables
         if(nid.eq.0) then
            call MPI_REDUCE(MPI_IN_PLACE,obs_time(1,ii),iqstep,
     $           MPI_REAL8,MPI_SUM,0,MPI_COMM_WORLD,IERR)
         else
            call MPI_REDUCE(obs_time(1,ii),obs_time(1,ii),iqstep,
     $           MPI_REAL8,MPI_SUM,0,MPI_COMM_WORLD,IERR)
         endif
      enddo

#endif



      if(nid.eq.0) then
         time = dt
         kk = 1
         if(ifPulse.eqv..true.) then

            do ii=1,iqstep
            
               call usertimedep(pulseFactor)
c     Now, do the fourier transform to save the value
!     Envelope*oscillations --- consider putting oscillations in user control?
               if(rwa.eqv..true.) then
                  fullPulse = 2*pulseFactor*
     $                 cos(pulse_omega_0*(time-pulse_t_0))
               else
                  fullPulse = pulseFactor
               endif
               
               do kk=1,nEh
                  en_t(kk) = en_t(kk)+fullPulse*
     $                 exp(ci*omega_quant(kk)*(time-pulse_t_0))
                  
                  do jj=1,numObservables
                     obs_freq(jj,kk) = obs_freq(jj,kk) +  
     $                    (exp(ci*omega_quant(kk)*(time-pulse_t_0))*
     $                    obs_time(ii,jj))
                  enddo
                  
               enddo
!     write(13,12)time*timeunit
!     $           ,(obs_time(ii,jj),jj=1,numObservables)
               
               time = time + dt
            enddo
         endif

         kk = 1
         time = dt
         do while(time.lt.timemax)
            if(ifPulse.eqv..true.) then
               call usertimedep(pulseFactor)
c     Now, do the fourier transform to save the value
!     Envelope*oscillations --- consider putting oscillations in user control?
               if(rwa.eqv..true.) then
                  fullPulse = 2*pulseFactor*
     $                 cos(pulse_omega_0*(time-pulse_t_0))
               else
                  fullPulse = pulseFactor
               endif
               if(abs(pulseFactor).lt.1e-10) pulseFactor=0

               write(13,12)time*timeunit,pulseFactor
     $              ,(obs_time(kk,jj),jj=1,numObservables)
            else
               write(13,12)time*timeunit
     $              ,(obs_time(kk,jj),jj=1,numObservables)
            endif
            kk = kk + obsPrintSkip
            time = time + obsPrintSkip*dt
         enddo

         time = dt
         do ii=1,iiqstep
            !Print populations
            write(12,12)time*timeunit
     $           ,(pop_time(ii,jj),jj=1,numberOfOps)

            time = time+printdt
         enddo
 12      format(1p20g20.8)
c     Print out results
         if(ifPulse.eqv..true.) then
            write(11,*)'#eV,sig_abs,sig_scatt,sig_tot'

!     Update this stuff!
!I'm not sure how to do this is a really user friendly manner
!FIXME! Hardcoded
            c_speed = 2.99792458d10 !cm/s
            d_per_au = 1.0d0 / 3.93430307d-1
            eesu_per_au = 5.14220652d17 / c_speed
            eps_med     = 2.25
            eV       = 27.21140 ! energy Eh  
            
            do ii=1,nEh
               rinc = 4.0*pi*(omega_quant(ii)/timeunit)/
     $              (c_speed*eps_med**0.5)*
     $              d_per_au*1.0d-18/eesu_per_au
               sig = (0,0)
               do jj=1,numObservables
                  sig = sig + obs_freq(jj,ii)
               enddo
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
               write(11,13) eV*omega_quant(ii),sigi,sigr,sigr+sigi 
            enddo
            
 13         format(1p5e20.10)
         endif
      close(12)
      close(11)
      endif

!$acc end data

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
         tmpfq= state_sums(ii,jj)*pulse_omega_0*rho(ii,jj)
         tmprr= 2*tmp_r1(ii,jj)-tmprv((jj-1)*nstate+ii)
!         if(ii.eq.jj+c_offset) tmprr = 0
         tmpii = tmpiv((jj-1)*nstate+ii)
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


c!$omp end parallel do
!$acc end parallel loop

!$acc end data

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
         print*,"Rho before"
         do ii=1,nstate
            write(6,'(100g25.15)') (rho_r(ii,jj), jj=1,nstate)
         enddo


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
         print*,"after mxm 1"
         do ii=1,nstate
            write(6,'(100g25.15)') (tmp_c1(ii,jj), jj=1,nstate)
         enddo


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
         print*,"after conj transpose"
         do ii=1,nstate
            write(6,'(100g25.15)') (tmp_c1(ii,jj), jj=1,nstate)
         enddo


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

         print*,"Rho after"
         do ii=1,nstate
            write(6,'(100g25.15)') (rho(ii,jj), jj=1,nstate)
         enddo

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
      integer ii,jj,iq,tilingNumber,curState,ierr
      complex*16 ci
      real*8     pop(maxOp),currentPop


      ci = (0.0,1.0)

!$acc data present(state_sums,rho_r,rho_i,rho)
!$acc parallel loop collapse(2)
      do jj=1,nstate_np
      do ii=1,nstate
         tmp_c1(ii,jj)= (rho_r(ii,jj)+ci*rho_i(ii,jj))
         if(ifPulse.eqv..true.) then
            tmp_c1(ii,jj) = tmp_c1(ii,jj)*
     $        *exp(-ci*state_sums(ii,jj)*pulse_omega_0*(time-pulse_t_0))
         endif
      enddo
      enddo
!$acc end parallel loop
!$acc update host(rho)
!$acc end data

      do iq=1,numberOfOps
         pop(iq) = 0
      enddo

c     Find populations 
      do ii=1,numberOfOps
         tilingNumber = oldN(ii)
         do jj=1,nstate_np
            curState = state(ii,mod(jj+c_offset-1,tilingNumber)+1)
            currentPop = real(tmp_c1(jj+c_offset,jj))
            !print*,curState,currentPop
            pop(ii) = pop(ii) + 
     $           curState*currentPop
         enddo
         if(abs(pop(ii)).le.1e-13) pop(ii) = 0.0
!     Save results
         pop_time(iiqstep,ii) = pop(ii)
      enddo
      
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
      subroutine cem_quantum3_observables(obs_freq,en_t)
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
      complex  en_t(nEh),obs_freq(maxOp,nEh)!Update with variable

      ci = (0.0,1.0)

!$acc data present(state_sums,rho_r,rho_i,rho)
!$acc parallel loop collapse(2)
      do jj=1,nstate_np
      do ii=1,nstate
            tmp_c1(ii,jj)= (rho_r(ii,jj)+ci*rho_i(ii,jj))
     $        *exp(-ci*state_sums(ii,jj)*pulse_omega_0*(time-pulse_t_0))
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

      
      do ii=1,numObservables
         obs_time(iqstep,ii) = observable(ii)
      enddo




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
      complex red_rho(2**3,2**3)
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
     $        *exp(-ci*state_sums(ii,jj)*pulse_omega_0*(time-pulse_t_0))
!            kk = kk+1
         enddo
      enddo

c$$$      call outmat(rho_r,nstate,nstate,'rho_r',nid)
c$$$      call outmat(rho_i,nstate,nstate,'rho_i',nid)
c$$$      if(nid.eq.0) call outmat(red_rho,8,8,'red_rho',nid)
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

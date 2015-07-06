c----------------------------------------------------------------------
c
c  USER SPECIFIED ROUTINES:
c
c     - boundary conditions
c     - initial conditions
c     - variable properties
c     - forcing function for fluid (f)
c     - forcing function for passive scalar (q)
c     - general purpose routine for checking errors etc.
c
c-----------------------------------------------------------------------
      subroutine useriniQ
c-----------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'QUANT'
      include 'mpif.h'

      real op(160,160)
      real omega,g_couple,eV,gamma_di,gamma_pi,gamma_s,mu_s,mu_q
      real c_speed,d_per_au,eesu_per_au,eps_med,fluence,tmpint
      real omega2,debye,omega3,radius,g_couple2,g_couple3
      integer qd1,qd1t,qd1n,sp,spt,spn
      integer qd2,qd2t,qd2n,ii,jj,kk,j0
      integer qd1t_sp,qd1_spt,sp2,sp2t,sp2n
      integer qd2t_sp,qd2_spt,ierr,qd2t_sp2,qd2_sp2t
      integer qd1t_spt,qd1_sp,sp2_qd1t,sp2t_qd1
      integer qd1_a_qd1t,qd2_a_qd2t,sp_a_spt
      integer qd3,qd3t,qd3n,qd3_a_qd3t,qd3t_sp,qd3_spt
      real hadamard(2,2)
      
      hadamard(1,1) = 1
      hadamard(1,2) = 2
      hadamard(2,1) = 3
      hadamard(2,2) = 4


      call quantum_tensor_product_um(hadamard,1,8)


      print*,'nid ini',nid

      !Read the simple input
      if(nid.eq.0) then
      open(unit=2,file='quantInput')
      read(2,*) fluence,g_couple,g_couple2,g_couple3
      close(unit=2)
      endif
      call MPI_BCAST(fluence,1,MPI_REAL8,0,MPI_COMM_WORLD,IERR)
      call MPI_BCAST(g_couple,1,MPI_REAL8,0,MPI_COMM_WORLD,IERR)
      call MPI_BCAST(g_couple2,1,MPI_REAL8,0,MPI_COMM_WORLD,IERR)
      call MPI_BCAST(g_couple3,1,MPI_REAL8,0,MPI_COMM_WORLD,IERR)
      
      eV       = 27.21140                ! energy Eh  
      debye     = 0.3934303070      

      mu_s     = 4d3*debye
      mu_q     = 1.3d1*debye

      gamma_pi = 1.9d-7/eV
      gamma_di = 2.0d-3/eV
      gamma_s  = 1.5d-1/eV

      omega    = 2.05/eV    
      omega2   = 2.05/eV
      
      g_couple  = g_couple*1e-3/eV
      g_couple2 = g_couple2*1e-3/eV
      g_couple3 = g_couple3*1e-3/eV

      fluence = fluence*1e-9*1e7
      pulse_omega_0 = 2.05/eV!omega

      debug         = 1
      
!     Parameters for energy spectrum

      Emax = 2.14
      Emin = 1.94
      nEh  = 100

!     Setup pulse things
      c_speed = 2.99792458d10 !cm/s
      d_per_au = 1.0d0 / 3.93430307d-1
      eesu_per_au = 5.14220652d17 / c_speed
      timeunit= 2.418884326505e-17        ! hbar/Eh(hartree) SI unit       
      eps_med     = 2.25
!      fluence     = 1.0e-9*1d7  ! nJ/cm**2
      pi          = 4.*ATAN(1.0)
      pulse_duration = 2e-14/timeunit
      rwa         = .true.
      timemax     = 10e-13/timeunit
      nqstep      = 10000
      pulse_t_0     = 1e-13/timeunit
      numsp       =32 
      ltonq       = 8

      tmpint = 5e-1 * sqrt(5d-1 * pi/(2d0*log(2d0)/pulse_duration**2))
     $     * (1e0 + exp(-5e-1 * pulse_omega_0 ** 2 / 
     $     (2d0*log(2d0)/pulse_duration**2)))*timeunit
      energy0_pls =sqrt(4*pi*fluence/(c_speed*sqrt(eps_med)*tmpint))
      energy0_pls = energy0_pls / eesu_per_au
      

      call quantum_setup_initialize()
      
      call quantum_create_op(2,qd1,qd1t,qd1n)
      call quantum_create_op(2,qd2,qd2t,qd2n)
      call quantum_create_op(2,qd3,qd3t,qd3n)
      call quantum_create_op(numsp,sp,spt,spn)

      call quantum_finalize_op_csr()

      call quantum_combine_op_csr(qd1t,sp,1,qd1t_sp)

      call quantum_combine_op_csr(qd1,spt,1,qd1_spt)
      call quantum_combine_op_csr(qd2t,sp,1,qd2t_sp)
      call quantum_combine_op_csr(qd2,spt,1,qd2_spt)

      call quantum_combine_op_csr(qd3t,sp,1,qd3t_sp)
      call quantum_combine_op_csr(qd3,spt,1,qd3_spt)

      call quantum_combine_op_csr(qd1,qd1t,0,qd1_a_qd1t)
      call quantum_combine_op_csr(qd2,qd2t,0,qd2_a_qd2t)
      call quantum_combine_op_csr(qd3,qd3t,0,qd3_a_qd3t)
      call quantum_combine_op_csr(sp,spt,0,sp_a_spt)
      print*,'after combine op',nid      
      !Oscillator terms
      call quantum_add_to_ham_csr(qd3n,omega,0)
      call quantum_add_to_ham_csr(qd2n,omega,0)
      call quantum_add_to_ham_csr(qd1n,omega,0)      
      call quantum_add_to_ham_csr(spn,omega,0)

      !Driving terms
      call quantum_add_to_ham_csr(qd1_a_qd1t,-mu_q,1)
      call quantum_add_to_ham_csr(qd2_a_qd2t,-mu_q,1)
      call quantum_add_to_ham_csr(qd3_a_qd3t,-mu_q,1)
      call quantum_add_to_ham_csr(sp_a_spt,-mu_s,1)

      !Coupling terms
      call quantum_add_to_ham_csr(qd1t_sp,-g_couple,0)
      call quantum_add_to_ham_csr(qd1_spt,-g_couple,0)

      call quantum_add_to_ham_csr(qd2t_sp,-g_couple2,0)
      call quantum_add_to_ham_csr(qd2_spt,-g_couple2,0)

!
      call quantum_add_to_ham_csr(qd3t_sp,-g_couple3,0)
      call quantum_add_to_ham_csr(qd3_spt,-g_couple3,0)
      print*,'after add to ham',nid      
      call quantum_ham_finalize_csr()
      print*,'after add to ham',nid      

c$$$      if(nid.eq.1) then
c$$$         do ii=1,ham_nnz
c$$$            ham_a(ii) = ham_i(ii) + ham_d(ii)
c$$$         enddo
c$$$
c$$$         do ii=1,nstate
c$$$            write(15,'(100g15.5)') (op(ii,jj), jj=1,nstate)
c$$$         enddo
c$$$      endif
c$$$      if(nid.eq.1) then
c$$$         do ii=1,ham_nnz
c$$$            ham_a(ii) = ham_i(ii) + ham_d(ii)
c$$$         enddo
c$$$
c$$$         do ii=1,nstate
c$$$            write(15,'(100g15.5)') (op(ii,jj), jj=1,nstate)
c$$$         enddo
c$$$      endif
c$$$
c$$$
c$$$      if(nid.eq.1) then
c$$$         do ii=1,ham_nnz
c$$$            ham_a(ii) = ham_i(ii) + ham_d(ii)
c$$$         enddo
c$$$
c$$$         do ii=1,nstate
c$$$            write(15,'(100g15.5)') (op(ii,jj), jj=1,nstate)
c$$$         enddo
c$$$      endif
c$$$      if(nid.eq.0) then
c$$$         do ii=1,ham_nnz
c$$$            ham_a(ii) = ham_i(ii) + ham_d(ii)
c$$$         enddo
c$$$         call csrdns(nstate,nstate,ham_a,ham_ja,ham_ia,op,nstate,ierr)
c$$$         do ii=1,nstate
c$$$            write(16,'(100g15.5)') (op(ii,jj), jj=1,nstate)
c$$$         enddo
c$$$      endif
c$$$      
c$$$      call exitt
c$$$

      call quantum_lin_add(qd1,qd1n,-gamma_pi/2)
      call quantum_lin_add(qd1n,qd1n,-gamma_di)

      call quantum_lin_add(qd2,qd2n,-gamma_pi/2)
      call quantum_lin_add(qd2n,qd2n,-gamma_di)

      call quantum_lin_add(qd3,qd3n,-gamma_pi/2)
      call quantum_lin_add(qd3n,qd3n,-gamma_di)

      call quantum_lin_add(sp,spn,-gamma_s/2)

      call quantum_lin_finalize()

      if(nid.eq.0) then
      do ii=1,num_nl_lin_ops
         print*,'l_length',lin_l_length(ii)
         print*,'nl_col_length',lin_nl_col_length(ii)
         do kk=1,lin_nl_col_length(ii)
            print*,'lin_nl_length',lin_nl_length(kk,ii)
         enddo
      enddo
      endif
!      call quantum3_debug8cores()
!      call exitt()

c$$$      call quantum_lin_gs_op()
c$$$      
c$$$      if(nid.eq.0) then
c$$$         do ii=1,nstate
c$$$            write(15,'(100g15.5)') (real(drho(ii,jj)),jj=1,nstate/np)
c$$$         enddo
c$$$         write(15,*)'imag'
c$$$         do ii=1,nstate
c$$$            write(15,'(100g15.5)') (imag(drho(ii,jj)),jj=1,nstate/np)
c$$$         enddo
c$$$      endif
c$$$      if(nid.eq.1) then
c$$$         do ii=1,nstate
c$$$            write(16,'(100g15.5)') (real(drho(ii,jj)),jj=1,nstate/np)
c$$$         enddo
c$$$         write(16,*)'imag'
c$$$         do ii=1,nstate
c$$$            write(16,'(100g15.5)') (imag(drho(ii,jj)),jj=1,nstate/np)
c$$$         enddo
c$$$      endif
c$$$
c$$$      call exitt()

!     Add observables
      call quantum_observable(sp_a_spt,mu_s)
      call quantum_observable(qd1_a_qd1t,mu_q)
      call quantum_observable(qd2_a_qd2t,mu_q)

      call cem_quantum3_propagation_rk

      return
      end



c-----------------------------------------------------------------------
      subroutine usertimedep(pulse)
      implicit none
      include 'SIZE'
      include 'QUANT'
      real pulse
      real alpha

      alpha = 2 * log(2.0)/ (pulse_duration)**2
      if(rwa.eqv..true.) then
         pulse = energy0_pls/2.0*exp(-alpha*(time-pulse_t_0)**2)
      else
         pulse = energy0_pls/2.0*exp(-alpha*(time-pulse_t_0)**2)
     $        *cos(pulse_omega_0*time)
      endif

!      if(nid.eq.1) then
!      print*,'1',time*timeunit,pulse
 !     print*,'2',eps_med,c_speed,eesu_per_au,timeunit,pi,time

!      endif
      return
      end

c-----------------------------------------------------------------------
      subroutine userchk
      include 'SIZE'
      include 'QUANT'


c ... compute errors


      if (iqstep.le.10.or.mod(iqstep,100).eq.0) then

c$$$        cpu_t1  = glsum(qrtime1,1)/(np*iqstep) ! total cpu/(np*iqstep)
c$$$        cpu_t2  = glsum(qrtime2,1)/(np*iqstep) ! hamiltonian/(np*iqstep)
c$$$        cpu_t3  = glsum(qrtime3,1)/(np*iqstep) ! linblad/(np*iqstep)
c$$$        cpu_t4  = glsum(qrtime4,1)/(np) ! fourier/(np)
c$$$
c$$$        comm_t1 = glsum(qctime1,1)/(np*iqstep) ! total gs_op/(np*iqstep)
c$$$        comm_t3 = glsum(qctime3,1)/(np*iqstep) ! total get  /(np*iqstep)

        if (nid.eq.0) then

        if(iqstep.eq.1) write(6,*)'iqstep nstate
     $gs_op get total ham lin fourier gs_op get total ham lin fourier'
         write(6,91) iqstep,nstate, 
     $               qctime1,qctime3,
     $               qrtime1,qrtime2, qrtime3, qrtime4,
     $               comm_t1,comm_t3,              
     $               cpu_t1,cpu_t2,cpu_t3,cpu_t4              

 91      format(/,i10,i10,12e10.3,' CPU: L2  '  )

        endif ! nid=0
      endif

      return
      end

c-----------------------------------------------------------------------
      subroutine userchk_acc
      include 'SIZE'
      include 'QUANT'


c ... compute errors


      if (iqstep.le.10.or.mod(iqstep,100).eq.0) then

c$$$        cpu_t1  = glsum(qrtime1,1)/(np*iqstep) ! total cpu/(np*iqstep)
c$$$        cpu_t2  = glsum(qrtime2,1)/(np*iqstep) ! hamiltonian/(np*iqstep)
c$$$        cpu_t3  = glsum(qrtime3,1)/(np*iqstep) ! linblad/(np*iqstep)
c$$$        cpu_t4  = glsum(qrtime4,1)/(np) ! fourier/(np)
c$$$
c$$$        comm_t1 = glsum(qctime1,1)/(np*iqstep) ! total gs_op/(np*iqstep)
c$$$        comm_t3 = glsum(qctime3,1)/(np*iqstep) ! total get  /(np*iqstep)

        if (nid.eq.0) then

        if(iqstep.eq.1) write(6,*)'iqstep level numqd numsp nstate
     $gs_op get total ham lin fourier gs_op get total ham lin fourier'
         write(6,91) iqstep, level,numqd,numsp,nstate, 
     $               qctime1,qctime3,
     $               qrtime1,qrtime2, qrtime3, qrtime4,
     $               comm_t1,comm_t3,    
     $               cpu_t1,cpu_t2,cpu_t3,cpu_t4    

 91      format(/,i10,3i7,i10,12e10.3,' CPU: L2  '  )

        endif ! nid=0
      endif

      return
      end 

c-----------------------------------------------------------------------

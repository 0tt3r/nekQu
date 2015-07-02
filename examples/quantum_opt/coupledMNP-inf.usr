
c-----------------------------------------------------------------------
      subroutine useriniQ
c-----------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'QUANT'
      include 'mpif.h'

      real g_couple_2_1,eV,gamma_sa,gamma_sb,mu_sa,mu_sb
      real c_speed,d_per_au,eesu_per_au,eps_med,fluence,tmpint
      real omega1,debye,omega2,radius,g_couple_1_1,g_couple_2_2
      integer ii,jj,kk
      !Operator handles below
      integer sa1,sa1t,sa1n,sa2,sa2t,sa2n
      integer sb1,sb1t,sb1n,sb2,sb2t,sb2n
      integer sa1t_sb1,sa1_sb1t,sa2t_sb1,sa2_sb1t
      integer sa1t_sb2,sa1_sb2t,sa2t_sb2,sa2_sb2t
      integer sa1_a_sa1t,sb1_a_sb1t
      
      eV       = 27.21140                ! energy Eh  
      debye     = 0.3934303070      

      mu_sa    = 1.7d3*debye
      mu_sb    = mu_sa
      gamma_sa  = 1.00d-1/eV
      gamma_sb = gamma_sa
      omega1   = 3.05/eV    
      omega2   = 3.111/eV

      g_couple_1_1 = 300.0e-3/eV
      g_couple_2_1 = 100.0e-3/eV
      g_couple_2_2 = 14.16e-3/eV
      pulse_omega_0 = 3.05/eV!omega

      debug         = 1
      
!     Parameters for energy spectrum

      Emax             = 3.5 
      Emin             = 2.5
      nEh              = 100
      nqstep           = 2500
!     Setup pulse things
      c_speed          = 2.99792458d10 !cm/s
      d_per_au         = 1.0d0 / 3.93430307d-1
      eesu_per_au      = 5.14220652d17 / c_speed
      timeunit         = 2.418884326505e-17        ! hbar/Eh(hartree) SI unit       
      eps_med          = 2.25
      fluence          = 50.0e-9*1d7  ! nJ/cm**2
      pi               = 4.*ATAN(1.0)
      pulse_duration   = 0.1e-14/timeunit
      rwa              = .true.
      calc_concurrence = .false.
      pulse_t_0        = 1e-13/timeunit
      timemax          = 2.5e-13/timeunit

      tmpint           = 5e-1 * sqrt(5d-1 * pi/(2d0*log(2d0)/
     $     pulse_duration**2))* (1e0 + exp(-5e-1 * pulse_omega_0 ** 2 / 
     $     (2d0*log(2d0)/pulse_duration**2)))*timeunit
      energy0_pls     =sqrt(4*pi*fluence/(c_speed*sqrt(eps_med)*tmpint))
      energy0_pls     = energy0_pls / eesu_per_au

      call quantum_setup_initialize()

      call quantum_create_op(4,sa1,sa1t,sa1n)
      call quantum_create_op(2,sa2,sa2t,sa2n)
      call quantum_create_op(4,sb1,sb1t,sb1n)
      call quantum_create_op(2,sb2,sb2t,sb2n)

      call quantum_finalize_op_csr()

      call quantum_combine_op_csr(sa1t,sb1,1,sa1t_sb1)
      call quantum_combine_op_csr(sa1,sb1t,1,sa1_sb1t)

      call quantum_combine_op_csr(sa2t,sb1,1,sa2t_sb1)
      call quantum_combine_op_csr(sa2,sb1t,1,sa2_sb1t)

      call quantum_combine_op_csr(sa1t,sb2,1,sa1t_sb2)
      call quantum_combine_op_csr(sa1,sb2t,1,sa1_sb2t)

      call quantum_combine_op_csr(sa2t,sb2,1,sa2t_sb2)
      call quantum_combine_op_csr(sa2,sb2t,1,sa2_sb2t)

      call quantum_combine_op_csr(sa1,sa1t,0,sa1_a_sa1t)
      call quantum_combine_op_csr(sb1,sb1t,0,sb1_a_sb1t)
      
      !Oscillator terms
      call quantum_add_to_ham_csr(sb2n,omega2,0)
      call quantum_add_to_ham_csr(sa2n,omega2,0)
      call quantum_add_to_ham_csr(sa1n,omega1,0)      
      call quantum_add_to_ham_csr(sb1n,omega1,0)

      !Driving terms
      call quantum_add_to_ham_csr(sa1_a_sa1t,-mu_sa,1)
      call quantum_add_to_ham_csr(sb1_a_sb1t,-mu_sb,1)

      !Coupling terms
      call quantum_add_to_ham_csr(sa1t_sb1,-g_couple_1_1,0)
      call quantum_add_to_ham_csr(sa1_sb1t,-g_couple_1_1,0)

      call quantum_add_to_ham_csr(sa2t_sb1,-g_couple_2_1,0)
      call quantum_add_to_ham_csr(sa2_sb1t,-g_couple_2_1,0)

      call quantum_add_to_ham_csr(sa1t_sb2,-g_couple_2_1,0)
      call quantum_add_to_ham_csr(sa1_sb2t,-g_couple_2_1,0)

      call quantum_add_to_ham_csr(sa2t_sb2,-g_couple_2_2,0)
      call quantum_add_to_ham_csr(sa2_sb2t,-g_couple_2_2,0)


      call quantum_ham_finalize_csr()

      call quantum_lin_add(sa1,sa1n,-gamma_sa/2)

      call quantum_lin_add(sa2,sa2n,-gamma_sa)
      call quantum_lin_add(sb2,sb2n,-gamma_sb/2)

      call quantum_lin_add(sb1,sb1n,-gamma_sb)

      call quantum_lin_finalize()

!     Add observables
      call quantum_observable(sa1_a_sa1t,mu_sa)
      call quantum_observable(sb1_a_sb1t,mu_sb)

      call cem_quantum3_propagation_rk
      call exitt

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
      if(rwa.eq..true.) then
         pulse = energy0_pls/2.0*exp(-alpha*(time-pulse_t_0)**2)
      else
         pulse = energy0_pls/2.0*exp(-alpha*(time-pulse_t_0)**2)
     $        *cos(pulse_omega_0*time)
      endif

      if(nid.eq.1) then
!      print*,'1',pulse,time
 !     print*,'2',eps_med,c_speed,eesu_per_au,timeunit,pi,time

      endif
      return
      end

c-----------------------------------------------------------------------
      subroutine userchk
      include 'SIZE'
      include 'QUANT'


c ... compute errors


      if (iqstep.le.10.or.mod(iqstep,100).eq.0) then

        ! cpu_t1  = glsum(qrtime1,1)/(np*iqstep) ! total cpu/(np*iqstep)
        ! cpu_t2  = glsum(qrtime2,1)/(np*iqstep) ! hamiltonian/(np*iqstep)
        ! cpu_t3  = glsum(qrtime3,1)/(np*iqstep) ! linblad/(np*iqstep)
        ! cpu_t4  = glsum(qrtime4,1)/(np) ! fourier/(np)

        ! comm_t1 = glsum(qctime1,1)/(np*iqstep) ! total gs_op/(np*iqstep)
        ! comm_t3 = glsum(qctime3,1)/(np*iqstep) ! total get  /(np*iqstep)

        if (nid.eq.0) then

        if(iqstep.eq.1) write(6,*)'iqstep nstate
     $gs_op get total ham lin fourier gs_op get total ham lin fourier'
         write(6,91) iqstep, nstate, 
     $               qctime1,qctime3,
     $               qrtime1,qrtime2, qrtime3, qrtime4,
     $               comm_t1,comm_t3,              
     $               cpu_t1,cpu_t2,cpu_t3,cpu_t4              

 91      format(/,2i10,12e10.3,' CPU: L2  '  )

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

        ! cpu_t1  = glsum(qrtime1,1)/(np*iqstep) ! total cpu/(np*iqstep)
        ! cpu_t2  = glsum(qrtime2,1)/(np*iqstep) ! hamiltonian/(np*iqstep)
        ! cpu_t3  = glsum(qrtime3,1)/(np*iqstep) ! linblad/(np*iqstep)
        ! cpu_t4  = glsum(qrtime4,1)/(np) ! fourier/(np)

        ! comm_t1 = glsum(qctime1,1)/(np*iqstep) ! total gs_op/(np*iqstep)
        ! comm_t3 = glsum(qctime3,1)/(np*iqstep) ! total get  /(np*iqstep)

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

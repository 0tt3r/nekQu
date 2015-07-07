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

      integer ii,jj,kk,numberOfQubits
      integer qubit(3,10) !qubit stores all of the handles
                          !qubit(1,ii) is the annihilation operator for qubit ii
                          !qubit(2,ii) is the creation operator for qubit ii
                          !qubit(3,ii) is the number operator for qubit ii
      
      debug         = 1
      
      numberOfQubits = 3
      call quantum_setup_initialize()
      useGates = .true.
      timeunit = 1
      timemax = 10
      nqstep  = 11
      !Create all of the ops
      do ii=1,numberOfQubits
         call quantum_create_op(2,qubit(1,ii),qubit(2,ii),qubit(3,ii))
      enddo

      call quantum_finalize_op_csr()

      call quantum_add_gate(1,numberOfQubits,5,3.0)
      call quantum_add_gate(2,numberOfQubits,1,5.0)
      call quantum_add_gate(3,numberOfQubits,1,7.0)


      call quantum_ham_finalize_csr()

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
      
      pulse = 0.0

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

c----------------------------------------------------------------------
c     hpm initialization
      subroutine hpm_setup
      include 'SIZE'
      include 'PARALLEL'
      include 'mpif.h'

#ifdef HPM
cc    call mpi_init(ierr)      ! this needs when hpm used independently 
      call hpm_init()
      call hpm_start("tst")
#endif      

      return
      end
c----------------------------------------------------------------------
c     hpm release
      subroutine hpm_release
      include 'SIZE'
      include 'PARALLEL'
      include 'mpif.h'

#ifdef HPM
      call hpm_stop("tst")
c     call hpm_print()         ! produce  per node
c     call hpm_print_flops()   ! per node
      call hpm_print_flops_agg()
cc    call mpi_finalize(ierr)  ! this needs when hpm used independently 
#endif      

      return
      end
c----------------------------------------------------------------------
      subroutine iniproc2
      include 'SIZE'
      include 'PARALLEL'
      include 'mpif.h'

      common /nekmpi/ nid_,np_,nekcomm,nekgroup,nekreal

      logical flag
      integer required, provided, ierr
      integer flag_gs_init

      call mpi_initialized(mpi_is_initialized, ierr) !  Initialize MPI
      if ( mpi_is_initialized .eq. 0 ) then
c        call MPI_INIT_THREAD(MPI_THREAD_MULTIPLE, provided, ierr)
         call mpi_init (ierr)
      endif

c     ! create communicator
      call init_nek_comm2
      np  = np_
      nid = nid_

c     if (nid.eq.0) write(6,10) MPI_THREAD_MULTIPLE,provided        
  10   format(' MPI_thread safety check: required=',i5,', provided=',i5)
      ! check upper tag size limit
      call mpi_attr_get(MPI_COMM_WORLD,MPI_TAG_UB,nval,flag,ierr)

      if (nval.lt.(10000+max(lp,lelg))) then
         if(nid.eq.0) write(6,*) 'ABORT: MPI_TAG_UB too small!'
         call exitt
      endif

      IF (NP.GT.LP) THEN
         WRITE(6,*)
     $   'ERROR: Code compiled for a max of',LP,' processors.'
         WRITE(6,*)
     $   'Recompile with LP =',NP,' or run with fewer processors.'
         WRITE(6,*)
     $   'Aborting in routine INIPROC.'
         call exitt
      endif
      ! set word size for REAL
      wdsize = 4
      eps    = 1.0e-12
      oneeps = 1.0+eps

      if (oneeps.ne.1.0) then
         wdsize=8
      else
         if(nid.eq.0)
     &     write(6,*) 'ABORT: single precision mode not supported!'
         call exitt
      endif
      nekreal = mpi_real
      if (wdsize.eq.8) nekreal = mpi_double_precision

      ifdblas = .false.
      if (wdsize.eq.8) ifdblas = .true.

      ! set word size for INTEGER
      ! HARDCODED since there is no secure way to detect an int overflow
      isize = 4

      ! set word size for LOGICAL
      lsize = 4

      ! set word size for CHARACTER
      csize = 1
c
      PID = 0
      NULLPID=0
      NODE0=0
      NODE= NID+1

      if (nid.eq.0) then
         write(6,*) 'Number of processors:',np
         WRITE(6,*) 'REAL    wdsize      :',WDSIZE
         WRITE(6,*) 'INTEGER wdsize      :',ISIZE
      endif
      call crystal_setup(cr_h,nekcomm,np)  ! set cr handle to new instance

      return
      end
c-----------------------------------------------------------------------
      subroutine create_comm(icomm)
      include 'mpif.h'

c     mpi_comm_world: communicator (handle)
c     itmp: subset of the group of  mpi_comm_world (handle)
c     icomm: new communicator (handle)
      call mpi_comm_group (mpi_comm_world,itmp,ierr)
      call mpi_comm_create (mpi_comm_world,itmp,icomm,ierr)
      call mpi_group_free (itmp,ierr)

      return
      end
c-----------------------------------------------------------------------
      subroutine init_nek_comm2
      include 'mpif.h'
      common /nekmpi/ nid_,np_,nekcomm,nekgroup,nekreal
C
      call create_comm(nekcomm) ! set up nekton specific communicator
c
      nid_  = mynode()
      np_   = numnodes()
c
      return
      end
c-----------------------------------------------------------------------
      subroutine init_nek_comm(nido,npo,wdsize)
      include 'mpif.h'
      common /nekmpi/ nid,np,nekcomm,nekgroup,nekreal
      integer wdsize
C
      call nek_comm            ! set up nekton specific communicator
c
      nid  = mynode()
      np   = numnodes()
      nido = nid
      npo  = np
c
      wdsize=4
      eps=1.0e-12
      oneeps = 1.0+eps
      if (oneeps.ne.1.0) wdsize=8
      nekreal = mpi_real
      if (wdsize.eq.8) nekreal = mpi_double_precision
c
      return
      end
c-----------------------------------------------------------------------
      subroutine gop( x, w, op, n)
c
c     Global vector commutative operation using spanning tree.
c
      INCLUDE 'CTIMER'
c
      include 'mpif.h'
      common /nekmpi/ nid,np,nekcomm,nekgroup,nekreal
c
      real x(n), w(n)
      character*3 op
c
      if (icalld.eq.0) then
        tgop =0.0d0
        ngop =0
        icalld=1
      endif
      ngop = ngop + 1
      etime1=dclock()
c
      if (op.eq.'+  ') then
c        call mpi_allreduce_(x,w,n,nekreal,mpi_sum ,nekcomm,ierr)
         call mpi_allreduce (x,w,n,nekreal,mpi_sum ,nekcomm,ierr)
      elseif (op.EQ.'M  ') then
c        call mpi_allreduce_(x,w,n,nekreal,mpi_max ,nekcomm,ierr)
         call mpi_allreduce (x,w,n,nekreal,mpi_max ,nekcomm,ierr)
      elseif (op.EQ.'m  ') then
c        call mpi_allreduce_(x,w,n,nekreal,mpi_min ,nekcomm,ierr)
         call mpi_allreduce (x,w,n,nekreal,mpi_min ,nekcomm,ierr)
      elseif (op.EQ.'*  ') then
c        call mpi_allreduce_(x,w,n,nekreal,mpi_prod,nekcomm,ierr)
         call mpi_allreduce (x,w,n,nekreal,mpi_prod,nekcomm,ierr)
      else
         write(6,*) nid,' OP ',op,' not supported.  ABORT in GOP.'
         call exitt
      endif
c
      !call copy(x,w,n)
c
      tgop =tgop +(dclock()-etime1)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine igop( x, w, op, n)
c
c     Global vector commutative operation using spanning tree.
c
      include 'mpif.h'
      common /nekmpi/ nid,np,nekcomm,nekgroup,nekreal
c
      integer x(n), w(n)
      character*3 op
c
      if (op.eq.'+  ') then
c       call mpi_allreduce_(x,w,n,mpi_integer,mpi_sum ,nekcomm,ierr)
        call mpi_allreduce (x,w,n,mpi_integer,mpi_sum ,nekcomm,ierr)
      elseif (op.EQ.'M  ') then
c       call mpi_allreduce_(x,w,n,mpi_integer,mpi_max ,nekcomm,ierr)
        call mpi_allreduce (x,w,n,mpi_integer,mpi_max ,nekcomm,ierr)
      elseif (op.EQ.'m  ') then
c       call mpi_allreduce_(x,w,n,mpi_integer,mpi_min ,nekcomm,ierr)
        call mpi_allreduce (x,w,n,mpi_integer,mpi_min ,nekcomm,ierr)
      elseif (op.EQ.'*  ') then
c       call mpi_allreduce_(x,w,n,mpi_integer,mpi_prod,nekcomm,ierr)
        call mpi_allreduce (x,w,n,mpi_integer,mpi_prod,nekcomm,ierr)
      else
        write(6,*) nid,' OP ',op,' not supported.  ABORT in GOP.'
        call exitt
      endif
c
      !call icopy(x,w,n)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine csend(mtype,buf,len,jnid,jpid)
      include 'mpif.h'
      common /nekmpi/ nid,np,nekcomm,nekgroup,nekreal
      real   buf(1)
c     call mpi_send_(buf,len,mpi_byte,jnid,mtype,nekcomm,ierr)
      call mpi_send (buf,len,mpi_byte,jnid,mtype,nekcomm,ierr)
      return
      END
c-----------------------------------------------------------------------
      subroutine crecv(   mtype, buf ,lenm )
      include 'mpif.h'
      common /nekmpi/ nid,np,nekcomm,nekgroup,nekreal
      integer status(mpi_status_size)
C
      real   buf(1)
      len = lenm
      jnid = mpi_any_source

      call mpi_recv (buf,len,mpi_byte
     $            ,jnid,mtype,nekcomm,status,ierr)
c
      if (len.gt.lenm) 
     $    write(6,*) nid,'long message in mpi_crecv:',len,lenm
c
      return
      END
c-----------------------------------------------------------------------
      integer function numnodes()
      include 'mpif.h'
      common /nekmpi/ nid,np,nekcomm,nekgroup,nekreal
c     call mpi_comm_size_(nekcomm, numnodes , ierr)
      call mpi_comm_size (nekcomm, numnodes , ierr)
      return
      end
c-----------------------------------------------------------------------
      integer function mynode()
      include 'mpif.h'
      common /nekmpi/ nid,np,nekcomm,nekgroup,nekreal
      integer myid
c     call mpi_comm_rank_(nekcomm, myid, ierr)
      call mpi_comm_rank (nekcomm, myid, ierr)
      mynode = myid
      return
      end
c-----------------------------------------------------------------------
      real*8 function dclock()
      include 'mpif.h'
c
      real   etime,q(2)
      save q
      data q /0.,0./
c
c     dclock=mpi_wtime_()
      dclock=mpi_wtime ()
c     dclock=etime(q)    ! for alpha
c
      return
      end
c-----------------------------------------------------------------------
      SUBROUTINE LBCAST(IFIF)
C
C  Broadcast logical variable to all processors.
C
      INCLUDE 'SIZE'
      INCLUDE 'PARALLEL'
      INCLUDE 'mpif.h'

      LOGICAL IFIF
C
      IF (NP.EQ.1) RETURN
C
      ITEM=0
      IF (IFIF) ITEM=1
      call bcast(item,isize)
      IFIF=.FALSE.
      IF (ITEM.EQ.1) IFIF=.TRUE.
      RETURN
      END

c-----------------------------------------------------------------------
      subroutine bcast(buf,len)
      include 'mpif.h'
      common /nekmpi/ nid,np,nekcomm,nekgroup,nekreal
      real   buf(len)
c     call mpi_bcast_(buf,len,mpi_byte,0,nekcomm,ierr)
      call mpi_bcast (buf,len,mpi_byte,0,nekcomm,ierr)
      return
      end
c-----------------------------------------------------------------------
      subroutine nek_comm
      include 'mpif.h'
      common /nekmpi/ nid,np,nekcomm,nekgroup,nekreal
c     call mpi_comm_group_(mpi_comm_world,nekgroup,ierr)
c     call mpi_comm_create_(mpi_comm_world,nekgroup,nekcomm,ierr)
c     call mpi_group_free_(nekgroup,ierr)
      call mpi_comm_group (mpi_comm_world,nekgroup,ierr)
      call mpi_comm_create (mpi_comm_world,nekgroup,nekcomm,ierr)
      call mpi_group_free (nekgroup,ierr)
      return
      end
c-----------------------------------------------------------------------
      function isend(msgtag,x,len,jnid,jpid)
c
c     Note: len in bytes
c
      integer x(len)
C
      include 'mpif.h'
      common /nekmpi/ nid,np,nekcomm,nekgroup,nekreal
C
c     call mpi_isend_(x,len,mpi_byte,jnid,msgtag
c    $       ,nekcomm,imsg,ierr)
      call mpi_isend (x,len,mpi_byte,jnid,msgtag
     $       ,nekcomm,imsg,ierr)
      isend = imsg
c     write(6,*) nid,' isend:',imsg,msgtag,len,jnid,(x(k),k=1,len/4)
c
      return
      end
c-----------------------------------------------------------------------
      function irecv(msgtag,x,len)
c
c     Note: len in bytes
c
      integer x(len)
C
      include 'mpif.h'
      common /nekmpi/ nid,np,nekcomm,nekgroup,nekreal
C
c     call mpi_irecv_(x,len,mpi_byte,mpi_any_source,msgtag
c    $       ,nekcomm,imsg,ierr)
      call mpi_irecv (x,len,mpi_byte,mpi_any_source,msgtag
     $       ,nekcomm,imsg,ierr)
      irecv = imsg
c     write(6,*) nid,' irecv:',imsg,msgtag,len
c
      return
      end
c-----------------------------------------------------------------------
      subroutine msgwait(imsg)
c
      include 'mpif.h'
      common /nekmpi/ nid,np,nekcomm,nekgroup,nekreal
      integer status(mpi_status_size)
c
c     write(6,*) nid,' msgwait:',imsg
c
c     call mpi_wait_(imsg,status,ierr)
      call mpi_wait (imsg,status,ierr)
c
      return
      end
c-----------------------------------------------------------------------
      subroutine close_unit(io)
      close (unit=io)
      return
      end
c-----------------------------------------------------------------------
      subroutine exitti(stringi,idata)
      character*1 stringi(132)
      character*1 stringo(132)
      character*11 s11
      include 'SIZE'
      include 'CTIMER'

      !call blank(stringo,132)
      !call chcopy(stringo,stringi,132)
      !len = indx1(stringo,'$',1)
      write(s11,11) idata
   11 format(1x,i10)
      !call chcopy(stringo(len),s11,11)

      if (nid.eq.0) write(6,1) (stringo(k),k=1,len+10)
    1 format('EXIT: ',132a1)

      call exitt

      return
      end
c-----------------------------------------------------------------------
      subroutine exitt
c     common /nekmpi/ mid,mp,nekcomm,nekgroup,nekreal
      include 'SIZE'
      include 'CTIMER'
      include 'mpif.h'

c     Communicate unhappiness to the other session

      call nekgsync()


#ifdef PAPI
      call nek_flops(papi_flops,papi_mflops)
#endif

      tstop  = dnekclock()
      ttotal = tstop-etimes
      nxyz   = nx1*ny1*nz1

      if (nid.eq.0) then
c        ifopen(unit=50,opened=ifopen)
c        if(ifopen) close(50)
         dtmp1 = 0
         dtmp2 = 0
         dtmp3 = 0
         if(istep.gt.0) then
           dgp   = nvtot
           dgp   = max(dgp,1.)
           dtmp1 = np*ttime/(dgp*max(istep,1))
           dtmp2 = ttime/max(istep,1)
c          dtmp3 = 1.*papi_flops/1e6
         endif
         write(6,*) ' '
         write(6,'(A)') 'call exitt: dying ...'
         write(6,*) ' '
         !call print_stack()
         write(6,*) ' '
         write(6,'(4(A,1p1e13.5,A,/))')
     &       'total elapsed time             : ',ttotal, ' sec'
     &      ,'total solver time incl. I/O    : ',ttime , ' sec'
     &      ,'time/timestep                  : ',dtmp2 , ' sec'
     &      ,'CPU seconds/timestep/gridpt    : ',dtmp1 , ' sec'
#ifdef PAPI
         write(6,'(2(A,1g13.5,/))')
     &       'Gflops                         : ',dtmp3/1000.
     &      ,'Gflops/s                       : ',papi_mflops/1000.
#endif
      endif

      !call flush_io
      call mpi_finalize (ierr)
 
      call exit(0)

      return
      end

c-----------------------------------------------------------------------
      integer function igl_running_sum(in)
      include 'mpif.h'
      common /nekmpi/ nid,np,nekcomm,nekgroup,nekreal
      integer status(mpi_status_size)
      integer x,w,r

      x = in  ! running sum
      w = in  ! working buff
      r = 0   ! recv buff

      call mpi_scan(x,r,1,mpi_integer,mpi_sum,nekcomm,ierr)
      igl_running_sum = r

      return
      end
c-----------------------------------------------------------------------
      subroutine platform_timer(ivb) ! mxm, ping-pong, and all_reduce timer

      include 'SIZE'


      call mxm_test_all(nid,ivb)  ! measure mxm times
c     call exitti('done mxm_test_all$',ivb)

      call comm_test(ivb)         ! measure message-passing and all-reduce times

      return
      end
c-----------------------------------------------------------------------
      subroutine comm_test(ivb) ! measure message-passing and all-reduce times
                                ! ivb = 0 --> minimal verbosity
                                ! ivb = 1 --> fully verbose
                                ! ivb = 2 --> smaller sample set(shorter)

      include 'SIZE'
      include 'PARALLEL'

      call gop_test(ivb)   ! added, Jan. 8, 2008

      !log_np=log2(np)
      np2 = 2**log_np
      if (np2.eq.np) call gp2_test(ivb)   ! added, Jan. 8, 2008

      io = 6
      n512 = min(512,np-1)

      do nodeb=1,n512
         call pingpong(alphas,betas,0,nodeb,.0005,io,ivb)
         if (nid.eq.0) write(6,1) nodeb,np,alphas,betas
    1    format(2i10,1p2e15.7,' alpha beta')
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine pingpong(alphas,betas,nodea,nodeb,dt,io,ivb)

      include 'SIZE'
      common /nekmpi/ mid,np,nekcomm,nekgroup,nekreal

      parameter  (lt=1)!lx1*ly1*lz1*lelt)
      parameter (mwd = 3*lt)
      common /scrns/ x(mwd),y(mwd)

      include 'mpif.h'
      integer status(mpi_status_size)

      character*10 fname

      if (nid.eq.nodea) then
         write(fname,3) np,nodeb
    3    format('t',i4.4,'.',i4.4)
         if (io.ne.6) open (unit=io,file=fname)
      endif

      call nekgsync
      call get_msg_vol(msg_vol,dt,nodea,nodeb) ! Est. msg vol for dt s

      nwds = 0
      if (nid.eq.nodea.and.ivb.gt.0) write(io,*)

      betas = 0  ! Reported inverse bandwidth
      count = 0

      do itest = 1,500

         nloop = msg_vol/(nwds+2)
         nloop = min(nloop,1000)
         nloop = max(nloop,1)

         len   = 8*nwds
     
         call ping_loop(t1,t0,len,nloop,nodea,nodeb,nid,x,y,x,y)

         if (nid.eq.nodea) then
            tmsg = (t1-t0)/(2*nloop)   ! 2*nloop--> Double Buffer
            tmsg = tmsg / 2.           ! one-way cost = 1/2 round-trip
            tpwd = tmsg                ! time-per-word
            if (nwds.gt.0) tpwd = tmsg/nwds
            if (ivb.gt.0) write(io,1) nodeb,np,nloop,nwds,tmsg,tpwd
    1       format(3i6,i12,1p2e16.8,' pg')

            if (nwds.eq.1) then
               alphas = tmsg
            elseif (nwds.gt.10000) then   ! "average" beta
               betas = (betas*count + tpwd)/(count+1)
               count = count + 1
            endif
         endif

         if (ivb.eq.2) then
            nwds = (nwds+1)*1.25
         else
            nwds = (nwds+1)*1.016
         endif
         if (nwds.gt.mwd) then
c        if (nwds.gt.1024) then
            if (nid.eq.nodea.and.io.ne.6) close(unit=io)
            call nekgsync
            return
         endif

      enddo

      if (nid.eq.nodea.and.io.ne.6) close(unit=io)
      call nekgsync

      return
      end
c-----------------------------------------------------------------------
      subroutine pingpongo(alphas,betas,nodea,nodeb,dt,io,ivb)

      include 'SIZE'
      common /nekmpi/ mid,np,nekcomm,nekgroup,nekreal

      parameter  (lt=1)!lx1*ly1*lz1*lelt)
      parameter (mwd = 3*lt)
      common /scrns_r/ x(mwd),y(mwd)
      real             x,y
      common /scrns_i/ ix(mwd),iy(mwd)
      integer          ix,iy
      include 'mpif.h'
      integer status(mpi_status_size)

      real*8  dnekclock
      character*10 fname

      if (nid.eq.nodea) then
         write(fname,3) np,nodeb
    3    format('t',i4.4,'.',i4.4)
         if (io.ne.6) open (unit=io,file=fname)
      endif

      call nekgsync
      call get_msg_vol(msg_vol,dt,nodea,nodeb) ! Est. msg vol for dt s

      nwds = 0
      if (nid.eq.nodea.and.ivb.gt.0) write(io,*)

      betas = 0  ! Reported inverse bandwidth
      count = 0

      do itest = 1,500
         call nekgsync
         nloop = msg_vol/(nwds+2)
         nloop = min(nloop,1000)
         nloop = max(nloop,1)

         len   = 8*nwds
         jnid = mpi_any_source

         if (nid.eq.nodea) then

            msg  = irecv(itest,iy,1)
            call csend(itest,x,1,nodeb,0)   ! Initiate send, to synch.
            call msgwait(msg)

            t0 = mpi_wtime ()
            do i=1,nloop
               call mpi_irecv(iy,len,mpi_byte,mpi_any_source,i
     $                        ,nekcomm,msg,ierr)
               call mpi_send (x,len,mpi_byte,nodeb,i,nekcomm,ierr)
               call mpi_wait (msg,status,ierr)
            enddo
            t1 = mpi_wtime ()
            tmsg = (t1-t0)/nloop
            tmsg = tmsg / 2.       ! Round-trip message time = twice one-way
            tpwd = tmsg
            if (nwds.gt.0) tpwd = tmsg/nwds
            if (ivb.gt.0) write(io,1) nodeb,np,nloop,nwds,tmsg,tpwd
    1       format(3i6,i12,1p2e16.8,' pg')

            if (nwds.eq.1) then
               alphas = tmsg
            elseif (nwds.gt.10000) then
               betas = (betas*count + tpwd)/(count+1)
               count = count + 1
            endif

         elseif (nid.eq.nodeb) then

            call crecv(itest,y,1)           ! Initiate send, to synch.
            call csend(itest,x,1,nodea,0)

            t0 = dnekclock()
            do i=1,nloop
               call mpi_recv (y,len,mpi_byte
     $               ,jnid,i,nekcomm,status,ierr)
               call mpi_send (x,len,mpi_byte,nodea,i,nekcomm,ierr)
            enddo
            t1 = dnekclock()
            tmsg = (t1-t0)/nloop

         endif

         nwds = (nwds+1)*1.016
         if (nwds.gt.mwd) then
            if (nid.eq.nodea.and.io.ne.6) close(unit=io)
            call nekgsync
            return
         endif

      enddo

      if (nid.eq.nodea.and.io.ne.6) close(unit=io)
      call nekgsync

      return
      end
c-----------------------------------------------------------------------
      subroutine get_msg_vol(msg_vol,dt,nodea,nodeb)
      include 'SIZE'
      common /nekmpi/ mid,np,nekcomm,nekgroup,nekreal
      parameter (lt=1)!lx1*ly1*lz1*lelt)
      common /scrns_r/  x(3*lt), y(3*lt)
      common /scrns_i/ ix(3*lt),iy(3*lt)
      real*8 dnekclock
      real   t0(1),t1(1),tmsg(1),tpwd
!
!     Est. msg vol for dt s
!
      msg_vol = 1000

      nwds  = min(1000,lt)
      nloop = 50
 
      tmsg(1) = 0.
      call gop(tmsg,t1,'+  ',1)

      len = 8*nwds
      if (nid.eq.nodea) then

         msg  = irecv(1,iy,1)
         call csend(1,x,1,nodeb,0)   ! Initiate send, to synch.
         call msgwait(msg)

         t0(1) = dnekclock()
         do i=1,nloop
            msg  = irecv(i,iy,len)
            call csend(i,x,len,nodeb,0)
            call msgwait(msg)
         enddo
         t1(1)   = dnekclock()
         tmsg(1) = (t1(1)-t0(1))/nloop
         tpwd = tmsg(1)/nwds

      elseif (nid.eq.nodeb) then

         call crecv(1,y,1)           ! Initiate send, to synch.
         call csend(1,x,1,nodea,0)

         t0(1) = dnekclock()
         do i=1,nloop
            call crecv(i,y,len)
            call csend(i,x,len,nodea,0)
         enddo
         t1(1)   = dnekclock()
         tmsg(1) = (t1(1)-t0(1))/nloop
         tmsg(1) = 0.
      endif

      call gop(tmsg,t1,'+  ',1)
      msg_vol = nwds*(dt/tmsg(1))
c     if (nid.eq.nodea) write(6,*) nid,msg_vol,nwds,dt,tmsg,' msgvol'

      return
      end
c-----------------------------------------------------------------------
      subroutine gop_test(ivb)
      include 'SIZE'
      common /nekmpi/ mid,np,nekcomm,nekgroup,nekreal
      include 'mpif.h'
      integer status(mpi_status_size)

      parameter  (lt=1)!lx1*ly1*lz1*lelt)
      parameter (mwd = 3*lt)
      common /scrns/ x(mwd),y(mwd)
      common /scruz/ times(2,500)

      !call rzero(x,mwd)

      nwds = 1
      do itest = 1,500
         call nekgsync

         t0 = mpi_wtime ()
         call gop(x,y,'+  ',nwds)
         call gop(x,y,'+  ',nwds)
         call gop(x,y,'+  ',nwds)
         call gop(x,y,'+  ',nwds)
         t1 = mpi_wtime ()

         tmsg = (t1-t0)/4 ! four calls
         tpwd = tmsg
         if (nwds.gt.0) tpwd = tmsg/nwds
         times(1,itest) = tmsg
         times(2,itest) = tpwd

         nwds = (nwds+1)*1.016
         if (nwds.gt.mwd) goto 101
      enddo
  101 continue


      if (nid.eq.0) then
         nwds = 1
         do itest=1,500
            if (ivb.gt.0.or.itest.eq.1) 
     $         write(6,1) np,nwds,(times(k,itest),k=1,2)
    1       format(i9,i12,1p2e16.8,' gop')
            nwds = (nwds+1)*1.016
            if (nwds.gt.mwd) goto 102
         enddo
  102    continue
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine gp2_test(ivb)

      include 'SIZE'
      include 'mpif.h'

      common /nekmpi/ mid,np,nekcomm,nekgroup,nekreal
      integer status(mpi_status_size)

      parameter  (lt=1)!lx1*ly1*lz1*lelt)
      parameter (mwd = 3*lt)
      common /scrns/ x(mwd),y(mwd)
      common /scruz/ times(2,500)

      !call rzero(x,mwd)

      nwds = 1
      do itest = 1,500
         call gp2(x,y,'+  ',1,nid,np)

         t0 = mpi_wtime ()
         call gp2(x,y,'+  ',nwds,nid,np)
         call gp2(x,y,'+  ',nwds,nid,np)
         call gp2(x,y,'+  ',nwds,nid,np)
         call gp2(x,y,'+  ',nwds,nid,np)
         t1 = mpi_wtime ()

         tmsg = (t1-t0)/4 ! four calls
         tpwd = tmsg
         if (nwds.gt.0) tpwd = tmsg/nwds
         times(1,itest) = tmsg
         times(2,itest) = tpwd

         nwds = (nwds+1)*1.016
         if (nwds.gt.mwd) goto 101
      enddo
  101 continue


      if (nid.eq.0) then
         nwds = 1
         do itest=1,500
            if (ivb.gt.0.or.itest.eq.1) 
     $         write(6,1) np,nwds,(times(k,itest),k=1,2)
    1       format(i9,i12,1p2e16.8,' gp2')
            nwds = (nwds+1)*1.016
            if (nwds.gt.mwd) goto 102
         enddo
  102    continue
      endif

      return
      end
c-----------------------------------------------------------------------
      integer function xor(m,n)
c
c  If NOT running on a parallel processor, it is sufficient to
c  have this routine return a value of XOR=1.
c
c  Pick one of the following:
c
c  UNIX 4.2, f77:
       XOR = OR(M,N)-AND(M,N)
c
c  Intel FTN286:
c     XOR = M.NEQV.N
c
c  Ryan-McFarland Fortran
C      XOR = IEOR(M,N)
c
c     XOR = 0
c     IF(M.EQ.1 .OR.  N.EQ.1) XOR=1
c     IF(M.EQ.0 .AND. N.EQ.0) XOR=0
c     IF(M.EQ.1 .AND. N.EQ.1) XOR=0
c     IF(M.GT.1 .OR.N.GT.1 .OR.M.LT.0.OR.N.LT.0) THEN
c        PRINT*,'ERROR IN XOR'
c        STOP
c     ENDIF
C
      return
      end
c-----------------------------------------------------------------------
      subroutine gp2( x, w, op, n, nid, np)
c
c     Global vector commutative operation using spanning tree.
c
c     Std. fan-in/fan-out

      real x(n), w(n)
      character*3 op

      integer bit, bytes, cnt, diff, spsize, i, 
     *   parent, troot, xor, root, lnp, log2
      logical ifgot

      integer type
      save    type
      data    type  /998/
      integer typer    

      type  = type+100
      if (type.gt.9992) type=type-998
      typer = type-1
      bytes = 8*n

      root    = 0
      troot   = max0((nid/np)*np, root)
      diff    = xor(nid,troot)
      nullpid = 0

c     Accumulate contributions from children, if any
      level2=1
    5 continue
         level=level2
         level2=level+level
         if (mod(nid,level2).ne.0) goto 20
            call crecv(type,w,bytes)
            if (op.eq.'+  ') then
               do i=1,n
                  x(i) = x(i) + w(i)
               enddo
            elseif (op.eq.'*  ') then
               do i=1,n
                  x(i) = x(i) * w(i)
               enddo
            elseif (op.eq.'M  ') then
               do i=1,n
                  x(i) = max(x(i),w(i))
               enddo
            elseif (op.eq.'m  ') then
               do i=1,n
                  x(i) = min(x(i),w(i))
               enddo
            endif
         if (level2.lt.np) goto 5

c     Pass result back to parent
   20 parent = nid-level
      if (nid .ne. 0) call csend(type,x,bytes,parent,nullpid)

c     Await final answer from node 0 via log_2 fan out
      level=np/2
      ifgot=.false.
      if (nid.eq.root) ifgot=.true.

      !lnp = log2(np)
      do i=1,lnp
        if (ifgot) then
           jnid=nid+level
           call csend(typer,x,bytes,jnid,nullpid)
        elseif (mod(nid,level).eq.0) then
           call crecv(typer,x,bytes)
           ifgot=.true.
        endif
        level=level/2
      enddo

      return
      end
c-----------------------------------------------------------------------
      subroutine ping_loop1(t1,t0,len,nloop,nodea,nodeb,nid,x,y)

      common /nekmpi/ mid,np,nekcomm,nekgroup,nekreal

      real x(1),y(1)

      include 'mpif.h'
      integer status(mpi_status_size)

      i=0
      if (nid.eq.nodea) then
         call nekgsync
         call mpi_irecv(y,len,mpi_byte,nodeb,i,nekcomm,msg,ierr)    ! 1b
         call mpi_send (x,len,mpi_byte,nodeb,i,nekcomm,ierr)        ! 1a
c        call mpi_rsend(x,len,mpi_byte,nodeb,i,nekcomm,ierr)        ! 1a
         call msgwait(msg)                                          ! 1b

         t0 = mpi_wtime ()
         do i=1,nloop
            call mpi_irecv(y,len,mpi_byte,nodeb,i,nekcomm,msg,ierr) ! 2b
            call mpi_send (x,len,mpi_byte,nodeb,i,nekcomm,ierr)     ! 2a
c           call mpi_rsend(x,len,mpi_byte,nodeb,i,nekcomm,ierr)     ! 2a
            call mpi_wait (msg,status,ierr)                         ! 2b
         enddo
         t1 = mpi_wtime ()

      elseif (nid.eq.nodeb) then

         call mpi_irecv(y,len,mpi_byte,nodea,i,nekcomm,msg,ierr)    ! 1a
         call nekgsync
         call mpi_wait (msg,status,ierr)                            ! 1a

         j=i
         do i=1,nloop
            call mpi_irecv(y,len,mpi_byte,nodea,i,nekcomm,msg,ierr) ! 2a
c           call mpi_rsend(x,len,mpi_byte,nodea,j,nekcomm,ierr)     ! 1b
            call mpi_send (x,len,mpi_byte,nodea,j,nekcomm,ierr)     ! 1b
            call mpi_wait (msg,status,ierr)                         ! 2a
            j=i
         enddo
c        call mpi_rsend(x,len,mpi_byte,nodea,j,nekcomm,ierr)        ! nb
         call mpi_send (x,len,mpi_byte,nodea,j,nekcomm,ierr)        ! nb

      else
         call nekgsync
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine ping_loop2(t1,t0,len,nloop,nodea,nodeb,nid,x,y)

      common /nekmpi/ mid,np,nekcomm,nekgroup,nekreal

      real x(1),y(1)

      include 'mpif.h'
      integer status(mpi_status_size)

      i=0
      if (nid.eq.nodea) then
         call nekgsync
         call mpi_irecv(y,len,mpi_byte,nodeb,i,nekcomm,msg,ierr)    ! 1b
         call mpi_send (x,len,mpi_byte,nodeb,i,nekcomm,ierr)        ! 1a
         call msgwait(msg)                                          ! 1b

         t0 = mpi_wtime ()
         do i=1,nloop
            call mpi_send (x,len,mpi_byte,nodeb,i,nekcomm,ierr)     ! 2a
            call mpi_irecv(y,len,mpi_byte,nodeb,i,nekcomm,msg,ierr) ! 2b
            call mpi_wait (msg,status,ierr)                         ! 2b
         enddo
         t1 = mpi_wtime ()

      elseif (nid.eq.nodeb) then

         call mpi_irecv(y,len,mpi_byte,nodea,i,nekcomm,msg,ierr)    ! 1a
         call nekgsync
         call mpi_wait (msg,status,ierr)                            ! 1a

         j=i
         do i=1,nloop
            call mpi_send (x,len,mpi_byte,nodea,j,nekcomm,ierr)     ! 1b
            call mpi_irecv(y,len,mpi_byte,nodea,i,nekcomm,msg,ierr) ! 2a
            call mpi_wait (msg,status,ierr)                         ! 2a
            j=i
         enddo
         call mpi_send (x,len,mpi_byte,nodea,j,nekcomm,ierr)        ! nb

      else
         call nekgsync
      endif

      return
      end
c-----------------------------------------------------------------------
      subroutine ping_loop(t1,t0,len,nloop,nodea,nodeb,nid,x1,y1,x2,y2)
c     Double Buffer : does 2*nloop timings

      common /nekmpi/ mid,np,nekcomm,nekgroup,nekreal

      real x1(1),y1(1),x2(1),y2(1)

      include 'mpif.h'
      integer status(mpi_status_size)

      itag=1
      if (nid.eq.nodea) then
         call mpi_irecv(y1,len,mpi_byte,nodeb,itag,nekcomm,msg1,ierr)   ! 1b 
         call nekgsync


         t0 = mpi_wtime ()
         do i=1,nloop
            call mpi_send (x1,len,mpi_byte,nodeb,itag,nekcomm,ierr)     ! 1a 
            call mpi_irecv(y2,len,mpi_byte,nodeb,itag,nekcomm,msg2,ierr)! 2b 
            call mpi_wait (msg1,status,ierr)                            ! 1b
            call mpi_send (x2,len,mpi_byte,nodeb,itag,nekcomm,ierr)     ! 2a 
            call mpi_irecv(y1,len,mpi_byte,nodeb,itag,nekcomm,msg1,ierr)! 3b 
            call mpi_wait (msg2,status,ierr)                            ! 2b
         enddo
         t1 = mpi_wtime ()
         call mpi_send (x1,len,mpi_byte,nodeb,itag,nekcomm,ierr)        ! nb
         call mpi_wait (msg1,status,ierr)                              ! nb

      elseif (nid.eq.nodeb) then

         call mpi_irecv(y1,len,mpi_byte,nodea,itag,nekcomm,msg1,ierr)   ! nb 
         call nekgsync


         do i=1,nloop
            call mpi_wait (msg1,status,ierr)                            ! 1a
            call mpi_send (x1,len,mpi_byte,nodea,itag,nekcomm,ierr)     ! 1b
            call mpi_irecv(y2,len,mpi_byte,nodea,itag,nekcomm,msg2,ierr)! 2a
            call mpi_wait (msg2,status,ierr)                            ! 2a 
            call mpi_send (x2,len,mpi_byte,nodea,itag,nekcomm,ierr)     ! 2b
            call mpi_irecv(y1,len,mpi_byte,nodea,itag,nekcomm,msg1,ierr)! 3a
         enddo
         call mpi_wait (msg1,status,ierr)                            ! 2a 
         call mpi_send (x1,len,mpi_byte,nodea,itag,nekcomm,ierr)        ! nb

      else
         call nekgsync
      endif

      return
      end

c-----------------------------------------------------------------------
      subroutine nekgsync()

      include 'mpif.h'
      common /nekmpi/ nid,np,nekcomm,nekgroup,nekreal

      call mpi_barrier(nekcomm,ierr)

      return
      end

c-----------------------------------------------------------------------
      real*8 function dnekclock()
      include 'mpif.h'
c
      dnekclock=mpi_wtime()
c
      return
      end
c-----------------------------------------------------------------------
      real*8 function dnekclock_sync()
      include 'mpif.h'
c
      call nekgsync()
      dnekclock_sync=mpi_wtime()
c
      return
      end

c-----------------------------------------------------------------------
      subroutine nek_flops(flops,mflops)
      real*4 rtime,ptime,mflops
      integer*8 flops

      call getflops_papi(flops,mflops)

      return
      end

c-----------------------------------------------------------------------
      subroutine getflops_papi(flops,mflops)
#ifdef PAPI
      include 'f77papi.h'
      real*4 rtime,ptime,mflops
      integer*8 flops

      call papif_flops(rtime,ptime,flops,mflops,ierr)
      if(ierr.gt.0) then
        flops = -1
        mflops = -1
      endif
#endif

      return
      end
c-----------------------------------------------------------------------
      integer*8 function i8gl_running_sum(in)
c
      include 'mpif.h'
      common /nekmpi/ nid,np,nekcomm,nekgroup,nekreal
      integer status(mpi_status_size)
      integer*8 x,r

      x = in  ! running sum
      r = 0   ! recv buff

      call mpi_scan(x,r,1,mpi_integer8,mpi_sum,nekcomm,ierr)
      i8gl_running_sum = r

      return
      end
c-----------------------------------------------------------------------


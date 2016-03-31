

      subroutine quantum3_debug8cores()
      implicit none
      include 'SIZE'
      include 'QUANT'
      integer ii,jj,kk
      complex  sig,tmpc1

      call cem_quantum3_initialize      
      call cem_quantum3_rk_storage
      call cem_quantum3_gs

      do ii=1,nEh
         en_t(ii) = (0.0,0.0)
         do jj=1,numObservables
            obs_freq(jj,ii) = (0.0,0.0)
         enddo
      enddo

c$$$      do jj=1,nstate/np
c$$$         do ii=1,nstate
c$$$            rho_r(ii,jj) = ii+nstate*(jj+c_offset-1)
c$$$            rho_i(ii,jj) = -ii-nstate*(jj+c_offset-1)
c$$$            rho(ii,jj)   = rho_r(ii,jj) + (0.0,1.0)*rho_i(ii,jj)
c$$$            drho(ii,jj)  = (0.0,0.0)
c$$$         enddo
c$$$      enddo
      iqstep = 1
      do while(iqstep.lt.20)
      do kk=1,5
      call cem_quantum3_hamiltonian()

      if(nid.eq.0) then
         write(10,*)'realham',iqstep
         do ii=1,nstate
            write(10,'(100g15.5)') (real(drho(ii,jj)),jj=1,nstate/8)
         enddo
         write(10,*)'imagham',iqstep
         do ii=1,nstate
            write(10,'(100g15.5)') (imag(drho(ii,jj)),jj=1,nstate/8)
         enddo

         if(np.eq.1) then
            write(11,*)'realham',iqstep
            do ii=1,nstate
               write(11,'(100g15.5)') (real(drho(ii,jj)),jj=nstate/8+1
     $              ,2*nstate/8)
            enddo
            write(11,*)'imagham',iqstep
            do ii=1,nstate
               write(11,'(100g15.5)') (imag(drho(ii,jj)),jj=nstate/8+1,
     $              2*nstate/8)
            enddo
            write(12,*)'realham',iqstep
            do ii=1,nstate
               write(12,'(100g15.5)') (real(drho(ii,jj)),jj=2*nstate/8+1
     $              ,3*nstate/8)
            enddo
            write(12,*)'imagham',iqstep
            do ii=1,nstate
               write(12,'(100g15.5)') (imag(drho(ii,jj)),jj=2*nstate/8+1
     $              ,3*nstate/8)
            enddo

            write(13,*)'realham',iqstep
            do ii=1,nstate
               write(13,'(100g15.5)') (real(drho(ii,jj)),jj=3*nstate/8+1
     $              ,4*nstate/8)
            enddo
            write(13,*)'imagham',iqstep
            do ii=1,nstate
               write(13,'(100g15.5)') (imag(drho(ii,jj)),jj=3*nstate/8+1
     $              ,4*nstate/8)
            enddo

            write(14,*)'realham',iqstep
            do ii=1,nstate
               write(14,'(100g15.5)') (real(drho(ii,jj)),jj=4*nstate/8+1
     $              ,5*nstate/8)
            enddo
            write(14,*)'imagham',iqstep
            do ii=1,nstate
               write(14,'(100g15.5)') (imag(drho(ii,jj)),jj=4*nstate/8+1
     $              ,5*nstate/8)
            enddo

            write(15,*)'realham',iqstep
            do ii=1,nstate
               write(15,'(100g15.5)') (real(drho(ii,jj)),jj=5*nstate/8+1
     $              ,6*nstate/8)
            enddo
            write(15,*)'imagham',iqstep
            do ii=1,nstate
               write(15,'(100g15.5)') (imag(drho(ii,jj)),jj=5*nstate/8+1
     $              ,6*nstate/8)
            enddo

            write(16,*)'realham',iqstep
            do ii=1,nstate
               write(16,'(100g15.5)') (real(drho(ii,jj)),jj=6*nstate/8+1
     $              ,7*nstate/8)
            enddo
            write(16,*)'imagham',iqstep
            do ii=1,nstate
               write(16,'(100g15.5)') (imag(drho(ii,jj)),jj=6*nstate/8+1
     $              ,7*nstate/8)
            enddo

            write(17,*)'realham',iqstep
            do ii=1,nstate
               write(17,'(100g15.5)') (real(drho(ii,jj)),jj=7*nstate/8+1
     $              ,8*nstate/8)
            enddo
            write(17,*)'imagham',iqstep
            do ii=1,nstate
               write(17,'(100g15.5)') (imag(drho(ii,jj)),jj=7*nstate/8+1
     $              ,8*nstate/8)
            enddo
         endif
      endif
      if(nid.eq.1) then
         write(11,*)'realham',iqstep
         do ii=1,nstate
            write(11,'(100g15.5)') (real(drho(ii,jj)),jj=1,nstate/np)
         enddo
         write(11,*)'imagham',iqstep
         do ii=1,nstate
            write(11,'(100g15.5)') (imag(drho(ii,jj)),jj=1,nstate/np)
         enddo
      endif
      if(nid.eq.2) then
         write(12,*)'realham',iqstep
         do ii=1,nstate
            write(12,'(100g15.5)') (real(drho(ii,jj)),jj=1,nstate/np)
         enddo
         write(12,*)'imagham',iqstep
         do ii=1,nstate
            write(12,'(100g15.5)') (imag(drho(ii,jj)),jj=1,nstate/np)
         enddo
      endif

      if(nid.eq.3) then
         write(13,*)'realham',iqstep
         do ii=1,nstate
            write(13,'(100g15.5)') (real(drho(ii,jj)),jj=1,nstate/np)
         enddo
         write(13,*)'imagham',iqstep
         do ii=1,nstate
            write(13,'(100g15.5)') (imag(drho(ii,jj)),jj=1,nstate/np)
         enddo
      endif

      if(nid.eq.4) then
         write(14,*)'realham',iqstep
         do ii=1,nstate
            write(14,'(100g15.5)') (real(drho(ii,jj)),jj=1,nstate/np)
         enddo
         write(14,*)'imagham',iqstep
         do ii=1,nstate
            write(14,'(100g15.5)') (imag(drho(ii,jj)),jj=1,nstate/np)
         enddo
      endif

      if(nid.eq.5) then
         write(15,*)'realham',iqstep
         do ii=1,nstate
            write(15,'(100g15.5)') (real(drho(ii,jj)),jj=1,nstate/np)
         enddo
         write(15,*)'imagham',iqstep
         do ii=1,nstate
            write(15,'(100g15.5)') (imag(drho(ii,jj)),jj=1,nstate/np)
         enddo
      endif

      if(nid.eq.6) then
         write(16,*)'realham',iqstep
         do ii=1,nstate
            write(16,'(100g15.5)') (real(drho(ii,jj)),jj=1,nstate/np)
         enddo
         write(16,*)'imagham',iqstep
         do ii=1,nstate
            write(16,'(100g15.5)') (imag(drho(ii,jj)),jj=1,nstate/np)
         enddo
      endif

      if(nid.eq.7) then
         write(17,*)'realham',iqstep
         do ii=1,nstate
            write(17,'(100g15.5)') (real(drho(ii,jj)),jj=1,nstate/np)
         enddo
         write(17,*)'imagham',iqstep
         do ii=1,nstate
            write(17,'(100g15.5)') (imag(drho(ii,jj)),jj=1,nstate/np)
         enddo
      endif

c$$$      do jj=1,nstate/np
c$$$         do ii=1,nstate
c$$$            rho_r(ii,jj) = real(drho(ii,jj))
c$$$            rho_i(ii,jj) = imag(drho(ii,jj))
c$$$            drho(ii,jj)  = (0.0,0.0)
c$$$         enddo
c$$$      enddo

      call quantum_lin_mpi_get()

      if(nid.eq.0) then
         write(10,*)'reallin',iqstep
         do ii=1,nstate
            write(10,'(100g15.5)') (real(drho(ii,jj)),jj=1,nstate/8)
         enddo
         write(10,*)'imag',iqstep
         do ii=1,nstate
            write(10,'(100g15.5)') (imag(drho(ii,jj)),jj=1,nstate/8)
         enddo

         if(np.eq.1) then
            write(11,*)'reallin',iqstep
            do ii=1,nstate
               write(11,'(100g15.5)') (real(drho(ii,jj)),jj=nstate/8+1
     $              ,2*nstate/8)
            enddo
            write(11,*)'imag',iqstep
            do ii=1,nstate
               write(11,'(100g15.5)') (imag(drho(ii,jj)),jj=nstate/8+1,
     $              2*nstate/8)
            enddo

            write(12,*)'reallin',iqstep
            do ii=1,nstate
               write(12,'(100g15.5)') (real(drho(ii,jj)),jj=2*nstate/8+1
     $              ,3*nstate/8)
            enddo
            write(12,*)'imag',iqstep
            do ii=1,nstate
               write(12,'(100g15.5)') (imag(drho(ii,jj)),jj=2*nstate/8+1
     $              ,3*nstate/8)
            enddo

            write(13,*)'reallin',iqstep
            do ii=1,nstate
               write(13,'(100g15.5)') (real(drho(ii,jj)),jj=3*nstate/8+1
     $              ,4*nstate/8)
            enddo
            write(13,*)'imag',iqstep
            do ii=1,nstate
               write(13,'(100g15.5)') (imag(drho(ii,jj)),jj=3*nstate/8+1
     $              ,4*nstate/8)
            enddo

            write(14,*)'reallin',iqstep
            do ii=1,nstate
               write(14,'(100g15.5)') (real(drho(ii,jj)),jj=4*nstate/8+1
     $              ,5*nstate/8)
            enddo
            write(14,*)'imag',iqstep
            do ii=1,nstate
               write(14,'(100g15.5)') (imag(drho(ii,jj)),jj=4*nstate/8+1
     $              ,5*nstate/8)
            enddo

            write(15,*)'reallin',iqstep
            do ii=1,nstate
               write(15,'(100g15.5)') (real(drho(ii,jj)),jj=5*nstate/8+1
     $              ,6*nstate/8)
            enddo
            write(15,*)'imag',iqstep
            do ii=1,nstate
               write(15,'(100g15.5)') (imag(drho(ii,jj)),jj=5*nstate/8+1
     $              ,6*nstate/8)
            enddo

            write(16,*)'reallin',iqstep
            do ii=1,nstate
               write(16,'(100g15.5)') (real(drho(ii,jj)),jj=6*nstate/8+1
     $              ,7*nstate/8)
            enddo
            write(16,*)'imag',iqstep
            do ii=1,nstate
               write(16,'(100g15.5)') (imag(drho(ii,jj)),jj=6*nstate/8+1
     $              ,7*nstate/8)
            enddo

            write(17,*)'reallin',iqstep
            do ii=1,nstate
               write(17,'(100g15.5)') (real(drho(ii,jj)),jj=7*nstate/8+1
     $              ,8*nstate/8)
            enddo
            write(17,*)'imag',iqstep
            do ii=1,nstate
               write(17,'(100g15.5)') (imag(drho(ii,jj)),jj=7*nstate/8+1
     $              ,8*nstate/8)
            enddo
         endif
      endif
      if(nid.eq.1) then
         write(11,*)'reallin',iqstep
         do ii=1,nstate
            write(11,'(100g15.5)') (real(drho(ii,jj)),jj=1,nstate/np)
         enddo
         write(11,*)'imag',iqstep
         do ii=1,nstate
            write(11,'(100g15.5)') (imag(drho(ii,jj)),jj=1,nstate/np)
         enddo
      endif
      if(nid.eq.2) then
         write(12,*)'reallin',iqstep
         do ii=1,nstate
            write(12,'(100g15.5)') (real(drho(ii,jj)),jj=1,nstate/np)
         enddo
         write(12,*)'imag',iqstep
         do ii=1,nstate
            write(12,'(100g15.5)') (imag(drho(ii,jj)),jj=1,nstate/np)
         enddo
      endif

      if(nid.eq.3) then
         write(13,*)'reallin',iqstep
         do ii=1,nstate
            write(13,'(100g15.5)') (real(drho(ii,jj)),jj=1,nstate/np)
         enddo
         write(13,*)'imag',iqstep
         do ii=1,nstate
            write(13,'(100g15.5)') (imag(drho(ii,jj)),jj=1,nstate/np)
         enddo
      endif

      if(nid.eq.4) then
         write(14,*)'reallin',iqstep
         do ii=1,nstate
            write(14,'(100g15.5)') (real(drho(ii,jj)),jj=1,nstate/np)
         enddo
         write(14,*)'imag',iqstep
         do ii=1,nstate
            write(14,'(100g15.5)') (imag(drho(ii,jj)),jj=1,nstate/np)
         enddo
      endif

      if(nid.eq.5) then
         write(15,*)'reallin',iqstep
         do ii=1,nstate
            write(15,'(100g15.5)') (real(drho(ii,jj)),jj=1,nstate/np)
         enddo
         write(15,*)'imag',iqstep
         do ii=1,nstate
            write(15,'(100g15.5)') (imag(drho(ii,jj)),jj=1,nstate/np)
         enddo
      endif

      if(nid.eq.6) then
         write(16,*)'reallin',iqstep
         do ii=1,nstate
            write(16,'(100g15.5)') (real(drho(ii,jj)),jj=1,nstate/np)
         enddo
         write(16,*)'imag',iqstep
         do ii=1,nstate
            write(16,'(100g15.5)') (imag(drho(ii,jj)),jj=1,nstate/np)
         enddo
      endif

      if(nid.eq.7) then
         write(17,*)'reallin',iqstep
         do ii=1,nstate
            write(17,'(100g15.5)') (real(drho(ii,jj)),jj=1,nstate/np)
         enddo
         write(17,*)'imag',iqstep
         do ii=1,nstate
            write(17,'(100g15.5)') (imag(drho(ii,jj)),jj=1,nstate/np)
         enddo
      endif

      call rk_quantum3_ab(kk)

      if(nid.eq.0) then
         write(10,*)'realab',iqstep
         do ii=1,nstate
            write(10,'(100g15.5)') (rho_r(ii,jj),jj=1,nstate/8)
         enddo
         write(10,*)'imagab',iqstep,kk
         do ii=1,nstate
            write(10,'(100g15.5)') (rho_i(ii,jj),jj=1,nstate/8)
         enddo

         if(np.eq.1) then
            write(11,*)'realab',iqstep
            do ii=1,nstate
               write(11,'(100g15.5)') (rho_r(ii,jj),jj=nstate/8+1
     $              ,2*nstate/8)
            enddo
            write(11,*)'imagab',iqstep,kk
            do ii=1,nstate
               write(11,'(100g15.5)') (rho_i(ii,jj),jj=nstate/8+1,
     $              2*nstate/8)
            enddo

            write(12,*)'realab',iqstep
            do ii=1,nstate
               write(12,'(100g15.5)') (rho_r(ii,jj),jj=2*nstate/8+1
     $              ,3*nstate/8)
            enddo
            write(12,*)'imagab',iqstep,kk
            do ii=1,nstate
               write(12,'(100g15.5)') (rho_i(ii,jj),jj=2*nstate/8+1
     $              ,3*nstate/8)
            enddo

            write(13,*)'realab',iqstep
            do ii=1,nstate
               write(13,'(100g15.5)') (rho_r(ii,jj),jj=3*nstate/8+1
     $              ,4*nstate/8)
            enddo
            write(13,*)'imagab',iqstep,kk
            do ii=1,nstate
               write(13,'(100g15.5)') (rho_i(ii,jj),jj=3*nstate/8+1
     $              ,4*nstate/8)
            enddo

            write(14,*)'realab',iqstep
            do ii=1,nstate
               write(14,'(100g15.5)') (rho_r(ii,jj),jj=4*nstate/8+1
     $              ,5*nstate/8)
            enddo
            write(14,*)'imagab',iqstep,kk
            do ii=1,nstate
               write(14,'(100g15.5)') (rho_i(ii,jj),jj=4*nstate/8+1
     $              ,5*nstate/8)
            enddo

            write(15,*)'realab',iqstep
            do ii=1,nstate
               write(15,'(100g15.5)') (rho_r(ii,jj),jj=5*nstate/8+1
     $              ,6*nstate/8)
            enddo
            write(15,*)'imagab',iqstep,kk
            do ii=1,nstate
               write(15,'(100g15.5)') (rho_i(ii,jj),jj=5*nstate/8+1
     $              ,6*nstate/8)
            enddo

            write(16,*)'realab',iqstep
            do ii=1,nstate
               write(16,'(100g15.5)') (rho_r(ii,jj),jj=6*nstate/8+1
     $              ,7*nstate/8)
            enddo
            write(16,*)'imagab',iqstep,kk
            do ii=1,nstate
               write(16,'(100g15.5)') (rho_i(ii,jj),jj=6*nstate/8+1
     $              ,7*nstate/8)
            enddo

            write(17,*)'realab',iqstep
            do ii=1,nstate
               write(17,'(100g15.5)') (rho_r(ii,jj),jj=7*nstate/8+1
     $              ,8*nstate/8)
            enddo
            write(17,*)'imagab',iqstep,kk
            do ii=1,nstate
               write(17,'(100g15.5)') (rho_i(ii,jj),jj=7*nstate/8+1
     $              ,8*nstate/8)
            enddo
         endif
      endif
      if(nid.eq.1) then
         write(11,*)'realab',iqstep
         do ii=1,nstate
            write(11,'(100g15.5)') (rho_r(ii,jj),jj=1,nstate/np)
         enddo
         write(11,*)'imagab',iqstep,kk
         do ii=1,nstate
            write(11,'(100g15.5)') (rho_i(ii,jj),jj=1,nstate/np)
         enddo
      endif
      if(nid.eq.2) then
         write(12,*)'realab',iqstep
         do ii=1,nstate
            write(12,'(100g15.5)') (rho_r(ii,jj),jj=1,nstate/np)
         enddo
         write(12,*)'imagab',iqstep,kk
         do ii=1,nstate
            write(12,'(100g15.5)') (rho_i(ii,jj),jj=1,nstate/np)
         enddo
      endif

      if(nid.eq.3) then
         write(13,*)'realab',iqstep
         do ii=1,nstate
            write(13,'(100g15.5)') (rho_r(ii,jj),jj=1,nstate/np)
         enddo
         write(13,*)'imagab',iqstep,kk
         do ii=1,nstate
            write(13,'(100g15.5)') (rho_i(ii,jj),jj=1,nstate/np)
         enddo
      endif

      if(nid.eq.4) then
         write(14,*)'realab',iqstep
         do ii=1,nstate
            write(14,'(100g15.5)') (rho_r(ii,jj),jj=1,nstate/np)
         enddo
         write(14,*)'imagab',iqstep,kk
         do ii=1,nstate
            write(14,'(100g15.5)') (rho_i(ii,jj),jj=1,nstate/np)
         enddo
      endif

      if(nid.eq.5) then
         write(15,*)'realab',iqstep
         do ii=1,nstate
            write(15,'(100g15.5)') (rho_r(ii,jj),jj=1,nstate/np)
         enddo
         write(15,*)'imagab',iqstep,kk
         do ii=1,nstate
            write(15,'(100g15.5)') (rho_i(ii,jj),jj=1,nstate/np)
         enddo
      endif

      if(nid.eq.6) then
         write(16,*)'realab',iqstep
         do ii=1,nstate
            write(16,'(100g15.5)') (rho_r(ii,jj),jj=1,nstate/np)
         enddo
         write(16,*)'imagab',iqstep,kk
         do ii=1,nstate
            write(16,'(100g15.5)') (rho_i(ii,jj),jj=1,nstate/np)
         enddo
      endif

      if(nid.eq.7) then
         write(17,*)'realab',iqstep
         do ii=1,nstate
            write(17,'(100g15.5)') (rho_r(ii,jj),jj=1,nstate/np)
         enddo
         write(17,*)'imagab',iqstep,kk
         do ii=1,nstate
            write(17,'(100g15.5)') (rho_i(ii,jj),jj=1,nstate/np)
         enddo
      endif

      enddo
      time = time+dt
      iqstep = iqstep+1
      call cem_quantum3_pop
         if(time.gt.timeOld) then
            call cem_quantum3_pop()
            timeOld = timeOld+printdt
         endif

      if(iqstep.gt.15) call exitt()
      enddo

      call exitt()
      return
      end

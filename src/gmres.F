      subroutine quantum_gmres
c     Solve the Helmholtz equation by right-preconditioned 
c     GMRES iteration.
      implicit none
      include 'SIZE'
      include 'QUANT'
      include 'mpif.h'
      integer  outer,iter,m,i,j,k,ii,jj,iconv,j0,ierr
      real     tol,alpha,l,temp,tolpss,rnorm,tolps
      real     c_gm(larnol),s_gm(larnol),gamma(larnol+1)
      

      tol = 1e-20 !move to user input

      iter  = 0
      m     = larnol

      tolps = tol
      tolpss= tolps
      iconv = 0
      do j=1,m !Hessenberg matrix
         do i=1,m
            qH(i,j) = 0.0
         enddo
      enddo
      !Initial guess
      do j=1,nstate_np
         do i=1,nstate
            j0 = (j-1)*nstate+i
            x_cg(0*nstate2+j0) = rho_r(i,j)
            x_cg(1*nstate2+j0) = rho_i(i,j)
         enddo
      enddo

      !Calculate initial res?
!      call quantum3_mult(res,x_cg,nstate2_2) ! w = A x
      iter = 1
      !I think res = b ?!
      outer = 0


      do while (iconv.eq.0.and.iter.lt.3500000)
         outer = outer+1

c$$$         if(iter.eq.0) then
c$$$            do i=1,nstate2_2
c$$$               r_cg(i) = 0!res(i)
c$$$            enddo
c$$$         else
            !update residual
c$$$            do i=1,nstate2_2
c$$$               r_cg(i) = 0!res(i)   !r = res
c$$$            enddo
            call quantum3_mult(w_gm,x_cg,nstate2_2) ! w = A x

            !Preconditioning with diagonal
!     Because we are solving 2N^2 real ODEs, we have
!     i[H,p] + L(p) => Ax
!                         where A = |          L          (HcrossI - IcrossH) |
!                                   | (HcrossI - IcrossH)          L          |
!     So, the diagonal is only L(p)
!
!     For L(p), the diagonal is simply the local terms (which is only
!     those with the number operator) - this means it is lin_comb_op!
!     So, diag_inv(A)ii,jj = 1/lin_comb_op(ii,jj)

c$$$            do j=1,nstate_np
c$$$               do i=1,nstate
c$$$                  j0 = (j-1)*nstate + i
c$$$                  x_cg(0*nstate2+j0) = x_cg(0*nstate2+j0)/
c$$$     $                 lin_comb_op(i,j)
c$$$               enddo
c$$$            enddo

c$$$            do i=1,nstate2_2
c$$$               r_cg(i) = r_cg(i) - w_gm(i)            ! r = r - w
c$$$            enddo
            do i=1,nstate2_2
               r_cg(i) = -w_gm(i)
            enddo
!         endif
         
         gamma(1) = 0.0
         do i=1,nstate2_2
            gamma(1) = gamma(1) + r_cg(i)*r_cg(i)   !gamma = (r,r)
         enddo
         call MPI_ALLREDUCE(MPI_IN_PLACE,gamma(1),1,MPI_REAL8,MPI_SUM,
     $        MPI_COMM_WORLD,IERR)

         gamma(1) = sqrt(gamma(1))                 ! gamma  = sqrt{ (r,r) }
                                                   !      1

         !check for lucky convergence
c         rnorm = 0.
         if(gamma(1) .eq. 0.) goto 9000
         if(nid.eq.0.and.mod(iter,1).eq.0) 
     $        print*,'gamma(1) ',gamma(1)
         !  v  = r / gamma
         !   1            1
         temp = 1./gamma(1)
         do i=1,nstate2_2
            qV(i,1) = r_cg(i)*temp
         enddo


!         write(6,*) 'start form m-th krylov subspace'
         do j=1,m
            iter = iter+1

            call quantum3_mult(w_gm,qV(1,j),nstate2_2)    ! w = A v

c           !modified Gram-Schmidt
            do i=1,j

               ! h    = (w,v )
               !  i,j       i
               qH(i,j) = 0
               do ii=1,nstate2_2
                  qH(i,j) = qH(i,j) + w_gm(ii) * qV(ii,i)
               enddo
               call MPI_ALLREDUCE(MPI_IN_PLACE,qH(i,j),1,MPI_REAL8,
     $              MPI_SUM,MPI_COMM_WORLD,IERR)

               ! w = w - h    v
               !          i,j  i
               do ii=1,nstate2_2
                  w_gm(ii) = w_gm(ii) - qH(i,j)*qV(ii,i)
               enddo

            enddo

            !apply Givens rotations to new column
            do i=1,j-1
               temp = qH(i,j)                   
               qH(i  ,j)=  c_gm(i)*temp + s_gm(i)*qH(i+1,j)  
               qH(i+1,j)= -s_gm(i)*temp + c_gm(i)*qH(i+1,j)
            enddo

            alpha = 0
            do ii=1,nstate2_2
               alpha = alpha + w_gm(ii)*w_gm(ii)
            enddo
            call MPI_ALLREDUCE(MPI_IN_PLACE,alpha,1,MPI_REAL8,
     $           MPI_SUM,MPI_COMM_WORLD,IERR)
                                       !            _______ 
            alpha = sqrt(alpha)        ! alpha =  \/ (w,w)

            if(alpha.eq.0.) goto 900  !converged

            l = sqrt(qH(j,j)*qH(j,j)+alpha*alpha)
            temp = 1./l
            c_gm(j) = qH(j,j) * temp
            s_gm(j) = alpha  * temp
            qH(j,j) = l
            gamma(j+1) = -s_gm(j) * gamma(j)
            gamma(j)   =  c_gm(j) * gamma(j)

            rnorm = abs(gamma(j+1))

c$$$            if (nid.eq.0) write(6,66) iter,tolpss,rnorm
c$$$   66       format(i5,1p2e12.5,i8,' Divergence')

            if (rnorm .lt. tolps) goto 900 !converged
            if (j.eq.m) goto 1000 !not converged, restart

            temp = 1./alpha
            ! v    = w / alpha
            !  j+1            
            do ii=1,nstate2_2
               qV(ii,j+1) = w_gm(ii)*temp
            enddo

         enddo
c        write(6,*) 'end of forming m-th krylov subspace'
  900    iconv = 1
 1000    continue

c        back substitution
c             -1
c        c = H   gamma

         do k=j,1,-1
            temp = gamma(k)
            do i=j,k+1,-1
               temp = temp - qH(k,i)*c_gm(i)
            enddo
            c_gm(k) = temp/qH(k,k)
         enddo
         !sum up Arnoldi vectors
         do i=1,j
            ! x = x + c  z
            !          i  i
            do ii=1,nstate2_2
               x_cg(ii) = x_cg(ii) + c_gm(i)*qV(ii,i)
            enddo
         enddo  
         !Copy current answer to rho, get current pop
         do jj=1,nstate_np
            do ii=1,nstate
               j0 = (jj-1)*nstate+ii
               rho_r(ii,jj) = x_cg(0*nstate2+j0)
               rho_i(ii,jj) = x_cg(1*nstate2+j0)
c$$$  print*,'rho_r(ii,jj)',ii+c_offset,jj,
c$$$  $           rho_r(ii,jj)
            enddo
         enddo
         call cem_quantum3_pop()

      enddo
 9000 continue

      !Copy answer to back to rho
      !Get the final, steady state populations
      do jj=1,nstate_np
         do ii=1,nstate
            j0 = (jj-1)*nstate+ii
            rho_r(ii,jj) = x_cg(0*nstate2+j0)
            rho_i(ii,jj) = x_cg(1*nstate2+j0)
c$$$            print*,'rho_r(ii,jj)',ii+c_offset,jj,
c$$$     $           rho_r(ii,jj)
         enddo
      enddo


      if (nid.eq.0) write(6,*) 'gmres-rnorm',rnorm
      if (nid.eq.0) write(6,*) iter,tolpss


      return
      end

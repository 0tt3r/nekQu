!*************************************************************
! Basic math/array routines. Imported from NEKCEM - mat1.F
!*************************************************************

c-----------------------------------------------------------------------
      subroutine rzero(a,n)
c-----------------------------------------------------------------------
      real a(n)
     
!$ACC DATA PRESENT(a(1:n))  if (is_present)
!$ACC PARALLEL LOOP   if (is_present)
      DO 100 I = 1, N
 100     a(I ) = 0.0
!$ACC END PARALLEL LOOP
!$ACC END DATA

      return
      end


c-----------------------------------------------------------------------
      SUBROUTINE COPY(A,B,N)
c-----------------------------------------------------------------------
      REAL A(N), B(N)
C
C
C
!$ACC DATA PRESENT(A,B) if(is_present)
!$ACC PARALLEL LOOP if(is_present)
      DO I = 1, N
         A(I) = B(I)
      ENDDO
!$ACC END PARALLEL LOOP
!$ACC END DATA
      RETURN
      END

!***********************************************************************
!     Copy real array to complex array
!***********************************************************************
c-----------------------------------------------------------------------
      subroutine copy_r2z(z,r,n)
c-----------------------------------------------------------------------
      complex*16 z(n)
      real*8     r(n)

      do i=1,n
         z(i)=r(i)
      enddo

      return
      end

!***********************************************************************
!     Copy complex array to complex array
!***********************************************************************
c-----------------------------------------------------------------------
      subroutine copy_z2z(z2,z1,n)
c-----------------------------------------------------------------------
      complex*16 z1(n),z2(n)

      do i=1,n
         z2(i)=z1(i)
      enddo

      return
      end

!***********************************************************************
!     
!***********************************************************************
c-----------------------------------------------------------------------
      subroutine z_exp(f,dt,n)
c-----------------------------------------------------------------------
      real*8     dt
      complex*16 f(n),a

      do i=1,n
         a    = dt*f(i)
         f(i) = exp(a)
      enddo

      return
      end

!***********************************************************************
!     
!***********************************************************************
c-----------------------------------------------------------------------
      subroutine z_rowscale(a,s,n)
c-----------------------------------------------------------------------
      complex*16 a(n,n),s(n)

      do i=1,n
      do j=1,n
         a(i,j) = s(i)*a(i,j)
      enddo
      enddo

      return
      end

c-----------------------------------------------------------------------
      subroutine c_mxm(a,n1,b,n2,c,n3) ! complex mxm
c-----------------------------------------------------------------------
      include 'SIZE'
c
      real a(1),b(1),c(1)   
      real c_one(2),c_zero(2)

      save c_one   ,c_zero
      data c_one  / 1. , 0. /
      data c_zero / 0. , 0. /

      !c=n1xn3  b=n2xn3  a=n1xn2

       call zgemm( 'N','N',n1,n3,n2,c_one,a,n1,b,n2,c_zero,c,n1) !(ifblas)
      !call cgemm( 'N','N',n1,n3,n2,c_one,a,n1,b,n2,c_zero,c,n1)

      return
      end

!***********************************************************************
!     Copy complex array to real array
!***********************************************************************
c-----------------------------------------------------------------------
      subroutine copy_z2r(r,z,n)
c-----------------------------------------------------------------------
      complex*16 z(n)
      real*8     r(n)

      do i=1,n
         r(i)=real(z(i))
      enddo

      return
      end

c$$$c-----------------------------------------------------------------------
c$$$      subroutine mxm(a,n1,b,n2,c,n3) ! complex mxm
c$$$c-----------------------------------------------------------------------
c$$$      include 'SIZE'
c$$$c
c$$$      real a(1),b(1),c(1)   
c$$$
c$$$      !c=n1xn3  b=n2xn3  a=n1xn2
c$$$
c$$$       call dgemm( 'N','N',n1,n3,n2,1.0,a,n1,b,n2,0.0,c,n1) !(ifblas)
c$$$      !call cgemm( 'N','N',n1,n3,n2,c_one,a,n1,b,n2,c_zero,c,n1)
c$$$
c$$$      return
c$$$      end

c-----------------------------------------------------------------------
      SUBROUTINE CMULT(A,CONST,N)
c-----------------------------------------------------------------------
      REAL A(N)
C
!$ACC DATA PRESENT(A) if(is_present)
!$ACC PARALLEL LOOP if(is_present)
      DO 100 I=1,N
         A(I)=A(I)*CONST
 100  CONTINUE
!$ACC END PARALLEL LOOP
!$ACC END DATA
      RETURN
      END


c-----------------------------------------------------------------------
      FUNCTION GLSC2(X,Y,N)
C
C     Perform inner-product in double precision
C

      include 'OPCTR'
      DIMENSION X(N), Y(N)
      DIMENSION TMP(1),WORK(1)
      real*8 ds,dx,dy
      ds = 0.0
!$ACC DATA PRESENT(x,y)  if (is_present)
!$ACC PARALLEL LOOP if (is_present)
      do 10 i=1,n
         dx=x(i)
         dy=y(i)
         ds=ds+dx*dy
   10 continue
!$ACC END PARALLEL LOOP
!$ACC END DATA
      tmp(1)=ds
      CALL GOP(TMP,WORK,'+  ',1)
      GLSC2 = TMP(1)
      RETURN
      END


c-----------------------------------------------------------------------
      SUBROUTINE CMULT2 (A,B,CONST,N)
      DIMENSION A(N),B(N)
!$ACC DATA PRESENT(A,B) if(is_present)
!$ACC PARALLEL LOOP if(is_present)
      DO 100 I=1,N
         A(I)=B(I)*CONST
 100  CONTINUE
!$ACC END PARALLEL LOOP
!$ACC END DATA
      RETURN
      END


c-----------------------------------------------------------------------
      SUBROUTINE ADD2S2(A,B,C1,N)
      DIMENSION A(N),B(N)
!$ACC DATA PRESENT(A,B) if(is_present)
!$ACC PARALLEL LOOP if(is_present)
      DO 100 I=1,N
        A(I)=A(I)+C1*B(I)
  100 CONTINUE
!$ACC END PARALLEL LOOP
!$ACC END DATA
      RETURN
      END

#
#  Copyright (c) 2009-2016, Jack Poulson
#  All rights reserved.
#
#  This file is part of Elemental and is under the BSD 2-Clause License, 
#  which can be found in the LICENSE file in the root directory, or at 
#  http://opensource.org/licenses/BSD-2-Clause
#
import El

m = 10000
n = 5000
rho = 10

display = False
worldRank = El.mpi.WorldRank()
worldSize = El.mpi.WorldSize()

def Rectang(height,width):
  A = El.DistSparseMatrix()
  A.Resize(height,width)
  localHeight = A.LocalHeight()
  A.Reserve(5*localHeight)
  for sLoc in xrange(localHeight):
    s = A.GlobalRow(sLoc)
    A.QueueLocalUpdate( sLoc, s%width, 11 )
    A.QueueLocalUpdate( sLoc, (s-1)%width, -1 )
    A.QueueLocalUpdate( sLoc, (s+1)%width,  2 )
    A.QueueLocalUpdate( sLoc, (s-height)%width, -3 )
    A.QueueLocalUpdate( sLoc, (s+height)%width,  4 )
    # The dense last column
    #A.QueueLocalUpdate( sLoc, width-1, -5/height );

  A.ProcessQueues()
  return A

A = Rectang(m,n)
b = El.DistMultiVec()
El.Gaussian( b, m, 1 )
if display:
  El.Display( A, "A" )
  El.Display( b, "b" )

ctrl = El.SOCPAffineCtrl_d()
ctrl.mehrotraCtrl.progress = True
ctrl.mehrotraCtrl.time = True
ctrl.mehrotraCtrl.solveCtrl.progress = True

# Solve *with* resolving the regularization
ctrl.mehrotraCtrl.resolveReg = True
startRNNLS = El.mpi.Time()
x = El.RNNLS( A, b, rho, ctrl )
endRNNLS = El.mpi.Time()
if worldRank == 0:
  print('RNNLS time (resolve reg.): {} seconds'.format(endRNNLS-startRNNLS))
if display:
  El.Display( x, "x" )

# Solve without resolving the regularization
startRNNLS = El.mpi.Time()
x = El.RNNLS( A, b, rho, ctrl )
endRNNLS = El.mpi.Time()
if worldRank == 0:
  print('RNNLS time (no resolve reg.): {} seconds'.format(endRNNLS-startRNNLS))
if display:
  El.Display( x, "x" )

e = El.DistMultiVec()
El.Copy( b, e )
El.Multiply( El.NORMAL, -1., A, x, 1., e )
eTwoNorm = El.Nrm2( e )
if worldRank == 0:
  print('|| A x - b ||_2 = {}'.format(eTwoNorm))

startNNLS = El.mpi.Time()
xNNLS = El.NNLS( A, b )
endNNLS = El.mpi.Time()
if worldRank == 0:
  print('NNLS time: {} seconds'.format(endNNLS-startNNLS))
if display:
  El.Display( xNNLS, "xNNLS" )
El.Copy( b, e )
El.Multiply( El.NORMAL, -1., A, xNNLS, 1., e )
eTwoNorm = El.Nrm2( e )
if worldRank == 0:
  print('|| A x_{{NNLS}} - b ||_2 = {}'.format(eTwoNorm))

startLS = El.mpi.Time()
xLS = El.LeastSquares( A, b )
endLS = El.mpi.Time()
if worldRank == 0:
  print('LS time: {} seconds'.format(endLS-startLS))
if display:
  El.Display( xLS, "xLS" )
El.Copy( b, e )
El.Multiply( El.NORMAL, -1., A, xLS, 1., e )
eTwoNorm = El.Nrm2( e )
if worldRank == 0:
  print('|| A x_{{LS}} - b ||_2 = {}'.format(eTwoNorm))

El.Finalize()

/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>

#include "../util.hpp"

namespace El {
namespace pdco {

template<typename Real>
Real Merit
( const Matrix<Real>& r1,
  const Matrix<Real>& r2,
  const Matrix<Real>& cL,
  const Matrix<Real>& cU )
{
  Matrix<Real> f;
  Zeros(f, 4, 1);
  f.Set(0, 0, FrobeniusNorm(r1));
  f.Set(1, 0, FrobeniusNorm(r2));
  f.Set(2, 0, FrobeniusNorm(cL));
  f.Set(3, 0, FrobeniusNorm(cU));

  return FrobeniusNorm(f);
}

template<typename Real>
Real MaxXStepSize
( const Matrix<Real>& x,
  const Matrix<Real>& dx,
  const Matrix<Real>& bl,
  const Matrix<Real>& bu,
  const vector<Int>& ixSetLow,
  const vector<Int>& ixSetUpp )
{
  Int n = x.Height();
  Real Inf = limits::Infinity<Real>();
  Real maxStep = Inf;
  Real maxStepL, maxStepU;
  Int ctrLow = 0;
  Int ctrUpp = 0;

  for( Int i=0; i < n; i++ )
  {
      maxStepL = Inf;
      maxStepU = Inf;
      if (ixSetLow[ctrLow] == i)
      {
          if (dx.Get(i,0) < 0)
              maxStepL = (bl.Get(i,0) - x.Get(i,0))/dx.Get(i,0);
          ctrLow++;
      }
      if (ixSetUpp[ctrUpp] == i)
      {
          if (dx.Get(i,0) > 0)
              maxStepU = (bu.Get(i,0) - x.Get(i,0))/dx.Get(i,0);
          ctrUpp++;
      }
      maxStep = Min(maxStep, Min(maxStepL, maxStepU));
  }
  return maxStep;
}

template<typename Real>
Real MaxZStepSize
( const Matrix<Real>& z,
  const Matrix<Real>& dz )
{
  Int n = z.Height();
  Real maxStep = limits::Infinity<Real>();
  for( Int i=0; i < n; i++ )
  {
      if (dz.Get(i,0) > 0)
          continue;
      maxStep = Min(maxStep, -z.Get(i,0)/dz.Get(i,0));
  }
  return maxStep;
}

template<typename Real>
bool Linesearch
( const PDCOObj<Real>& phi,
  const Real& mu,
  const Matrix<Real>& A,
  const Matrix<Real>& b, 
  const Matrix<Real>& bl,
  const Matrix<Real>& bu,
  const Matrix<Real>& D1,
  const Matrix<Real>& D2,
        Matrix<Real>& x,
        Matrix<Real>& y,
        Matrix<Real>& z1,
        Matrix<Real>& z2,
        Matrix<Real>& r1,
        Matrix<Real>& r2,
        Real& center,
        Real& Cinf0,
        Matrix<Real>& cL,
        Matrix<Real>& cU,
        Real& stepx,
        Real& stepz,
  const Matrix<Real>& dx,
  const Matrix<Real>& dy,
  const Matrix<Real>& dz1,
  const Matrix<Real>& dz2,
  const vector<Int>& ixSetLow,
  const vector<Int>& ixSetUpp,
  const vector<Int>& ixSetFix,
  const PDCOCtrl<Real>& ctrl )
{
    DEBUG_ONLY(CSE cse("pdco::Linesearch"))
    stepx  = MaxXStepSize(x, dx, bl, bu, ixSetLow, ixSetUpp);
    Real stepz1 = MaxZStepSize(z1, dz1); // No upper bound constraint
    Real stepz2 = MaxZStepSize(z2, dz2); // No upper bound constraint
    
    stepz = Min(stepz1, stepz2);
    stepx = Min(ctrl.stepTol*stepx, Real(1));
    stepz = Min(ctrl.stepTol*stepz, Real(1));

    if( false && ctrl.print )
    {
        Output("  == Beginning linesearch ==");
        Output("    Backtracking? : ", ctrl.backtrack);
        Output("    Max x-step = ", stepx);
        Output("    Max z-step = ", stepz);
    }

    if( ctrl.stepSame )
    {
        stepx = Min(stepx, stepz);
        stepz = stepx;
    }

    Real merit = Merit(r1, r2, cL, cU);

    if( false && ctrl.print )
        Output("    Current merit = ", merit);

    Real meritNew;

    Matrix<Real> xNew, yNew, z1New, z2New, grad;

    bool success = false;

    // TODO: Fix updating logic here
    // Residuals are probably messed

    for( Int i=0 ; i < ctrl.maxLSIts; i++ )
    {
        // Need to scale?

        Copy(x, xNew);
        Copy(y, yNew);
        Copy(z1, z1New);
        Copy(z2, z2New);
        Axpy(stepx, dx, xNew); // xNew = x + stepx*dx
        Axpy(stepx, dy, yNew); // yNew = y + stepx*dy
        Axpy(stepz, dz1, z1New); // z1New = z1 + stepz*dz1
        Axpy(stepz, dz2, z2New); // z2New = z2 + stepz*dz2

        // Compute residuals
        // Residual vectors to be populated
        phi.grad(x, grad); // get gradient
        ResidualPD(A, ixSetLow, ixSetUpp, ixSetFix,
          b, D1, D2, grad, xNew, yNew, z1New, z2New, r1, r2);

        ResidualC(mu, ixSetLow, ixSetUpp, bl, bu, xNew, z1New, z2New, center, Cinf0, cL, cU);

        meritNew = Merit(r1, r2, cL, cU);

        if( false && ctrl.print )
            Output("      New merit = ", meritNew);

        Real step = Min(stepx, stepz);

        if ( ~ctrl.backtrack || meritNew <= (Real(1) - ctrl.eta*step) * merit )
        {
            Copy(xNew, x);
            Copy(yNew, y);
            Copy(z1New, z1);
            Copy(z2New, z2);
            success = true;
            break;
        }

        // Set stepx = stepz if not already equal
        // Maintain equality for rest of linesearch
        if( (i==0) && (stepx != stepz) )
            stepx = step;

        stepx *= ctrl.lsparam;
        stepz = stepx;
    }

    return success;
}

template<typename Real>
bool Linesearch
( const PDCOObj<Real>& phi,
  const Real& mu,
  const SparseMatrix<Real>& A,
  const Matrix<Real>& b, 
  const Matrix<Real>& bl,
  const Matrix<Real>& bu,
  const Matrix<Real>& D1,
  const Matrix<Real>& D2,
        Matrix<Real>& x,
        Matrix<Real>& y,
        Matrix<Real>& z1,
        Matrix<Real>& z2,
        Matrix<Real>& r1,
        Matrix<Real>& r2,
        Real& center,
        Real& Cinf0,
        Matrix<Real>& cL,
        Matrix<Real>& cU,
        Real& stepx,
        Real& stepz,
  const Matrix<Real>& dx,
  const Matrix<Real>& dy,
  const Matrix<Real>& dz1,
  const Matrix<Real>& dz2,
  const vector<Int>& ixSetLow,
  const vector<Int>& ixSetUpp,
  const vector<Int>& ixSetFix,
  const PDCOCtrl<Real>& ctrl )
{
    DEBUG_ONLY(CSE cse("pdco::Linesearch"))
    stepx  = MaxXStepSize(x, dx, bl, bu, ixSetLow, ixSetUpp);
    Real stepz1 = MaxZStepSize(z1, dz1); // No upper bound constraint
    Real stepz2 = MaxZStepSize(z2, dz2); // No upper bound constraint
    
    stepz = Min(stepz1, stepz2);
    stepx = Min(ctrl.stepTol*stepx, Real(1));
    stepz = Min(ctrl.stepTol*stepz, Real(1));

    if( false && ctrl.print )
    {
        Output("  == Beginning linesearch ==");
        Output("    Backtracking? : ", ctrl.backtrack);
        Output("    Max x-step = ", stepx);
        Output("    Max z-step = ", stepz);
    }

    if( ctrl.stepSame )
    {
        stepx = Min(stepx, stepz);
        stepz = stepx;
    }

    Real merit = Merit(r1, r2, cL, cU);

    if( false && ctrl.print )
        Output("    Current merit = ", merit);

    Real meritNew;

    Matrix<Real> xNew, yNew, z1New, z2New, grad;

    bool success = false;

    // TODO: Fix updating logic here
    // Residuals are probably messed

    for( Int i=0 ; i < ctrl.maxLSIts; i++ )
    {
        // Need to scale?

        Copy(x, xNew);
        Copy(y, yNew);
        Copy(z1, z1New);
        Copy(z2, z2New);
        Axpy(stepx, dx, xNew); // xNew = x + stepx*dx
        Axpy(stepx, dy, yNew); // yNew = y + stepx*dy
        Axpy(stepz, dz1, z1New); // z1New = z1 + stepz*dz1
        Axpy(stepz, dz2, z2New); // z2New = z2 + stepz*dz2

        // Compute residuals
        // Residual vectors to be populated
        phi.grad(x, grad); // get gradient
        ResidualPD(A, ixSetLow, ixSetUpp, ixSetFix,
          b, D1, D2, grad, xNew, yNew, z1New, z2New, r1, r2);

        ResidualC(mu, ixSetLow, ixSetUpp, bl, bu, xNew, z1New, z2New, center, Cinf0, cL, cU);

        meritNew = Merit(r1, r2, cL, cU);

        if( false && ctrl.print )
            Output("      New merit = ", meritNew);

        Real step = Min(stepx, stepz);

        if ( ~ctrl.backtrack || meritNew <= (Real(1) - ctrl.eta*step) * merit )
        {
            Copy(xNew, x);
            Copy(yNew, y);
            Copy(z1New, z1);
            Copy(z2New, z2);
            success = true;
            break;
        }

        // Set stepx = stepz if not already equal
        // Maintain equality for rest of linesearch
        if( (i==0) && (stepx != stepz) )
            stepx = step;

        stepx *= ctrl.lsparam;
        stepz = stepx;
    }

    return success;
}


#define PROTO(Real) \
  template bool Linesearch \
  ( const PDCOObj<Real>& phi, \
    const Real& mu, \
    const Matrix<Real>& A, \
    const Matrix<Real>& b, \
    const Matrix<Real>& bl, \
    const Matrix<Real>& bu, \
    const Matrix<Real>& D1, \
    const Matrix<Real>& D2, \
          Matrix<Real>& x, \
          Matrix<Real>& y, \
          Matrix<Real>& z1, \
          Matrix<Real>& z2, \
          Matrix<Real>& r1, \
          Matrix<Real>& r2, \
          Real& center, \
          Real& Cinf0, \
          Matrix<Real>& cL, \
          Matrix<Real>& cU, \
          Real& stepx, \
          Real& stepz, \
    const Matrix<Real>& dx, \
    const Matrix<Real>& dy, \
    const Matrix<Real>& dz1, \
    const Matrix<Real>& dz2, \
    const vector<Int>& ixSetLow, \
    const vector<Int>& ixSetUpp, \
    const vector<Int>& ixSetFix, \
    const PDCOCtrl<Real>& ctrl ); \
  template bool Linesearch \
  ( const PDCOObj<Real>& phi, \
    const Real& mu, \
    const SparseMatrix<Real>& A, \
    const Matrix<Real>& b, \
    const Matrix<Real>& bl, \
    const Matrix<Real>& bu, \
    const Matrix<Real>& D1, \
    const Matrix<Real>& D2, \
          Matrix<Real>& x, \
          Matrix<Real>& y, \
          Matrix<Real>& z1, \
          Matrix<Real>& z2, \
          Matrix<Real>& r1, \
          Matrix<Real>& r2, \
          Real& center, \
          Real& Cinf0, \
          Matrix<Real>& cL, \
          Matrix<Real>& cU, \
          Real& stepx, \
          Real& stepz, \
    const Matrix<Real>& dx, \
    const Matrix<Real>& dy, \
    const Matrix<Real>& dz1, \
    const Matrix<Real>& dz2, \
    const vector<Int>& ixSetLow, \
    const vector<Int>& ixSetUpp, \
    const vector<Int>& ixSetFix, \
    const PDCOCtrl<Real>& ctrl ); \
  template Real Merit \
  ( const Matrix<Real>& r1, \
    const Matrix<Real>& r2, \
    const Matrix<Real>& cL, \
    const Matrix<Real>& cU ); \
  template Real MaxXStepSize \
  ( const Matrix<Real>& x, \
    const Matrix<Real>& dx, \
    const Matrix<Real>& bl, \
    const Matrix<Real>& bu, \
    const vector<Int>& ixSetLow, \
    const vector<Int>& ixSetUpp ); \
  template Real MaxZStepSize \
  ( const Matrix<Real>& z, \
    const Matrix<Real>& dz );

#define EL_NO_INT_PROTO
#define EL_NO_COMPLEX_PROTO
#define EL_ENABLE_DOUBLEDOUBLE
#define EL_ENABLE_QUADDOUBLE
#define EL_ENABLE_QUAD
#define EL_ENABLE_BIGFLOAT
#include "El/macros/Instantiate.h"

} // namespace pdco
} // namespace El

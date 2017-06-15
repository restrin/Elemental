/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License, 
   which can be found in the LICENSE file in the root directory, or at 
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El-lite.hpp>

namespace El {

template<>
string TypeName<bool>()
{ return string("bool"); }
template<>
string TypeName<char>()
{ return string("char"); }
template<>
string TypeName<char*>()
{ return string("char*"); }
template<>
string TypeName<const char*>()
{ return string("const char*"); }
template<>
string TypeName<string>()
{ return string("string"); }
template<>
string TypeName<unsigned>()
{ return string("unsigned"); }
template<>
string TypeName<unsigned long>()
{ return string("unsigned long"); }
template<>
string TypeName<unsigned long long>()
{ return string("unsigned long long"); }
template<>
string TypeName<int>()
{ return string("int"); }
template<>
string TypeName<long int>()
{ return string("long int"); }
template<>
string TypeName<long long int>()
{ return string("long long int"); }
template<>
string TypeName<float>()
{ return string("float"); }
template<>
string TypeName<double>()
{ return string("double"); }
#ifdef EL_HAVE_QD
template<>
string TypeName<DoubleDouble>()
{ return string("DoubleDouble"); }
template<>
string TypeName<QuadDouble>()
{ return string("QuadDouble"); }
#endif
#ifdef EL_HAVE_QUAD
template<>
string TypeName<Quad>()
{ return string("Quad"); }
#endif
#ifdef EL_HAVE_MPC
template<>
string TypeName<BigInt>()
{ return string("BigInt"); }
template<>
string TypeName<BigFloat>()
{ return string("BigFloat"); }
#endif

// Basic element manipulation and I/O
// ==================================

// Pretty-printing
// ---------------

// TODO: Move into core/imports/quadmath.hpp?
#ifdef EL_HAVE_QUAD
ostream& operator<<( ostream& os, const Quad& alpha )
{
    char str[128];
    quadmath_snprintf( str, 128, "%Qe", alpha );
    os << str;
    return os;
}

istream& operator>>( istream& is, Quad& alpha )
{
    string token;
    is >> token; 
    alpha = strtoflt128( token.c_str(), NULL );
    return is;
}
#endif

// Conjugate
// ---------
#ifdef EL_HAVE_QD
Complex<DoubleDouble> Conj( const Complex<DoubleDouble>& alpha )
{
    Complex<DoubleDouble> alphaConj;
    alphaConj.realPart = alpha.realPart; 
    alphaConj.imagPart = -alpha.imagPart;
    return alphaConj;
}
Complex<QuadDouble> Conj( const Complex<QuadDouble>& alpha )
{
    Complex<QuadDouble> alphaConj;
    alphaConj.realPart = alpha.realPart; 
    alphaConj.imagPart = -alpha.imagPart;
    return alphaConj;
}

void Conj
( const Complex<DoubleDouble>& alpha,
        Complex<DoubleDouble>& alphaConj )
{
    alphaConj.realPart = alpha.realPart; 
    alphaConj.imagPart = -alpha.imagPart;
}
void Conj
( const Complex<QuadDouble>& alpha,
        Complex<QuadDouble>& alphaConj )
{
    alphaConj.realPart = alpha.realPart; 
    alphaConj.imagPart = -alpha.imagPart;
}
#endif
#ifdef EL_HAVE_MPC
Complex<BigFloat> Conj( const Complex<BigFloat>& alpha )
{
    Complex<BigFloat> alphaConj;
    Conj( alpha, alphaConj );
    return alphaConj;
}

void Conj( const Complex<BigFloat>& alpha, Complex<BigFloat>& alphaConj )
{
    mpc_conj( alphaConj.Pointer(), alpha.LockedPointer(), mpc::RoundingMode() );
}
#endif

// Return the complex argument
// ---------------------------
#ifdef EL_HAVE_QUAD
template<>
Quad Arg( const Complex<Quad>& alphaPre )
{
    __complex128 alpha;
    __real__(alpha) = alphaPre.real();
    __imag__(alpha) = alphaPre.imag();

    return cargq(alpha);
}
#endif
#ifdef EL_HAVE_MPC
template<>
BigFloat Arg( const Complex<BigFloat>& alpha )
{
    BigFloat arg;
    arg.SetPrecision( alpha.Precision() );
    mpc_arg( arg.Pointer(), alpha.LockedPointer(), mpfr::RoundingMode() );
    return arg;
}
#endif

// Construct a complex number from its polar coordinates
// -----------------------------------------------------
#ifdef EL_HAVE_QD
Complex<DoubleDouble>
ComplexFromPolar( const DoubleDouble& r, const DoubleDouble& theta )
{
    const DoubleDouble realPart = r*Cos(theta);
    const DoubleDouble imagPart = r*Sin(theta);
    return Complex<DoubleDouble>(realPart,imagPart);
}
Complex<QuadDouble>
ComplexFromPolar( const QuadDouble& r, const QuadDouble& theta )
{
    const QuadDouble realPart = r*Cos(theta);
    const QuadDouble imagPart = r*Sin(theta);
    return Complex<QuadDouble>(realPart,imagPart);
}
#endif
#ifdef EL_HAVE_QUAD
Complex<Quad> ComplexFromPolar( const Quad& r, const Quad& theta )
{
    const Quad realPart = r*cosq(theta);
    const Quad imagPart = r*sinq(theta);
    return Complex<Quad>(realPart,imagPart);
}
#endif
#ifdef EL_HAVE_MPC
Complex<BigFloat> ComplexFromPolar( const BigFloat& r, const BigFloat& theta )
{
    BigFloat sinTheta, cosTheta;
    sinTheta.SetPrecision( theta.Precision() );
    cosTheta.SetPrecision( theta.Precision() );
    mpfr_sin_cos
    ( sinTheta.Pointer(),
      cosTheta.Pointer(),
      theta.LockedPointer(),
      mpfr::RoundingMode() );
    return Complex<BigFloat>(r*cosTheta,r*sinTheta);
}
#endif

// Magnitude and sign
// ==================
#ifdef EL_HAVE_QD
DoubleDouble Abs( const DoubleDouble& alpha ) EL_NO_EXCEPT
{ return fabs(alpha); }

QuadDouble Abs( const QuadDouble& alpha ) EL_NO_EXCEPT
{ return fabs(alpha); }

DoubleDouble Abs( const Complex<DoubleDouble>& alpha ) EL_NO_EXCEPT
{
    DoubleDouble realPart=alpha.realPart, imagPart=alpha.imagPart;
    const DoubleDouble scale = Max(Abs(realPart),Abs(imagPart));
    if( scale == DoubleDouble(0) )
        return scale;
    realPart /= scale;
    imagPart /= scale;
    return scale*Sqrt(realPart*realPart + imagPart*imagPart);
}

QuadDouble Abs( const Complex<QuadDouble>& alpha ) EL_NO_EXCEPT
{
    QuadDouble realPart=alpha.realPart, imagPart=alpha.imagPart;
    const QuadDouble scale = Max(Abs(realPart),Abs(imagPart));
    if( scale == QuadDouble(0) )
        return scale;
    realPart /= scale;
    imagPart /= scale;
    return scale*Sqrt(realPart*realPart + imagPart*imagPart);
}
#endif

#ifdef EL_HAVE_QUAD
Quad Abs( const Quad& alpha ) EL_NO_EXCEPT { return fabsq(alpha); }

Quad Abs( const Complex<Quad>& alphaPre ) EL_NO_EXCEPT
{ 
    __complex128 alpha;
    __real__(alpha) = alphaPre.real();
    __imag__(alpha) = alphaPre.imag();
    return cabsq(alpha); 
}
#endif

#ifdef EL_HAVE_MPC
BigInt Abs( const BigInt& alpha ) EL_NO_EXCEPT
{
    BigInt absAlpha;
    absAlpha.SetMinBits( alpha.NumBits() );
    mpz_abs( absAlpha.Pointer(), alpha.LockedPointer() );
    return absAlpha;
}

BigFloat Abs( const BigFloat& alpha ) EL_NO_EXCEPT
{
    BigFloat absAlpha;
    absAlpha.SetPrecision( alpha.Precision() );
    mpfr_abs( absAlpha.Pointer(), alpha.LockedPointer(), mpfr::RoundingMode() );
    return absAlpha;
}

BigFloat Abs( const Complex<BigFloat>& alpha ) EL_NO_EXCEPT
{
    BigFloat absAlpha;
    absAlpha.SetPrecision( alpha.Precision() );
    mpc_abs( absAlpha.Pointer(), alpha.LockedPointer(), mpfr::RoundingMode() );
    return absAlpha;
}
#endif

#ifdef EL_HAVE_MPC
BigInt Sgn( const BigInt& alpha, bool symmetric ) EL_NO_EXCEPT
{
    const int numBits = alpha.NumBits();
    const int result = mpz_cmp_si( alpha.LockedPointer(), 0 );
    if( result < 0 )
        return BigInt(-1,numBits);
    else if( result > 0 || !symmetric )
        return BigInt(1,numBits);
    else
        return BigInt(0,numBits);
}

BigFloat Sgn( const BigFloat& alpha, bool symmetric ) EL_NO_EXCEPT
{
    mpfr_prec_t prec = alpha.Precision();
    mpfr_sign_t sign = alpha.Sign();
    if( sign < 0 )
        return BigFloat(-1,prec);
    else if( sign > 0 || !symmetric )
        return BigFloat(1,prec);
    else
        return BigFloat(0,prec);
}
#endif

// Exponentiation
// ==============
double Exp( const Int& alpha ) EL_NO_EXCEPT { return std::exp(alpha); }

#ifdef EL_HAVE_QD
DoubleDouble Exp( const DoubleDouble& alpha ) EL_NO_EXCEPT
{ return exp(alpha); }

QuadDouble Exp( const QuadDouble& alpha ) EL_NO_EXCEPT
{ return exp(alpha); }
#endif

#ifdef EL_HAVE_QUAD
Quad Exp( const Quad& alpha ) EL_NO_EXCEPT { return expq(alpha); }

Complex<Quad> Exp( const Complex<Quad>& alphaPre ) EL_NO_EXCEPT
{
    __complex128 alpha;
    __real__(alpha) = alphaPre.real();
    __imag__(alpha) = alphaPre.imag();

    __complex128 alphaExp = cexpq(alpha);
    return Complex<Quad>(crealq(alphaExp),cimagq(alphaExp)); 
}
#endif

#ifdef EL_HAVE_MPC
BigFloat Exp( const BigFloat& alpha ) EL_NO_EXCEPT
{
    BigFloat beta;
    beta.SetPrecision( alpha.Precision() );
    mpfr_exp( beta.Pointer(), alpha.LockedPointer(), mpfr::RoundingMode() );
    return beta;
}

Complex<BigFloat> Exp( const Complex<BigFloat>& alpha ) EL_NO_EXCEPT
{
    Complex<BigFloat> beta;
    beta.SetPrecision( alpha.Precision() );
    mpc_exp( beta.Pointer(), alpha.LockedPointer(), mpc::RoundingMode() );
    return beta;
}
#endif

#ifdef EL_HAVE_QD
// The QD implementation of 'pow' appears to have severe bugs when raising
// non-positive numbers to non-integer powers, as the implementation is:
//
//   dd_real pow(const dd_real &a, const dd_real &b) {
//       return exp(b * log(a));
//   }
//
DoubleDouble Pow( const DoubleDouble& alpha, const DoubleDouble& beta )
{
    if( alpha > DoubleDouble(0) )
    {
        return Exp( beta * Log(alpha) );
    }
    else if( alpha == DoubleDouble(0) )
    {
        return DoubleDouble(0);
    }
    else
    {
        RuntimeError(alpha,"^",beta," not yet supported");
        return DoubleDouble(0);
    }
}
DoubleDouble Pow( const DoubleDouble& alpha, const int& beta )
{ return pow(alpha,beta); }

QuadDouble Pow( const QuadDouble& alpha, const QuadDouble& beta )
{
    if( alpha > QuadDouble(0) )
    {
        return Exp( beta * Log(alpha) );
    }
    else if( alpha == QuadDouble(0) )
    {
        return QuadDouble(0);
    }
    else
    {
        RuntimeError(alpha,"^",beta," not yet supported");
        return QuadDouble(0);
    }
}
QuadDouble Pow( const QuadDouble& alpha, const int& beta )
{ return pow(alpha,beta); }
#endif

#ifdef EL_HAVE_QUAD
Quad Pow( const Quad& alpha, const Quad& beta )
{ return powq(alpha,beta); }

Quad Pow( const Quad& alpha, const Int& beta )
{ return powq(alpha,Quad(beta)); }

Complex<Quad> Pow( const Complex<Quad>& alphaPre, const Complex<Quad>& betaPre )
{
    __complex128 alpha, beta;
    __real__(alpha) = alphaPre.real();
    __imag__(alpha) = alphaPre.imag();
    __real__(beta)  = betaPre.real();
    __imag__(beta)  = betaPre.imag();

    __complex128 gamma = cpowq(alpha,beta);
    return Complex<Quad>(crealq(gamma),cimagq(gamma));
}

Complex<Quad> Pow( const Complex<Quad>& alphaPre, const Quad& betaPre )
{
    __complex128 alpha, beta;
    __real__(alpha) = alphaPre.real();
    __imag__(alpha) = alphaPre.imag();
    __real__(beta)  = betaPre;
    __imag__(beta)  = 0;

    __complex128 gamma = cpowq(alpha,beta);
    return Complex<Quad>(crealq(gamma),cimagq(gamma));
}

Complex<Quad> Pow( const Complex<Quad>& alpha, const Int& beta )
{
    return Pow( alpha, Quad(beta) );
}
#endif

#ifdef EL_HAVE_MPC
void Pow( const BigInt& alpha, const BigInt& beta, BigInt& gamma )
{
    mpz_pow_ui( gamma.Pointer(), alpha.LockedPointer(), (unsigned long)(beta) );
}

void Pow( const BigInt& alpha, const unsigned& beta, BigInt& gamma )
{
    mpz_pow_ui( gamma.Pointer(), alpha.LockedPointer(), beta );
}

void Pow( const BigInt& alpha, const unsigned long& beta, BigInt& gamma )
{
    mpz_pow_ui( gamma.Pointer(), alpha.LockedPointer(), beta );
}

void Pow( const BigInt& alpha, const unsigned long long& beta, BigInt& gamma )
{
    EL_DEBUG_ONLY(
      if( beta > static_cast<unsigned long long>(ULONG_MAX) )
      {
          RuntimeError("Excessively large exponent for Pow: ",beta); 
      }
    )
    unsigned long betaLong = static_cast<unsigned long>(beta);
    mpz_pow_ui( gamma.Pointer(), alpha.LockedPointer(), betaLong );
}

BigInt Pow( const BigInt& alpha, const BigInt& beta )
{
    BigInt gamma;
    Pow( alpha, beta, gamma );
    return gamma;
}

BigInt Pow( const BigInt& alpha, const unsigned& beta )
{
    BigInt gamma;
    Pow( alpha, beta, gamma );
    return gamma;
}

BigInt Pow( const BigInt& alpha, const unsigned long& beta )
{
    BigInt gamma;
    Pow( alpha, beta, gamma );
    return gamma;
}

BigInt Pow( const BigInt& alpha, const unsigned long long& beta )
{
    BigInt gamma;
    Pow( alpha, beta, gamma );
    return gamma;
}

void Pow( const BigFloat& alpha, const BigFloat& beta, BigFloat& gamma )
{
    mpfr_pow
    ( gamma.Pointer(),
      alpha.LockedPointer(),
      beta.LockedPointer(), mpfr::RoundingMode() );
}

void Pow( const BigFloat& alpha, const unsigned& beta, BigFloat& gamma )
{
    mpfr_pow_ui
    ( gamma.Pointer(),
      alpha.LockedPointer(),
      beta, mpfr::RoundingMode() );
}

void Pow( const BigFloat& alpha, const unsigned long& beta, BigFloat& gamma )
{
    mpfr_pow_ui
    ( gamma.Pointer(),
      alpha.LockedPointer(),
      beta, mpfr::RoundingMode() );
}

void Pow
( const BigFloat& alpha, const unsigned long long& beta, BigFloat& gamma )
{
    if( beta <= static_cast<unsigned long long>(ULONG_MAX) )
    {
        unsigned long betaLong = static_cast<unsigned long>(beta);
        mpfr_pow_ui
        ( gamma.Pointer(),
          alpha.LockedPointer(),
          betaLong,
          mpfr::RoundingMode() );
    }
    else
    {
        BigFloat betaBig(beta);
        mpfr_pow
        ( gamma.Pointer(),
          alpha.LockedPointer(),
          betaBig.LockedPointer(), mpfr::RoundingMode() );
    }
}

void Pow( const BigFloat& alpha, const int& beta, BigFloat& gamma )
{
    mpfr_pow_si
    ( gamma.Pointer(),
      alpha.LockedPointer(),
      beta, mpfr::RoundingMode() );
}

void Pow( const BigFloat& alpha, const long int& beta, BigFloat& gamma )
{
    mpfr_pow_si
    ( gamma.Pointer(),
      alpha.LockedPointer(),
      beta, mpfr::RoundingMode() );
}

void Pow( const BigFloat& alpha, const long long int& beta, BigFloat& gamma )
{
    if( (beta >=0 && beta <= static_cast<long long int>(LONG_MAX)) ||
        (beta < 0 && beta >= static_cast<long long int>(LONG_MIN)) )
    {
        long int betaLong = static_cast<long int>(beta);
        mpfr_pow_si
        ( gamma.Pointer(),
          alpha.LockedPointer(),
          betaLong, mpfr::RoundingMode() );
    }
    else
    {
        BigFloat betaBig(beta);
        mpfr_pow
        ( gamma.Pointer(),
          alpha.LockedPointer(),
          betaBig.LockedPointer(), mpfr::RoundingMode() );
    }
}

void Pow( const BigFloat& alpha, const BigInt& beta, BigFloat& gamma )
{
    mpfr_pow_z
    ( gamma.Pointer(),
      alpha.LockedPointer(),
      beta.LockedPointer(), mpfr::RoundingMode() );
}

BigFloat Pow( const BigFloat& alpha, const BigFloat& beta )
{
    BigFloat gamma;
    Pow( alpha, beta, gamma );
    return gamma;
}

Complex<BigFloat>
Pow( const Complex<BigFloat>& alpha, const Complex<BigFloat>& beta )
{
    Complex<BigFloat> gamma;
    mpc_pow
    ( gamma.Pointer(),
      alpha.LockedPointer(),
      beta.LockedPointer(),
      mpc::RoundingMode() );
    return gamma;
}

BigFloat Pow( const BigFloat& alpha, const unsigned& beta )
{
    BigFloat gamma;
    Pow( alpha, beta, gamma );
    return gamma;
}

BigFloat Pow( const BigFloat& alpha, const unsigned long& beta )
{
    BigFloat gamma;
    Pow( alpha, beta, gamma );
    return gamma;
}

BigFloat Pow( const BigFloat& alpha, const unsigned long long& beta )
{
    BigFloat gamma;
    Pow( alpha, beta, gamma );
    return gamma;
}

BigFloat Pow( const BigFloat& alpha, const int& beta )
{
    BigFloat gamma;
    Pow( alpha, beta, gamma );
    return gamma;
}

BigFloat Pow( const BigFloat& alpha, const long int& beta )
{
    BigFloat gamma;
    Pow( alpha, beta, gamma );
    return gamma;
}

BigFloat Pow( const BigFloat& alpha, const long long int& beta )
{
    BigFloat gamma;
    Pow( alpha, beta, gamma );
    return gamma;
}

BigFloat Pow( const BigFloat& alpha, const BigInt& beta )
{
    BigFloat gamma;
    Pow( alpha, beta, gamma );
    return gamma;
}

Complex<BigFloat>
Pow( const Complex<BigFloat>& alpha, const BigFloat& beta )
{
    Complex<BigFloat> gamma;
    mpc_pow_fr
    ( gamma.Pointer(),
      alpha.LockedPointer(),
      beta.LockedPointer(),
      mpc::RoundingMode() );
    return gamma;
}
#endif

// Inverse exponentiation
// ----------------------
#ifdef EL_HAVE_QD
DoubleDouble Log( const DoubleDouble& alpha )
{ return log(alpha); }

QuadDouble Log( const QuadDouble& alpha )
{ return log(alpha); }
#endif

#ifdef EL_HAVE_QUAD
Quad Log( const Quad& alpha ) { return logq(alpha); }

Complex<Quad> Log( const Complex<Quad>& alphaPre )
{
    __complex128 alpha;
    __real__(alpha) = alphaPre.real();
    __imag__(alpha) = alphaPre.imag();

    __complex128 logAlpha = clogq(alpha);
    return Complex<Quad>(crealq(logAlpha),cimagq(logAlpha));
}
#endif

#ifdef EL_HAVE_MPC
double Log( const BigInt& alpha )
{
    return double(Log(BigFloat(alpha)));
}

BigFloat Log( const BigFloat& alpha )
{
    BigFloat beta;
    beta.SetPrecision( alpha.Precision() );
    mpfr_log( beta.Pointer(), alpha.LockedPointer(), mpfr::RoundingMode() );
    return beta;
}

Complex<BigFloat> Log( const Complex<BigFloat>& alpha )
{
    Complex<BigFloat> beta;
    beta.SetPrecision( alpha.Precision() );
    mpc_log( beta.Pointer(), alpha.LockedPointer(), mpc::RoundingMode() );
    return beta;
}
#endif

#ifdef EL_HAVE_QD
DoubleDouble Log2( const DoubleDouble& alpha )
{ return Log(alpha)/DoubleDouble(dd_real::_log2); }

QuadDouble Log2( const QuadDouble& alpha )
{ return Log(alpha)/QuadDouble(qd_real::_log2); }

Complex<DoubleDouble> Log2( const Complex<DoubleDouble>& alpha )
{ return Log(alpha)/DoubleDouble(dd_real::_log2); }

Complex<QuadDouble> Log2( const Complex<QuadDouble>& alpha )
{ return Log(alpha)/QuadDouble(qd_real::_log2); }
#endif

#ifdef EL_HAVE_QUAD
Quad Log2( const Quad& alpha )
{ return log2q(alpha); }
#endif

#ifdef EL_HAVE_MPC
double Log2( const BigInt& alpha )
{
    return double(Log2(BigFloat(alpha)));
}

BigFloat Log2( const BigFloat& alpha )
{
    BigFloat log2Alpha;
    log2Alpha.SetPrecision( alpha.Precision() );
    mpfr_log2
    ( log2Alpha.Pointer(), alpha.LockedPointer(), mpfr::RoundingMode() );
    return log2Alpha;
}

Complex<BigFloat> Log2( const Complex<BigFloat>& alpha )
{
    auto logAlpha = Log( alpha );
    return logAlpha / Log(BigFloat(2));
}
#endif

#ifdef EL_HAVE_QD
DoubleDouble Log10( const DoubleDouble& alpha )
{ return Log(alpha)/DoubleDouble(dd_real::_log10); }

QuadDouble Log10( const QuadDouble& alpha )
{ return Log(alpha)/QuadDouble(qd_real::_log10); }

Complex<DoubleDouble> Log10( const Complex<DoubleDouble>& alpha )
{ return Log(alpha)/DoubleDouble(dd_real::_log10); }

Complex<QuadDouble> Log10( const Complex<QuadDouble>& alpha )
{ return Log(alpha)/QuadDouble(qd_real::_log10); }
#endif

#ifdef EL_HAVE_QUAD
Quad Log10( const Quad& alpha )
{ return log10q(alpha); }
#endif

#ifdef EL_HAVE_MPC
double Log10( const BigInt& alpha )
{
    return double(Log10(BigFloat(alpha)));
}

BigFloat Log10( const BigFloat& alpha )
{
    BigFloat log10Alpha;
    log10Alpha.SetPrecision( alpha.Precision() );
    mpfr_log10
    ( log10Alpha.Pointer(), alpha.LockedPointer(), mpfr::RoundingMode() );
    return log10Alpha;
}

Complex<BigFloat> Log10( const Complex<BigFloat>& alpha )
{
    Complex<BigFloat> log10Alpha;
    log10Alpha.SetPrecision( alpha.Precision() );
    mpc_log10
    ( log10Alpha.Pointer(), alpha.LockedPointer(), mpc::RoundingMode() );
    return log10Alpha;
}
#endif

template<>
Int Sqrt( const Int& alpha ) { return Int(sqrt(alpha)); }

#ifdef EL_HAVE_QD
DoubleDouble Sqrt( const DoubleDouble& alpha ) { return sqrt(alpha); }

QuadDouble Sqrt( const QuadDouble& alpha ) { return sqrt(alpha); }
#endif

#ifdef EL_HAVE_QUAD
Quad Sqrt( const Quad& alpha ) { return sqrtq(alpha); }

Complex<Quad> Sqrt( const Complex<Quad>& alphaPre )
{ 
    __complex128 alpha;
    __real__(alpha) = alphaPre.real();
    __imag__(alpha) = alphaPre.imag();

    __complex128 sqrtAlpha = csqrtq(alpha);
    return Complex<Quad>(crealq(sqrtAlpha),cimagq(sqrtAlpha));
}
#endif

#ifdef EL_HAVE_MPC
void Sqrt( const BigInt& alpha, BigInt& alphaSqrt )
{ mpz_sqrt( alphaSqrt.Pointer(), alpha.LockedPointer() ); }

void Sqrt( const BigFloat& alpha, BigFloat& alphaSqrt )
{
    mpfr_sqrt
    ( alphaSqrt.Pointer(), alpha.LockedPointer(), mpfr::RoundingMode() );
}

void Sqrt( const Complex<BigFloat>& alpha, Complex<BigFloat>& alphaSqrt )
{
    mpc_sqrt
    ( alphaSqrt.Pointer(), alpha.LockedPointer(), mpc::RoundingMode() );
}

BigInt Sqrt( const BigInt& alpha )
{
    BigInt alphaSqrt;
    Sqrt( alpha, alphaSqrt );
    return alphaSqrt;
}

BigFloat Sqrt( const BigFloat& alpha )
{
    BigFloat alphaSqrt;
    alphaSqrt.SetPrecision( alpha.Precision() );
    Sqrt( alpha, alphaSqrt );
    return alphaSqrt;
}

Complex<BigFloat> Sqrt( const Complex<BigFloat>& alpha )
{
    Complex<BigFloat> alphaSqrt;
    alphaSqrt.SetPrecision( alpha.Precision() );
    Sqrt( alpha, alphaSqrt );
    return alphaSqrt;
}
#endif

template<>
unsigned ISqrt( const unsigned& alpha )
{ return static_cast<unsigned>(std::sqrt(alpha)); }
template<>
unsigned long ISqrt( const unsigned long& alpha )
{ return static_cast<unsigned long>(std::sqrt(alpha)); }
template<>
int ISqrt( const int& alpha )
{ return static_cast<int>(std::sqrt(alpha)); }
template<>
long int ISqrt( const long int& alpha )
{ return static_cast<long int>(std::sqrt(alpha)); }

#ifdef EL_HAVE_MPC
void ISqrt( const BigInt& alpha, BigInt& alphaSqrt )
{
    mpz_sqrt( alphaSqrt.Pointer(), alpha.LockedPointer() );
}

template<>
BigInt ISqrt( const BigInt& alpha )
{
    BigInt alphaSqrt;
    ISqrt( alpha, alphaSqrt );
    return alphaSqrt;
}
#endif

// Trigonometric
// =============
double Cos( const Int& alpha ) { return std::cos(alpha); }

#ifdef EL_HAVE_QD
DoubleDouble Cos( const DoubleDouble& alpha ) { return cos(alpha); }

QuadDouble Cos( const QuadDouble& alpha ) { return cos(alpha); }
#endif

#ifdef EL_HAVE_QUAD
Quad Cos( const Quad& alpha ) { return cosq(alpha); }

Complex<Quad> Cos( const Complex<Quad>& alphaPre )
{
    __complex128 alpha;
    __real__(alpha) = alphaPre.real();
    __imag__(alpha) = alphaPre.imag();
    
    __complex128 cosAlpha = ccosq(alpha);
    return Complex<Quad>(crealq(cosAlpha),cimagq(cosAlpha));
}
#endif

#ifdef EL_HAVE_MPC
BigFloat Cos( const BigFloat& alpha )
{
    BigFloat beta;
    beta.SetPrecision( alpha.Precision() );
    mpfr_cos( beta.Pointer(), alpha.LockedPointer(), mpfr::RoundingMode() );
    return beta;
}

Complex<BigFloat> Cos( const Complex<BigFloat>& alpha )
{
    Complex<BigFloat> beta;
    beta.SetPrecision( alpha.Precision() );
    mpc_cos( beta.Pointer(), alpha.LockedPointer(), mpc::RoundingMode() );
    return beta;
}
#endif

double Sin( const Int& alpha ) { return std::sin(alpha); }

#ifdef EL_HAVE_QD
DoubleDouble Sin( const DoubleDouble& alpha ) { return sin(alpha); }

QuadDouble Sin( const QuadDouble& alpha ) { return sin(alpha); }
#endif

#ifdef EL_HAVE_QUAD
Quad Sin( const Quad& alpha ) { return sinq(alpha); }

Complex<Quad> Sin( const Complex<Quad>& alphaPre )
{
    __complex128 alpha;
    __real__(alpha) = alphaPre.real();
    __imag__(alpha) = alphaPre.imag();
    
    __complex128 sinAlpha = csinq(alpha);
    return Complex<Quad>(crealq(sinAlpha),cimagq(sinAlpha));
}
#endif

#ifdef EL_HAVE_MPC
BigFloat Sin( const BigFloat& alpha )
{
    BigFloat beta;
    beta.SetPrecision( alpha.Precision() );
    mpfr_sin( beta.Pointer(), alpha.LockedPointer(), mpfr::RoundingMode() );
    return beta;
}

Complex<BigFloat> Sin( const Complex<BigFloat>& alpha )
{
    Complex<BigFloat> beta;
    beta.SetPrecision( alpha.Precision() );
    mpc_sin( beta.Pointer(), alpha.LockedPointer(), mpc::RoundingMode() );
    return beta;
}
#endif

double Tan( const Int& alpha ) { return std::tan(alpha); }

#ifdef EL_HAVE_QD
DoubleDouble Tan( const DoubleDouble& alpha ) { return tan(alpha); }

QuadDouble Tan( const QuadDouble& alpha ) { return tan(alpha); }
#endif

#ifdef EL_HAVE_QUAD
Quad Tan( const Quad& alpha ) { return tanq(alpha); }

Complex<Quad> Tan( const Complex<Quad>& alphaPre )
{
    __complex128 alpha;
    __real__(alpha) = alphaPre.real();
    __imag__(alpha) = alphaPre.imag();
    
    __complex128 tanAlpha = ctanq(alpha);
    return Complex<Quad>(crealq(tanAlpha),cimagq(tanAlpha));
}
#endif

#ifdef EL_HAVE_MPC
BigFloat Tan( const BigFloat& alpha )
{
    BigFloat beta;
    beta.SetPrecision( alpha.Precision() );
    mpfr_tan( beta.Pointer(), alpha.LockedPointer(), mpfr::RoundingMode() );
    return beta;
}

Complex<BigFloat> Tan( const Complex<BigFloat>& alpha )
{
    Complex<BigFloat> beta;
    beta.SetPrecision( alpha.Precision() );
    mpc_tan( beta.Pointer(), alpha.LockedPointer(), mpc::RoundingMode() );
    return beta;
}
#endif

// Inverse trigonometric
// ---------------------
double Acos( const Int& alpha ) { return std::acos(alpha); }

#ifdef EL_HAVE_QD
DoubleDouble Acos( const DoubleDouble& alpha ) { return acos(alpha); }

QuadDouble Acos( const QuadDouble& alpha ) { return acos(alpha); }
#endif

#ifdef EL_HAVE_QUAD
Quad Acos( const Quad& alpha ) { return acosq(alpha); }

Complex<Quad> Acos( const Complex<Quad>& alphaPre )
{
    __complex128 alpha;
    __real__(alpha) = alphaPre.real();
    __imag__(alpha) = alphaPre.imag();
    
    __complex128 acosAlpha = cacosq(alpha);
    return Complex<Quad>(crealq(acosAlpha),cimagq(acosAlpha));
}
#endif

#ifdef EL_HAVE_MPC
BigFloat Acos( const BigFloat& alpha )
{
    BigFloat beta;
    beta.SetPrecision( alpha.Precision() );
    mpfr_acos( beta.Pointer(), alpha.LockedPointer(), mpfr::RoundingMode() );
    return beta;
}

Complex<BigFloat> Acos( const Complex<BigFloat>& alpha )
{
    Complex<BigFloat> beta;
    beta.SetPrecision( alpha.Precision() );
    mpc_acos( beta.Pointer(), alpha.LockedPointer(), mpc::RoundingMode() );
    return beta;
}
#endif

double Asin( const Int& alpha ) { return std::asin(alpha); }

#ifdef EL_HAVE_QD
DoubleDouble Asin( const DoubleDouble& alpha ) { return asin(alpha); }

QuadDouble Asin( const QuadDouble& alpha ) { return asin(alpha); }
#endif

#ifdef EL_HAVE_QUAD
Quad Asin( const Quad& alpha ) { return asinq(alpha); }

Complex<Quad> Asin( const Complex<Quad>& alphaPre )
{
    __complex128 alpha;
    __real__(alpha) = alphaPre.real();
    __imag__(alpha) = alphaPre.imag();
    
    __complex128 asinAlpha = casinq(alpha);
    return Complex<Quad>(crealq(asinAlpha),cimagq(asinAlpha));
}
#endif

#ifdef EL_HAVE_MPC
BigFloat Asin( const BigFloat& alpha )
{
    BigFloat beta;
    beta.SetPrecision( alpha.Precision() );
    mpfr_asin( beta.Pointer(), alpha.LockedPointer(), mpfr::RoundingMode() );
    return beta;
}

Complex<BigFloat> Asin( const Complex<BigFloat>& alpha )
{
    Complex<BigFloat> beta;
    beta.SetPrecision( alpha.Precision() );
    mpc_asin( beta.Pointer(), alpha.LockedPointer(), mpc::RoundingMode() );
    return beta;
}
#endif

double Atan( const Int& alpha ) { return std::atan(alpha); }

#ifdef EL_HAVE_QD
DoubleDouble Atan( const DoubleDouble& alpha ) { return atan(alpha); }

QuadDouble Atan( const QuadDouble& alpha ) { return atan(alpha); }
#endif

#ifdef EL_HAVE_QUAD
Quad Atan( const Quad& alpha ) { return atanq(alpha); }

Complex<Quad> Atan( const Complex<Quad>& alphaPre )
{
    __complex128 alpha;
    __real__(alpha) = alphaPre.real();
    __imag__(alpha) = alphaPre.imag();
    
    __complex128 atanAlpha = catanq(alpha);
    return Complex<Quad>(crealq(atanAlpha),cimagq(atanAlpha));
}
#endif

#ifdef EL_HAVE_MPC
BigFloat Atan( const BigFloat& alpha )
{
    BigFloat beta;
    beta.SetPrecision( alpha.Precision() );
    mpfr_atan( beta.Pointer(), alpha.LockedPointer(), mpfr::RoundingMode() );
    return beta;
}

Complex<BigFloat> Atan( const Complex<BigFloat>& alpha )
{
    Complex<BigFloat> beta;
    beta.SetPrecision( alpha.Precision() );
    mpc_atan( beta.Pointer(), alpha.LockedPointer(), mpc::RoundingMode() );
    return beta;
}
#endif

double Atan2( const Int& y, const Int& x ) { return std::atan2( y, x ); }

#ifdef EL_HAVE_QD
DoubleDouble Atan2( const DoubleDouble& y, const DoubleDouble& x )
{ return atan2(y,x); }

QuadDouble Atan2( const QuadDouble& y, const QuadDouble& x )
{ return atan2(y,x); }
#endif

#ifdef EL_HAVE_QUAD
Quad Atan2( const Quad& y, const Quad& x ) { return atan2q( y, x ); }
#endif

#ifdef EL_HAVE_MPC
BigFloat Atan2( const BigFloat& y, const BigFloat& x )
{
    BigFloat alpha;
    alpha.SetPrecision( y.Precision() );
    mpfr_atan2
    ( alpha.Pointer(),
      y.LockedPointer(),
      x.LockedPointer(), mpfr::RoundingMode() );
    return alpha;
}
#endif

// Hyperbolic
// ==========
double Cosh( const Int& alpha ) { return std::cosh(alpha); }

#ifdef EL_HAVE_QD
DoubleDouble Cosh( const DoubleDouble& alpha ) { return cosh(alpha); }

QuadDouble Cosh( const QuadDouble& alpha ) { return cosh(alpha); }
#endif

#ifdef EL_HAVE_QUAD
Quad Cosh( const Quad& alpha ) { return coshq(alpha); }

Complex<Quad> Cosh( const Complex<Quad>& alphaPre )
{
    __complex128 alpha;
    __real__(alpha) = alphaPre.real();
    __imag__(alpha) = alphaPre.imag();
    
    __complex128 coshAlpha = ccoshq(alpha);
    return Complex<Quad>(crealq(coshAlpha),cimagq(coshAlpha));
}
#endif

#ifdef EL_HAVE_MPC
BigFloat Cosh( const BigFloat& alpha )
{
    BigFloat beta;
    beta.SetPrecision( alpha.Precision() );
    mpfr_cosh( beta.Pointer(), alpha.LockedPointer(), mpfr::RoundingMode() );
    return beta;
}

Complex<BigFloat> Cosh( const Complex<BigFloat>& alpha )
{
    Complex<BigFloat> beta;
    beta.SetPrecision( alpha.Precision() );
    mpc_cosh( beta.Pointer(), alpha.LockedPointer(), mpc::RoundingMode() );
    return beta;
}
#endif

double Sinh( const Int& alpha ) { return std::sinh(alpha); }

#ifdef EL_HAVE_QD
DoubleDouble Sinh( const DoubleDouble& alpha ) { return sinh(alpha); }

QuadDouble Sinh( const QuadDouble& alpha ) { return sinh(alpha); }
#endif

#ifdef EL_HAVE_QUAD
Quad Sinh( const Quad& alpha ) { return sinhq(alpha); }

Complex<Quad> Sinh( const Complex<Quad>& alphaPre )
{
    __complex128 alpha;
    __real__(alpha) = alphaPre.real();
    __imag__(alpha) = alphaPre.imag();
    
    __complex128 sinhAlpha = csinhq(alpha);
    return Complex<Quad>(crealq(sinhAlpha),cimagq(sinhAlpha));
}
#endif

#ifdef EL_HAVE_MPC
BigFloat Sinh( const BigFloat& alpha )
{
    BigFloat beta;
    beta.SetPrecision( alpha.Precision() );
    mpfr_sinh( beta.Pointer(), alpha.LockedPointer(), mpfr::RoundingMode() );
    return beta;
}

Complex<BigFloat> Sinh( const Complex<BigFloat>& alpha )
{
    Complex<BigFloat> beta;
    beta.SetPrecision( alpha.Precision() );
    mpc_sinh( beta.Pointer(), alpha.LockedPointer(), mpc::RoundingMode() );
    return beta;
}
#endif

double Tanh( const Int& alpha ) { return std::tanh(alpha); }

#ifdef EL_HAVE_QD
DoubleDouble Tanh( const DoubleDouble& alpha ) { return tanh(alpha); }

QuadDouble Tanh( const QuadDouble& alpha ) { return tanh(alpha); }
#endif

#ifdef EL_HAVE_QUAD
Quad Tanh( const Quad& alpha ) { return tanhq(alpha); }

Complex<Quad> Tanh( const Complex<Quad>& alphaPre )
{
    __complex128 alpha;
    __real__(alpha) = alphaPre.real();
    __imag__(alpha) = alphaPre.imag();
    
    __complex128 tanhAlpha = ctanhq(alpha);
    return Complex<Quad>(crealq(tanhAlpha),cimagq(tanhAlpha));
}
#endif

#ifdef EL_HAVE_MPC
BigFloat Tanh( const BigFloat& alpha )
{
    BigFloat beta;
    beta.SetPrecision( alpha.Precision() );
    mpfr_tanh( beta.Pointer(), alpha.LockedPointer(), mpfr::RoundingMode() );
    return beta;
}

Complex<BigFloat> Tanh( const Complex<BigFloat>& alpha )
{
    Complex<BigFloat> beta;
    beta.SetPrecision( alpha.Precision() );
    mpc_tanh( beta.Pointer(), alpha.LockedPointer(), mpc::RoundingMode() );
    return beta;
}
#endif

// Inverse hyperbolic
// ------------------
double Acosh( const Int& alpha ) { return std::acosh(alpha); }

#ifdef EL_HAVE_QD
DoubleDouble Acosh( const DoubleDouble& alpha ) { return acosh(alpha); }

QuadDouble Acosh( const QuadDouble& alpha ) { return acosh(alpha); }
#endif

#ifdef EL_HAVE_QUAD
Quad Acosh( const Quad& alpha ) { return acoshq(alpha); }

Complex<Quad> Acosh( const Complex<Quad>& alphaPre )
{
    __complex128 alpha;
    __real__(alpha) = alphaPre.real();
    __imag__(alpha) = alphaPre.imag();
    
    __complex128 acoshAlpha = cacoshq(alpha);
    return Complex<Quad>(crealq(acoshAlpha),cimagq(acoshAlpha));
}
#endif

#ifdef EL_HAVE_MPC
BigFloat Acosh( const BigFloat& alpha )
{
    BigFloat beta;
    beta.SetPrecision( alpha.Precision() );
    mpfr_acosh( beta.Pointer(), alpha.LockedPointer(), mpfr::RoundingMode() );
    return beta;
}

Complex<BigFloat> Acosh( const Complex<BigFloat>& alpha )
{
    Complex<BigFloat> beta;
    beta.SetPrecision( alpha.Precision() );
    mpc_acosh( beta.Pointer(), alpha.LockedPointer(), mpc::RoundingMode() );
    return beta;
}
#endif

double Asinh( const Int& alpha ) { return std::asinh(alpha); }

#ifdef EL_HAVE_QD
DoubleDouble Asinh( const DoubleDouble& alpha ) { return asinh(alpha); }

QuadDouble Asinh( const QuadDouble& alpha ) { return asinh(alpha); }
#endif

#ifdef EL_HAVE_QUAD
Quad Asinh( const Quad& alpha ) { return asinhq(alpha); }

Complex<Quad> Asinh( const Complex<Quad>& alphaPre )
{
    __complex128 alpha;
    __real__(alpha) = alphaPre.real();
    __imag__(alpha) = alphaPre.imag();
    
    __complex128 asinhAlpha = casinhq(alpha);
    return Complex<Quad>(crealq(asinhAlpha),cimagq(asinhAlpha));
}
#endif

#ifdef EL_HAVE_MPC
BigFloat Asinh( const BigFloat& alpha )
{
    BigFloat beta;
    beta.SetPrecision( alpha.Precision() );
    mpfr_asinh( beta.Pointer(), alpha.LockedPointer(), mpfr::RoundingMode() );
    return beta;
}

Complex<BigFloat> Asinh( const Complex<BigFloat>& alpha )
{
    Complex<BigFloat> beta;
    beta.SetPrecision( alpha.Precision() );
    mpc_asinh( beta.Pointer(), alpha.LockedPointer(), mpc::RoundingMode() );
    return beta;
}
#endif

double Atanh( const Int& alpha ) { return std::atanh(alpha); }

#ifdef EL_HAVE_QD
DoubleDouble Atanh( const DoubleDouble& alpha ) { return atanh(alpha); }

QuadDouble Atanh( const QuadDouble& alpha ) { return atanh(alpha); }
#endif

#ifdef EL_HAVE_QUAD
Quad Atanh( const Quad& alpha ) { return atanhq(alpha); }

Complex<Quad> Atanh( const Complex<Quad>& alphaPre )
{
    __complex128 alpha;
    __real__(alpha) = alphaPre.real();
    __imag__(alpha) = alphaPre.imag();
    
    __complex128 atanhAlpha = catanhq(alpha);
    return Complex<Quad>(crealq(atanhAlpha),cimagq(atanhAlpha));
}
#endif

#ifdef EL_HAVE_MPC
BigFloat Atanh( const BigFloat& alpha )
{
    BigFloat beta;
    beta.SetPrecision( alpha.Precision() );
    mpfr_atanh( beta.Pointer(), alpha.LockedPointer(), mpfr::RoundingMode() );
    return beta;
}

Complex<BigFloat> Atanh( const Complex<BigFloat>& alpha )
{
    Complex<BigFloat> beta;
    beta.SetPrecision( alpha.Precision() );
    mpc_atanh( beta.Pointer(), alpha.LockedPointer(), mpc::RoundingMode() );
    return beta;
}
#endif

// Rounding
// ========

// Round to the nearest integer
// ----------------------------
template<>
Int Round( const Int& alpha ) { return alpha; }
#ifdef EL_HAVE_QD
template<>
DoubleDouble Round( const DoubleDouble& alpha ) { return nint(alpha); }
template<>
QuadDouble Round( const QuadDouble& alpha ) { return nint(alpha); }
#endif
#ifdef EL_HAVE_QUAD
template<>
Quad Round( const Quad& alpha ) { return rintq(alpha); }
#endif
#ifdef EL_HAVE_MPC
template<>
BigInt Round( const BigInt& alpha ) { return alpha; }
template<>
BigFloat Round( const BigFloat& alpha )
{ 
    BigFloat alphaRound;
    alphaRound.SetPrecision( alpha.Precision() );
    mpfr_round( alphaRound.Pointer(), alpha.LockedPointer() );
    return alphaRound;
}
#endif

// Ceiling
// -------
template<> Int Ceil( const Int& alpha ) { return alpha; }
#ifdef EL_HAVE_QD
template<> DoubleDouble Ceil( const DoubleDouble& alpha )
{ return ceil(alpha); }
template<> QuadDouble Ceil( const QuadDouble& alpha )
{ return ceil(alpha); }
#endif
#ifdef EL_HAVE_QUAD
template<> Quad Ceil( const Quad& alpha ) { return ceilq(alpha); }
#endif
#ifdef EL_HAVE_MPC
template<>
BigInt Ceil( const BigInt& alpha ) { return alpha; }
template<>
BigFloat Ceil( const BigFloat& alpha )
{
    BigFloat alphaCeil;
    alphaCeil.SetPrecision( alpha.Precision() );
    mpfr_ceil( alphaCeil.Pointer(), alpha.LockedPointer() );
    return alphaCeil;
}
#endif

// Floor
// -----
template<> Int Floor( const Int& alpha ) { return alpha; }
#ifdef EL_HAVE_QD
template<> DoubleDouble Floor( const DoubleDouble& alpha )
{ return floor(alpha); }
template<> QuadDouble Floor( const QuadDouble& alpha )
{ return floor(alpha); }
#endif
#ifdef EL_HAVE_QUAD
template<> Quad Floor( const Quad& alpha ) { return floorq(alpha); }
#endif
#ifdef EL_HAVE_MPC
template<>
BigInt Floor( const BigInt& alpha ) { return alpha; }
template<> BigFloat Floor( const BigFloat& alpha )
{
    BigFloat alphaFloor;
    alphaFloor.SetPrecision( alpha.Precision() );
    mpfr_floor( alphaFloor.Pointer(), alpha.LockedPointer() );
    return alphaFloor;
}
#endif

// Pi
// ==
#ifdef EL_HAVE_QD
template<> DoubleDouble Pi<DoubleDouble>() { return dd_real::_pi; }
template<> QuadDouble Pi<QuadDouble>() { return qd_real::_pi; }
#endif
#ifdef EL_HAVE_QUAD
template<> Quad Pi<Quad>() { return M_PIq; }
#endif
#ifdef EL_HAVE_MPC
template<>
BigFloat Pi<BigFloat>()
{
    BigFloat pi;
    mpfr_const_pi( pi.Pointer(), mpfr::RoundingMode() );
    return pi;
}

BigFloat Pi( mpfr_prec_t prec )
{
    BigFloat pi;
    pi.SetPrecision( prec );
    mpfr_const_pi( pi.Pointer(), mpfr::RoundingMode() );
    return pi;
}
#endif

// Gamma
// =====
#ifdef EL_HAVE_QD
template<>
DoubleDouble Gamma( const DoubleDouble& alpha )
{
    // TODO: Decide a more precise means of handling this
    return Gamma(to_double(alpha));
}
template<>
QuadDouble Gamma( const QuadDouble& alpha )
{
    // TODO: Decide a more precise means of handling this
    return Gamma(to_double(alpha));
}
#endif
#ifdef EL_HAVE_QUAD
template<>
Quad Gamma( const Quad& alpha ) { return tgammaq(alpha); }
#endif
#ifdef EL_HAVE_MPC
template<> BigFloat Gamma( const BigFloat& alpha )
{
    BigFloat gammaAlpha;
    gammaAlpha.SetPrecision( alpha.Precision() );
    mpfr_gamma
    ( gammaAlpha.Pointer(), alpha.LockedPointer(), mpfr::RoundingMode() );
    return gammaAlpha;
}
#endif

#ifdef EL_HAVE_QD
template<>
DoubleDouble LogGamma( const DoubleDouble& alpha )
{
    // TODO: Decide a more precise means of handling this
    return LogGamma(to_double(alpha));
}
template<>
QuadDouble LogGamma( const QuadDouble& alpha )
{
    // TODO: Decide a more precise means of handling this
    return LogGamma(to_double(alpha));
}
#endif
#ifdef EL_HAVE_QUAD
template<>
Quad LogGamma( const Quad& alpha ) { return lgammaq(alpha); }
#endif
#ifdef EL_HAVE_MPC
template<> BigFloat LogGamma( const BigFloat& alpha )
{
    BigFloat logGammaAlpha;
    logGammaAlpha.SetPrecision( alpha.Precision() );
    int signp;
    mpfr_lgamma
    ( logGammaAlpha.Pointer(), &signp,
      alpha.LockedPointer(), mpfr::RoundingMode() );
    return logGammaAlpha;
}
#endif

} // namespace El

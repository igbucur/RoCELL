#include "RoCELL.h"

arma::vec log_likelihood_gradient_scale_free(
    arma::mat sncsm, unsigned N, 
    double sb21, double sb31, double sb32, double sc24, double sc34,
    double v1, double v2, double v3) {
  
  double k1 = 1.0 / v1;
  double k2 = 1.0 / v2;
  double k3 = 1.0 / v3;
  
  double den = 1 + sc24 * sc24 + sc34 * sc34;
  double z = sc24 * sb31 - sc34 * sb21;
  double y = sc34 + sc24 * sb32;
  
  double p11 = k1 * (den + sb21 * sb21 + sb31 * sb31 + z * z);
  double p12 = sqrt(k1 * k2) * (- sb21 + sb31 * sb32 + z * y);
  double p13 = - sqrt(k1 * k3) * (sb31 + sc24 * z);
  double p22 = k2 * (1 + sb32 * sb32 + y * y);
  double p23 = - sqrt(k2 * k3) * (sb32 + sc24 * y);
  double p33 = k3 * (1 + sc24 * sc24);
  
  double term = sncsm(0, 0) * p11 + 2 * sncsm(0, 1) * p12 + 2 * sncsm(0, 2) * p13 + 
    sncsm(1, 1) * p22 + 2 * sncsm(1, 2) * p23 + sncsm(2, 2) * p33;
  
  arma::vec grad(8);
  
  // derivative wrt sb21
  double dsb21p11 =  2 * k1 * (sb21 - sc34 * z);
  double dsb21p12 = - sqrt(k1 * k2) * (1 + sc34 * y);
  double dsb21p13 = sqrt(k1 * k3) * sc24 * sc34;
  
  // derivative wrt sb31  
  double dsb31p11 = 2 * k1 * (sb31 + sc24 * z);
  double dsb31p12 = sqrt(k1 * k2) * (sb32 + sc24 * y);
  double dsb31p13 = - sqrt(k1 * k3) * (1 + sc24 * sc24);
  
  // derivative wrt sb32
  double dsb32p12 = sqrt(k1 * k2) * (sb31 + sc24 * z);
  double dsb32p22 = 2 * k2 * (sb32 + sc24 * y);
  double dsb32p23 = - sqrt(k2 * k3) * (1 + sc24 * sc24);
  
  // derivative wrt sc24
  double dsc24p11 = 2 * k1 * (sc24 + sb31 * z);
  double dsc24p12 = sqrt(k1 * k2) * (sb31 * y + sb32 * z);
  double dsc24p13 = - sqrt(k1 * k3) * (z + sc24 * sb31);
  double dsc24p22 = 2 * k2 * sb32 * y;
  double dsc24p23 = - sqrt(k2 * k3) * (y + sc24 * sb32);
  double dsc24p33 = 2 * k3 * sc24;
  
  double dsc24term = sncsm(0, 0) * dsc24p11 + 2 * sncsm(0, 1) * dsc24p12 + 
    2 * sncsm(0, 2) * dsc24p13 + sncsm(1, 1) * dsc24p22 +
    2 * sncsm(1, 2) * dsc24p23 + sncsm(2, 2) * dsc24p33;
  
  // derivative wrt sc34
  double dsc34p11 = 2 * k1 * (sc34 - sb21 * z);
  double dsc34p12 = sqrt(k1 * k2) * (z - sb21 * y);
  double dsc34p13 = sqrt(k1 * k3) * sc24 * sb21;
  double dsc34p22 = 2 * k2 * y;
  double dsc34p23 = - sqrt(k2 * k3) * sc24;
  
  double dsc34term = sncsm(0, 0) * dsc34p11 + 2 * sncsm(0, 1) * dsc34p12 + 
    2 * sncsm(0, 2) * dsc34p13 + sncsm(1, 1) * dsc34p22 + 2 * sncsm(1, 2) * dsc34p23;
  
  // derivative wrt k1
  
  double dk1p11 = p11 / k1;
  double dk1p12 = p12 / (2 * k1);
  double dk1p13 = p13 / (2 * k1);
  
  double dk1term = sncsm(0, 0) * dk1p11 + 2 * sncsm(0, 1) * dk1p12 + 2 * sncsm(0, 2) * dk1p13;
  
  // derivative wrt k2
  
  double dk2p12 = p12 / (2 * k2);
  double dk2p22 = p22 / k2;
  double dk2p23 = p23 / (2 * k2);
  
  double dk2term = 2 * sncsm(0, 1) * dk2p12 + sncsm(1, 1) * dk2p22 + 2 * sncsm(1, 2) * dk2p23;
  
  // derivative wrt k3
  
  double dk3p13 = p13 / (2 * k3);
  double dk3p23 = p23 / (2 * k3);
  double dk3p33 = p33 / k3;
  
  double dk3term = 2 * sncsm(0, 2) * dk3p13 + 2 * sncsm(1, 2) * dk3p23 + sncsm(2, 2) * dk3p33;
  
  grad[0] = - (sncsm(0, 0) * dsb21p11 + 2 * sncsm(0, 1) * dsb21p12 + 2 * sncsm(0, 2) * dsb21p13) * N / (2 * den);
  grad[1] = - (sncsm(0, 0) * dsb31p11 + 2 * sncsm(0, 1) * dsb31p12 + 2 * sncsm(0, 2) * dsb31p13) * N / (2 * den);
  grad[2] = - (2 * sncsm(0, 1) * dsb32p12 + sncsm(1, 1) * dsb32p22 + 2 * sncsm(1, 2) * dsb32p23) * N / (2 * den);
  grad[3] = - (dsc24term * den - term * 2 * sc24) * N / (2 * den * den) - sc24 * N / den;
  grad[4] = - (dsc34term * den - term * 2 * sc34) * N / (2 * den * den) - sc34 * N / den;
  grad[5] = ( dk1term * N / (2 * den) - N / (2 * k1) ) * (k1 * k1);
  grad[6] = ( dk2term * N / (2 * den) - N / (2 * k2) ) * (k2 * k2);
  grad[7] = ( dk3term * N / (2 * den) - N / (2 * k3) ) * (k3 * k3);
  
  return grad;
}


arma::mat log_likelihood_hessian_scale_free(
    arma::mat sncsm, unsigned N, 
    double sb21, double sb31, double sb32, double sc24, double sc34,
    double v1, double v2, double v3) {
  
  double k1 = 1.0 / v1;
  double k2 = 1.0 / v2;
  double k3 = 1.0 / v3;
  
  double den = 1 + sc24 * sc24 + sc34 * sc34;
  double z = sc24 * sb31 - sc34 * sb21;
  double y = sc34 + sc24 * sb32;
  
  double p11 = k1 * (den + sb21 * sb21 + sb31 * sb31 + z * z);
  double p12 = sqrt(k1 * k2) * (- sb21 + sb31 * sb32 + z * y);
  double p13 = - sqrt(k1 * k3) * (sb31 + sc24 * z);
  double p22 = k2 * (1 + sb32 * sb32 + y * y);
  double p23 = - sqrt(k2 * k3) * (sb32 + sc24 * y);
  double p33 = k3 * (1 + sc24 * sc24);
  
  
  double term = sncsm(0, 0) * p11 + 2 * sncsm(0, 1) * p12 + 2 * sncsm(0, 2) * p13 + 
    sncsm(1, 1) * p22 + 2 * sncsm(1, 2) * p23 + sncsm(2, 2) * p33;
  
  arma::mat hess(8, 8);
  
  
  // derivative wrt sb21
  double dsb21p11 =  2 * k1 * (sb21 - sc34 * z);
  double dsb21p12 = - sqrt(k1 * k2) * (1 + sc34 * y);
  double dsb21p13 = sqrt(k1 * k3) * sc24 * sc34;
  
  double dsb21term = sncsm(0, 0) * dsb21p11  + 2 * sncsm(0, 1) * dsb21p12 + 2 * sncsm(0, 2) * dsb21p13;
  
  // derivative wrt sb21^2
  double dsb21sb21p11 = 2 * k1 * (1 + sc34 * sc34);
  
  // derivative wrt sb21 sb31
  double dsb21sb31p11 = - 2 * k1 * sc34 * sc24;
  
  // derivative wrt sb21 sb32
  double dsb21sb32p12 = - sqrt(k1 * k2) * sc34 * sc24;
  
  // derivative wrt sb21 sc24
  double dsb21sc24p11 = - 2 * k1 * sc34 * sb31;
  double dsb21sc24p12 = - sqrt(k1 * k2) * sc34 * sb32;
  double dsb21sc24p13 = sqrt(k1 * k3) * sc34;
  
  double dsb21sc24term = sncsm(0, 0) * dsb21sc24p11 + 2 * sncsm(0, 1) * dsb21sc24p12 + 2 * sncsm(0, 2) * dsb21sc24p13;
  
  // derivative wrt sb21 sc34
  double dsb21sc34p11 = 2 * k1 * (sc34 * sb21 - z);
  double dsb21sc34p12 = - sqrt(k1 * k2) * (sc34 + y);
  double dsb21sc34p13 = sqrt(k1 * k3) * sc24;
  
  double dsb21sc34term = sncsm(0, 0) * dsb21sc34p11 + 2 * sncsm(0, 1) * dsb21sc34p12 + 2 * sncsm(0, 2) * dsb21sc34p13;
  
  // derivative wrt sb21 k1
  double dsb21k1p11 = dsb21p11 / k1;
  double dsb21k1p12 = dsb21p12 / (2 * k1);
  double dsb21k1p13 = dsb21p13 / (2 * k1);
  double dsb21k1term = sncsm(0, 0) * dsb21k1p11 + 2 * sncsm(0, 1) * dsb21k1p12 + 2 * sncsm(0, 2) * dsb21k1p13;
  
  // derivative wrt sb21 k2
  double dsb21k2term = sncsm(0, 1) * dsb21p12 / k2;
  
  // derivative wrt sb21 k3
  double dsb21k3term = sncsm(0, 2) * dsb21p13 / k3;
  
  // ------
  
  // derivative wrt sb31  
  double dsb31p11 = 2 * k1 * (sb31 + sc24 * z);
  double dsb31p12 = sqrt(k1 * k2) * (sb32 + sc24 * y);
  double dsb31p13 = - sqrt(k1 * k3) * (1 + sc24 * sc24);
  
  double dsb31term = sncsm(0, 0) * dsb31p11  + 2 * sncsm(0, 1) * dsb31p12 + 2 * sncsm(0, 2) * dsb31p13;
  
  // derivative wrt sb31^2
  double dsb31sb31p11 = 2 * k1 * (1 + sc24 * sc24);
  
  // derivative wrt sb31 sb32
  double dsb31sb32p12 = sqrt(k1 * k2) * (1 + sc24 * sc24);
  
  // derivative wrt sb31 sc24
  double dsb31sc24p11 = 2 * k1 * (z + sc24 * sb31);
  double dsb31sc24p12 = sqrt(k1 * k2) * (y + sc24 * sb32);
  double dsb31sc24p13 = - 2 * sqrt(k1 * k3) * sc24;
  
  double dsb31sc24term = sncsm(0, 0) * dsb31sc24p11 + 2 * sncsm(0, 1) * dsb31sc24p12 + 2 * sncsm(0, 2) * dsb31sc24p13;
  
  // derivative wrt sb31 sc34
  double dsb31sc34p11 = - 2 * k1 * sc24 * sb21;
  double dsb31sc34p12 = sqrt(k1 * k2) * sc24;
  
  double dsb31sc34term = sncsm(0, 0) * dsb31sc34p11 + 2 * sncsm(0, 1) * dsb31sc34p12;
  
  // derivative wrt sb31 k1
  double dsb31k1p11 = dsb31p11 / k1;
  double dsb31k1p12 = dsb31p12 / (2 * k1);
  double dsb31k1p13 = dsb31p13 / (2 * k1);
  
  double dsb31k1term = sncsm(0, 0) * dsb31k1p11 + 2 * sncsm(0, 1) * dsb31k1p12 + 2 * sncsm(0, 2) * dsb31k1p13;
  
  // derivative wrt sb31 k2
  double dsb31k2term = sncsm(0, 1) * dsb31p12 / k2;
  
  // derivative wrt sb31 k3
  double dsb31k3term = sncsm(0, 2) * dsb31p13 / k3;
  
  // ------
  
  // derivative wrt sb32
  double dsb32p12 = sqrt(k1 * k2) * (sb31 + sc24 * z);
  double dsb32p22 = 2 * k2 * (sb32 + sc24 * y);
  double dsb32p23 = - sqrt(k2 * k3) * (1 + sc24 * sc24);
  
  double dsb32term = 2 * sncsm(0, 1) * dsb32p12  + sncsm(1, 1) * dsb32p22 + 2 * sncsm(1, 2) * dsb32p23;
  
  // derivative wrt sb32^2
  double dsb32sb32p22 = 2 * k2 * (1 + sc24 * sc24);
  
  // derivative wrt sb32 sc24
  double dsb32sc24p12 = sqrt(k1 * k2) * (z + sc24 * sb31);
  double dsb32sc24p22 = 2 * k2 * (y + sc24 * sb32);
  double dsb32sc24p23 = - 2 * sqrt(k2 * k3) * sc24;
  
  double dsb32sc24term = 2 * sncsm(0, 1) * dsb32sc24p12 + sncsm(1, 1) * dsb32sc24p22 + 2 * sncsm(1, 2) * dsb32sc24p23;
  
  // derivative wrt sb32 sc34
  double dsb32sc34p12 = - sqrt(k1 * k2) * sc24 * sb21;
  double dsb32sc34p22 = 2 * k2 * sc24;
  
  double dsb32sc34term = 2 * sncsm(0, 1) * dsb32sc34p12 + sncsm(1, 1) * dsb32sc34p22;
  
  // derivative wrt sb32 k1
  double dsb32k1term = sncsm(0, 1) * dsb32p12 / k1;
  
  // derivative wrt sb32 k2
  double dsb32k2p12 = dsb32p12 / (2 * k2);
  double dsb32k2p22 = dsb32p22 / k2;
  double dsb32k2p23 = dsb32p23 / (2 * k2);
  
  double dsb32k2term = 2 * sncsm(0, 1) * dsb32k2p12 + sncsm(1, 1) * dsb32k2p22 + 2 * sncsm(1, 2) * dsb32k2p23;
  
  // derivative wrt sb32 k3
  double dsb32k3term = sncsm(1, 2) * dsb32p23 / k3;
  
  // ------
  
  // derivative wrt sc24
  double dsc24p11 = 2 * k1 * (sc24 + sb31 * z);
  double dsc24p12 = sqrt(k1 * k2) * (sb31 * y + sb32 * z);
  double dsc24p13 = - sqrt(k1 * k3) * (z + sc24 * sb31);
  double dsc24p22 = 2 * k2 * sb32 * y;
  double dsc24p23 = - sqrt(k2 * k3) * (y + sc24 * sb32);
  double dsc24p33 = 2 * k3 * sc24;
  
  double dsc24term = sncsm(0, 0) * dsc24p11 + 2 * sncsm(0, 1) * dsc24p12 + 
    2 * sncsm(0, 2) * dsc24p13 + sncsm(1, 1) * dsc24p22 +
    2 * sncsm(1, 2) * dsc24p23 + sncsm(2, 2) * dsc24p33;
  
  // derivative wrt sc24^2
  double dsc24sc24p11 = 2 * k1 * (1 + sb31 * sb31);
  double dsc24sc24p12 = 2 * sqrt(k1 * k2) * sb31 * sb32;
  double dsc24sc24p13 = - 2 * sqrt(k1 * k3) * sb31;
  double dsc24sc24p22 = 2 * k2 * sb32 * sb32;
  double dsc24sc24p23 = - 2 * sqrt(k2 * k3) * sb32;
  double dsc24sc24p33 = 2 * k3;
  
  double dsc24sc24term = sncsm(0, 0) * dsc24sc24p11 + 2 * sncsm(0, 1) * dsc24sc24p12 + 2 * sncsm(0, 2) * dsc24sc24p13 + 
    sncsm(1, 1) * dsc24sc24p22 + 2 * sncsm(1, 2) * dsc24sc24p23 + sncsm(2, 2) * dsc24sc24p33;
  
  // derivative wrt sc24 sc34
  double dsc24sc34p11 = - 2 * k1 * sb31 * sb21;
  double dsc24sc34p12 = sqrt(k1 * k2) * (sb31 - sb32 * sb21);
  double dsc24sc34p13 = sqrt(k1 * k3) * sb21;
  double dsc24sc34p22 = 2 * k2 * sb32;
  double dsc24sc34p23 = - sqrt(k2 * k3);
  
  double dsc24sc34term = sncsm(0, 0) * dsc24sc34p11 + 2 * sncsm(0, 1) * dsc24sc34p12 + 
    2 * sncsm(0, 2) * dsc24sc34p13 + sncsm(1, 1) * dsc24sc34p22 + 2 * sncsm(1, 2) * dsc24sc34p23;
  
  // derivative wrt sc24 k1
  double dsc24k1term = (sncsm(0, 0) * dsc24p11 + sncsm(0, 1) * dsc24p12 + sncsm(0, 2) * dsc24p13) / k1;
  
  // derivative wrt sc24 k2
  double dsc24k2term = (sncsm(1, 1) * dsc24p22 + sncsm(0, 1) * dsc24p12 + sncsm(1, 2) * dsc24p23) / k2;
  
  // derivative wrt sc24 k3
  double dsc24k3term = (sncsm(2, 2) * dsc24p33 + sncsm(1, 2) * dsc24p23 + sncsm(0, 2) * dsc24p13) / k3;
  
  // ------
  
  // derivative wrt sc34
  double dsc34p11 = 2 * k1 * (sc34 - sb21 * z);
  double dsc34p12 = sqrt(k1 * k2) * (z - sb21 * y);
  double dsc34p13 = sqrt(k1 * k3) * sc24 * sb21;
  double dsc34p22 = 2 * k2 * y;
  double dsc34p23 = - sqrt(k2 * k3) * sc24;
  
  double dsc34term = sncsm(0, 0) * dsc34p11 + 2 * sncsm(0, 1) * dsc34p12 + 
    2 * sncsm(0, 2) * dsc34p13 + sncsm(1, 1) * dsc34p22 + 2 * sncsm(1, 2) * dsc34p23;
  
  // derivative wrt sc34^2
  double dsc34sc34p11 = 2 * k1 * (1 + sb21 * sb21);
  double dsc34sc34p12 = - 2 * sqrt(k1 * k2) * sb21;
  double dsc34sc34p22 = 2 * k2;
  
  double dsc34sc34term = sncsm(0, 0) * dsc34sc34p11 + 2 * sncsm(0, 1) * dsc34sc34p12 + sncsm(1, 1) * dsc34sc34p22;
  
  // derivative wrt sc34 k1
  double dsc34k1term = (sncsm(0, 0) * dsc34p11 + sncsm(0, 1) * dsc34p12 + sncsm(0, 2) * dsc34p13) / k1;
  
  // derivative wrt sc34 k2
  double dsc34k2term = (sncsm(1, 1) * dsc34p22 + sncsm(0, 1) * dsc34p12 + sncsm(1, 2) * dsc34p23) / k2;
  
  // derivative wrt sc34 k3
  double dsc34k3term = (sncsm(1, 2) * dsc34p23 + sncsm(0, 2) * dsc34p13) / k3;
  
  // ------
  
  // derivative wrt k1
  
  double dk1p11 = p11 / k1;
  double dk1p12 = p12 / (2 * k1);
  double dk1p13 = p13 / (2 * k1);
  
  double dk1term = sncsm(0, 0) * dk1p11 + 2 * sncsm(0, 1) * dk1p12 + 2 * sncsm(0, 2) * dk1p13;
  
  // derivative wrt k1^2
  double dk1k1p12 = - dk1p12 / (2 * k1);
  double dk1k1p13 = - dk1p13 / (2 * k1);
  double dk1k1term = 2 * sncsm(0, 1) * dk1k1p12 + 2 * sncsm(0, 2) * dk1k1p13;
  
  // derivative wrt k1 k2
  double dk1k2term = sncsm(0, 1) * dk1p12 / k2;
  
  // derivative wrt k1 k3
  double dk1k3term = sncsm(0, 2) * dk1p13 / k3;
  
  // ------
  
  // derivative wrt k2
  
  double dk2p12 = p12 / (2 * k2);
  double dk2p22 = p22 / k2;
  double dk2p23 = p23 / (2 * k2);
  
  double dk2term = 2 * sncsm(0, 1) * dk2p12 + sncsm(1, 1) * dk2p22 + 2 * sncsm(1, 2) * dk2p23;
  
  // derivative wrt k2^2
  double dk2k2p12 = - dk2p12 / (2 * k2);
  double dk2k2p23 = - dk2p23 / (2 * k2);
  double dk2k2term = 2 * sncsm(0, 1) * dk2k2p12 + 2 * sncsm(1, 2) * dk2k2p23;
  
  // derivative wrt k2 k3
  double dk2k3term = sncsm(1, 2) * dk2p23 / k3;
  
  
  // ------
  
  // derivative wrt k3
  
  double dk3p13 = p13 / (2 * k3);
  double dk3p23 = p23 / (2 * k3);
  double dk3p33 = p33 / k3;
  
  double dk3term = 2 * sncsm(0, 2) * dk3p13 + 2 * sncsm(1, 2) * dk3p23 + sncsm(2, 2) * dk3p33;
  
  // derivative wrt k3^2
  double dk3k3p13 = - dk3p13 / (2 * k3);
  double dk3k3p23 = - dk3p23 / (2 * k3);
  double dk3k3term = 2 * sncsm(0, 2) * dk3k3p13 + 2 * sncsm(1, 2) * dk3k3p23;
  
  // grad[0] = - (sncsm(0, 0) * dsb21p11 + 2 * sncsm(0, 1) * dsb21p12 + 2 * sncsm(0, 2) * dsb21p13) * N / (2 * den);
  // grad[1] = - (sncsm(0, 0) * dsb31p11 + 2 * sncsm(0, 1) * dsb31p12 + 2 * sncsm(0, 2) * dsb31p13) * N / (2 * den);
  // grad[2] = - (2 * sncsm(0, 1) * dsb32p12 + sncsm(1, 1) * dsb32p22 + 2 * sncsm(1, 2) * dsb32p23) * N / (2 * den);
  // grad[3] = - (dsc24term * den - term * 2 * sc24) * N / (2 * den * den) - sc24 * N / den;
  // grad[4] = - (dsc34term * den - term * 2 * sc34) * N / (2 * den * den) - sc34 * N / den;
  // grad[5] = ( dk1term * N / (2 * den) - N / (2 * k1) ) * (k1 * k1);
  // grad[6] = ( dk2term * N / (2 * den) - N / (2 * k2) ) * (k2 * k2);
  // grad[7] = ( dk3term * N / (2 * den) - N / (2 * k3) ) * (k3 * k3);
  
  hess(0, 0) = - sncsm(0, 0) * dsb21sb21p11 * N / (2 * den);
  hess(0, 1) = hess(1, 0) = - sncsm(0, 0) * dsb21sb31p11 * N / (2 * den);
  hess(0, 2) = hess(2, 0) = - 2 * sncsm(0, 1) * dsb21sb32p12 * N / (2 * den);
  hess(0, 3) = hess(3, 0) = - (dsb21sc24term * den - dsb21term * 2 * sc24) * N / (2 * den * den);
  hess(0, 4) = hess(4, 0) = - (dsb21sc34term * den - dsb21term * 2 * sc34) * N / (2 * den * den);
  hess(0, 5) = hess(5, 0) = dsb21k1term * N * k1 * k1 / (2 * den);
  hess(0, 6) = hess(6, 0) = dsb21k2term * N * k2 * k2 / (2 * den);
  hess(0, 7) = hess(7, 0) = dsb21k3term * N * k3 * k3 / (2 * den);
  
  hess(1, 1) = - sncsm(0, 0) * dsb31sb31p11 * N / (2 * den);
  hess(1, 2) = hess(2, 1) = - 2 * sncsm(0, 1) * dsb31sb32p12 * N / (2 * den);
  hess(1, 3) = hess(3, 1) = - (dsb31sc24term * den - dsb31term * 2 * sc24) * N / (2 * den * den);
  hess(1, 4) = hess(4, 1) = - (dsb31sc34term * den - dsb31term * 2 * sc34) * N / (2 * den * den);
  hess(1, 5) = hess(5, 1) = dsb31k1term * N * k1 * k1 / (2 * den);
  hess(1, 6) = hess(6, 1) = dsb31k2term * N * k2 * k2 / (2 * den);
  hess(1, 7) = hess(7, 1) = dsb31k3term * N * k3 * k3 / (2 * den);
  
  hess(2, 2) = hess(2, 2) = - sncsm(1, 1) * dsb32sb32p22 * N / (2 * den);
  hess(2, 3) = hess(3, 2) = - (dsb32sc24term * den - dsb32term * 2 * sc24) * N / (2 * den * den);
  hess(2, 4) = hess(4, 2) = - (dsb32sc34term * den - dsb32term * 2 * sc34) * N / (2 * den * den);
  hess(2, 5) = hess(5, 2) = dsb32k1term * N * k1 * k1 / (2 * den);
  hess(2, 6) = hess(6, 2) = dsb32k2term * N * k2 * k2 / (2 * den);
  hess(2, 7) = hess(7, 2) = dsb32k3term * N * k3 * k3 / (2 * den);
  
  hess(3, 3) = - (dsc24sc24term * den * den - 2 * term * den - 4 * dsc24term * sc24 * den + 8 * term * sc24 * sc24) * N / (2 * den * den * den) - N * (den - 2 * sc24 * sc24) / (den * den);
  hess(3, 4) = hess(4, 3) = - (dsc24sc34term * den * den - 2 * dsc24term * sc34 * den - 2 * dsc34term * sc24 * den + 8 * term * sc24 * sc34) * N / (2 * den * den * den) + 2 * sc24 * sc34 * N / (den * den);
  hess(3, 5) = hess(5, 3) = (dsc24k1term * den - dk1term * 2 * sc24) * N * k1 * k1 / (2 * den * den);
  hess(3, 6) = hess(6, 3) = (dsc24k2term * den - dk2term * 2 * sc24) * N * k2 * k2 / (2 * den * den);
  hess(3, 7) = hess(7, 3) = (dsc24k3term * den - dk3term * 2 * sc24) * N * k3 * k3 / (2 * den * den);
  
  hess(4, 4) = - (dsc34sc34term * den * den - 2 * term * den - 4 * dsc34term * sc34 * den + 8 * term * sc34 * sc34) * N / (2 * den * den * den) - N * (den - 2 * sc34 * sc34) / (den * den);
  hess(4, 5) = hess(5, 4) = (dsc34k1term * den - dk1term * 2 * sc34) * N * k1 * k1 / (2 * den * den);
  hess(4, 6) = hess(6, 4) = (dsc34k2term * den - dk2term * 2 * sc34) * N * k2 * k2 / (2 * den * den);
  hess(4, 7) = hess(7, 4) = (dsc34k3term * den - dk3term * 2 * sc34) * N * k3 * k3 / (2 * den * den);
  
  hess(5, 5) = -(( dk1term * N / (2 * den) - N / (2 * k1) ) * 2 * k1 * k1 * k1 + (dk1k1term * N / (2 * den) + N / (2 * k1 * k1)) * k1 * k1 * k1 * k1);
  hess(5, 6) = hess(6, 5) = - dk1k2term * N * k1 * k1 * k2 * k2 / (2 * den);
  hess(5, 7) = hess(7, 5) = - dk1k3term * N * k1 * k1 * k3 * k3 / (2 * den);
  
  hess(6, 6) = -(( dk2term * N / (2 * den) - N / (2 * k2) ) * 2 * k2 * k2 * k2 + (dk2k2term * N / (2 * den) + N / (2 * k2 * k2)) * k2 * k2 * k2 * k2);
  hess(6, 7) = hess(7, 6) = - dk2k3term * N * k2 * k2 * k3 * k3 / (2 * den);
  
  hess(7, 7) = -(( dk3term * N / (2 * den) - N / (2 * k3) ) * 2 * k3 * k3 * k3 + (dk3k3term * N / (2 * den) + N / (2 * k3 * k3)) * k3 * k3 * k3 * k3);
  
  return hess;
}

arma::mat log_likelihood_hessian_scale_free_confounders(
    arma::mat sncsm, unsigned N, 
    double b21, double b31, double b32, double sc24, double sc34,
    double v1, double v2, double v3) {
 

  double sb21 = b21 * sqrt(v1 / v2);
  double sb31 = b31 * sqrt(v1 / v3);
  double sb32 = b32 * sqrt(v2 / v3);
  
  arma::vec grad = log_likelihood_gradient_scale_free(sncsm, N, sb21, sb31, sb32, sc24, sc34, v1, v2, v3);
  arma::mat orig = log_likelihood_hessian_scale_free(sncsm, N, sb21, sb31, sb32, sc24, sc34, v1, v2, v3);
  
  arma::mat hess(8, 8);
  
  hess(0, 0) = orig(0, 0) * v1 / v2;
  hess(0, 1) = hess(1, 0) = orig(0, 1) * v1 / sqrt(v2 * v3);
  hess(0, 2) = hess(2, 0) = orig(0, 2) * sqrt(v1 / v3);
  hess(0, 3) = hess(3, 0) = orig(0, 3) * sqrt(v1 / v2);
  hess(0, 4) = hess(4, 0) = orig(0, 4) * sqrt(v1 / v2);
  hess(0, 5) = hess(5, 0) = sqrt(v1 / v2) * (orig(0, 5) + orig(0, 0) * sb21 / (2 * v1) + orig(0, 1) * sb31 / (2 * v1) + grad(0) / (2 * v1));
  hess(0, 6) = hess(6, 0) = sqrt(v1 / v2) * (orig(0, 6) - orig(0, 0) * sb21 / (2 * v2) + orig(0, 2) * sb32 / (2 * v2) - grad(0) / (2 * v2));
  hess(0, 7) = hess(7, 0) = sqrt(v1 / v2) * (orig(0, 7) - orig(0, 1) * sb31 / (2 * v3) - orig(0, 2) * sb32 / (2 * v3)); 
  
  hess(1, 1) = orig(1, 1) * v1 / v3;
  hess(1, 2) = hess(2, 1) = orig(1, 2) * sqrt(v1 * v2) / v3;
  hess(1, 3) = hess(3, 1) = orig(1, 3) * sqrt(v1 / v3);
  hess(1, 4) = hess(4, 1) = orig(1, 4) * sqrt(v1 / v3);
  hess(1, 5) = hess(5, 1) = sqrt(v1 / v3) * (orig(1, 5) + orig(1, 0) * sb21 / (2 * v1) + orig(1, 1) * sb31 / (2 * v1) + grad(1) / (2 * v1));
  hess(1, 6) = hess(6, 1) = sqrt(v1 / v3) * (orig(1, 6) - orig(1, 0) * sb21 / (2 * v2) + orig(1, 2) * sb32 / (2 * v2));
  hess(1, 7) = hess(7, 1) = sqrt(v1 / v3) * (orig(1, 7) - orig(1, 1) * sb31 / (2 * v3) - orig(1, 2) * sb32 / (2 * v3) - grad(1) / (2 * v3));
  
  hess(2, 2) = orig(2, 2) * v2 / v3;
  hess(2, 3) = hess(3, 2) = orig(2, 3) * sqrt(v2 / v3);
  hess(2, 4) = hess(4, 2) = orig(2, 4) * sqrt(v2 / v3);
  hess(2, 5) = hess(5, 2) = sqrt(v2 / v3) * (orig(2, 5) + orig(2, 0) * sb21 / (2 * v1) + orig(2, 1) * sb31 / (2 * v1));
  hess(2, 6) = hess(6, 2) = sqrt(v2 / v3) * (orig(2, 6) - orig(2, 0) * sb21 / (2 * v2) + orig(2, 2) * sb32 / (2 * v2) + grad(2) / (2 * v2));
  hess(2, 7) = hess(7, 2) = sqrt(v2 / v3) * (orig(2, 7) - orig(2, 1) * sb31 / (2 * v3) - orig(2, 2) * sb32 / (2 * v3) - grad(2) / (2 * v3));
  
  hess(3, 3) = orig(3, 3);
  hess(3, 4) = hess(4, 3) = orig(3, 4);
  hess(3, 5) = hess(5, 3) = orig(3, 5) + orig(3, 0) * sb21 / (2 * v1) + orig(3, 1) * sb31 / (2 * v1);
  hess(3, 6) = hess(6, 3) = orig(3, 6) - orig(3, 0) * sb21 / (2 * v2) + orig(3, 2) * sb32 / (2 * v2);
  hess(3, 7) = hess(7, 3) = orig(3, 7) - orig(3, 1) * sb31 / (2 * v3) - orig(3, 2) * sb32 / (2 * v3);
  
  hess(4, 4) = orig(4, 4);
  hess(4, 5) = hess(5, 4) = orig(4, 5) + orig(4, 0) * sb21 / (2 * v1) + orig(4, 1) * sb31 / (2 * v1);
  hess(4, 6) = hess(6, 4) = orig(4, 6) - orig(4, 0) * sb21 / (2 * v2) + orig(4, 2) * sb32 / (2 * v2);
  hess(4, 7) = hess(7, 4) = orig(4, 7) - orig(4, 1) * sb31 / (2 * v3) - orig(4, 2) * sb32 / (2 * v3);
  
  hess(5, 5) = orig(5, 5) + (orig(0, 0) * sb21 * sb21 + orig(1, 1) * sb31 * sb31 + 2 * orig(0, 1) * sb21 * sb31 - grad(0) * sb21 - grad(1) * sb31) / (4 * v1 * v1) + (orig(5, 0) * sb21 + orig(5, 1) * sb31) / v1;
  hess(5, 6) = hess(6, 5) = orig(5, 6) + (orig(5, 2) * sb32 - orig(5, 0) * sb21) / (2 * v2) + (sb31 * (orig(1, 6) + (orig(1, 2) * sb32 - orig(1, 0) * sb21) / (2 * v2)) + sb21 * (orig(0, 6) + (orig(0, 2) * sb32 - orig(0, 0) * sb21) / (2 * v2))) / (2 * v1) - grad(0) * sb21 / (4 * v1 * v2);
  hess(5, 7) = hess(7, 5) = orig(5, 7) - (orig(5, 2) * sb32 + orig(5, 1) * sb31) / (2 * v3) + (sb31 * (orig(1, 7) - (orig(1, 2) * sb32 + orig(1, 1) * sb31) / (2 * v3)) + sb21 * (orig(0, 7) - (orig(0, 2) * sb32 + orig(0, 1) * sb31) / (2 * v3))) / (2 * v1) - grad(1) * sb31 / (4 * v1 * v3);
  
  hess(6, 6) = orig(6, 6) + (orig(0, 0) * sb21 * sb21 + orig(2, 2) * sb32 * sb32 - 2 * orig(0, 2) * sb21 * sb32 + 3 * grad(0) * sb21 - grad(2) * sb32) / (4 * v2 * v2) + (- orig(6, 0) * sb21 + orig(6, 2) * sb32) / v2;
  hess(6, 7) = hess(7, 6) = orig(6, 7) - (orig(6, 2) * sb32 + orig(6, 1) * sb31) / (2 * v3) + (sb32 * (orig(2, 7) - (orig(2, 2) * sb32 + orig(2, 1) * sb31) / (2 * v3)) - sb21 * (orig(0, 7) - (orig(0, 2) * sb32 + orig(0, 1) * sb31) / (2 * v3))) / (2 * v2) - grad(2) * sb32 / (4 * v2 * v3);
  
  hess(7, 7) = orig(7, 7) + (orig(1, 1) * sb31 * sb31 + orig(2, 2) * sb32 * sb32 + 2 * orig(1, 2) * sb31 * sb32 + 3 * grad(1) * sb31 + 3 * grad(2) * sb32) / (4 * v3 * v3) - (orig(7, 1) * sb31 + orig(7, 2) * sb32) / v3;
 
  return hess;
}



#include <stdio.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <stdlib.h>
#include "multinest.h"
#include "RoCELL.h"

#include <boost/math/distributions/normal.hpp>
boost::math::normal dist(0.0, 1.0);

// Input structure
typedef struct {
  arma::mat covariance;
  double slab_precision;
  double spike_precision;
  int model; // 0 - Slab, 1 - Spike, 2 - Reverse
} Config;
// 
// 
// /**
//  *
//  */
// Eigen::VectorXd get_params(Eigen::MatrixXd Sigma, double c24, double c34) {
//   
//   Eigen::MatrixXd B = Eigen::MatrixXd::Zero(3, 3);
//   Eigen::VectorXd AV = Eigen::VectorXd::Zero(6);
//   
//   B(0, 0) = 1;
//   B(1, 1) = 1 + c24 * c24;
//   B(1, 2) = B(2, 1) = c24 * c34;
//   B(2, 2) = 1 + c34 * c34;
//   
//   Eigen::MatrixXd Q = Sigma.llt().matrixL();
//   Eigen::MatrixXd U = B.llt().matrixL();
//   Eigen::MatrixXd R = Q * U.inverse();
//   
//   Eigen::VectorXd sqV = R.diagonal();
//   
//   
//   Eigen::MatrixXd A = Eigen::MatrixXd::Identity(3, 3) - sqV.asDiagonal() * U * Q.inverse();
//   
//   AV[0] = A(1, 0);
//   AV[1] = A(2, 0);
//   AV[2] = A(2, 1);
//   AV[3] = sqV[0] * sqV[0];
//   AV[4] = sqV[1] * sqV[1];
//   AV[5] = sqV[2] * sqV[2];
//   
//   return AV;
// }
// 
// Eigen::VectorXd logLikelihoodObservedGradientVarianceScaleFree(
//     Eigen::MatrixXd sncsm, int N, 
//     double b12, double b13, double b23, double c24, double c34,
//     double v1, double v2, double v3) {
//   
//   double k1 = 1.0 / v1;
//   double k2 = 1.0 / v2;
//   double k3 = 1.0 / v3;
//   
//   double den = 1 + c24 * c24 + c34 * c34;
//   double z = c24 * b13 - c34 * b12;
//   double y = c34 + c24 * b23;
//   
//   double p11 = k1 * (den + b12 * b12 + b13 * b13 + z * z);
//   double p12 = sqrt(k1 * k2) * (- b12 + b13 * b23 + z * y);
//   double p13 = - sqrt(k1 * k3) * (b13 + c24 * z);
//   double p22 = k2 * (1 + b23 * b23 + y * y);
//   double p23 = - sqrt(k2 * k3) * (b23 + c24 * y);
//   double p33 = k3 * (1 + c24 * c24);
//   
//   double term = sncsm(0, 0) * p11 + 2 * sncsm(0, 1) * p12 + 2 * sncsm(0, 2) * p13 + 
//     sncsm(1, 1) * p22 + 2 * sncsm(1, 2) * p23 + sncsm(2, 2) * p33;
//   
//   Eigen::VectorXd grad(8);
//   
//   // derivative wrt b12
//   double db12p11 =  2 * k1 * (b12 - c34 * z);
//   double db12p12 = - sqrt(k1 * k2) * (1 + c34 * y);
//   double db12p13 = sqrt(k1 * k3) * c24 * c34;
//   
//   // derivative wrt b13  
//   double db13p11 = 2 * k1 * (b13 + c24 * z);
//   double db13p12 = sqrt(k1 * k2) * (b23 + c24 * y);
//   double db13p13 = - sqrt(k1 * k3) * (1 + c24 * c24);
//   
//   // derivative wrt b23
//   double db23p12 = sqrt(k1 * k2) * (b13 + c24 * z);
//   double db23p22 = 2 * k2 * (b23 + c24 * y);
//   double db23p23 = - sqrt(k2 * k3) * (1 + c24 * c24);
//   
//   // derivative wrt c24
//   double dc24p11 = 2 * k1 * (c24 + b13 * z);
//   double dc24p12 = sqrt(k1 * k2) * (b13 * y + b23 * z);
//   double dc24p13 = - sqrt(k1 * k3) * (z + c24 * b13);
//   double dc24p22 = 2 * k2 * b23 * y;
//   double dc24p23 = - sqrt(k2 * k3) * (y + c24 * b23);
//   double dc24p33 = 2 * k3 * c24;
//   
//   double dc24term = sncsm(0, 0) * dc24p11 + 2 * sncsm(0, 1) * dc24p12 + 
//     2 * sncsm(0, 2) * dc24p13 + sncsm(1, 1) * dc24p22 +
//     2 * sncsm(1, 2) * dc24p23 + sncsm(2, 2) * dc24p33;
//   
//   // derivative wrt c34
//   double dc34p11 = 2 * k1 * (c34 - b12 * z);
//   double dc34p12 = sqrt(k1 * k2) * (z - b12 * y);
//   double dc34p13 = sqrt(k1 * k3) * c24 * b12;
//   double dc34p22 = 2 * k2 * y;
//   double dc34p23 = - sqrt(k2 * k3) * c24;
//   
//   double dc34term = sncsm(0, 0) * dc34p11 + 2 * sncsm(0, 1) * dc34p12 + 
//     2 * sncsm(0, 2) * dc34p13 + sncsm(1, 1) * dc34p22 + 2 * sncsm(1, 2) * dc34p23;
//   
//   // derivative wrt k1
//   
//   double dk1p11 = p11 / k1;
//   double dk1p12 = p12 / (2 * k1);
//   double dk1p13 = p13 / (2 * k1);
//   
//   double dk1term = sncsm(0, 0) * dk1p11 + 2 * sncsm(0, 1) * dk1p12 + 2 * sncsm(0, 2) * dk1p13;
//   
//   // derivative wrt k2
//   
//   double dk2p12 = p12 / (2 * k2);
//   double dk2p22 = p22 / k2;
//   double dk2p23 = p23 / (2 * k2);
//   
//   double dk2term = 2 * sncsm(0, 1) * dk2p12 + sncsm(1, 1) * dk2p22 + 2 * sncsm(1, 2) * dk2p23;
//   
//   // derivative wrt k3
//   
//   double dk3p13 = p13 / (2 * k3);
//   double dk3p23 = p23 / (2 * k3);
//   double dk3p33 = p33 / k3;
//   
//   double dk3term = 2 * sncsm(0, 2) * dk3p13 + 2 * sncsm(1, 2) * dk3p23 + sncsm(2, 2) * dk3p33;
//   
//   grad[0] = - (sncsm(0, 0) * db12p11 + 2 * sncsm(0, 1) * db12p12 + 2 * sncsm(0, 2) * db12p13) * N / (2 * den);
//   grad[1] = - (sncsm(0, 0) * db13p11 + 2 * sncsm(0, 1) * db13p12 + 2 * sncsm(0, 2) * db13p13) * N / (2 * den);
//   grad[2] = - (2 * sncsm(0, 1) * db23p12 + sncsm(1, 1) * db23p22 + 2 * sncsm(1, 2) * db23p23) * N / (2 * den);
//   grad[3] = - (dc24term * den - term * 2 * c24) * N / (2 * den * den) - c24 * N / den;
//   grad[4] = - (dc34term * den - term * 2 * c34) * N / (2 * den * den) - c34 * N / den;
//   grad[5] = ( dk1term * N / (2 * den) - N / (2 * k1) ) * (k1 * k1);
//   grad[6] = ( dk2term * N / (2 * den) - N / (2 * k2) ) * (k2 * k2);
//   grad[7] = ( dk3term * N / (2 * den) - N / (2 * k3) ) * (k3 * k3);
//   
//   return grad;
// }
// 
// Eigen::MatrixXd logLikelihoodObservedHessianVarianceScaleFree(
//     Eigen::MatrixXd sncsm, int N, 
//     double b12, double b13, double b23, double c24, double c34,
//     double v1, double v2, double v3) {
//   
//   double k1 = 1.0 / v1;
//   double k2 = 1.0 / v2;
//   double k3 = 1.0 / v3;
//   
//   double den = 1 + c24 * c24 + c34 * c34;
//   double z = c24 * b13 - c34 * b12;
//   double y = c34 + c24 * b23;
//   
//   double p11 = k1 * (den + b12 * b12 + b13 * b13 + z * z);
//   double p12 = sqrt(k1 * k2) * (- b12 + b13 * b23 + z * y);
//   double p13 = - sqrt(k1 * k3) * (b13 + c24 * z);
//   double p22 = k2 * (1 + b23 * b23 + y * y);
//   double p23 = - sqrt(k2 * k3) * (b23 + c24 * y);
//   double p33 = k3 * (1 + c24 * c24);
//   
//   
//   double term = sncsm(0, 0) * p11 + 2 * sncsm(0, 1) * p12 + 2 * sncsm(0, 2) * p13 + 
//     sncsm(1, 1) * p22 + 2 * sncsm(1, 2) * p23 + sncsm(2, 2) * p33;
//   
//   Eigen::MatrixXd hess(8, 8);
//   
//   
//   // derivative wrt b12
//   double db12p11 =  2 * k1 * (b12 - c34 * z);
//   double db12p12 = - sqrt(k1 * k2) * (1 + c34 * y);
//   double db12p13 = sqrt(k1 * k3) * c24 * c34;
//   
//   double db12term = sncsm(0, 0) * db12p11  + 2 * sncsm(0, 1) * db12p12 + 2 * sncsm(0, 2) * db12p13;
//   
//   // derivative wrt b12^2
//   double db12b12p11 = 2 * k1 * (1 + c34 * c34);
//   
//   // derivative wrt b12 b13
//   double db12b13p11 = - 2 * k1 * c34 * c24;
//   
//   // derivative wrt b12 b23
//   double db12b23p12 = - sqrt(k1 * k2) * c34 * c24;
//   
//   // derivative wrt b12 c24
//   double db12c24p11 = - 2 * k1 * c34 * b13;
//   double db12c24p12 = - sqrt(k1 * k2) * c34 * b23;
//   double db12c24p13 = sqrt(k1 * k3) * c34;
//   
//   double db12c24term = sncsm(0, 0) * db12c24p11 + 2 * sncsm(0, 1) * db12c24p12 + 2 * sncsm(0, 2) * db12c24p13;
//   
//   // derivative wrt b12 c34
//   double db12c34p11 = 2 * k1 * (c34 * b12 - z);
//   double db12c34p12 = - sqrt(k1 * k2) * (c34 + y);
//   double db12c34p13 = sqrt(k1 * k3) * c24;
//   
//   double db12c34term = sncsm(0, 0) * db12c34p11 + 2 * sncsm(0, 1) * db12c34p12 + 2 * sncsm(0, 2) * db12c34p13;
//   
//   // derivative wrt b12 k1
//   double db12k1p11 = db12p11 / k1;
//   double db12k1p12 = db12p12 / (2 * k1);
//   double db12k1p13 = db12p13 / (2 * k1);
//   double db12k1term = sncsm(0, 0) * db12k1p11 + 2 * sncsm(0, 1) * db12k1p12 + 2 * sncsm(0, 2) * db12k1p13;
//   
//   // derivative wrt b12 k2
//   double db12k2term = sncsm(0, 1) * db12p12 / k2;
//   
//   // derivative wrt b12 k3
//   double db12k3term = sncsm(0, 2) * db12p13 / k3;
//   
//   // ------
//   
//   // derivative wrt b13  
//   double db13p11 = 2 * k1 * (b13 + c24 * z);
//   double db13p12 = sqrt(k1 * k2) * (b23 + c24 * y);
//   double db13p13 = - sqrt(k1 * k3) * (1 + c24 * c24);
//   
//   double db13term = sncsm(0, 0) * db13p11  + 2 * sncsm(0, 1) * db13p12 + 2 * sncsm(0, 2) * db13p13;
//   
//   // derivative wrt b13^2
//   double db13b13p11 = 2 * k1 * (1 + c24 * c24);
//   
//   // derivative wrt b13 b23
//   double db13b23p12 = sqrt(k1 * k2) * (1 + c24 * c24);
//   
//   // derivative wrt b13 c24
//   double db13c24p11 = 2 * k1 * (z + c24 * b13);
//   double db13c24p12 = sqrt(k1 * k2) * (y + c24 * b23);
//   double db13c24p13 = - 2 * sqrt(k1 * k3) * c24;
//   
//   double db13c24term = sncsm(0, 0) * db13c24p11 + 2 * sncsm(0, 1) * db13c24p12 + 2 * sncsm(0, 2) * db13c24p13;
//   
//   // derivative wrt b13 c34
//   double db13c34p11 = - 2 * k1 * c24 * b12;
//   double db13c34p12 = sqrt(k1 * k2) * c24;
//   
//   double db13c34term = sncsm(0, 0) * db13c34p11 + 2 * sncsm(0, 1) * db13c34p12;
//   
//   // derivative wrt b13 k1
//   double db13k1p11 = db13p11 / k1;
//   double db13k1p12 = db13p12 / (2 * k1);
//   double db13k1p13 = db13p13 / (2 * k1);
//   
//   double db13k1term = sncsm(0, 0) * db13k1p11 + 2 * sncsm(0, 1) * db13k1p12 + 2 * sncsm(0, 2) * db13k1p13;
//   
//   // derivative wrt b13 k2
//   double db13k2term = sncsm(0, 1) * db13p12 / k2;
//   
//   // derivative wrt b13 k3
//   double db13k3term = sncsm(0, 2) * db13p13 / k3;
//   
//   // ------
//   
//   // derivative wrt b23
//   double db23p12 = sqrt(k1 * k2) * (b13 + c24 * z);
//   double db23p22 = 2 * k2 * (b23 + c24 * y);
//   double db23p23 = - sqrt(k2 * k3) * (1 + c24 * c24);
//   
//   double db23term = 2 * sncsm(0, 1) * db23p12  + sncsm(1, 1) * db23p22 + 2 * sncsm(1, 2) * db23p23;
//   
//   // derivative wrt b23^2
//   double db23b23p22 = 2 * k2 * (1 + c24 * c24);
//   
//   // derivative wrt b23 c24
//   double db23c24p12 = sqrt(k1 * k2) * (z + c24 * b13);
//   double db23c24p22 = 2 * k2 * (y + c24 * b23);
//   double db23c24p23 = - 2 * sqrt(k2 * k3) * c24;
//   
//   double db23c24term = 2 * sncsm(0, 1) * db23c24p12 + sncsm(1, 1) * db23c24p22 + 2 * sncsm(1, 2) * db23c24p23;
//   
//   // derivative wrt b23 c34
//   double db23c34p12 = - sqrt(k1 * k2) * c24 * b12;
//   double db23c34p22 = 2 * k2 * c24;
//   
//   double db23c34term = 2 * sncsm(0, 1) * db23c34p12 + sncsm(1, 1) * db23c34p22;
//   
//   // derivative wrt b23 k1
//   double db23k1term = sncsm(0, 1) * db23p12 / k1;
//   
//   // derivative wrt b23 k2
//   double db23k2p12 = db23p12 / (2 * k2);
//   double db23k2p22 = db23p22 / k2;
//   double db23k2p23 = db23p23 / (2 * k2);
//   
//   double db23k2term = 2 * sncsm(0, 1) * db23k2p12 + sncsm(1, 1) * db23k2p22 + 2 * sncsm(1, 2) * db23k2p23;
//   
//   // derivative wrt b23 k3
//   double db23k3term = sncsm(1, 2) * db23p23 / k3;
//   
//   // ------
//   
//   // derivative wrt c24
//   double dc24p11 = 2 * k1 * (c24 + b13 * z);
//   double dc24p12 = sqrt(k1 * k2) * (b13 * y + b23 * z);
//   double dc24p13 = - sqrt(k1 * k3) * (z + c24 * b13);
//   double dc24p22 = 2 * k2 * b23 * y;
//   double dc24p23 = - sqrt(k2 * k3) * (y + c24 * b23);
//   double dc24p33 = 2 * k3 * c24;
//   
//   double dc24term = sncsm(0, 0) * dc24p11 + 2 * sncsm(0, 1) * dc24p12 + 
//     2 * sncsm(0, 2) * dc24p13 + sncsm(1, 1) * dc24p22 +
//     2 * sncsm(1, 2) * dc24p23 + sncsm(2, 2) * dc24p33;
//   
//   // derivative wrt c24^2
//   double dc24c24p11 = 2 * k1 * (1 + b13 * b13);
//   double dc24c24p12 = 2 * sqrt(k1 * k2) * b13 * b23;
//   double dc24c24p13 = - 2 * sqrt(k1 * k3) * b13;
//   double dc24c24p22 = 2 * k2 * b23 * b23;
//   double dc24c24p23 = - 2 * sqrt(k2 * k3) * b23;
//   double dc24c24p33 = 2 * k3;
//   
//   double dc24c24term = sncsm(0, 0) * dc24c24p11 + 2 * sncsm(0, 1) * dc24c24p12 + 2 * sncsm(0, 2) * dc24c24p13 + 
//     sncsm(1, 1) * dc24c24p22 + 2 * sncsm(1, 2) * dc24c24p23 + sncsm(2, 2) * dc24c24p33;
//   
//   // derivative wrt c24 c34
//   double dc24c34p11 = - 2 * k1 * b13 * b12;
//   double dc24c34p12 = sqrt(k1 * k2) * (b13 - b23 * b12);
//   double dc24c34p13 = sqrt(k1 * k3) * b12;
//   double dc24c34p22 = 2 * k2 * b23;
//   double dc24c34p23 = - sqrt(k2 * k3);
//   
//   double dc24c34term = sncsm(0, 0) * dc24c34p11 + 2 * sncsm(0, 1) * dc24c34p12 + 
//     2 * sncsm(0, 2) * dc24c34p13 + sncsm(1, 1) * dc24c34p22 + 2 * sncsm(1, 2) * dc24c34p23;
//   
//   // derivative wrt c24 k1
//   double dc24k1term = (sncsm(0, 0) * dc24p11 + sncsm(0, 1) * dc24p12 + sncsm(0, 2) * dc24p13) / k1;
//   
//   // derivative wrt c24 k2
//   double dc24k2term = (sncsm(1, 1) * dc24p22 + sncsm(0, 1) * dc24p12 + sncsm(1, 2) * dc24p23) / k2;
//   
//   // derivative wrt c24 k3
//   double dc24k3term = (sncsm(2, 2) * dc24p33 + sncsm(1, 2) * dc24p23 + sncsm(0, 2) * dc24p13) / k3;
//   
//   // ------
//   
//   // derivative wrt c34
//   double dc34p11 = 2 * k1 * (c34 - b12 * z);
//   double dc34p12 = sqrt(k1 * k2) * (z - b12 * y);
//   double dc34p13 = sqrt(k1 * k3) * c24 * b12;
//   double dc34p22 = 2 * k2 * y;
//   double dc34p23 = - sqrt(k2 * k3) * c24;
//   
//   double dc34term = sncsm(0, 0) * dc34p11 + 2 * sncsm(0, 1) * dc34p12 + 
//     2 * sncsm(0, 2) * dc34p13 + sncsm(1, 1) * dc34p22 + 2 * sncsm(1, 2) * dc34p23;
//   
//   // derivative wrt c34^2
//   double dc34c34p11 = 2 * k1 * (1 + b12 * b12);
//   double dc34c34p12 = - 2 * sqrt(k1 * k2) * b12;
//   double dc34c34p22 = 2 * k2;
//   
//   double dc34c34term = sncsm(0, 0) * dc34c34p11 + 2 * sncsm(0, 1) * dc34c34p12 + sncsm(1, 1) * dc34c34p22;
//   
//   // derivative wrt c34 k1
//   double dc34k1term = (sncsm(0, 0) * dc34p11 + sncsm(0, 1) * dc34p12 + sncsm(0, 2) * dc34p13) / k1;
//   
//   // derivative wrt c34 k2
//   double dc34k2term = (sncsm(1, 1) * dc34p22 + sncsm(0, 1) * dc34p12 + sncsm(1, 2) * dc34p23) / k2;
//   
//   // derivative wrt c34 k3
//   double dc34k3term = (sncsm(1, 2) * dc34p23 + sncsm(0, 2) * dc34p13) / k3;
//   
//   // ------
//   
//   // derivative wrt k1
//   
//   double dk1p11 = p11 / k1;
//   double dk1p12 = p12 / (2 * k1);
//   double dk1p13 = p13 / (2 * k1);
//   
//   double dk1term = sncsm(0, 0) * dk1p11 + 2 * sncsm(0, 1) * dk1p12 + 2 * sncsm(0, 2) * dk1p13;
//   
//   // derivative wrt k1^2
//   double dk1k1p12 = - dk1p12 / (2 * k1);
//   double dk1k1p13 = - dk1p13 / (2 * k1);
//   double dk1k1term = 2 * sncsm(0, 1) * dk1k1p12 + 2 * sncsm(0, 2) * dk1k1p13;
//   
//   // derivative wrt k1 k2
//   double dk1k2term = sncsm(0, 1) * dk1p12 / k2;
//   
//   // derivative wrt k1 k3
//   double dk1k3term = sncsm(0, 2) * dk1p13 / k3;
//   
//   // ------
//   
//   // derivative wrt k2
//   
//   double dk2p12 = p12 / (2 * k2);
//   double dk2p22 = p22 / k2;
//   double dk2p23 = p23 / (2 * k2);
//   
//   double dk2term = 2 * sncsm(0, 1) * dk2p12 + sncsm(1, 1) * dk2p22 + 2 * sncsm(1, 2) * dk2p23;
//   
//   // derivative wrt k2^2
//   double dk2k2p12 = - dk2p12 / (2 * k2);
//   double dk2k2p23 = - dk2p23 / (2 * k2);
//   double dk2k2term = 2 * sncsm(0, 1) * dk2k2p12 + 2 * sncsm(1, 2) * dk2k2p23;
//   
//   // derivative wrt k2 k3
//   double dk2k3term = sncsm(1, 2) * dk2p23 / k3;
//   
//   
//   // ------
//   
//   // derivative wrt k3
//   
//   double dk3p13 = p13 / (2 * k3);
//   double dk3p23 = p23 / (2 * k3);
//   double dk3p33 = p33 / k3;
//   
//   double dk3term = 2 * sncsm(0, 2) * dk3p13 + 2 * sncsm(1, 2) * dk3p23 + sncsm(2, 2) * dk3p33;
//   
//   // derivative wrt k3^2
//   double dk3k3p13 = - dk3p13 / (2 * k3);
//   double dk3k3p23 = - dk3p23 / (2 * k3);
//   double dk3k3term = 2 * sncsm(0, 2) * dk3k3p13 + 2 * sncsm(1, 2) * dk3k3p23;
//   
//   // grad[0] = - (sncsm(0, 0) * db12p11 + 2 * sncsm(0, 1) * db12p12 + 2 * sncsm(0, 2) * db12p13) * N / (2 * den);
//   // grad[1] = - (sncsm(0, 0) * db13p11 + 2 * sncsm(0, 1) * db13p12 + 2 * sncsm(0, 2) * db13p13) * N / (2 * den);
//   // grad[2] = - (2 * sncsm(0, 1) * db23p12 + sncsm(1, 1) * db23p22 + 2 * sncsm(1, 2) * db23p23) * N / (2 * den);
//   // grad[3] = - (dc24term * den - term * 2 * c24) * N / (2 * den * den) - c24 * N / den;
//   // grad[4] = - (dc34term * den - term * 2 * c34) * N / (2 * den * den) - c34 * N / den;
//   // grad[5] = ( dk1term * N / (2 * den) - N / (2 * k1) ) * (k1 * k1);
//   // grad[6] = ( dk2term * N / (2 * den) - N / (2 * k2) ) * (k2 * k2);
//   // grad[7] = ( dk3term * N / (2 * den) - N / (2 * k3) ) * (k3 * k3);
//   
//   hess(0, 0) = - sncsm(0, 0) * db12b12p11 * N / (2 * den);
//   hess(0, 1) = hess(1, 0) = - sncsm(0, 0) * db12b13p11 * N / (2 * den);
//   hess(0, 2) = hess(2, 0) = - 2 * sncsm(0, 1) * db12b23p12 * N / (2 * den);
//   hess(0, 3) = hess(3, 0) = - (db12c24term * den - db12term * 2 * c24) * N / (2 * den * den);
//   hess(0, 4) = hess(4, 0) = - (db12c34term * den - db12term * 2 * c34) * N / (2 * den * den);
//   hess(0, 5) = hess(5, 0) = db12k1term * N * k1 * k1 / (2 * den);
//   hess(0, 6) = hess(6, 0) = db12k2term * N * k2 * k2 / (2 * den);
//   hess(0, 7) = hess(7, 0) = db12k3term * N * k3 * k3 / (2 * den);
//   
//   hess(1, 1) = - sncsm(0, 0) * db13b13p11 * N / (2 * den);
//   hess(1, 2) = hess(2, 1) = - 2 * sncsm(0, 1) * db13b23p12 * N / (2 * den);
//   hess(1, 3) = hess(3, 1) = - (db13c24term * den - db13term * 2 * c24) * N / (2 * den * den);
//   hess(1, 4) = hess(4, 1) = - (db13c34term * den - db13term * 2 * c34) * N / (2 * den * den);
//   hess(1, 5) = hess(5, 1) = db13k1term * N * k1 * k1 / (2 * den);
//   hess(1, 6) = hess(6, 1) = db13k2term * N * k2 * k2 / (2 * den);
//   hess(1, 7) = hess(7, 1) = db13k3term * N * k3 * k3 / (2 * den);
//   
//   hess(2, 2) = hess(2, 2) = - sncsm(1, 1) * db23b23p22 * N / (2 * den);
//   hess(2, 3) = hess(3, 2) = - (db23c24term * den - db23term * 2 * c24) * N / (2 * den * den);
//   hess(2, 4) = hess(4, 2) = - (db23c34term * den - db23term * 2 * c34) * N / (2 * den * den);
//   hess(2, 5) = hess(5, 2) = db23k1term * N * k1 * k1 / (2 * den);
//   hess(2, 6) = hess(6, 2) = db23k2term * N * k2 * k2 / (2 * den);
//   hess(2, 7) = hess(7, 2) = db23k3term * N * k3 * k3 / (2 * den);
//   
//   hess(3, 3) = - (dc24c24term * den * den - 2 * term * den - 4 * dc24term * c24 * den + 8 * term * c24 * c24) * N / (2 * den * den * den) - N * (den - 2 * c24 * c24) / (den * den);
//   hess(3, 4) = hess(4, 3) = - (dc24c34term * den * den - 2 * dc24term * c34 * den - 2 * dc34term * c24 * den + 8 * term * c24 * c34) * N / (2 * den * den * den) + 2 * c24 * c34 * N / (den * den);
//   hess(3, 5) = hess(5, 3) = (dc24k1term * den - dk1term * 2 * c24) * N * k1 * k1 / (2 * den * den);
//   hess(3, 6) = hess(6, 3) = (dc24k2term * den - dk2term * 2 * c24) * N * k2 * k2 / (2 * den * den);
//   hess(3, 7) = hess(7, 3) = (dc24k3term * den - dk3term * 2 * c24) * N * k3 * k3 / (2 * den * den);
//   
//   hess(4, 4) = - (dc34c34term * den * den - 2 * term * den - 4 * dc34term * c34 * den + 8 * term * c34 * c34) * N / (2 * den * den * den) - N * (den - 2 * c34 * c34) / (den * den);
//   hess(4, 5) = hess(5, 4) = (dc34k1term * den - dk1term * 2 * c34) * N * k1 * k1 / (2 * den * den);
//   hess(4, 6) = hess(6, 4) = (dc34k2term * den - dk2term * 2 * c34) * N * k2 * k2 / (2 * den * den);
//   hess(4, 7) = hess(7, 4) = (dc34k3term * den - dk3term * 2 * c34) * N * k3 * k3 / (2 * den * den);
//   
//   hess(5, 5) = -(( dk1term * N / (2 * den) - N / (2 * k1) ) * 2 * k1 * k1 * k1 + (dk1k1term * N / (2 * den) + N / (2 * k1 * k1)) * k1 * k1 * k1 * k1);
//   hess(5, 6) = hess(6, 5) = - dk1k2term * N * k1 * k1 * k2 * k2 / (2 * den);
//   hess(5, 7) = hess(7, 5) = - dk1k3term * N * k1 * k1 * k3 * k3 / (2 * den);
//   
//   hess(6, 6) = -(( dk2term * N / (2 * den) - N / (2 * k2) ) * 2 * k2 * k2 * k2 + (dk2k2term * N / (2 * den) + N / (2 * k2 * k2)) * k2 * k2 * k2 * k2);
//   hess(6, 7) = hess(7, 6) = - dk2k3term * N * k2 * k2 * k3 * k3 / (2 * den);
//   
//   hess(7, 7) = -(( dk3term * N / (2 * den) - N / (2 * k3) ) * 2 * k3 * k3 * k3 + (dk3k3term * N / (2 * den) + N / (2 * k3 * k3)) * k3 * k3 * k3 * k3);
//   
//   return hess;
// }
// 
// 
// Eigen::MatrixXd logLikelihoodObservedHessianVarianceScaleFreeConfounders(
//     Eigen::MatrixXd sncsm, int N, 
//     double a12, double a13, double a23, double c24, double c34,
//     double v1, double v2, double v3) {
//   
//   double b12 = a12 * sqrt(v1 / v2);
//   double b13 = a13 * sqrt(v1 / v3);
//   double b23 = a23 * sqrt(v2 / v3);
//   
//   Eigen::VectorXd grad = logLikelihoodObservedGradientVarianceScaleFree(sncsm, N, b12, b13, b23, c24, c34, v1, v2, v3);
//   Eigen::MatrixXd orig = logLikelihoodObservedHessianVarianceScaleFree(sncsm, N, b12, b13, b23, c24, c34, v1, v2, v3);
//   
//   Eigen::MatrixXd hess(8, 8);
//   
//   hess(0, 0) = orig(0, 0) * v1 / v2;
//   hess(0, 1) = hess(1, 0) = orig(0, 1) * v1 / sqrt(v2 * v3);
//   hess(0, 2) = hess(2, 0) = orig(0, 2) * sqrt(v1 / v3);
//   hess(0, 3) = hess(3, 0) = orig(0, 3) * sqrt(v1 / v2);
//   hess(0, 4) = hess(4, 0) = orig(0, 4) * sqrt(v1 / v2);
//   hess(0, 5) = hess(5, 0) = sqrt(v1 / v2) * (orig(0, 5) + orig(0, 0) * b12 / (2 * v1) + orig(0, 1) * b13 / (2 * v1) + grad(0) / (2 * v1));
//   hess(0, 6) = hess(6, 0) = sqrt(v1 / v2) * (orig(0, 6) - orig(0, 0) * b12 / (2 * v2) + orig(0, 2) * b23 / (2 * v2) - grad(0) / (2 * v2));
//   hess(0, 7) = hess(7, 0) = sqrt(v1 / v2) * (orig(0, 7) - orig(0, 1) * b13 / (2 * v3) - orig(0, 2) * b23 / (2 * v3)); 
//   
//   hess(1, 1) = orig(1, 1) * v1 / v3;
//   hess(1, 2) = hess(2, 1) = orig(1, 2) * sqrt(v1 * v2) / v3;
//   hess(1, 3) = hess(3, 1) = orig(1, 3) * sqrt(v1 / v3);
//   hess(1, 4) = hess(4, 1) = orig(1, 4) * sqrt(v1 / v3);
//   hess(1, 5) = hess(5, 1) = sqrt(v1 / v3) * (orig(1, 5) + orig(1, 0) * b12 / (2 * v1) + orig(1, 1) * b13 / (2 * v1) + grad(1) / (2 * v1));
//   hess(1, 6) = hess(6, 1) = sqrt(v1 / v3) * (orig(1, 6) - orig(1, 0) * b12 / (2 * v2) + orig(1, 2) * b23 / (2 * v2));
//   hess(1, 7) = hess(7, 1) = sqrt(v1 / v3) * (orig(1, 7) - orig(1, 1) * b13 / (2 * v3) - orig(1, 2) * b23 / (2 * v3) - grad(1) / (2 * v3));
//   
//   hess(2, 2) = orig(2, 2) * v2 / v3;
//   hess(2, 3) = hess(3, 2) = orig(2, 3) * sqrt(v2 / v3);
//   hess(2, 4) = hess(4, 2) = orig(2, 4) * sqrt(v2 / v3);
//   hess(2, 5) = hess(5, 2) = sqrt(v2 / v3) * (orig(2, 5) + orig(2, 0) * b12 / (2 * v1) + orig(2, 1) * b13 / (2 * v1));
//   hess(2, 6) = hess(6, 2) = sqrt(v2 / v3) * (orig(2, 6) - orig(2, 0) * b12 / (2 * v2) + orig(2, 2) * b23 / (2 * v2) + grad(2) / (2 * v2));
//   hess(2, 7) = hess(7, 2) = sqrt(v2 / v3) * (orig(2, 7) - orig(2, 1) * b13 / (2 * v3) - orig(2, 2) * b23 / (2 * v3) - grad(2) / (2 * v3));
//   
//   hess(3, 3) = orig(3, 3);
//   hess(3, 4) = hess(4, 3) = orig(3, 4);
//   hess(3, 5) = hess(5, 3) = orig(3, 5) + orig(3, 0) * b12 / (2 * v1) + orig(3, 1) * b13 / (2 * v1);
//   hess(3, 6) = hess(6, 3) = orig(3, 6) - orig(3, 0) * b12 / (2 * v2) + orig(3, 2) * b23 / (2 * v2);
//   hess(3, 7) = hess(7, 3) = orig(3, 7) - orig(3, 1) * b13 / (2 * v3) - orig(3, 2) * b23 / (2 * v3);
//   
//   hess(4, 4) = orig(4, 4);
//   hess(4, 5) = hess(5, 4) = orig(4, 5) + orig(4, 0) * b12 / (2 * v1) + orig(4, 1) * b13 / (2 * v1);
//   hess(4, 6) = hess(6, 4) = orig(4, 6) - orig(4, 0) * b12 / (2 * v2) + orig(4, 2) * b23 / (2 * v2);
//   hess(4, 7) = hess(7, 4) = orig(4, 7) - orig(4, 1) * b13 / (2 * v3) - orig(4, 2) * b23 / (2 * v3);
//   
//   hess(5, 5) = orig(5, 5) + (orig(0, 0) * b12 * b12 + orig(1, 1) * b13 * b13 + 2 * orig(0, 1) * b12 * b13 - grad(0) * b12 - grad(1) * b13) / (4 * v1 * v1) + (orig(5, 0) * b12 + orig(5, 1) * b13) / v1;
//   hess(5, 6) = hess(6, 5) = orig(5, 6) + (orig(5, 2) * b23 - orig(5, 0) * b12) / (2 * v2) + (b13 * (orig(1, 6) + (orig(1, 2) * b23 - orig(1, 0) * b12) / (2 * v2)) + b12 * (orig(0, 6) + (orig(0, 2) * b23 - orig(0, 0) * b12) / (2 * v2))) / (2 * v1) - grad(0) * b12 / (4 * v1 * v2);
//   hess(5, 7) = hess(7, 5) = orig(5, 7) - (orig(5, 2) * b23 + orig(5, 1) * b13) / (2 * v3) + (b13 * (orig(1, 7) - (orig(1, 2) * b23 + orig(1, 1) * b13) / (2 * v3)) + b12 * (orig(0, 7) - (orig(0, 2) * b23 + orig(0, 1) * b13) / (2 * v3))) / (2 * v1) - grad(1) * b13 / (4 * v1 * v3);
//   
//   hess(6, 6) = orig(6, 6) + (orig(0, 0) * b12 * b12 + orig(2, 2) * b23 * b23 - 2 * orig(0, 2) * b12 * b23 + 3 * grad(0) * b12 - grad(2) * b23) / (4 * v2 * v2) + (- orig(6, 0) * b12 + orig(6, 2) * b23) / v2;
//   hess(6, 7) = hess(7, 6) = orig(6, 7) - (orig(6, 2) * b23 + orig(6, 1) * b13) / (2 * v3) + (b23 * (orig(2, 7) - (orig(2, 2) * b23 + orig(2, 1) * b13) / (2 * v3)) - b12 * (orig(0, 7) - (orig(0, 2) * b23 + orig(0, 1) * b13) / (2 * v3))) / (2 * v2) - grad(2) * b23 / (4 * v2 * v3);
//   
//   hess(7, 7) = orig(7, 7) + (orig(1, 1) * b13 * b13 + orig(2, 2) * b23 * b23 + 2 * orig(1, 2) * b13 * b23 + 3 * grad(1) * b13 + 3 * grad(2) * b23) / (4 * v3 * v3) - (orig(7, 1) * b13 + orig(7, 2) * b23) / v3;
//   
//   return hess;
// }
// 
// 
// double log_spike_and_slab(double alpha, double vtail, double varrow, double beta = 0.5, double k1 = 1, double k2 = 1e2) {
// 
//   double var1 = varrow / vtail / k1;
//   double var2 = varrow / vtail / k2;
//   
//   if (fabs(beta) < 1e-6) return - 0.5 * (alpha * alpha / var2 + log(2 * M_PI * var2));
// 
//   return log(beta * exp(-(alpha * alpha) / (2 * var1)) / sqrt(2 * M_PI * var1) + (1 - beta) * exp(-(alpha * alpha) / (2 * var2)) / sqrt(2 * M_PI * var2));
// }
// 
// Eigen::MatrixXd getSigma(double a12, double a13, double a23, double a42, double a43, double v1, double v2, double v3, double v4) {
//   Eigen::MatrixXd Sigma(3, 3);
//   
//   std::cout << "Call to getSigma" << std::endl;
//   
//   double a123 = a13 + a12 * a23;
//   double a423 = a43 + a42 * a23;
//   
//   Sigma(0, 0) = v1;
//   Sigma(0, 1) = Sigma(1, 0) = a12 * v1;
//   Sigma(0, 2) = Sigma(2, 0) = a123 * v1;
//   Sigma(1, 1) = a42 * a42 * v4 + a12 * a12 * v1 + v2;
//   Sigma(1, 2) = Sigma(2, 1) = a42 * a423 * v4 + a12 * a123 * v1 + a23 * v2;
//   Sigma(2, 2) = a423 * a423 * v4 + a123 * a123 * v1 + a23 * a23 * v2 + v3;
//   
//   return Sigma;
// }


/******************************************** loglikelihood routine ****************************************************/

// MultiNest wrapper for RoCELL log-likelihood
//
// Input arguments
// ndim 						= dimensionality (total number of free parameters) of the problem
// npars 						= total number of free plus derived parameters
// context						void pointer, any additional information
//
// Input/Output arguments
// Cube[npars] 			= on entry has the ndim parameters in unit-hypercube
//	 						on exit, the physical parameters plus copy any derived parameters you want to store with the free parameters
//	 
// Output arguments
// lnew 						= loglikelihood


void LogLike(double *Cube, int &ndim, int &npars, double &lnew, void *context)
{
  // Read Config structure
  Config* config = (Config*) context;
   
  // Read scaled confounding coefficients, which are N(0, 1) distributed.
  double sc24 = quantile(dist, Cube[0]);
  double sc34 = quantile(dist, Cube[1]);


  arma::vec AV = get_params(config->covariance, sc24, sc34);
  
  // std::cout << "AV: " << AV << std::endl << std::endl;

  arma::mat hess = log_likelihood_hessian_scale_free_confounders(config->covariance, 1, AV[0], AV[1], AV[2], sc24, sc34, AV[3], AV[4], AV[5]);
 
  arma::mat negHessReduced(6, 6);

  // std::cout << "before block" << std::endl;

  negHessReduced.submat(0, 0, 2, 2) = - hess.submat(0, 0, 2, 2);
  negHessReduced.submat(3, 0, 5, 2) = - hess.submat(5, 0, 7, 2);
  negHessReduced.submat(0, 3, 2, 5) = - hess.submat(0, 5, 2, 7);
  negHessReduced.submat(3, 3, 5, 5) = - hess.submat(5, 5, 7, 7);
  
  // std::cout << hess << std::endl;
  // std::cout << negHessReduced << std::endl;

  double ld = 0;
  arma::mat U = chol(negHessReduced, "lower");
  for (unsigned i = 0; i < U.n_rows; ++i)
    ld += log(U(i,i));
  ld *= 2;
  
  // TODO: Improve model handling
  double w32 = -1;
  if (config->model == 0) w32 = 1;
  else if (config->model == 1) w32 = 0;
  else if (config->model == 2) w32 = 0.5;

  lnew = - ld / 2 +
    log_spike_and_slab(AV[0], AV[3], AV[4], 0.5, config->slab_precision, config->spike_precision) +
    log_spike_and_slab(AV[1], AV[3], AV[5], 0.5, config->slab_precision, config->spike_precision) +
    log_spike_and_slab(AV[2], AV[4], AV[5], w32, config->slab_precision, config->spike_precision) +
    (- log(AV[3]) - log(AV[4]) - log(AV[5])); //+
}

/***********************************************************************************************************************/




/************************************************* dumper routine ******************************************************/

// The dumper routine will be called every updInt*10 iterations
// MultiNest doesn not need to the user to do anything. User can use the arguments in whichever way he/she wants
//
//
// Arguments:
//
// nSamples 						= total number of samples in posterior distribution
// nlive 						= total number of live points
// nPar 						= total number of parameters (free + derived)
// physLive[1][nlive * (nPar + 1)] 			= 2D array containing the last set of live points (physical parameters plus derived parameters) along with their loglikelihood values
// posterior[1][nSamples * (nPar + 2)] 			= posterior distribution containing nSamples points. Each sample has nPar parameters (physical + derived) along with the their loglike value & posterior probability
// paramConstr[1][4*nPar]:
// paramConstr[0][0] to paramConstr[0][nPar - 1] 	= mean values of the parameters
// paramConstr[0][nPar] to paramConstr[0][2*nPar - 1] 	= standard deviation of the parameters
// paramConstr[0][nPar*2] to paramConstr[0][3*nPar - 1] = best-fit (maxlike) parameters
// paramConstr[0][nPar*4] to paramConstr[0][4*nPar - 1] = MAP (maximum-a-posteriori) parameters
// maxLogLike						= maximum loglikelihood value
// logZ							= log evidence value from the default (non-INS) mode
// INSlogZ						= log evidence value from the INS mode
// logZerr						= error on log evidence value
// context						void pointer, any additional information

void dumper(int &nSamples, int &nlive, int &nPar, double **physLive, double **posterior, double **paramConstr, double &maxLogLike, double &logZ, double &INSlogZ, double &logZerr, void *context)
{
	// convert the 2D Fortran arrays to C++ arrays
	
	
	// the posterior distribution
	// postdist will have nPar parameters in the first nPar columns & loglike value & the posterior probability in the last two columns
	
	int i, j;
	
	double postdist[nSamples][nPar + 2];
	for( i = 0; i < nPar + 2; i++ )
		for( j = 0; j < nSamples; j++ )
			postdist[j][i] = posterior[0][i * nSamples + j];
	
	
	
	// last set of live points
	// pLivePts will have nPar parameters in the first nPar columns & loglike value in the last column
	
	double pLivePts[nlive][nPar + 1];
	for( i = 0; i < nPar + 1; i++ )
		for( j = 0; j < nlive; j++ )
			pLivePts[j][i] = physLive[0][i * nlive + j];
}

/***********************************************************************************************************************/




/************************************************** Main program *******************************************************/



int main(int argc, char *argv[])
{
  // basic check for command line arguments
  if (argc < 9) {
    std::cout << "Missing command line arguments!" << std::endl;
    std::exit(1);
  }  



	// set the MultiNest sampling parameters
	int IS = 0;					// do Nested Importance Sampling?
	
	int mmodal = 1;					// do mode separation?
	
	int ceff = 0;					// run in constant efficiency mode?
	
	int nlive = atoi(argv[1]);				// number of live points
	
	double efr = 0.8;				// set the required efficiency
	
	double tol = 0.5;				// tol, defines the stopping criteria
	
	int ndims = 2;					// dimensionality (no. of free parameters)
	
	int nPar = 2;					// total no. of parameters including free & derived parameters
	
	int nClsPar = 2;				// no. of parameters to do mode separation on
	
	int updInt = 1000;				// after how many iterations feedback is required & the output files should be updated
							// note: posterior files are updated & dumper routine is called after every updInt*10 iterations
	
	double Ztol = -1E90;				// all the modes with logZ < Ztol are ignored
	
	int maxModes = 100;				// expected max no. of modes (used only for memory allocation)
	
	int pWrap[ndims];				// which parameters to have periodic boundary conditions?
	for(int i = 0; i < ndims; i++) pWrap[i] = 0;
	
  char root[100] = "output_RoCELL-";			// root for output files

	int seed = -1;					// random no. generator seed, if < 0 then take the seed from system clock
	
	int fb = 1;					// need feedback on standard output?
	
	int resume = 0;					// resume from a previous job?
	
	int outfile = 1;				// write output files?
	
	int initMPI = 1;				// initialize MPI routines?, relevant only if compiling with MPI
							// set it to F if you want your main program to handle MPI initialization
	
	double logZero = -1E90;				// points with loglike < logZero will be ignored by MultiNest
	
	int maxiter = 0;				// max no. of iterations, a non-positive value means infinity. MultiNest will terminate if either it 
							// has done max no. of iterations or convergence criterion (defined through tol) has been satisfied
	
  // double b21 = strtod(argv[3], NULL);
  // double b31 = strtod(argv[4], NULL);
  // double b32 = strtod(argv[5], NULL);
  // double c24 = strtod(argv[6], NULL);
  // double c34 = strtod(argv[7], NULL);

  Config config;
  
  config.model = atoi(argv[2]);
  config.slab_precision = strtod(argv[3], NULL);
  config.spike_precision = strtod(argv[4], NULL);
  
  // config.covariance = getSigma(b21, b31, b32, c24, c34, 1, 1, 1, 1);
  config.covariance = arma::mat(3, 3, arma::fill::zeros);
  config.covariance(0, 0) = strtod(argv[5], NULL);
  config.covariance(0, 1) = config.covariance(1, 0) = strtod(argv[6], NULL);
  config.covariance(0, 2) = config.covariance(2, 0) = strtod(argv[7], NULL);
  config.covariance(1, 1) = strtod(argv[8], NULL);
  config.covariance(1, 2) = config.covariance(2, 1) = strtod(argv[9], NULL);
  config.covariance(2, 2) = strtod(argv[10], NULL);

//   if (reverse) {
//     Eigen::MatrixXd temp = config.covariance;
// 	  config.covariance(1, 1) = temp(2, 2);
// 	  config.covariance(2, 2) = temp(1, 1);
// 	  config.covariance(1, 2) = temp(2, 1);
// 	  config.covariance(2, 1) = temp(1, 2);
// 	  config.covariance(0, 2) = temp(0, 1);
// 	  config.covariance(0, 1) = temp(0, 2);
// 	  config.covariance(2, 0) = temp(1, 0);
// 	  config.covariance(1, 0) = temp(2, 0);
//   }


  std::cout << config.covariance << std::endl;

	
	void *context = &config;				// not required by MultiNest, any additional information user wants to pass

	
	double C[2] = {0.5, 0.5};
	double ll = -1.0;
	 
	LogLike(C, ndims, nPar, ll, context);
	std::cout << ll << std::endl;

	C[0] = 0.25;
	LogLike(C, ndims, nPar, ll, context);
	std::cout << ll << std::endl;
	
	// calling MultiNest
  nested::run(IS, mmodal, ceff, nlive, tol, efr, ndims, nPar, nClsPar, maxModes, updInt, Ztol, root, seed, pWrap, fb, resume, outfile, initMPI, logZero, maxiter, LogLike, dumper, context);

}

/***********************************************************************************************************************/



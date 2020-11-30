#pragma once
#include <ros/assert.h>
#include <iostream>
#include <eigen3/Eigen/Dense>

#include "../utility/utility.h"
#include "../estimator/parameters.h"
#include "../estimator/estimator.h"
#include "integration_base.h"
#include <ceres/ceres.h>

// class uwb_has_initialed : public ceres::SizedCostFunction<1, 7, 9>
// {
//   public:
//     uwb_has_initialed(double td, double uwb_t, const Eigen::Vector3d &result_uwb_p, const Eigen::Vector3d &temp_a, const Eigen::Vector3d &g)
//     {
//     	d = td;
//         t = uwb_t;
//         tem_result_uwb_p = result_uwb_p;
//         tem_a = temp_a;
//         temp_g = g;
//     }
//     virtual bool Evaluate(double const *paraPose, double const *paraSpeedBias, double residuals, double **jacobians) const
//     {
//         Eigen::Quaterniond Qi(paraPose[6], paraPose[3], paraPose[4], paraPose[5]);
//         Eigen::Vector3d Pi(paraPose[0], paraPose[1], paraPose[2]);
//         Eigen::Vector3d Vi(paraSpeedBias[0], paraSpeedBias[1], paraSpeedBias[2]);
//         Matrix3d r = Qi.toRotationMatrix();
//         Vector3d P = Pi + Vi * t - 0.5 * temp_g * t * t + r * tem_a - tem_result_uwb_p;
//         double residual(residuals);
//         residual = d - P.norm();
//         Eigen::Map<Eigen::Matrix<double, 1, 1, Eigen::RowMajor>> jacobian_bias(jacobians[0]);
//         jacobians = 
//     	return true;
//     }

//     private:
//         double d;
//         Vector3d tem_result_uwb_p;
//         Vector3d tem_a;
//         double* apha;
//         double t;
//         Vector3d temp_g;

    
// };
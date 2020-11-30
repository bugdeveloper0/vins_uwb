#include <ros/assert.h>
#include <iostream>
#include <eigen3/Eigen/Dense>

#include "../utility/utility.h"
#include "../estimator/parameters.h"
#include "../estimator/estimator.h"
#include "integration_base.h"
#include <ceres/ceres.h>
struct uwb_initialing
{
  private:
    double d;
    Vector3d position;
    double uwb_err;
  public:
    uwb_initialing(double Dd, Eigen::Vector3d &Position, double uwb_err)
    {
      d = Dd;
      position = Position;
      uwb_err = uwb_err;
    }
    template <typename T>
    // 在里面需要对相机的p和v进行优化吗,为什么有两个const，不是要优化吗
    bool operator()(const T * uwb_p, T *residual) const
    {
        Eigen::Matrix<T, 3, 1> uwb_p1(uwb_p[0], uwb_p[1], uwb_p[2]);
        Eigen::Matrix<T, 3, 1> position1((T)position[0], (T)position[1], (T)position[2]);
        // residuals[0] = (T)1;
        residual[0] = T(10) * (T(d) - T((uwb_p1 - position1).norm())) / (T(uwb_err) + T(0.001));
        // residual[0] = T(10) * (T(d) - T((uwb_p1 - position1).norm()));
        return true;
    }
    /*     static ceres::CostFunction *Creat(const double d, Vector3d p)
    {
        return (new ceres::AutoDiffCostFunction<uwb_initialing, 1, 3>(
            new uwb_initialing(d, p)));
    } */
};
struct uwb_has_initialed
{
  private:
    double d;
    Vector3d temp_result_uwb_p;
    Vector3d tem_a;
    double* apha;
    double t;
    Vector3d temp_g;

  public:
    uwb_has_initialed(double td, double uwb_t, Eigen::Vector3d &result_uwb_p, const Eigen::Vector3d &temp_a, const Eigen::Vector3d &g)
        //: d(td), t(uwb_t), apha(a), temp_g(tem_g)
    {
      d = td;
      t = uwb_t;
      temp_result_uwb_p = result_uwb_p;
      tem_a = temp_a;
      temp_g = g;
    }

    template <typename T>
    bool operator()(const T * paraPose, const T * paraSpeedBias, T *residuals) const
    {
        // // r和a相乘得到预计分的值
        Eigen::Quaternion<T> Qi(paraPose[6], paraPose[3], paraPose[4], paraPose[5]);
        Eigen::Matrix<T, 3, 1> Pi(paraPose[0], paraPose[1], paraPose[2]);
        Eigen::Matrix<T, 3, 1> Vi(paraSpeedBias[0], paraSpeedBias[1], paraSpeedBias[2]);
        Eigen::Matrix<T, 3, 1> tmp_a((T)tem_a[0], (T)tem_a[1], (T)tem_a[2]);
        // Eigen::Matrix<T, 3, 1> Gg((T)temp_g[0], (T)temp_g[1], (T)temp_g[2]);
        // Eigen::Matrix<T, 3, 1> tmp_result_uwb_p((T)temp_result_uwb_p[0], (T)temp_result_uwb_p[1], (T)temp_result_uwb_p[2]);
        //Eigen::Matrix<T, 3, 3> r = Qi.toRotationMatrix();
        Eigen::Matrix<T, 3, 1> P;
        Eigen::Matrix<T, 3, 1> rA;
        rA = Qi * tmp_a;
        P.x() = Pi.x() + t * Vi.x() - 0.5 * temp_g.x() *t + rA.x() - temp_result_uwb_p.x();
        // cout << "shijiweizhiX" << (T)Pi.x() << endl;
        P.y() = Pi.y() + t * Vi.y() - 0.5 * temp_g.y() *t + rA.y() - temp_result_uwb_p.y();
        // cout << "shijiweizhiY" << (T)Pi.y() << endl;
        P.z() = Pi.z() + t * Vi.z() - 0.5 * temp_g.z() *t + rA.z() - temp_result_uwb_p.z();
        // cout << "shijiweizhiZ" << (T)Pi.z() << endl;
        // P = Pi + Vi * T(t) - 0.5 * Gg * T(t) * T(t) + rA - tmp_result_uwb_p;
        
        residuals[0] =  (T)10*(T(d) - T(P.norm()));
        // if(ceres::IsNaN(residuals[0]))
        // {
        //   residuals[0] = T(0);
        //   cout <<"butaimiao" << endl;
        // }
        // cout << "liushu" << residuals[0];
        // residuals[0] = T(0);
        return true;
    }

};
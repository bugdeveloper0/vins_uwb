/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "estimator.h"
#include "../utility/visualization.h"



Estimator::Estimator() : f_manager{Rs}
{
    ROS_INFO("init begins");
    initThreadFlag = false;
    clearState();
}

Estimator::~Estimator()
{
    if (MULTIPLE_THREAD)
    {
        processThread.join();
        printf("join thread \n");
    }
}

void Estimator::clearState()
{
    mProcess.lock();
    while (!accBuf.empty())
        accBuf.pop();
    while (!gyrBuf.empty())
        gyrBuf.pop();
    while (!featureBuf.empty())
        featureBuf.pop();

    prevTime = -1;
    curTime = 0;
    openExEstimation = 0;
    initP = Eigen::Vector3d(0, 0, 0);
    initR = Eigen::Matrix3d::Identity();
    inputImageCnt = 0;
    initFirstPoseFlag = false;
    cout << "qingchuzhuangtai" << endl;
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
        {
            delete pre_integrations[i];
        }
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;

    uwb_initialed = false;
    result_uwb_p.setZero();
    uwb_p[0] = -3;
    uwb_p[1] = 10;
    uwb_p[2] = 5;
    lastd = 0;
    latestd = 0;
    lastp.setZero();
    latestp.setZero();
    tmp_p_means.setZero();
    d_means = 0;
    uwb_err_means = 0;
    means_count = 0;

    initial_timestamp = 0;
    all_image_frame.clear();

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;

    mProcess.unlock();
}

void Estimator::setParameter()
{
    mProcess.lock();
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
        cout << " exitrinsic cam " << i << endl
             << ric[i] << endl
             << tic[i].transpose() << endl;
    }
    f_manager.setRic(ric);
    ProjectionTwoFrameOneCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTwoFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionOneFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
    g = G;
    cout << "set g " << g.transpose() << endl;
    featureTracker.readIntrinsicParameter(CAM_NAMES);

    std::cout << "MULTIPLE_THREAD is " << MULTIPLE_THREAD << '\n';
    if (MULTIPLE_THREAD && !initThreadFlag)
    {
        initThreadFlag = true;
        processThread = std::thread(&Estimator::processMeasurements, this);
    }
    mProcess.unlock();
}

void Estimator::changeSensorType(int use_imu, int use_stereo)
{
    bool restart = false;
    mProcess.lock();
    if (!use_imu && !use_stereo)
        printf("at least use two sensors! \n");
    else
    {
        if (USE_IMU != use_imu)
        {
            USE_IMU = use_imu;
            if (USE_IMU)
            {
                // reuse imu; restart system
                restart = true;
            }
            else
            {
                if (last_marginalization_info != nullptr)
                    delete last_marginalization_info;

                tmp_pre_integration = nullptr;
                last_marginalization_info = nullptr;
                last_marginalization_parameter_blocks.clear();
            }
        }

        STEREO = use_stereo;
        printf("use imu %d use stereo %d\n", USE_IMU, STEREO);
    }
    mProcess.unlock();
    if (restart)
    {
        clearState();
        setParameter();
    }
}

void Estimator::inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1)
{
    inputImageCnt++;
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    TicToc featureTrackerTime;

    if (_img1.empty())
        featureFrame = featureTracker.trackImage(t, _img);
    else
        featureFrame = featureTracker.trackImage(t, _img, _img1);
    //printf("featureTracker time: %f\n", featureTrackerTime.toc());

    if (SHOW_TRACK)
    {
        cv::Mat imgTrack = featureTracker.getTrackImage();
        pubTrackImage(imgTrack, t);
    }

    if (MULTIPLE_THREAD)
    {
        if (inputImageCnt % 2 == 0)
        {
            mBuf.lock();
            featureBuf.push(make_pair(t, featureFrame));
            mBuf.unlock();
        }
    }
    else
    {
        mBuf.lock();
        featureBuf.push(make_pair(t, featureFrame));
        mBuf.unlock();
        TicToc processTime;
        processMeasurements();
        printf("process time: %f\n", processTime.toc());
    }
}

void Estimator::inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity)
{
    mBuf.lock();
    accBuf.push(make_pair(t, linearAcceleration));
    gyrBuf.push(make_pair(t, angularVelocity));
    
    //printf("input imu with time %f \n", t);
    mBuf.unlock();

    if (solver_flag == NON_LINEAR)
    {
        mPropagate.lock();
        fastPredictIMU(t, linearAcceleration, angularVelocity);
        pubLatestOdometry(latest_P, latest_Q, latest_V, t);
        mPropagate.unlock();
    }
}

void Estimator::inputUWB(double t, double d, int Id, double uwb_err)
{
    if (Id == 202)
    {
    mBuf.lock();
    uwbBuf.push(make_pair(make_pair(t, d), uwb_err));
    cout << "t" << t << "   d" << d << "    uwb_err" << uwb_err << endl;
    mBuf.unlock();
    }
}

void Estimator::inputFeature(double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame)
{
    mBuf.lock();
    featureBuf.push(make_pair(t, featureFrame));
    mBuf.unlock();

    if (!MULTIPLE_THREAD)
        processMeasurements();
}

bool Estimator::getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector,
                               vector<pair<double, Eigen::Vector3d>> &gyrVector,
                               queue<pair<double, Eigen::Vector3d>> &accBuf_copy, queue<pair<double, Eigen::Vector3d>> &gyrBuf_copy,
                               vector<vector<pair<double, Eigen::Vector3d>>> &uwb_accVector,
                               vector<vector<pair<double, Eigen::Vector3d>>> &uwb_gyrVector,
                               queue<pair<pair<double, double>, double>> &uwb_d)
{
    if (accBuf.empty())
    {
        printf("not receive imu\n");
        return false;
    }
    if (t1 <= accBuf.back().first)
    {
        while (accBuf.front().first <= t0)
        {
            accBuf.pop();
            gyrBuf.pop();
        }
        accBuf_copy1 = accBuf;
        gyrBuf_copy1 = gyrBuf;
        while (accBuf.front().first < t1)
        {
            accVector.push_back(accBuf.front());
            accBuf.pop();
            gyrVector.push_back(gyrBuf.front());
            gyrBuf.pop();
        }
        accVector.push_back(accBuf.front());
        gyrVector.push_back(gyrBuf.front());


        //我加的
        if (uwbBuf.empty())
        {
            printf("not receive uwb\n");
            return false;
        }
        while (uwbBuf.front().first.first <= t0)
        {
            uwbBuf.pop();
        }

        size_t u = uwbBuf.size();
        cout << "处理前uwb大小" << u << endl;
        vector<pair<double, Eigen::Vector3d>> tem_uwb_accVector[u];
        vector<pair<double, Eigen::Vector3d>> tem_uwb_gyrVector[u];
        for (size_t i = 0; i < u; i++)
        {
            accBuf_copy = accBuf_copy1;
            gyrBuf_copy = gyrBuf_copy1;
            if (uwbBuf.front().first.first < t1)
            {
                while (accBuf_copy.front().first < uwbBuf.front().first.first)
                {
                    tem_uwb_accVector[i].push_back(accBuf_copy.front());
                    accBuf_copy.pop();
                    tem_uwb_gyrVector[i].push_back(gyrBuf_copy.front());
                    gyrBuf_copy.pop();
                }
                tem_uwb_accVector[i].push_back(accBuf_copy.front());
                tem_uwb_gyrVector[i].push_back(gyrBuf_copy.front());
                uwb_accVector.push_back(tem_uwb_accVector[i]);
                uwb_gyrVector.push_back(tem_uwb_gyrVector[i]);
                uwb_d.push(uwbBuf.front());
                uwbBuf.pop();
            }
        }
    }
    else
    {
        printf("wait for imu\n");
        return false;
    }
}


bool Estimator::IMUAvailable(double t)
{
    if (!accBuf.empty() && t <= accBuf.back().first)
        return true;
    else
        return false;
}

void Estimator::processMeasurements()
{
    while (1)
    {
        //printf("process measurments\n");
        pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>> feature;
        vector<pair<double, Eigen::Vector3d>> accVector, gyrVector;
        queue<pair<pair<double, double>, double>> uwb_d;
        vector<vector<pair<double, Eigen::Vector3d>>> uwb_accVector, uwb_gyrVector;
        if (!featureBuf.empty())
        {
            feature = featureBuf.front();
            curTime = feature.first + td;
            while (1)
            {
                if ((!USE_IMU || IMUAvailable(feature.first + td)))
                    break;
                else
                {
                    printf("wait for imu ... \n");
                    if (!MULTIPLE_THREAD)
                        return;
                    std::chrono::milliseconds dura(5);
                    std::this_thread::sleep_for(dura);
                }
            }
            
            mBuf.lock();
            if (USE_IMU)
            {
                getIMUInterval(prevTime, curTime, accVector, gyrVector, accBuf_copy, gyrBuf_copy, uwb_accVector, uwb_gyrVector, uwb_d);

                // getUWBInterval(prevTime, curTime, uwb_accVector, uwb_gyrVector, uwb_d);

            }
            featureBuf.pop();
            mBuf.unlock(); //关于线程的知识我还不知道  uwb要不要开一个线程



            if (USE_IMU)
            {
                if (!initFirstPoseFlag)
                    initFirstIMUPose(accVector);    //这个地方的初始化是对整个程序的初始化吗  而不是对每一次循环的初始化
                for (size_t i = 0; i < accVector.size(); i++)
                {
                    double dt;
                    if (i == 0)
                        dt = accVector[i].first - prevTime;
                    else if (i == accVector.size() - 1)
                        dt = curTime - accVector[i - 1].first;
                    else
                        dt = accVector[i].first - accVector[i - 1].first;
                    processIMU(accVector[i].first, dt, accVector[i].second, gyrVector[i].second);
                }
            }
            

            if (USE_IMU)
            {
                size_t u = uwb_accVector.size();
                IntegrationBase *tem_uwb_pre_integrations[u];
                std::vector<double> tem_uwb_dt_buf[u];
                std::vector<Vector3d>  tem_uwb_linear_acceleration_buf[u];
                std::vector<Eigen::Vector3d>  tem_uwb_angular_velocity_buf[u];
                vector<J_uwb> tem_uwb_vector;
                /* vector<IntegrationBase *> temp_uwb_pre_integrations;
                vector<std::vector<double>> temp_uwb_dt_buf;
                vector<std::vector<Eigen::Vector3d>> temp_uwb_linear_acceleration_buf;
                vector<std::vector<Eigen::Vector3d>> temp_uwb_angular_velocity_buf; */
                queue<pair<pair<double, double>, double>> uwbBuf_copy1 = uwb_d;
                queue<pair<pair<double, double>, double>> uwbBuf_copy = uwb_d;
                Vector3d uwb_Ps[u];
                Vector3d uwb_Vs[u];
                Matrix3d uwb_Rs[u];
                for (size_t i = 0; i < u; i++)
                {
                    uwb_acc_0 = accVector[0].second;
                    uwb_gyr_0 = gyrVector[0].second;
                    uwb_Ps[i] = Ps[frame_count]; //这种定义对吗
                    uwb_Vs[i] = Vs[frame_count];
                    uwb_Rs[i] = Rs[frame_count];
                    tem_uwb_pre_integrations[i] = new IntegrationBase{uwb_acc_0, uwb_gyr_0, Bas[frame_count], Bgs[frame_count]};
                    for (size_t j = 0; j < uwb_accVector[i].size(); j++)
                    {
                        double dt;
                        if (j == 0)
                            dt = uwb_accVector[i][j].first - prevTime;
                        else if (j == uwb_accVector[i].size() - 1)
                        {
                            dt = uwbBuf_copy1.front().first.first - uwb_accVector[i][j - 1].first;
                            uwbBuf_copy1.pop();
                        }
                        else
                            dt = uwb_accVector[i][j].first - uwb_accVector[i][j - 1].first;

                        //这里的uwb_pre_integrations不会定义   因为i这个值暂时不能确定  这是个比较麻烦的问题

                        if (frame_count != 0)
                        {
                            tem_uwb_pre_integrations[i]->push_back(dt, uwb_accVector[i][j].second, uwb_gyrVector[i][j].second);
                            //if(solver_flag != NON_LINEAR)
                            tem_uwb_dt_buf[i].push_back(dt);
                            tem_uwb_linear_acceleration_buf[i].push_back(uwb_accVector[i][j].second);
                            tem_uwb_angular_velocity_buf[i].push_back(uwb_gyrVector[i][j].second);
                            Vector3d uwb_un_acc_0 = uwb_Rs[i] * (uwb_acc_0 - Bas[frame_count]) - g;
                            Vector3d uwb_un_gyr = 0.5 * (uwb_gyr_0 + uwb_gyrVector[i][j].second) - Bgs[frame_count];
                            uwb_Rs[i] *= Utility::deltaQ(uwb_un_gyr * dt).toRotationMatrix();
                            Vector3d uwb_un_acc_1 = uwb_Rs[i] * (uwb_accVector[i][j].second - Bas[frame_count]) - g;
                            Vector3d uwb_un_acc = 0.5 * (uwb_un_acc_0 + uwb_un_acc_1);
                            uwb_Ps[i] += dt * uwb_Vs[i] + 0.5 * dt * dt * uwb_un_acc;
                            uwb_Vs[i] += dt * uwb_un_acc;
                        }
                        uwb_acc_0 = uwb_accVector[i][j].second;
                        uwb_gyr_0 = uwb_gyrVector[i][j].second;
                    }
                    // cout << "puanduan" << endl << uwbBuf_copy.front().second << endl << tem_uwb_pre_integrations[i]->sum_dt << endl
                    // cout << "所有的uwb约束pvd" << endl << uwb_Ps[i] << endl <<  uwb_Vs[i] << "   uwbjuli" << uwbBuf_copy.front().second << endl;
                    struct J_uwb a = {uwbBuf_copy.front().first.second, tem_uwb_pre_integrations[i]->sum_dt, uwb_Ps[i], uwb_Vs[i], uwb_Rs[i],
                                      tem_uwb_pre_integrations[i]->delta_p};

                    
                    struct init_J_uwb init = {uwbBuf_copy.front().first.second, uwb_Ps[i], uwbBuf_copy.front().second};
                    tmp_init_uwb_vector.push_back(init);
                    size_t m = tmp_init_uwb_vector.size();
                    if (m == 5)
                    {
                        double d = 0;
                        for (size_t i = 0; i < m; i++)
                        {
                            d += tmp_init_uwb_vector[i].d;
                        }
                        d = d / m;
                        for (size_t i = 0; i < m; i++)
                        {
                            if(abs(d - tmp_init_uwb_vector[i].d) > 0.5)
                            {
                                tmp_init_uwb_vector[i].d = -1;
                            }
                        }
                        for (size_t i = 0; i < m; i++)
                        {
                            if(tmp_init_uwb_vector[i].d > 0)
                            {
                                tmp_init_uwb_vector1.push(tmp_init_uwb_vector[i]);
                            }
                        }
                        tmp_init_uwb_vector.clear();
                    }

                    latestp = tmp_init_uwb_vector1.front().p;
                    if (!uwb_initialed)
                    {
                        cout << "标定时uwb约束pv" << endl
                             << uwb_Ps[i] << endl
                             << uwb_Vs[i] << endl;
                        cout << "锚点优化前d和位置" << endl
                             << uwbBuf_copy.front().second << endl
                             << uwb_Ps[i] << endl;
                        if (!tmp_init_uwb_vector1.empty())
                        {
                            d_means += tmp_init_uwb_vector1.front().d;
                            tmp_p_means.x() += tmp_init_uwb_vector1.front().p.x();
                            tmp_p_means.y() += tmp_init_uwb_vector1.front().p.y();
                            tmp_p_means.z() += tmp_init_uwb_vector1.front().p.z();
                            if (uwb_err_means < tmp_init_uwb_vector1.front().uwb_err)
                            {
                                uwb_err_means = tmp_init_uwb_vector1.front().uwb_err;
                            }
                            means_count++;
                            tmp_init_uwb_vector1.pop();
                            if ((lastp - latestp).norm() > 0.1)
                            {
                                d_means = d_means / means_count;
                                tmp_p_means.x() = tmp_p_means.x() / means_count;
                                tmp_p_means.y() = tmp_p_means.y() / means_count;
                                tmp_p_means.z() = tmp_p_means.z() / means_count;
                                struct init_J_uwb init = {d_means, tmp_p_means, uwb_err_means};
                                lastp = latestp;
                                init_uwb_vector.push_back(init);
                                tmp_p_means.setZero();
                                d_means = 0;
                                uwb_err_means = 0;
                                means_count = 0;
                            }
                        }
                    }
                    else
                    {
                        init_uwb_vector.clear();
                    }


                    // latestd = uwb_Ps[i].norm();
                    // if (!uwb_initialed)
                    // {
                    //     if (abs(lastd - latestd) > 0.1)
                    //     { 
                    //         struct init_J_uwb init = {uwbBuf_copy.front().first.second, uwb_Ps[i], uwbBuf_copy.front().second};
                    //         init_uwb_vector.push_back(init);
                    //         lastd = latestd;
                    //     }
                    // }
                    // else
                    // {
                    //     init_uwb_vector.clear();
                    // }



                    if (isnan(tem_uwb_pre_integrations[i]->delta_p.x()))
                    {
                        cout << "chushile" << endl;
                    }
                    uwb_vector[frame_count].push_back(a);                                     //这个用来判断锚点是否合格;
                    uwb_pre_integrations[frame_count].push_back(tem_uwb_pre_integrations[i]); //这个作用域怎么搞  出去岂不是不能赋值
                    uwb_dt_buf[frame_count].push_back(tem_uwb_dt_buf[i]);
                    uwb_linear_acceleration_buf[frame_count].push_back(tem_uwb_linear_acceleration_buf[i]);
                    uwb_angular_velocity_buf[frame_count].push_back(tem_uwb_angular_velocity_buf[i]);
                    uwbBuf_copy.pop();
                }
                // cout << "frame_count是" << frame_count << endl;
                // uwb_vector[frame_count].push_back(tem_uwb_vector);
                // uwb_pre_integrations[frame_count].push_back(temp_uwb_pre_integrations);
                // uwb_dt_buf[frame_count].push_back(temp_uwb_dt_buf);
                // uwb_linear_acceleration_buf[frame_count].push_back(temp_uwb_linear_acceleration_buf);
                // uwb_angular_velocity_buf[frame_count].push_back(temp_uwb_angular_velocity_buf);
                // tem_uwb_vector.clear();
            }

            mProcess.lock();
            
            processImage(feature.second, feature.first);
            prevTime = curTime;
            // cout << "ceshi8" << endl;
            printStatistics(*this, 0);

            std_msgs::Header header;
            header.frame_id = "world";
            header.stamp = ros::Time(feature.first);

            pubOdometry(*this, header);
            pubKeyPoses(*this, header);
            pubCameraPose(*this, header);
            pubPointCloud(*this, header);
            pubKeyframe(*this);
            pubTF(*this, header);
            mProcess.unlock();
        }

        if (!MULTIPLE_THREAD)
            break;

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}



void Estimator::initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector)
{
    printf("init first imu pose\n");
    initFirstPoseFlag = true;
    //return;
    Eigen::Vector3d averAcc(0, 0, 0);
    int n = (int)accVector.size();
    for (size_t i = 0; i < accVector.size(); i++)
    {
        averAcc = averAcc + accVector[i].second;
    }
    averAcc = averAcc / n;
    printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());
    Matrix3d R0 = Utility::g2R(averAcc);
    double yaw = Utility::R2ypr(R0).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    Rs[0] = R0;
    cout << "init R0 " << endl
         << Rs[0] << endl;
    //Vs[0] = Vector3d(5, 0, 0);
}

void Estimator::initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r)
{
    Ps[0] = p;
    Rs[0] = r;
    initP = p;
    initR = r;
}

void Estimator::processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);  
        //if(solver_flag != NON_LINEAR)
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);  //这里是不是不对啊  为什么要push两次

        dt_buf[frame_count].push_back(dt);   // 还有这里   这个部分到底是数组还是什么
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count;
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
        
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double header)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
    {
        marginalization_flag = MARGIN_OLD;
        //printf("keyframe\n");
    }
    else
    {
        marginalization_flag = MARGIN_SECOND_NEW;
        //printf("non-keyframe\n");
    }

    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;

    ImageFrame imageframe(image, header);
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(header, imageframe));
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};


    if (ESTIMATE_EXTRINSIC == 2)
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl
                                                               << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    if (solver_flag == INITIAL)
    {
        // monocular + IMU initilization
        if (!STEREO && USE_IMU)
        {
            if (frame_count == WINDOW_SIZE)
            {
                bool result = false;
                if (ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1)
                {
                    result = initialStructure();
                    initial_timestamp = header;
                }
                if (result)
                {
                    optimization();
                    updateLatestStates();
                    solver_flag = NON_LINEAR;
                    slideWindow();
                    ROS_INFO("Initialization finish!");
                }
                else
                    slideWindow();
            }
        }

        // stereo + IMU initilization
        if (STEREO && USE_IMU)
        {
            
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            if (frame_count == WINDOW_SIZE)
            {
                map<double, ImageFrame>::iterator frame_it;
                int i = 0;
                for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++)
                {
                    frame_it->second.R = Rs[i];
                    frame_it->second.T = Ps[i];
                    i++;
                }
                solveGyroscopeBias(all_image_frame, Bgs);
                for (int i = 0; i <= WINDOW_SIZE; i++)
                {
                    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
                }
           /*      for (int i = 0; i <= frame_count; i++)
                {
                    for (size_t j = 0; j < uwb_vector[i].size(); j++)
                    {
                        uwb_pre_integrations[i][j]->repropagate(Vector3d::Zero(), Bgs[i]);
                    }
                } */
                
                optimization();
                updateLatestStates();
                solver_flag = NON_LINEAR;
                cout << "ceshi1" << endl;
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }

        // stereo only initilization
        if (STEREO && !USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            optimization();

            if (frame_count == WINDOW_SIZE)
            {
                
                optimization();
                updateLatestStates();
                solver_flag = NON_LINEAR;
                
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }

        if (frame_count < WINDOW_SIZE)
        {
            frame_count++;
            int prev_frame = frame_count - 1;
            Ps[frame_count] = Ps[prev_frame];
            Vs[frame_count] = Vs[prev_frame];
            Rs[frame_count] = Rs[prev_frame];
            Bas[frame_count] = Bas[prev_frame];
            Bgs[frame_count] = Bgs[prev_frame];
        }
    }
    else
    {
        TicToc t_solve;
        if (!USE_IMU)
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
        f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
        tmp_uwb_count = uwb_count;
        uwb_count = init_uwb_vector.size();
        int delta = uwb_count - tmp_uwb_count;
        cout << "puanduan" << Rs[frame_count - 1] << endl << Ps[frame_count - 1] << endl << Vs[frame_count - 1] << endl;
        optimization();
        // tmp_init_uwb_vector.clear();
        if (!uwb_initialed && uwb_count >= 6 && delta > 0)
        {

            optimization1();
            cout << "result_uwb_p" << endl
                 << result_uwb_p << endl;
            cout << "puanduan2" << Rs[frame_count - 1] << endl
                 << Ps[frame_count - 1] << endl
                 << Vs[frame_count - 1] << endl;

            if ((result_uwb_p - tem_result_uwb_p).norm() < 0.005)
            {
                if (real_uwb_count > 40)
                {
                    uwb_initialed = true;
                    cout << "biaodingwancheng" << endl;
                }
            }


            // cout << "uwb_count" << uwb_count << endl;
            cv::Mat matA(uwb_count, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matAt(3, uwb_count, CV_32F, cv::Scalar::all(0));
            cv::Mat matAtA(3, 3, CV_32F, cv::Scalar::all(0));
            for (int j = 0; j < uwb_count; j++)
            {
                // float arx = (init_uwb_vector[j].d / tem_result_uwb_p.norm()) - (tem_result_uwb_p.x() - init_uwb_vector[j].p.x()) / ((init_uwb_vector[j].p - tem_result_uwb_p).norm());
                // float ary = (init_uwb_vector[j].d / tem_result_uwb_p.norm()) - (tem_result_uwb_p.y() - init_uwb_vector[j].p.y()) / ((init_uwb_vector[j].p - tem_result_uwb_p).norm());
                // float arz = (init_uwb_vector[j].d / tem_result_uwb_p.norm()) - (tem_result_uwb_p.z() - init_uwb_vector[j].p.z()) / ((init_uwb_vector[j].p - tem_result_uwb_p).norm());
                float arx = - (result_uwb_p.x() - init_uwb_vector[j].p.x()) / ((init_uwb_vector[j].p - result_uwb_p).norm());
                float ary = - (result_uwb_p.y() - init_uwb_vector[j].p.y()) / ((init_uwb_vector[j].p - result_uwb_p).norm());
                float arz = - (result_uwb_p.z() - init_uwb_vector[j].p.z()) / ((init_uwb_vector[j].p - result_uwb_p).norm());
                matA.at<float>(j, 0) = arx;
                matA.at<float>(j, 1) = ary;
                matA.at<float>(j, 2) = arz;
            }
            cv::transpose(matA, matAt);
            matAtA = matAt * matA;
            cv::Mat matE(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(3, 3, CV_32F, cv::Scalar::all(0));
            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);
            cout << "uwb优化测试chushi" << matE.at<float>(0, 0) <<  "dier" << matE.at<float>(0, 1) << "disam" << matE.at<float>(0, 2) << endl;
            // if (uwb_count == 8)
            // {
            //     cout << "tezhengzhi" << matE.at<float>(0, 0) << matE.at<float>(0, 1) << matE.at<float>(0, 2);
            //     if (matE.at<float>(0, 0) < 5)
            //     {
            //         cout << "uwb优化测试chushi" << matE.at<float>(0, 0) << endl;
            //         // init_uwb_vector.clear();
            //         // tmp_uwb_count = 0;
            //     }
            // }
            // else
            // {
            //     if (matE.at<float>(0, 0) < 10)
            //     {
            //         cout << "uwb优化测试4" << matE.at<float>(0, 0) << endl;
            //         init_uwb_vector[uwb_count - 1].d = -1;
            //         // break;
            //     }
            // }
            // if ((result_uwb_p - tem_result_uwb_p).norm() < 0.01)
            // {
            //     if (Vs[frame_count - 1].norm() > 0.2 && real_uwb_count > 30)
            //     {
            //         uwb_initialed = true;
            //         cout << "biaodingwancheng" << endl;
            //     }
            // } //这一步可以协助边缘化那里吗
        }

        set<int> removeIndex;
        outliersRejection(removeIndex);
        f_manager.removeOutlier(removeIndex);
        if (!MULTIPLE_THREAD)
        {
            featureTracker.removeOutliers(removeIndex);
            predictPtsInNextFrame();
          }

        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }

        slideWindow();
        f_manager.removeFailures();
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
        updateLatestStates();
    }
}

bool Estimator::initialStructure()
{
    TicToc t_sfm;
    //check imu observibility
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        //ROS_WARN("IMU variation %f!", var);
        if (var < 0.25)
        {
            ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }
    // global sfm
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    }
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    GlobalSFM sfm;
    if (!sfm.construct(frame_count + 1, Q, T, l,
                       relative_R, relative_T,
                       sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin();
    for (int i = 0; frame_it != all_image_frame.end(); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if ((frame_it->first) == Headers[i])
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if ((frame_it->first) > Headers[i])
        {
            i++;
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = -R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if (it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if (pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp, tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }
}

bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    //solve scale
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if (!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i]].R;
        Vector3d Pi = all_image_frame[Headers[i]].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i]].is_key_frame = true;
    }

    double s = (x.tail<1>())(0);
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if (frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());

    f_manager.clearDepth();
    f_manager.triangulate(frame_count, Ps, Rs, tic, ric);

    return true;
}

bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;
            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            if (average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        if (USE_IMU)
        {
            para_SpeedBias[i][0] = Vs[i].x();
            para_SpeedBias[i][1] = Vs[i].y();
            para_SpeedBias[i][2] = Vs[i].z();

            para_SpeedBias[i][3] = Bas[i].x();
            para_SpeedBias[i][4] = Bas[i].y();
            para_SpeedBias[i][5] = Bas[i].z();

            para_SpeedBias[i][6] = Bgs[i].x();
            para_SpeedBias[i][7] = Bgs[i].y();
            para_SpeedBias[i][8] = Bgs[i].z();
        }
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);

    para_Td[0][0] = td;
}

void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }

    if (USE_IMU)
    {
        Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                         para_Pose[0][3],
                                                         para_Pose[0][4],
                                                         para_Pose[0][5])
                                                 .toRotationMatrix());
        double y_diff = origin_R0.x() - origin_R00.x();
        //TODO
        Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
        if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
        {
            ROS_DEBUG("euler singular point!");
            rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                           para_Pose[0][3],
                                           para_Pose[0][4],
                                           para_Pose[0][5])
                                   .toRotationMatrix()
                                   .transpose();
        }

        for (int i = 0; i <= WINDOW_SIZE; i++)
        {

            Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

            Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                        para_Pose[i][1] - para_Pose[0][1],
                                        para_Pose[i][2] - para_Pose[0][2]) +
                    origin_P0;

            Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                        para_SpeedBias[i][1],
                                        para_SpeedBias[i][2]);

            Bas[i] = Vector3d(para_SpeedBias[i][3],
                              para_SpeedBias[i][4],
                              para_SpeedBias[i][5]);

            Bgs[i] = Vector3d(para_SpeedBias[i][6],
                              para_SpeedBias[i][7],
                              para_SpeedBias[i][8]);
        }
    }
    else
    {
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

            Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
        }
    }

    if (USE_IMU)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            tic[i] = Vector3d(para_Ex_Pose[i][0],
                              para_Ex_Pose[i][1],
                              para_Ex_Pose[i][2]);
            ric[i] = Quaterniond(para_Ex_Pose[i][6],
                                 para_Ex_Pose[i][3],
                                 para_Ex_Pose[i][4],
                                 para_Ex_Pose[i][5])
                         .normalized()
                         .toRotationMatrix();
        }
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);

    if (USE_IMU)
        td = para_Td[0][0];
}

bool Estimator::failureDetection()
{
    return false;
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        //ROS_INFO(" big translation");
        //return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        //ROS_INFO(" big z translation");
        //return true;
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}

//我加的
void Estimator::optimization1()
{
    ceres::Problem problem;
    tem_result_uwb_p = result_uwb_p;
    real_uwb_count = 0;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    // int u = uwb_vector[frame_count - 1].size();
    // cout << "uwb优化测试5" << "速度" << uwb_vector[frame_count - 1][0].v.norm() << endl;
    for (size_t i = 0; i < init_uwb_vector.size(); i++)
    {
        cout << "biaodingceshi" << endl;
        if (init_uwb_vector[i].d > 0)
        {
            cout << "uwb优化测试3" << endl << init_uwb_vector[i].d << endl << init_uwb_vector[i].p << endl << uwb_p[0] << endl <<uwb_p[1] << endl << uwb_p[2] << "  wucha" << init_uwb_vector[i].uwb_err;
            ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<uwb_initialing, 1, 3>(new uwb_initialing(init_uwb_vector[i].d, init_uwb_vector[i].p, init_uwb_vector[i].uwb_err));
            problem.AddResidualBlock(cost_function, loss_function, uwb_p);
            real_uwb_count ++;

        }
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 10;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    
    ceres::Solve(options, &problem, &summary);
    cout << summary.FullReport();
    result_uwb_p << uwb_p[0], uwb_p[1], uwb_p[2];
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
}

void Estimator::optimization()
{
    TicToc t_whole, t_prepare;
    vector2double();

    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = NULL;
    loss_function = new ceres::HuberLoss(1.0);
    //loss_function = new ceres::CauchyLoss(1.0 / FOCAL_LENGTH);
    //ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
    for (int i = 0; i < frame_count + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        if (USE_IMU)
            problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    if (!USE_IMU)
        problem.SetParameterBlockConstant(para_Pose[0]);

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if ((ESTIMATE_EXTRINSIC && frame_count == WINDOW_SIZE && Vs[0].norm() > 0.2) || openExEstimation)
        {
            //ROS_INFO("estimate extinsic param");
            openExEstimation = 1;
        }
        else
        {
            //ROS_INFO("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
    }
    problem.AddParameterBlock(para_Td[0], 1);

    if (!ESTIMATE_TD || Vs[0].norm() < 0.2)
        problem.SetParameterBlockConstant(para_Td[0]);

    if (last_marginalization_info && last_marginalization_info->valid)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }
    if (USE_IMU)
    {
        for (int i = 0; i < frame_count; i++)
        {
            int j = i + 1;
            if (pre_integrations[j]->sum_dt > 10.0)
                continue;
            IMUFactor *imu_factor = new IMUFactor(pre_integrations[j]);
            problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
        }
    }

    //我加的
    if (uwb_initialed)
    {
        for (int i = 0; i < frame_count; i++)
        {
            for (size_t j = 0; j < uwb_vector[i].size(); j++)
            {
                if (uwb_vector[i][j].d > 0 && uwb_vector[i][j].v.norm() > 0.1)
                {
                    cout << "uwb优化测试1" << endl;
                    cout << uwb_vector[i][j].d <<  endl << uwb_vector[i][j].uwb_t << endl << result_uwb_p << endl << g << endl;
                    cout << "与积分" << uwb_vector[i][j].a << endl;
                    ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<uwb_has_initialed, 1, 7, 9>
                    (new uwb_has_initialed(uwb_vector[i][j].d, uwb_vector[i][j].uwb_t, result_uwb_p, uwb_vector[i][j].a, g));
                    problem.AddResidualBlock(cost_function, NULL, para_Pose[i], para_SpeedBias[i]);
                }
                else
                {
                    cout << "beipaoqi" << endl;
                }
                
            }
        }
    }
    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;

        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;
                ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
            }

            if (STEREO && it_per_frame.is_stereo)
            {
                Vector3d pts_j_right = it_per_frame.pointRight;
                if (imu_i != imu_j)
                {
                    ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                                           it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
                else
                {
                    ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                                           it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f, loss_function, para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
            }
            f_m_cnt++;
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    //printf("prepare for ceres: %f \n", t_prepare.toc());

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    //printf("solver costs: %f \n", t_solver.toc());
    
    double2vector();
    //printf("frame_count: %d \n", frame_count);

    cout << "uwb优化测试2" << endl;
    if (frame_count < WINDOW_SIZE)
        return;

    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        if (last_marginalization_info && last_marginalization_info->valid)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        if (USE_IMU)
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor *imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                               vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                               vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (it_per_id.used_num < 4)
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i != imu_j)
                    {
                        Vector3d pts_j = it_per_frame.point;
                        ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                                                  it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                       vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    if (STEREO && it_per_frame.is_stereo)
                    {
                        Vector3d pts_j_right = it_per_frame.pointRight;
                        if (imu_i != imu_j)
                        {
                            ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                                                   it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                           vector<int>{0, 4});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                        else
                        {
                            ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                                                   it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                           vector<int>{2});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                    }
                }
            }
        }

        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());

        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            if (USE_IMU)
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

        addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
    }
    else
    {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info && last_marginalization_info->valid)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());

            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    if (USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    if (USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
        }
    }
    //printf("whole marginalization costs: %f \n", t_whole_marginalization.toc());
    //printf("whole time for ceres: %f \n", t_whole.toc());
}

void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0];
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Headers[i] = Headers[i + 1];
                Rs[i].swap(Rs[i + 1]);
                Ps[i].swap(Ps[i + 1]);
                
                if (USE_IMU)
                {
                    std::swap(pre_integrations[i], pre_integrations[i + 1]);

                    dt_buf[i].swap(dt_buf[i + 1]);
                    linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                    angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                    Vs[i].swap(Vs[i + 1]);
                    Bas[i].swap(Bas[i + 1]);
                    Bgs[i].swap(Bgs[i + 1]);
                    // cout << "ceshi2" << endl;
                    //我加的  锚点边缘化
                    uwb_pre_integrations[i].swap(uwb_pre_integrations[i+1]);
                    uwb_dt_buf[i].swap(uwb_dt_buf[i+1]);
                    uwb_linear_acceleration_buf[i].swap(uwb_linear_acceleration_buf[i+1]);
                    uwb_angular_velocity_buf[i].swap(uwb_angular_velocity_buf[i+1]);
                    uwb_vector[i].swap(uwb_vector[i + 1]);
                    // cout << "ceshi3" << endl;
                }
            }
            uwb_vector[WINDOW_SIZE].clear();
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

            if (USE_IMU)
            {
                Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
                Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
                Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

                
                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();

                //我加的
                for (size_t i = 0; i < uwb_pre_integrations[WINDOW_SIZE].size(); i++)
                {
                    delete uwb_pre_integrations[WINDOW_SIZE][i];
                }
                uwb_pre_integrations[WINDOW_SIZE].clear();
                uwb_dt_buf[WINDOW_SIZE].clear();
                uwb_linear_acceleration_buf[WINDOW_SIZE].clear();
                uwb_angular_velocity_buf[WINDOW_SIZE].clear();
            }

            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                all_image_frame.erase(all_image_frame.begin(), it_0);
            }
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];

            //我加的
            
            uwb_dt_buf[frame_count - 1].clear();
            uwb_linear_acceleration_buf[frame_count - 1].clear();
            uwb_linear_acceleration_buf[frame_count - 1].clear();
            uwb_vector[frame_count - 1].swap(uwb_vector[frame_count]);
            // cout << "ceshi" << endl;
            for (size_t i = 0; i < uwb_pre_integrations[frame_count - 1].size(); i++)
            {
                delete uwb_pre_integrations[frame_count - 1][i];
                // cout << "1" << uwb_pre_integrations[frame_count - 1].size() << endl; 
            }
            uwb_pre_integrations[frame_count - 1].clear();
            // cout << "3" <<endl;
            size_t z = uwb_pre_integrations[frame_count].size();
            IntegrationBase *tem_uwb_pre_integrations[z];
            for (size_t j = 0; j < z; j++)
            {
                // cout << "2" <<endl;
                tem_uwb_pre_integrations[j] = new IntegrationBase{pre_integrations[frame_count - 1]->acc_0, pre_integrations[frame_count - 1]->gyr_0,
                 Bas[frame_count], Bgs[frame_count]};
                // uwb_dt_buf[frame_count - 1][j].assign(dt_buf[frame_count].begin(), dt_buf[frame_count].end());
                // uwb_linear_acceleration_buf[frame_count - 1][j].assign(linear_acceleration_buf[frame_count].begin(), linear_acceleration_buf[frame_count].end());
                // uwb_angular_velocity_buf[frame_count - 1][j].assign(angular_velocity_buf[frame_count].begin(), angular_velocity_buf[frame_count].end());
                uwb_pre_integrations[frame_count - 1].push_back(tem_uwb_pre_integrations[j]);
                // cout << "ceshi7" << endl;
            }
            // for (unsigned int i = 0; i < uwb_dt_buf[frame_count].size(); i++)
            // {
            //     vector<double> tmp_dt = uwb_dt_buf[frame_count][i];
            //     vector<Vector3d> tmp_linear_acceleration = uwb_linear_acceleration_buf[frame_count][i];
            //     vector<Vector3d> tmp_angular_velocity = uwb_angular_velocity_buf[frame_count][i];
            //     uwb_dt_buf[frame_count - 1].push_back(tmp_dt);
            //     uwb_linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
            //     uwb_angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            // }
            uwb_dt_buf[frame_count - 1].swap(uwb_dt_buf[frame_count]);
            uwb_linear_acceleration_buf[frame_count - 1].swap(uwb_linear_acceleration_buf[frame_count]);
            uwb_angular_velocity_buf[frame_count - 1].swap(uwb_angular_velocity_buf[frame_count]);
            
            //我加的
            for (size_t i = 0; i < z; i++)
            {
                for (size_t j = 0; j < uwb_dt_buf[frame_count - 1][i].size(); j++)
                {
                    double tmp_dt = uwb_dt_buf[frame_count -1][i][j];
                    Vector3d tmp_linear_acceleration = uwb_linear_acceleration_buf[frame_count - 1][i][j];
                    Vector3d tmp_angular_velocity = uwb_angular_velocity_buf[frame_count - 1][i][j];

                    uwb_pre_integrations[frame_count - 1][i]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);
                }
                uwb_vector[frame_count - 1][i].uwb_t = uwb_pre_integrations[frame_count - 1][i]->sum_dt + pre_integrations[frame_count - 1]->sum_dt;
                uwb_vector[frame_count - 1][i].a = uwb_pre_integrations[frame_count - 1][i]->delta_p;
                if (isnan(uwb_vector[frame_count - 1][i].a.x()))
                {
                    cout << "chushile2" << endl;
                }
            }
       
            
            for (size_t i = 0; i < uwb_pre_integrations[WINDOW_SIZE].size(); i++)
            {
                delete uwb_pre_integrations[WINDOW_SIZE][i];
                // cout << "ceshi6" << endl;
            }
            uwb_pre_integrations[WINDOW_SIZE].clear();
            uwb_dt_buf[WINDOW_SIZE].clear();
            uwb_linear_acceleration_buf[WINDOW_SIZE].clear();
            uwb_angular_velocity_buf[WINDOW_SIZE].clear();
            uwb_vector[WINDOW_SIZE].clear();
            // cout << "ceshi5" << endl;

            if (USE_IMU)
            {
                for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
                {
                    double tmp_dt = dt_buf[frame_count][i];
                    Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                    Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                    pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                    dt_buf[frame_count - 1].push_back(tmp_dt);
                    linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                    angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
                }

                Vs[frame_count - 1] = Vs[frame_count];
                Bas[frame_count - 1] = Bas[frame_count];
                Bgs[frame_count - 1] = Bgs[frame_count];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();

            
            }
            slideWindowNew();
        }
    }
}

void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}

void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}

void Estimator::getPoseInWorldFrame(Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[frame_count];
    T.block<3, 1>(0, 3) = Ps[frame_count];
}

void Estimator::getPoseInWorldFrame(int index, Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[index];
    T.block<3, 1>(0, 3) = Ps[index];
}

void Estimator::predictPtsInNextFrame()
{
    //printf("predict pts in next frame\n");
    if (frame_count < 2)
        return;
    // predict next pose. Assume constant velocity motion
    Eigen::Matrix4d curT, prevT, nextT;
    getPoseInWorldFrame(curT);
    getPoseInWorldFrame(frame_count - 1, prevT);
    nextT = curT * (prevT.inverse() * curT);
    map<int, Eigen::Vector3d> predictPts;

    for (auto &it_per_id : f_manager.feature)
    {
        if (it_per_id.estimated_depth > 0)
        {
            int firstIndex = it_per_id.start_frame;
            int lastIndex = it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1;
            //printf("cur frame index  %d last frame index %d\n", frame_count, lastIndex);
            if ((int)it_per_id.feature_per_frame.size() >= 2 && lastIndex == frame_count)
            {
                double depth = it_per_id.estimated_depth;
                Vector3d pts_j = ric[0] * (depth * it_per_id.feature_per_frame[0].point) + tic[0];
                Vector3d pts_w = Rs[firstIndex] * pts_j + Ps[firstIndex];
                Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3));
                Vector3d pts_cam = ric[0].transpose() * (pts_local - tic[0]);
                int ptsIndex = it_per_id.feature_id;
                predictPts[ptsIndex] = pts_cam;
            }
        }
    }
    featureTracker.setPrediction(predictPts);
    //printf("estimator output %d predict pts\n",(int)predictPts.size());
}

double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                    Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj,
                                    double depth, Vector3d &uvi, Vector3d &uvj)
{
    Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}

void Estimator::outliersRejection(set<int> &removeIndex)
{
    //return;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        double err = 0;
        int errCnt = 0;
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;
        feature_index++;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        double depth = it_per_id.estimated_depth;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;
                double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                                     Rs[imu_j], Ps[imu_j], ric[0], tic[0],
                                                     depth, pts_i, pts_j);
                err += tmp_error;
                errCnt++;
                //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
            }
            // need to rewrite projecton factor.........
            if (STEREO && it_per_frame.is_stereo)
            {

                Vector3d pts_j_right = it_per_frame.pointRight;
                if (imu_i != imu_j)
                {
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                                         Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                         depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
                else
                {
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                                         Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                         depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
            }
        }
        double ave_err = err / errCnt;
        if (ave_err * FOCAL_LENGTH > 3)
            removeIndex.insert(it_per_id.feature_id);
    }
}

void Estimator::fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity)
{
    double dt = t - latest_time;
    latest_time = t;
    Eigen::Vector3d un_acc_0 = latest_Q * (latest_acc_0 - latest_Ba) - g;
    Eigen::Vector3d un_gyr = 0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg;
    latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt);
    Eigen::Vector3d un_acc_1 = latest_Q * (linear_acceleration - latest_Ba) - g;
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    latest_P = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
    latest_V = latest_V + dt * un_acc;
    latest_acc_0 = linear_acceleration;
    latest_gyr_0 = angular_velocity;
}

void Estimator::updateLatestStates()
{
    mPropagate.lock();
    latest_time = Headers[frame_count] + td;
    latest_P = Ps[frame_count];
    latest_Q = Rs[frame_count];
    latest_V = Vs[frame_count];
    latest_Ba = Bas[frame_count];
    latest_Bg = Bgs[frame_count];
    latest_acc_0 = acc_0;
    latest_gyr_0 = gyr_0;
    mBuf.lock();
    queue<pair<double, Eigen::Vector3d>> tmp_accBuf = accBuf;
    queue<pair<double, Eigen::Vector3d>> tmp_gyrBuf = gyrBuf;
    mBuf.unlock();
    while (!tmp_accBuf.empty())
    {
        double t = tmp_accBuf.front().first;
        Eigen::Vector3d acc = tmp_accBuf.front().second;
        Eigen::Vector3d gyr = tmp_gyrBuf.front().second;
        fastPredictIMU(t, acc, gyr);
        tmp_accBuf.pop();
        tmp_gyrBuf.pop();
    }
    mPropagate.unlock();
}

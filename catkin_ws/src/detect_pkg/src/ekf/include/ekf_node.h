#include <Eigen/Dense>
#include <iostream>
#include <ros/console.h>
#pragma once
class KalmanFilter {
public:
    // 状态维度：6 (px, py, pz, vx, vy, vz)，观测维度：3 (x, y, z)
    static constexpr int STATE_DIM = 6;
    static constexpr int MEASURE_DIM = 3;

    /**
     * 构造函数
     * @param dt 采样时间间隔 (秒)，必须与你的处理帧率匹配
     * @param pos_std 过程噪声中位置分量的标准差，控制模型信任度
     * @param vel_std 过程噪声中速度分量的标准差，控制模型信任度
     * @param obs_std 观测噪声的标准差，控制观测信任度（最重要！）
     */
    KalmanFilter(double dt = 1.0 / 30.0, double pos_std = 0.1, double vel_std = 0.5, double obs_std = 0.2)
        : dt_(dt), pos_std(pos_std), vel_std(vel_std), is_initialized_(false){
        // 初始化状态向量和协方差矩阵
        x_.setZero();           // 状态: [px, py, pz, vx, vy, vz]
        P_.setIdentity();       // 状态协方差
        P_ = P_ * 500.0;        // 初始不确定性设大，让滤波器快速收敛

        // 状态转移矩阵 F (匀速模型)
        F_.setIdentity();
        F_(0, 3) = dt;
        F_(1, 4) = dt;
        F_(2, 5) = dt;

        // 观测矩阵 H (只观测位置)
        H_.setZero();
        H_(0, 0) = 1.0;
        H_(1, 1) = 1.0;
        H_(2, 2) = 1.0;

        // 过程噪声协方差 Q
        Q_.setZero();

        // 观测噪声协方差 R
        R_.setIdentity();
        R_ = R_ * (obs_std * obs_std);

        // 单位矩阵 (重用，避免重复构造)
        I_.setIdentity();
    }

    /**
     * 初始化滤波器（通常在获得第一个有效观测时调用）
     * @param first_measurement 第一个观测到的3D位置 [x, y, z]
     */
    void init(const Eigen::Vector3d& first_measurement) {
        x_.head<3>() = first_measurement; // 位置初始化
        x_.tail<3>().setZero();           // 速度初始化为0
        is_initialized_ = true;
        std::cout << "[Kalman] Filter initialized with position: "
                  << first_measurement.transpose() << std::endl;
    }

    /**
     * 核心：预测步骤
     */
    void predict(double curr_time) {
        if (!is_initialized_) {
            std::cerr << "[Kalman] Warning: Predict called before initialization. Skipping." << std::endl;
            return;
        }

        double dt = (last_time > 0) ? (curr_time - last_time) : 1 / 30.;
        if (dt > 1) {
            last_time = curr_time;
            reset();
            return;
        }

        ROS_INFO("Delta time: %lf", dt);
        dt_ = dt;
        F_(0, 3) = dt;
        F_(1, 4) = dt;
        F_(2, 5) = dt;

        // 加入速度阻尼衰减，假设速度在约0.2秒内衰减一半(半衰期机制)，避免丢失观测时发散
        double damping = std::exp(-dt * 3.46); // ln(2)/0.2 ≈ 3.46
        F_(3, 3) = damping;
        F_(4, 4) = damping;
        F_(5, 5) = damping;

        double dt2 = dt * dt;
        double dt3 = dt2 * dt;
        // 对于连续时间噪声模型 \dot{p} = v, \dot{v} = w (w是白噪声)
        // 离散化后的Q通常为 [ (dt3/3)*σ_v^2, (dt2/2)*σ_v^2;
        //                   (dt2/2)*σ_v^2,  dt*σ_v^2    ]
        // 其中σ_v是速度过程噪声标准差(vel_std_)
        double pos_noise_var = vel_std * vel_std * dt3 / 3.0;
        double vel_noise_var = vel_std * vel_std * dt;
        double pos_vel_cov = vel_std * vel_std * dt2 / 2.0; // 位置与速度的噪声协方差

        Q_.setZero();
        // 位置噪声
        Q_(0, 0) = Q_(1, 1) = Q_(2, 2) = pos_noise_var;
        // 速度噪声
        Q_(3, 3) = Q_(4, 4) = Q_(5, 5) = vel_noise_var;
        // 位置与速度的噪声相关性 (非对角块)
        Q_(0, 3) = Q_(1, 4) = Q_(2, 5) = pos_vel_cov;
        Q_(3, 0) = Q_(4, 1) = Q_(5, 2) = pos_vel_cov;

        // 状态预测: x = F * x
        x_ = F_ * x_;
        // 协方差预测: P = F * P * F^T + Q
        P_ = F_ * P_ * F_.transpose() + Q_;
    }

    /**
     * 核心：更新步骤
     * @param measurement 新的观测值 [x, y, z]
     * @return 更新后的平滑状态 [px, py, pz, vx, vy, vz]
     */
    Eigen::Matrix<double, STATE_DIM, 1> update(const Eigen::Vector3d& measurement, double curr_time) {
        if (!valid_input)
        {
            ROS_INFO("Velocity: %lf, %lf, %lf", x_(3), x_(4), x_(5));
            ROS_INFO("Position: %lf, %lf, %lf", x_(0), x_(1), x_(2));
            return x_;
        }
        // 计算观测残差 (innovation): y = z - H * x
        Eigen::Vector3d y = measurement - H_ * x_;

        // 计算卡尔曼增益: K = P * H^T * (H * P * H^T + R)^(-1)
        Eigen::Matrix3d S = H_ * P_ * H_.transpose() + R_;


        double nis = y.transpose() * S.inverse() * y;

        if (nis < chi2_threshold) {
            Eigen::Matrix<double, STATE_DIM, 3> K = P_ * H_.transpose() * S.inverse();

            // 状态更新: x = x + K * y
            x_ = x_ + K * y;

            // 协方差更新: P = (I - K * H) * P
            // 使用约瑟夫公式 (Joseph form) 保证数值稳定性，尤其适合嵌入式系统
            Eigen::Matrix<double, STATE_DIM, STATE_DIM> I_KH = I_ - K * H_;
            P_ = I_KH * P_ * I_KH.transpose() + K * R_ * K.transpose();
            last_time  = curr_time;
        }

        ROS_INFO("Velocity: %lf, %lf, %lf", x_(3), x_(4), x_(5));
        ROS_INFO("Position: %lf, %lf, %lf", x_(0), x_(1), x_(2));
        return x_; // 返回平滑后的完整状态
    }

    /**
     * 一步封装：预测 + 更新 (最常用接口)
     * @param measurement 新的观测值
     * @param dt 从数据包中得到的真值时间间隔
     * @return 平滑后的状态
     */
    Eigen::Matrix<double, STATE_DIM, 1> predictAndUpdate(const Eigen::Vector3d& measurement, const double curr_time) {
        const double xm = measurement(0), ym = measurement(1), zm = measurement(2);
        valid_input = !(xm == 0 && ym == 0 && zm == 0);

        if (!is_initialized_ and valid_input) {
            init(measurement);
            last_time = curr_time;
            return x_;
        }

        predict(curr_time);
        return update(measurement, curr_time);
    }

    /**
     * 获取当前状态
     * @param get_velocity 是否同时返回速度 (true返回6维，false返回3维位置)
     */
    Eigen::VectorXd getState(bool get_velocity = false) const {
        if (!get_velocity) {
            return x_.head<3>(); // 只返回位置
        }
        return x_; // 返回完整状态 (位置+速度)
    }

    /**
     * 重置滤波器 (当目标丢失后重新出现时使用)
     * @param new_pos 新的初始位置 (可选)
     */
    void reset(const Eigen::Vector3d& new_pos = Eigen::Vector3d::Zero()) {
        x_.setZero();
        P_.setIdentity();
        P_ = P_ * 500.0;
        if (new_pos != Eigen::Vector3d::Zero()) {
            x_.head<3>() = new_pos;
        }
        is_initialized_ = (new_pos != Eigen::Vector3d::Zero());
    }

private:
    double dt_;                             // 采样时间间隔
    bool valid_input = false;
    double pos_std;
    double vel_std;
    bool is_initialized_;                   // 初始化标志
    const double chi2_threshold = 7.815;
    double last_time = -1;

    // 核心矩阵
    Eigen::Matrix<double, STATE_DIM, 1> x_; // 状态向量
    Eigen::Matrix<double, STATE_DIM, STATE_DIM> P_; // 状态协方差
    Eigen::Matrix<double, STATE_DIM, STATE_DIM> F_; // 状态转移矩阵
    Eigen::Matrix<double, MEASURE_DIM, STATE_DIM> H_; // 观测矩阵
    Eigen::Matrix<double, STATE_DIM, STATE_DIM> Q_; // 过程噪声协方差
    Eigen::Matrix<double, MEASURE_DIM, MEASURE_DIM> R_; // 观测噪声协方差
    Eigen::Matrix<double, STATE_DIM, STATE_DIM> I_; // 单位矩阵 (用于更新)
};
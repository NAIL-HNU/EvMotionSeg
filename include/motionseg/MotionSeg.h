#ifndef MOTIONSEG_MOTIONSEG_H
#define MOTIONSEG_MOTIONSEG_H

#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cmath>
#include <string>
#include <list>
#include <map>
#include <random>
#include "GCoptimization.h"
#include "LinkedBlockList.h"
#include <set>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <unordered_set>
#include <ros/ros.h>


#include <ceres/ceres.h>

#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>


class AffineCostFunction : public ceres::SizedCostFunction<1, 4> {
public:
    AffineCostFunction(double nx, double ny, double u, double v)
        : nx_(nx), ny_(ny), u_(u), v_(v) {}

    ~AffineCostFunction() override {}

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override {
        // Parse parameters
        const double m0 = parameters[0][0];
        const double theta_deg = parameters[0][1];
        const double tx = parameters[0][2];
        const double ty = parameters[0][3];

        // Convert angle from degrees to radians
        const double theta_rad = theta_deg * M_PI / 180.0;
        const double cos_t = cos(theta_rad);
        const double sin_t = sin(theta_rad);

        // Compute residual
        const double Rx = (m0 * cos_t - 1) * nx_ + (-m0 * sin_t) * ny_ + tx;
        const double Ry = (m0 * sin_t) * nx_ + (m0 * cos_t - 1) * ny_ + ty;
        residuals[0] = u_ * Rx + v_ * Ry - (u_ * u_ + v_ * v_);

        // Compute Jacobian matrix
        if (jacobians != nullptr && jacobians[0] != nullptr) {
            double* jac = jacobians[0];
            // Derivative w.r.t m0
            jac[0] = u_ * (cos_t * nx_ - sin_t * ny_) + v_ * (sin_t * nx_ + cos_t * ny_);
            // Derivative w.r.t theta (includes radian conversion)
            jac[1] = m0 * (u_ * (-sin_t * nx_ - cos_t * ny_) + v_ * (cos_t * nx_ - sin_t * ny_)) * (M_PI / 180.0);
            // Derivative w.r.t tx
            jac[2] = u_;
            // Derivative w.r.t ty
            jac[3] = v_;
        }

        return true;
    }

private:
    const double nx_, ny_, u_, v_; // Data point parameters
};


struct Vector2iHash {
    size_t operator()(const Eigen::Vector2i& key) const {
        return std::hash<int>()(key.x()) ^ (std::hash<int>()(key.y()) << 1);
    }
};



struct NormalFlowNode {    
    double timestamp;               
    Eigen::Vector2i origin_xy;     
    Eigen::Vector4d normal_flow;  
    
    std::vector<Eigen::Vector2i> associated_flows;
    std::vector<int> neighbors;

    NormalFlowNode(double ts,
                const Eigen::Vector2i& origin,
                const Eigen::Vector4d& nf)
        : timestamp(ts),
        origin_xy(origin),
        normal_flow(nf){}
};

struct LabelInfo {
    Eigen::Vector4d motion_param; 
    std::vector<int> nodes_id;  // Associated node indices
    size_t count = 0;

    LabelInfo(const Eigen::Vector4d& motion_param_): motion_param(motion_param_) {}

    void AddNode(int node_id) {
        nodes_id.push_back(node_id);
        count++;
    }

    void MergeLabel(std::vector<int> new_nodes_id) {
        nodes_id.insert(nodes_id.end(), new_nodes_id.begin(), new_nodes_id.end());
        count += new_nodes_id.size();
    }

    void Clear() {
        count = 0;
        nodes_id.clear();
    }
};


struct BoundingBox{
    Eigen::Vector2d center;
    Eigen::Vector2d radius;
    LabelInfo label;

    BoundingBox(const Eigen::Vector2d& center_, const Eigen::Vector2d& radius_, const LabelInfo& label_): 
                                                        center(center_), radius(radius_), label(label_) {}

    bool IsValid() const {
        if(radius.x() <= 0 || radius.y() <= 0) return false;
        return true;
    }

    void UpdateBox(double interval){
        Eigen::MatrixXd R(2, 2);
        R << label.motion_param[0] * cos(label.motion_param[1] * M_PI / 180) - 1, -label.motion_param[0] * sin(label.motion_param[1] * M_PI / 180),
            label.motion_param[0] * sin(label.motion_param[1] * M_PI / 180), label.motion_param[0] * cos(label.motion_param[1] * M_PI / 180) - 1;
        

        Eigen::Vector2d n = {label.motion_param[2], label.motion_param[3]};
        Eigen::Vector2d x = {center.x(), center.y()};
        center = x + (R * x + n) * interval;
        // radius = radius * 0.9;
    }
};

struct Message{
    double timestamp;
    std::vector<NormalFlowNode> nodes;
    std::vector<BoundingBox> boxes;
};

class MotionSeg {
public:
    MotionSeg(ros::NodeHandle &nh, ros::NodeHandle nh_private);
    virtual ~MotionSeg();

    void GenerationLoop();

    void ProcessingLoop();

    void remove_duplicate_neighbors(std::map<size_t, std::list<size_t> >& mEdges_);
    
    void downsamplingBG(std::vector<NormalFlowNode>& Nodes);

    bool isMotionDifferent(const Eigen::Vector4d& a, const Eigen::Vector4d& b,
                     double scale_eps = 0.1,     // Scale factor error threshold (relative)
                     double rotation_eps_deg = 10.0, // Rotation angle error threshold (degrees)
                     double length_ratio_eps = 2.0,
                     double angle_eps = 45.0);

    void GenerateMotionCandidates(std::vector<LabelInfo>& candidates, const Eigen::Vector4d& motion_param);

    void FindBoundingBox(BoundingBox& box, LabelInfo& label);

    void PredictBoundingBox();

    void PredictLabels();

    void ClearEmptyLabels();

    double ComputeMotionError(const NormalFlowNode& node, const Eigen::Vector4d & motion_param);

    void NonLinearSolver( LabelInfo& label);

    void FindNeighbors(std::vector<NormalFlowNode>& Nodes);

    bool OptimizeLabel();

    void SelectLabels();

    std::vector<NormalFlowNode> processEvents(
        double start_time,
        double end_time);
    
    bool RunGraphCut( 
        int& run_times
    );

    void Visualization(int t);

    void Clear();


    std::ofstream cost_ofs, timestamp_ofs;
    std::ifstream evt_fs, un_evt_fs, flow_fs;
    double first_event_time_ = 0.0;
    double first_event_time_offset_ = 0.00;

    bool is_first_event_time_ = true;
    std::string data_file_path_;

    std::thread ProcessingThread, GenerationThread;
    std::queue<std::vector<NormalFlowNode>> data_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    bool stop_thread_ = false;

    int count = 0;
    int count_for_vis = 0;

    double interval_;
    int width_, height_;
    std::vector<BoundingBox> boxes;
    std::vector<LabelInfo> labels, last_labels;
    std::vector<NormalFlowNode> nodes, removed_nodes;

    double data_term_, smooth_term_, label_term_;

    int downsample_rate_;

    double fx_, fy_;

    double node_thres_ = 5;

    int GraphCutIteration_;
    int MotionSegIteration_;

    long int all_event_count = 0;
    long int invalid_event_count = 0;

};


#endif
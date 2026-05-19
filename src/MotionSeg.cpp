#include "motionseg/MotionSeg.h"
#include "delaunator.hpp"


MotionSeg::MotionSeg(ros::NodeHandle &nh, ros::NodeHandle nh_private)
{

    nh_private.param<std::string>("data_file_path", data_file_path_," ");
    nh_private.param<double>("interval", interval_, 0.0333);
    nh_private.param<int>("width", width_, 346);
    nh_private.param<int>("height", height_, 260);
    nh_private.param<double>("data_term", data_term_, 1);
    nh_private.param<double>("smooth_term", smooth_term_, 7000);
    nh_private.param<double>("label_term", label_term_, 90000);
    nh_private.param<int>("downsample_rate", downsample_rate_, 2);
    nh_private.param<double>("fx", fx_, 250);
    nh_private.param<double>("fy", fy_, 250);
    nh_private.param<int>("GraphCutIteration", GraphCutIteration_, 10);
    nh_private.param<int>("MotionSegIteration", MotionSegIteration_, 4);
    
    std::cout <<"interval_: " << interval_ << std::endl;
    std::cout << "data_file_path: " << data_file_path_ << std::endl;

    evt_fs.open(data_file_path_+"events.txt");
    un_evt_fs.open(data_file_path_+"undistorted_normalized_xy.txt");
    flow_fs.open(data_file_path_+"flow_xy.txt");

    timestamp_ofs.open(data_file_path_+"timestamp.csv");

    if (!evt_fs.is_open() || !un_evt_fs.is_open() || !flow_fs.is_open()) {
        throw std::runtime_error("Cannot open input files");
    }
    
    timestamp_ofs.close();
    timestamp_ofs.open(data_file_path_+"timestamp.csv", std::ios::app);


    count = 0;
    count_for_vis = count;

    GenerationThread = std::thread(&MotionSeg::GenerationLoop, this);
    ProcessingThread = std::thread(&MotionSeg::ProcessingLoop, this);
    
}


MotionSeg::~MotionSeg()
{
    stop_thread_ = true;
    queue_cv_.notify_all();
    if (ProcessingThread.joinable()) {
        ProcessingThread.join();
    }
    if (GenerationThread.joinable()) {
        GenerationThread.join();
    }
}


void MotionSeg::GenerationLoop()
{
    ros::Rate r(1/interval_);
    while (ros::ok())
    {
      ros::Time sync_time_ = ros::Time::now();
      if (count < 300)
      {
        std::vector<NormalFlowNode> Nodes;
        Nodes = processEvents(
            interval_*count ,  // Start time
            interval_*(count+1) // End time
            );
        if (Nodes.size() < node_thres_)
        {
            count++;
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            data_queue_.push(Nodes);
        }
        
        queue_cv_.notify_one();  // Notify consumer
        count++;
      }else
      {
        std::cout << "finish" << std::endl;
        break;
      }

      r.sleep();
    }
}

void MotionSeg::ProcessingLoop()
{
    while (ros::ok()&& !stop_thread_) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            // Wait for queue non-empty or stop signal
            queue_cv_.wait(lock, [this]() {
                return !data_queue_.empty() || stop_thread_;
            });

            nodes = data_queue_.front();
            data_queue_.pop();
        }

        auto start = std::chrono::high_resolution_clock::now();

        

        if (boxes.size() > 0)
            PredictLabels();

        for(int i = 0; i < boxes.size(); i++)
            labels.push_back(boxes[i].label);

        int target_size = boxes.empty() ? 12 : 6 + labels.size();
        for (const auto& node : nodes) {  // Using range-based for loop
            if (labels.size() >= target_size) break;
            GenerateMotionCandidates(labels, 
                Eigen::Vector4d(1, 0, node.normal_flow[2], node.normal_flow[3]));
        }

        int run_times = 0;
        for (int iter = 0; iter < MotionSegIteration_; ++iter) {
            if(RunGraphCut(run_times))
                break;
        }

         // Code under test
        auto end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        std::cout <<"t: " << count << " run times: " << run_times << "  total time: " << total_time * 1e-3 << " ms" << std::endl;
        std::cout << std::endl;

        Visualization(count_for_vis);
        count_for_vis++;
        SelectLabels();
        
        PredictBoundingBox();

        Clear();
    }
}

void MotionSeg::remove_duplicate_neighbors(std::map<size_t, std::list<size_t> >& mEdges_) {
    for(auto it = mEdges_.begin(); it != mEdges_.end(); ++it) {
        it->second.sort();
        it->second.unique();
    }
}

void MotionSeg::downsamplingBG(std::vector<NormalFlowNode>& Nodes) {
    int size = Nodes.size();
    int ransac_times = 6;

    // Define distribution range [start, end)
    Eigen::MatrixXd A(6, 6);
    Eigen::VectorXd x(6);
    Eigen::VectorXd b(6);

    Eigen::VectorXd x_best(6);
    
    Eigen::MatrixXd A_all(Nodes.size(), 6);
    Eigen::VectorXd b_all(Nodes.size());
    Eigen::VectorXd error_all(Nodes.size());

    double error_best = 0;
    Eigen::MatrixXd x_all(6, ransac_times);

    for(int i = 0; i < Nodes.size(); i++)
    {
        A_all(i, 0) = Nodes[i].normal_flow[2] * Nodes[i].normal_flow[0];
        A_all(i, 1) = Nodes[i].normal_flow[2] * Nodes[i].normal_flow[1];
        A_all(i, 2) = Nodes[i].normal_flow[2];
        A_all(i, 3) = Nodes[i].normal_flow[3] * Nodes[i].normal_flow[0];
        A_all(i, 4) = Nodes[i].normal_flow[3] * Nodes[i].normal_flow[1];
        A_all(i, 5) = Nodes[i].normal_flow[3];
        b_all(i) = Nodes[i].normal_flow.segment(2, 2).norm();
    }


    for(int i = 0; i < ransac_times; i++){
        std::random_device rd;  // Used to generate random seed
        std::mt19937 gen(rd()); // Using Mersenne Twister algorithm
        std::uniform_int_distribution<> dist(0, size - 1);
        std::vector<int> indices;
        for (int j = 0; j < 6; ++j) {
            indices.push_back(dist(gen));
        }

        
        for (size_t i = 0; i < indices.size(); ++i) {
            A(i, 0) = Nodes[indices[i]].normal_flow[2] * Nodes[indices[i]].normal_flow[0];
            A(i, 1) = Nodes[indices[i]].normal_flow[2] * Nodes[indices[i]].normal_flow[1];
            A(i, 2) = Nodes[indices[i]].normal_flow[2];
            A(i, 3) = Nodes[indices[i]].normal_flow[3] * Nodes[indices[i]].normal_flow[0];
            A(i, 4) = Nodes[indices[i]].normal_flow[3] * Nodes[indices[i]].normal_flow[1];
            A(i, 5) = Nodes[indices[i]].normal_flow[3];
            b(i) = Nodes[indices[i]].normal_flow.segment(2, 2).norm();

        }
        x = A.colPivHouseholderQr().solve(b); 
        error_all = (A_all * x - b_all).array().abs();
        // error_median(i) = error_all.median();
        double error_median = error_all.mean();
        if(i == 0){
            x_best = x;
            error_best = error_median;
        }
        else{
            if(error_median < error_best){
                x_best = x;
                error_best = error_median;
            }
        }
    }

    error_all = (A_all * x_best - b_all).array().abs();


    const int remove_count = static_cast<int>(size * 0.2);
    std::vector<int> indices(size);
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices by corresponding values in ascending order
    std::sort(indices.begin(), indices.end(),
              [&error_all](int a, int b) { return error_all[a] < error_all[b]; });

    // Collect indices to remove (top 25%)
    std::vector<int> to_remove(indices.begin(), indices.begin() + remove_count);

    // removed_nodes.reserve(to_remove.size());
    std::sort(to_remove.rbegin(), to_remove.rend());

    for (int idx : to_remove) {
        // removed_nodes.emplace(removed_nodes.begin(), Nodes[idx]);  // Insert at beginning
        Nodes.erase(Nodes.begin() + idx);                          // Remove element
    }
    // removed_nodes.clear();
    // std::cout << "downsamplingBG: " << size << " -> " << nodes.size() << std::endl;
}

bool MotionSeg::isMotionDifferent(const Eigen::Vector4d& a, const Eigen::Vector4d& b, 
                     double scale_eps,     // Scale factor error threshold (relative)
                     double rotation_eps_deg, // Rotation angle error threshold (degrees)
                     double length_ratio_eps,
                     double angle_eps) {


    double scale_diff = std::abs(a[0] - b[0]) / std::min(a[0], b[0]);
    if (scale_diff > scale_eps) return true;

    // Compare rotation angles (handle 360-degree period)
    double rotation_diff = std::abs(a[1] - b[1]);
    rotation_diff = std::min(rotation_diff, 360.0 - rotation_diff);
    if (rotation_diff > rotation_eps_deg) return true;


    const double len_a = a.segment(2, 2).norm();
    const double len_b = b.segment(2, 2).norm();

    // Length ratio difference
    const double ratio = std::max(len_a, len_b) / std::min(len_a, len_b);
    if (ratio >= length_ratio_eps) return true;

    // Angle difference calculation
    const double dot = a.segment(2, 2).dot(b.segment(2, 2));
    const double cos_theta = dot / (len_a * len_b);
    const double angle = std::acos(std::clamp(cos_theta, -1.0, 1.0)) * 180.0 / CV_PI;
    return angle >= angle_eps;
}


void MotionSeg::GenerateMotionCandidates(std::vector<LabelInfo>& candidates, const Eigen::Vector4d& motion_param) {
    if (candidates.empty()) {
        candidates.push_back(LabelInfo(motion_param));
        return;
    }

    bool should_add = true;
    for (const auto& it : candidates) {
        if (!isMotionDifferent(motion_param, it.motion_param)) {
            should_add = false;
            break;
        }
    }

    if (should_add) {
        candidates.push_back(LabelInfo(motion_param));
    }
}

void MotionSeg::FindBoundingBox(BoundingBox& box, LabelInfo& label) {
    std::vector<Eigen::Vector2d> points;
    for(int i = 0; i < label.nodes_id.size(); i++)
    {
        points.push_back(nodes[label.nodes_id[i]].normal_flow.segment(0, 2));
    }

    // if (points.empty()) return AABB();

    double max_percent = 0.65;
    double min_percent = 0.5;
    int gridLevels = 4;

    // Phase 1: Global multi-level candidate region detection
    Eigen::Vector2d globalMin = points[0], globalMax = points[0];
    for (const auto& p : points) {
        globalMin = globalMin.cwiseMin(p);
        globalMax = globalMax.cwiseMax(p);
    }

    Eigen::Vector2d currentMin = globalMin;
    Eigen::Vector2d currentMax = globalMax;
    Eigen::Vector2d bestCenter = (globalMin + globalMax) * 0.5; // Initial center
    double bestDensity = 0.0;
    double bestGridSize = 0.0;

    for (int level = 0; level < gridLevels; ++level) {
        const double width_ = currentMax.x() - currentMin.x();
        const double height_ = currentMax.y() - currentMin.y();
        
        // Termination condition: region cannot be subdivided further
        if (width_ < 1e-9 || height_ < 1e-9) break;

        // Dynamically compute sub-block size
        constexpr int subDivisions = 2;
        const double subWidth = width_ / subDivisions;
        const double subHeight = height_ / subDivisions;

        // Count sub-block density
        std::array<int, 4> cellCounts{}; // 2x2 grid uses 4 elements storage
        for (const auto& p : points) {
            // Filter points outside current region
            if (p.x() < currentMin.x() || p.x() >= currentMax.x() ||
                p.y() < currentMin.y() || p.y() >= currentMax.y()) continue;

            // Compute grid index
            const int i = static_cast<int>((p.x() - currentMin.x()) / subWidth);
            const int j = static_cast<int>((p.y() - currentMin.y()) / subHeight);
            cellCounts[i + j * subDivisions]++; // Row-major storage
        }

        // Find densest sub-block index
        int maxIndex = 0;
        for (int k = 1; k < 4; ++k) {
            if (cellCounts[k] > cellCounts[maxIndex]) {
                maxIndex = k;
            }
        }

        // Compute sub-block boundaries
        const int i = maxIndex % subDivisions;
        const int j = maxIndex / subDivisions;
        const Eigen::Vector2d blockMin(
            currentMin.x() + i * subWidth,
            currentMin.y() + j * subHeight
        );
        const Eigen::Vector2d blockMax = blockMin + Eigen::Vector2d(subWidth, subHeight);

        // Update final result
        bestGridSize = subWidth; // Record current level grid size
        bestDensity = cellCounts[maxIndex] / (subWidth * subHeight);
        bestCenter = (blockMin + blockMax) * 0.5;

        // Shrink to selected sub-block
        currentMin = blockMin;
        currentMax = blockMax;
    }

    // Phase 2: Select best candidate as initial seed
    // auto bestCandidate = std::max_element(candidates.begin(), candidates.end(),
    //     [](const Candidate& a, const Candidate& b) { return a.density < b.density; });

    struct Candidate {
        Eigen::Vector2d center;
        double density;
        double gridSize;
    };

    struct AABB {
        double minX, minY, maxX, maxY;
        AABB(double a = 0, double b = 0, double c = 0, double d = 0) 
            : minX(a), minY(b), maxX(c), maxY(d) {}
    };

    struct PairHash {
        size_t operator()(const std::pair<long, long>& p) const {
            return (size_t)(p.first * 1812433253 + p.second);
        }
    };

    Candidate bestCandidate;
    bestCandidate.center = bestCenter;
    bestCandidate.density = bestDensity;
    bestCandidate.gridSize = bestGridSize;

    // Phase 3: Region growing based on global data
    const double finalGridSize = bestCandidate.gridSize;
    std::unordered_map<std::pair<int, int>, int, PairHash> globalGrid;
    
    // Reconstruct global fine grid
    for (const auto& p : points) {
        int i = static_cast<int>((p.x() - globalMin.x()) / finalGridSize);
        int j = static_cast<int>((p.y() - globalMin.y()) / finalGridSize);
        globalGrid[{i, j}]++;
    }

    // Initialize growing starting point
    // std::cout << "Total points: " << points.size() << "\n";
    const int targetCount = ceil(points.size() * max_percent);
    const int minCount = ceil(points.size() * min_percent);
    // std::cout << "Target count: " << targetCount << "\n";
    std::unordered_map<std::pair<int, int>, bool, PairHash> visited;
    
    // Compute seed grid coordinates
    int seedI = static_cast<int>((bestCandidate.center.x() - globalMin.x()) / finalGridSize);
    int seedJ = static_cast<int>((bestCandidate.center.y() - globalMin.y()) / finalGridSize);
    const int maxGridI = static_cast<int>(std::ceil((globalMax.x() - globalMin.x()) / finalGridSize));
    const int maxGridJ = static_cast<int>(std::ceil((globalMax.y() - globalMin.y()) / finalGridSize));
    
    std::priority_queue<std::pair<int, std::pair<int, int>>> pq;
    pq.push({globalGrid[{seedI, seedJ}], {seedI, seedJ}});
    visited[{seedI, seedJ}] = true;

    AABB result;
    int total = 0;
    bool initialized = false;

    // Four-neighborhood directions
    const int di[] = {-1, 1, 0, 0};
    const int dj[] = {0, 0, -1, 1};

     // Minimum point count threshold

    while (!pq.empty() && total < targetCount) {
        auto [cnt, key] = pq.top();
        pq.pop();
        // total += cnt;

        // Update AABB
        double x1 = globalMin.x() + key.first * finalGridSize;
        double y1 = globalMin.y() + key.second * finalGridSize;
        double x2 = x1 + finalGridSize;
        double y2 = y1 + finalGridSize;

        if (!initialized) {
            result = AABB(x1, y1, x2, y2);
            initialized = true;
        } else {
            result.minX = std::min(result.minX, x1);
            result.minY = std::min(result.minY, y1);
            result.maxX = std::max(result.maxX, x2);
            result.maxY = std::max(result.maxY, y2);
            // std::cout <<__LINE__ <<std::endl;
        }

        total = 0;
    
        // Compute covered grid range
        const int startI = std::max(0, static_cast<int>((result.minX - globalMin.x()) / finalGridSize));
        const int endI = std::min(
            static_cast<int>((result.maxX - globalMin.x()) / finalGridSize) + 1,
            static_cast<int>((globalMax.x() - globalMin.x()) / finalGridSize)
        );
        
        const int startJ = std::max(0, static_cast<int>((result.minY - globalMin.y()) / finalGridSize));
        const int endJ = std::min(
            static_cast<int>((result.maxY - globalMin.y()) / finalGridSize) + 1,
            static_cast<int>((globalMax.y() - globalMin.y()) / finalGridSize)
        );

        // Traverse covered region
        for (int i = startI; i < endI; ++i) {
            for (int j = startJ; j < endJ; ++j) {
                if (globalGrid.count({i, j})) {
                    total += globalGrid[{i, j}];
                }
            }
        }


        // Neighborhood expansion
        for (int d = 0; d < 4; ++d) {
            const int ni = key.first + di[d];
            const int nj = key.second + dj[d];
            
            // Boundary check
            if (ni < 0 || nj < 0 || ni >= maxGridI || nj >= maxGridJ) {
                continue; // Skip invalid coordinates
            }
            
            std::pair<int, int> neighbor(ni, nj);
            
            if (!visited[neighbor]) {
                const int neighborCnt = globalGrid.count(neighbor) ? globalGrid[neighbor] : 0;
                
                // Dual expansion strategy
                if (total < minCount || neighborCnt > 0) {
                    visited[neighbor] = true;
                    pq.push({neighborCnt, neighbor});
                }
            }
        }
    }

    // return result;
    box.center = Eigen::Vector2d((result.minX + result.maxX) * 0.5, (result.minY + result.maxY) * 0.5);
    box.radius = Eigen::Vector2d((result.maxX - result.minX) * 0.5, (result.maxY - result.minY) * 0.5);

}

void MotionSeg::PredictBoundingBox()
{
    boxes.clear();
    for(int i = 0; i < labels.size(); i++)
    {
        BoundingBox box(Eigen::Vector2d(0, 0), Eigen::Vector2d(0, 0), labels[i]);
        FindBoundingBox(box, labels[i]);
        if(box.IsValid())
            boxes.push_back(box);
    }


    for(int i = 0; i < boxes.size(); i++)
    {
        boxes[i].UpdateBox(interval_);
    }


}

void MotionSeg::PredictLabels()
{
    for(int i = 0; i < boxes.size(); i++)
    {
        boxes[i].label.Clear();
    }

    for(int i = 0; i < nodes.size(); i++)
    {
        for(int j = 0; j < boxes.size(); j++)
        {
            Eigen::Vector2d xy = nodes[i].normal_flow.segment(0, 2);
            if(xy.x() > boxes[j].center.x() - boxes[j].radius.x() && xy.x() < boxes[j].center.x() + boxes[j].radius.x() &&
               xy.y() > boxes[j].center.y() - boxes[j].radius.y() && xy.y() < boxes[j].center.y() + boxes[j].radius.y())
            {
                boxes[j].label.AddNode(i);
            }
        }
    }

    for(int i = 0; i < boxes.size(); i++)
    {
        if(boxes[i].label.nodes_id.size() > 0)
        {
            NonLinearSolver(boxes[i].label);
        }
    }
}

void MotionSeg::ClearEmptyLabels() 
{
    labels.erase(
    std::remove_if(labels.begin(), labels.end(),
                   [](const auto& label) { return label.count == 0; }),
    labels.end());
    std::cout << "finished clear labels" << std::endl;
}


double MotionSeg::ComputeMotionError(const NormalFlowNode& node, const Eigen::Vector4d & motion_param) {
    
    Eigen::MatrixXd R(2, 3);
    R << motion_param[0] * cos(motion_param[1] * M_PI / 180) - 1, -motion_param[0] * sin(motion_param[1] * M_PI / 180) , motion_param[2],
         motion_param[0] * sin(motion_param[1] * M_PI / 180),      motion_param[0] * cos(motion_param[1] * M_PI / 180) - 1, motion_param[3];

    Eigen::Vector2d n = {node.normal_flow[2], node.normal_flow[3]};
    Eigen::Vector3d x = {node.normal_flow[0], node.normal_flow[1], 1};

    double error = abs(n.transpose() * R * x - n.squaredNorm()); 
    return error ;
}


void MotionSeg::FindNeighbors(std::vector<NormalFlowNode>& Nodes) 
{
    std::vector<double> coords;
    coords.reserve(width_ * height_ * 2);

    //events
    for(int i = 0; i < Nodes.size(); i++){
        coords.push_back(Nodes[i].origin_xy[0]);
        coords.push_back(Nodes[i].origin_xy[1]);
    }

    delaunator::Delaunator d(coords);
    std::map<size_t, std::list<size_t> > mEdges_;
    std::set<size_t> setVerticesTMP;
    
    std::vector<size_t> v2dNbVertices_;
    v2dNbVertices_.reserve(d.triangles.size() * 3);
    for(size_t i = 0; i < d.triangles.size(); i+=3)
    {
        cv::Point2i p1, p2, p3;
        p1.x = (int)d.coords[2 * d.triangles[i]];
        p1.y = (int)d.coords[2 * d.triangles[i] + 1];
        p2.x = (int)d.coords[2 * d.triangles[i + 1]];
        p2.y = (int)d.coords[2 * d.triangles[i + 1] + 1];
        p3.x = (int)d.coords[2 * d.triangles[i + 2]];
        p3.y = (int)d.coords[2 * d.triangles[i + 2] + 1];

        size_t ID1 = p1.x + p1.y * width_;
        size_t ID2 = p2.x + p2.y * width_;
        size_t ID3 = p3.x + p3.y * width_;

        if(setVerticesTMP.find(ID1) == setVerticesTMP.end())
        {
            setVerticesTMP.insert(ID1);
            v2dNbVertices_.push_back(ID1);
        }
        if(setVerticesTMP.find(ID2) == setVerticesTMP.end())
        {
            setVerticesTMP.insert(ID2);
            v2dNbVertices_.push_back(ID2);
        }
        if(setVerticesTMP.find(ID3) == setVerticesTMP.end())
        {
            setVerticesTMP.insert(ID3);
            v2dNbVertices_.push_back(ID3);
        }

        auto it = mEdges_.find(ID1);
        if(it == mEdges_.end())
        {
            mEdges_.emplace(ID1, std::list<size_t>());
            mEdges_.find(ID1)->second.push_back(ID2);
            mEdges_.find(ID1)->second.push_back(ID3);
        }
        else
        {
            it->second.push_back(ID2);
            it->second.push_back(ID3);
        }

        it = mEdges_.find(ID2);
        if(it == mEdges_.end())
        {
            mEdges_.emplace(ID2, std::list<size_t>());
            mEdges_.find(ID2)->second.push_back(ID1);
            mEdges_.find(ID2)->second.push_back(ID3);
        }
        else
        {
            it->second.push_back(ID1);
            it->second.push_back(ID3);
        }

        it = mEdges_.find(ID3);
        if(it == mEdges_.end())
        {
            mEdges_.emplace(ID3, std::list<size_t>());
            mEdges_.find(ID3)->second.push_back(ID1);
            mEdges_.find(ID3)->second.push_back(ID2);
        }
        else
        {
            it->second.push_back(ID1);
            it->second.push_back(ID2);
        }
    } 



    remove_duplicate_neighbors(mEdges_);

    std::unordered_map<Eigen::Vector2i, int, Vector2iHash> coord_index_map;
    for (size_t i = 0; i < Nodes.size(); ++i) {
        coord_index_map[Nodes[i].origin_xy] = static_cast<int>(i);
    }

    for (auto it = mEdges_.begin(); it != mEdges_.end(); ++it) {
        int uy = it->first / width_;
        int ux = it->first - uy * width_;
        Eigen::Vector2i coord(ux, uy);
        
        auto iter = coord_index_map.find(coord);
        if (iter == coord_index_map.end()) {
            std::cout << "mEdges error1 " << std::endl;
            continue;
        }
        int index = iter->second;

        // Process neighbors
        for (auto neighbor : it->second) {
            int vy = neighbor / width_;
            int vx = neighbor - vy * width_;
            Eigen::Vector2i n_coord(vx, vy);
            
            auto n_iter = coord_index_map.find(n_coord);
            if (n_iter == coord_index_map.end()) {
                std::cout << "mEdges error2" << std::endl;
                continue;
            }
            
            Nodes[index].neighbors.push_back(n_iter->second);
        }
    }
}

std::vector<NormalFlowNode> MotionSeg::processEvents(
    double start_time,
    double end_time) 
{
    std::vector<NormalFlowNode> Nodes;

    using GridValue = std::pair<NormalFlowNode, std::vector<Eigen::Vector2i>>;
    std::unordered_map<std::string, GridValue> grid_map;
    std::string evt_line, un_evt_line, flow_line;

    std::ostringstream filename_ori;
    std::ostringstream filename_sampled;

    // In data processing loop
    while (std::getline(evt_fs, evt_line)
        && std::getline(un_evt_fs, un_evt_line) 
        && std::getline(flow_fs, flow_line)
        ) 
    {
        all_event_count++;

        bool is_valid_event = true;
        dvs_msgs::Event evt;
        double ux, uy, fx, fy;
        std::istringstream evt_ss(evt_line);
        std::istringstream un_evt_ss(un_evt_line);
        std::istringstream flow_ss(flow_line);
        double timestamp;  
        std::string fx_str, fy_str;

        if (!(evt_ss >> timestamp >> evt.x >> evt.y >> evt.polarity)) is_valid_event = false;
        if (!(flow_ss >> fx_str >> fy_str) || fx_str == "nan" || fy_str == "nan")  is_valid_event = false;
        if (!(un_evt_ss >>  ux >> uy))  is_valid_event = false;

        if(is_first_event_time_)
        {
            first_event_time_ = timestamp + first_event_time_offset_;
            is_first_event_time_ = false;
        }

        if (timestamp < start_time + first_event_time_) is_valid_event = false;

        

        if (timestamp > end_time + first_event_time_) break;
        if (!is_valid_event) continue;

        evt.ts = ros::Time(timestamp);

        

        // // Intrinsics for EVIMO
        // fx = std::stof(fx_str) *257.68;
        // fy = std::stof(fy_str) * 257.625; 
      
        // Intrinsics for hkust_EMS
        fx = std::stof(fx_str) * fx_;
        fy = std::stof(fy_str) * fy_;
        // Intrinsics for DistSurf
        // fx = std::stof(fx_str) * 289.774042631025;
        // fy = std::stof(fy_str) * 288.809833981223;
        // Intrinsics for EED
        // fx = std::stof(fx_str) * 250;
        // fy = std::stof(fy_str) * 250;

        // Lu
        // fx = fx * 257.68;
        // fy = fy * 257.625;

        const Eigen::Vector4d normal_flow(ux, uy, fx, fy);
        const Eigen::Vector2i origin_xy(evt.x, evt.y);

        const int grid_x = evt.x / downsample_rate_;
        const int grid_y = evt.y / downsample_rate_;
        const std::string grid_key = std::to_string(grid_x) + "_" + std::to_string(grid_y);

        if (auto it = grid_map.find(grid_key); it != grid_map.end()) {
            auto& [node, flows] = it->second;
            
            if (evt.ts.toSec() > node.timestamp) {
                flows.emplace_back(
                    node.origin_xy
                );
                
                node.normal_flow = normal_flow;
                node.origin_xy = origin_xy;
                node.timestamp = evt.ts.toSec();
            } else {

                flows.emplace_back(origin_xy);
            }
        } else {
            grid_map.emplace(grid_key, 
                std::make_pair(
                    NormalFlowNode(evt.ts.toSec(), origin_xy, normal_flow),
                    std::vector<Eigen::Vector2i>{}
                )
            );
        }
    }

    for (auto& [key, value] : grid_map) {
        auto& [node, flows] = value;
        node.associated_flows = std::move(flows);
        Nodes.push_back(std::move(node));
    }



    std::cout << "nodes size: " << Nodes.size() << std::endl;
    if (Nodes.size() < node_thres_)
        return Nodes;

    downsamplingBG(Nodes);
    FindNeighbors(Nodes);


    return Nodes;
}


void MotionSeg::NonLinearSolver(LabelInfo& label)
{
    ceres::Problem problem;
    
    double params[4] = {label.motion_param[0], label.motion_param[1], label.motion_param[2], label.motion_param[3]}; // m0, theta, tx, ty

    double cost_before = 0;
    for(int i = 0; i < label.nodes_id.size(); i++)
    {
        Eigen::Vector4d motion_param(params[0], params[1], params[2], params[3]);
        double error = ComputeMotionError(nodes[label.nodes_id[i]], motion_param);
        cost_before += error*error;
    }

    for (int i = 0; i < label.nodes_id.size(); ++i) {

        ceres::CostFunction* cost_function = new AffineCostFunction(nodes[label.nodes_id[i]].normal_flow[0], 
                                                                    nodes[label.nodes_id[i]].normal_flow[1],
                                                                    nodes[label.nodes_id[i]].normal_flow[2], 
                                                                    nodes[label.nodes_id[i]].normal_flow[3]);
        ceres::LossFunction* loss = new ceres::HuberLoss(1e-6);
        problem.AddResidualBlock(cost_function, loss, params);
    }

   
    ceres::Solver::Options options;
    options.minimizer_type = ceres::TRUST_REGION;  
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;   
    options.use_nonmonotonic_steps = true; 
    options.max_num_iterations = 50;              
    options.function_tolerance = 1e-3;              
    options.gradient_tolerance = 1e-3;            
    options.parameter_tolerance = 1e-3;            


    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    double cost_after = 0;
    for(int i = 0; i < label.nodes_id.size(); i++)
    {
        Eigen::Vector4d motion_param(params[0], params[1], params[2], params[3]);
        double error = ComputeMotionError(nodes[label.nodes_id[i]], motion_param);
        cost_after += error*error;
    }

    if(cost_after < cost_before)
    {
        label.motion_param << params[0], params[1], params[2], params[3];
    }
    else if(cost_after == cost_before)
        std::cout << "nonlinear not change " << std::endl;
    else
        std::cout << "nonlinear failed " << std::endl;

}

bool MotionSeg::OptimizeLabel() {
    bool is_converged = true;
    for(auto it = labels.begin(); it != labels.end(); it++)
    {
        if(it->count == 0)
            continue;
        Eigen::Vector4d motion_param = it->motion_param;
        NonLinearSolver(*it);

        if(is_converged == true && isMotionDifferent(motion_param, it->motion_param, 0.01, 0.1, 1.05, 5))
            is_converged = false;
    }


    return is_converged;
}

void MotionSeg::Visualization(int t)
{

    std::sort(labels.begin(), labels.end(), 
              [](const LabelInfo& a, const LabelInfo& b) {
                  return a.nodes_id.size() > b.nodes_id.size(); 
              });



    cv::Mat canvas(height_, width_, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat canvas_next(height_, width_, CV_8UC3, cv::Scalar(255, 255, 255));
    std::array<cv::Scalar, 8> COLORS = {
        cv::Scalar(128,128,128), cv::Scalar(0,0,255),    
        cv::Scalar(255,0,0), cv::Scalar(0,255,0),  
        cv::Scalar(255,0,255), cv::Scalar(0,165,255),
        cv::Scalar(255,255,0), cv::Scalar(128,0,128) 
    };

    

    int color_index = 0;


    double timestamp = nodes[labels[0].nodes_id[0]].timestamp;

    for (size_t i = 0; i < labels.size(); ++i) {

        std::cout <<"labes id: " << i << " count: " << labels[i].count << std::endl;

        for(int j = 0; j <labels[i].nodes_id.size(); j++)
        {
                if(nodes[labels[i].nodes_id[j]].timestamp <= timestamp)
                    timestamp = nodes[labels[i].nodes_id[j]].timestamp;
                cv::circle(canvas, cv::Point2f(nodes[labels[i].nodes_id[j]].origin_xy[0], nodes[labels[i].nodes_id[j]].origin_xy[1]), 1, COLORS[color_index], cv::FILLED);
        }
        
        for(int j = 0; j <labels[i].nodes_id.size(); j++)
        {
            Eigen::Vector2d p = {nodes[labels[i].nodes_id[j]].normal_flow[0], nodes[labels[i].nodes_id[j]].normal_flow[1]};
            Eigen::MatrixXd R(2, 2);
            R << labels[i].motion_param[0] * cos(labels[i].motion_param[1] * M_PI / 180) - 1, -labels[i].motion_param[0] * sin(labels[i].motion_param[1] * M_PI / 180),
                labels[i].motion_param[0] * sin(labels[i].motion_param[1] * M_PI / 180), labels[i].motion_param[0] * cos(labels[i].motion_param[1] * M_PI / 180) - 1;
            
            Eigen::Vector2d n = {labels[i].motion_param[2], labels[i].motion_param[3]};
            Eigen::Vector2d p_next = p + (R * p + n) *(timestamp + 0.015 - nodes[labels[i].nodes_id[j]].timestamp );

            cv::circle(canvas_next, cv::Point2f(p[0], p[1]), 1, COLORS[color_index], cv::FILLED);
        }
        color_index++;

    }

    for(int i = 0; i < boxes.size(); i++)
    {
        cv::rectangle(canvas_next, cv::Point2f(boxes[i].center[0] + boxes[i].radius[0], boxes[i].center[1] + boxes[i].radius[1]),
                                cv::Point2f(boxes[i].center[0] - boxes[i].radius[0], boxes[i].center[1] - boxes[i].radius[1])  , cv::Scalar(0, 0, 0), 2);
    }
    timestamp_ofs.precision(9);
    timestamp_ofs <<  std::fixed << timestamp << std::endl;

    std::stringstream ss;
    ss << t;
    std::string filename = data_file_path_ + "results/" + ss.str() + ".png";

    cv::imwrite(filename, canvas);
    filename = data_file_path_+"results_next/" + ss.str() + "_next.png";
    cv::imwrite(filename, canvas_next);
}

bool MotionSeg::RunGraphCut(
    int& run_times 
) {
     
    int num_sites = nodes.size();
    int num_labels = labels.size();
    std::cout <<"sites: " << num_sites << " labels: " << num_labels << std::endl;
     
    if(num_sites == 0 || num_labels <= 1)
        return true;
     
    auto step1 = std::chrono::high_resolution_clock::now();
    // ========== GCO initialization ==========
    GCoptimizationGeneralGraph gco(num_sites, num_labels);
    
    if(run_times == 0)
    {
        for(int i = 0; i < num_sites; i++)
            gco.setLabel(i, 1);
    }
    else
    {
        for(int i = 0; i < num_labels; i++)
        {
            if (labels[i].count == 0)
                continue;
            for(auto it = labels[i].nodes_id.begin(); it != labels[i].nodes_id.end(); it++)
            {
                gco.setLabel(*it, i);
            }
        }
    }



    for (int site = 0; site < num_sites; ++site) {
        for (int label = 0; label < num_labels; ++label) {
            double cost = ComputeMotionError(nodes[site], labels[label].motion_param) * data_term_;
            gco.setDataCost(site, label, cost);
        }
    }
    

    for ( int l1 = 0; l1 < num_labels; l1++ )
        for (int l2 = 0; l2 < num_labels; l2++ ){
            double cost = l1==l2  ? 0:smooth_term_;
            gco.setSmoothCost(l1,l2,cost); 
        }
    
    gco.setLabelCost(label_term_);


    for (int site = 0; site < num_sites; ++site) {
        for (int neighbor : nodes[site].neighbors) {
            gco.setNeighbors(site, neighbor);
        }
    }

    auto step1_1 = std::chrono::high_resolution_clock::now();
    double time1_1 = std::chrono::duration_cast<std::chrono::microseconds>(step1_1 - step1).count();
    std::cout <<"gco energy begin: " << gco.compute_energy() << std::endl;
    gco.expansion(GraphCutIteration_);
    std::cout <<"gco energy end: " << gco.compute_energy() << std::endl;

    auto step2 = std::chrono::high_resolution_clock::now();
    double time1 = std::chrono::duration_cast<std::chrono::microseconds>(step2 - step1_1).count();

    for(int i = 0; i < labels.size(); i++)
    {
        labels[i].nodes_id.clear();
    }
    for (int i = 0; i < num_sites; ++i) {
        int id = gco.whatLabel(i);
        labels[id].nodes_id.push_back(i);
    }
    int valid_labels = 0;
    for (int i = 0; i < num_labels; i++)
    {
        labels[i].count = labels[i].nodes_id.size();
        if(labels[i].count != 0)
            valid_labels++;
    }
    std::cout <<"valid labels: " << valid_labels << std::endl;
    auto step3 = std::chrono::high_resolution_clock::now();
    double time2 = std::chrono::duration_cast<std::chrono::microseconds>(step3 - step2).count();
    
    bool is_converged = OptimizeLabel();
    auto step4 = std::chrono::high_resolution_clock::now();
    double time3 = std::chrono::duration_cast<std::chrono::microseconds>(step4 - step3).count();

    for(int i = 0; i < labels.size(); i++)
    {
        if(labels[i].count != 0)
            std::cout << "labels i: " << i << " count: " << labels[i].count << std::endl;
    }

    ClearEmptyLabels();

    run_times++;
    return is_converged;
}

void MotionSeg::SelectLabels()
{
    int total_count = 0;
    for(int i = 0; i < labels.size(); i++)
        total_count += labels[i].count;

    last_labels.clear();

    auto it = std::partition(labels.begin(), labels.end(),
        [total_count](const LabelInfo& label) {
            return !(label.count == 0 || 
                    label.count > total_count * 0.2 || 
                    label.count < total_count * 0.02);
        });

    std::move(it, labels.end(), std::back_inserter(last_labels));
    std::cout <<"last labels size: "<< last_labels.size() << std::endl;
    labels.erase(it, labels.end());
    
    // labels.clear();
}

void MotionSeg::Clear()
{
    nodes.clear();
    removed_nodes.clear();
    labels.clear();
    labels.swap(last_labels);
}



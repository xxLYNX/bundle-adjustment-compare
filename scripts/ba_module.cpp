// scripts/ba_module.cpp
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/manifold.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <chrono>
#include <iomanip>
#include <sys/resource.h>
#include <unistd.h>

// Very tiny JSON helper: assumes {"fx": ..., "fy": ..., "cx": ..., "cy": ...}
struct Intrinsics {
    double fx, fy, cx, cy;
};

struct Observation {
    int frame_id;
    int point_id;
    double u;
    double v;
};

struct ReprojectionError {
    ReprojectionError(double u_meas, double v_meas,
                      double fx, double fy, double cx, double cy)
        : u_meas_(u_meas), v_meas_(v_meas),
          fx_(fx), fy_(fy), cx_(cx), cy_(cy) {}

    template<typename T>
    bool operator()(const T* const camera,  // [qw,qx,qy,qz, tx,ty,tz]
                    const T* const point,   // [X,Y,Z]
                    T* residuals) const {

        const T* q = camera;      // qw,qx,qy,qz
        const T* t = camera + 4;  // tx,ty,tz

        T pw[3] = { point[0], point[1], point[2] };

        T pc[3];
        ceres::QuaternionRotatePoint(q, pw, pc);

        pc[0] += t[0];
        pc[1] += t[1];
        pc[2] += t[2];

        T xp = pc[0] / pc[2];
        T yp = pc[1] / pc[2];

        T u_pred = T(fx_) * xp + T(cx_);
        T v_pred = T(fy_) * yp + T(cy_);

        residuals[0] = u_pred - T(u_meas_);
        residuals[1] = v_pred - T(v_meas_);
        return true;
    }

    static ceres::CostFunction* Create(double u_meas, double v_meas,
                                       double fx, double fy, double cx, double cy) {
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 7, 3>(
            new ReprojectionError(u_meas, v_meas, fx, fy, cx, cy));
    }

    double u_meas_, v_meas_;
    double fx_, fy_, cx_, cy_;
};

// Get current memory usage in MB
double GetMemoryUsageMB() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss / 1024.0;  // Convert KB to MB
}

bool LoadIntrinsics(const std::string& path, Intrinsics& K) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Failed to open intrinsics file: " << path << "\n";
        return false;
    }
    std::string json((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());

    auto find_val = [&](const std::string& key) -> double {
        auto pos = json.find("\"" + key + "\"");
        if (pos == std::string::npos) {
            throw std::runtime_error("Key not found in intrinsics JSON: " + key);
        }
        pos = json.find(':', pos);
        if (pos == std::string::npos) {
            throw std::runtime_error("Malformed JSON near key: " + key);
        }
        std::size_t end = json.find_first_of(",}", pos + 1);
        std::string num = json.substr(pos + 1, end - pos - 1);
        return std::stod(num);
    };

    try {
        K.fx = find_val("fx");
        K.fy = find_val("fy");
        K.cx = find_val("cx");
        K.cy = find_val("cy");
    } catch (const std::exception& e) {
        std::cerr << "Error parsing intrinsics JSON: " << e.what() << "\n";
        return false;
    }
    return true;
}

bool LoadPoses(const std::string& path,
               std::vector<std::array<double,7>>& cameras) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Failed to open poses file: " << path << "\n";
        return false;
    }
    cameras.clear();
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        int frame_id;
        double tx, ty, tz, qx, qy, qz, qw;
        if (!(iss >> frame_id >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) {
            std::cerr << "Could not parse pose line: " << line << "\n";
            return false;
        }
        std::array<double,7> cam;
        cam[0] = qw;
        cam[1] = qx;
        cam[2] = qy;
        cam[3] = qz;
        cam[4] = tx;
        cam[5] = ty;
        cam[6] = tz;
        cameras.push_back(cam);
    }
    return true;
}

bool LoadPoints(const std::string& path,
                std::vector<std::array<double,3>>& points) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Failed to open points file: " << path << "\n";
        return false;
    }
    points.clear();
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        int pid;
        double X, Y, Z;
        if (!(iss >> pid >> X >> Y >> Z)) {
            std::cerr << "Could not parse point line: " << line << "\n";
            return false;
        }
        if (pid >= static_cast<int>(points.size())) {
            points.resize(pid + 1);
        }
        points[pid] = {X, Y, Z};
    }
    return true;
}

bool LoadObservations(const std::string& path,
                      std::vector<Observation>& obs) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "Failed to open observations file: " << path << "\n";
        return false;
    }
    obs.clear();
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        Observation o;
        if (!(iss >> o.frame_id >> o.point_id >> o.u >> o.v)) {
            std::cerr << "Could not parse observation line: " << line << "\n";
            return false;
        }
        obs.push_back(o);
    }
    return true;
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);

    // Parse solver type from command line (default: dogleg)
    std::string solver_type = "dogleg";
    if (argc > 1) {
        solver_type = argv[1];
        if (solver_type != "lm" && solver_type != "dogleg") {
            std::cerr << "Invalid solver type: " << solver_type << "\n";
            std::cerr << "Usage: " << argv[0] << " [lm|dogleg]\n";
            return 1;
        }
    }

    // Get root directory from environment variable, fallback to HOME/slam
    const char* ba_root_env = std::getenv("BA_ROOT");
    std::string root = ba_root_env ? std::string(ba_root_env) : (std::string(std::getenv("HOME")) + "/slam");
    std::string out  = root + "/output";

    std::string poses_path = out + "/ba_robotcar_poses.txt";
    std::string points_path = out + "/ba_robotcar_points.txt";
    std::string obs_path    = out + "/ba_robotcar_observations.txt";
    std::string intr_path   = out + "/ba_intrinsics_stereo_left.json";

    Intrinsics K;
    if (!LoadIntrinsics(intr_path, K)) {
        return 1;
    }
    std::cout << "Loaded intrinsics: fx=" << K.fx << " fy=" << K.fy
              << " cx=" << K.cx << " cy=" << K.cy << "\n";

    std::vector<std::array<double,7>> cameras;
    std::vector<std::array<double,3>> points;
    std::vector<Observation> observations;

    if (!LoadPoses(poses_path, cameras)) return 1;
    if (!LoadPoints(points_path, points)) return 1;
    if (!LoadObservations(obs_path, observations)) return 1;

    std::cout << "Loaded " << cameras.size() << " camera poses (frames), "
              << points.size() << " points, "
              << observations.size() << " observations.\n";

    ceres::Problem problem;

    for (auto& cam : cameras) {
        auto* se3_manifold = new ceres::ProductManifold(
			new ceres::QuaternionManifold(),
			new ceres::EuclideanManifold<3>());
	// Add pose block (size 7) with manifold (ambient size 7)
	problem.AddParameterBlock(cam.data(), 7, se3_manifold);
    }
    for (auto& pt : points) {
        problem.AddParameterBlock(pt.data(), 3);
    }

    if (!cameras.empty()) {
        problem.SetParameterBlockConstant(cameras[0].data());
    }

    // Use Huber loss to handle outliers robustly
    // Threshold of 2.0 pixels - errors beyond this are downweighted
    ceres::LossFunction* loss = new ceres::HuberLoss(2.0);

    for (const auto& o : observations) {
        if (o.frame_id < 0 || o.frame_id >= static_cast<int>(cameras.size())) continue;
        if (o.point_id < 0 || o.point_id >= static_cast<int>(points.size())) continue;

        ceres::CostFunction* cost =
            ReprojectionError::Create(o.u, o.v,
                                      K.fx, K.fy, K.cx, K.cy);

        problem.AddResidualBlock(
            cost,
            loss,
            cameras[o.frame_id].data(),
            points[o.point_id].data());
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 500;
    // Using Ceres default tolerances for realistic comparison

    // Set trust region strategy based on command line argument
    if (solver_type == "lm") {
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        std::cout << "Using Levenberg-Marquardt solver\n";
    } else {
        options.trust_region_strategy_type = ceres::DOGLEG;
        std::cout << "Using Dogleg solver\n";
    }

    // Measure memory before optimization
    double mem_before_mb = GetMemoryUsageMB();
    
    // Start timing for BA optimization only
    auto ba_start = std::chrono::high_resolution_clock::now();
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    auto ba_end = std::chrono::high_resolution_clock::now();
    double ba_time_sec = std::chrono::duration<double>(ba_end - ba_start).count();
    
    // Measure memory after optimization
    double mem_after_mb = GetMemoryUsageMB();
    double mem_used_mb = mem_after_mb - mem_before_mb;

    std::cout << summary.BriefReport() << "\n";
    
    // Write detailed analysis to file
    std::string output_filename = out + "/" + solver_type + "_BA.txt";
    std::ofstream analysis_file(output_filename);
    if (!analysis_file) {
        std::cerr << "Warning: Could not write analysis to " << output_filename << "\n";
    }
    
    // Helper lambda to write to both stdout and file
    auto write_line = [&](const std::string& line) {
        std::cout << line;
        if (analysis_file) analysis_file << line;
    };
    
    // Detailed Performance Analysis
    write_line("\n========================================\n");
    write_line("    BUNDLE ADJUSTMENT ANALYSIS\n");
    write_line("========================================\n");
    
    std::ostringstream oss;
    oss << "Solver: " << (solver_type == "lm" ? "Levenberg-Marquardt" : "Dogleg") << "\n";
    write_line(oss.str()); oss.str("");
    
    oss << "BA Optimization Time: " << ba_time_sec << " seconds\n";
    write_line(oss.str()); oss.str("");
    
    oss << "Total Solver Time: " << summary.total_time_in_seconds << " seconds\n";
    write_line(oss.str()); oss.str("");
    
    write_line("\nProblem Size:\n");
    oss << "  - Camera Poses: " << cameras.size() << "\n";
    write_line(oss.str()); oss.str("");
    
    oss << "  - 3D Points: " << points.size() << "\n";
    write_line(oss.str()); oss.str("");
    
    oss << "  - Observations: " << observations.size() << "\n";
    write_line(oss.str()); oss.str("");
    
    oss << "  - Parameters: " << (cameras.size() * 7 + points.size() * 3) << "\n";
    write_line(oss.str()); oss.str("");
    
    write_line("\nConvergence:\n");
    int total_iters = summary.num_successful_steps + summary.num_unsuccessful_steps;
    oss << "  - Total Iterations: " << total_iters << "\n";
    write_line(oss.str()); oss.str("");
    
    oss << "  - Successful Steps: " << summary.num_successful_steps << "\n";
    write_line(oss.str()); oss.str("");
    
    oss << "  - Unsuccessful Steps: " << summary.num_unsuccessful_steps << "\n";
    write_line(oss.str()); oss.str("");
    
    oss << "  - Initial Cost: " << summary.initial_cost << "\n";
    write_line(oss.str()); oss.str("");
    
    oss << "  - Final Cost: " << summary.final_cost << "\n";
    write_line(oss.str()); oss.str("");
    
    write_line("\nMemory Usage:\n");
    oss << "  - Peak Memory (BA): " << std::fixed << std::setprecision(2) << mem_used_mb << " MB\n";
    write_line(oss.str()); oss.str("");
    
    oss << "  - Total Process Memory: " << std::fixed << std::setprecision(2) << mem_after_mb << " MB\n";
    write_line(oss.str()); oss.str("");
    
    double cost_reduction = summary.initial_cost - summary.final_cost;
    double cost_reduction_pct = (cost_reduction / summary.initial_cost) * 100.0;
    oss << "  - Cost Reduction: " << cost_reduction << " (" 
        << std::fixed << std::setprecision(2) << cost_reduction_pct << "%)\n";
    write_line(oss.str()); oss.str("");
    
    // RMS reprojection error (2 residuals per observation)
    double rms_error = std::sqrt(summary.final_cost / observations.size());
    write_line("\nAccuracy:\n");
    oss << "  - RMS Reprojection Error: " << std::fixed << std::setprecision(3) 
        << rms_error << " pixels\n";
    write_line(oss.str()); oss.str("");
    
    oss << "  - Average Error per Observation: " << std::fixed << std::setprecision(3)
        << (summary.final_cost / observations.size()) << " pixels²\n";
    write_line(oss.str()); oss.str("");
    
    write_line("\nTermination: ");
    if (summary.termination_type == ceres::CONVERGENCE) {
        write_line("✓ CONVERGED (gradient/cost/parameter tolerance met)\n");
    } else if (summary.termination_type == ceres::NO_CONVERGENCE) {
        write_line("⚠ MAX ITERATIONS (solver could continue if given more iterations)\n");
    } else if (summary.termination_type == ceres::USER_SUCCESS) {
        write_line("✓ USER SUCCESS\n");
    } else {
        oss << "⚠ " << summary.termination_type << "\n";
        write_line(oss.str()); oss.str("");
    }
    
    write_line("\nPerformance:\n");
    oss << "  - Time per Iteration: " << std::fixed << std::setprecision(3)
        << (ba_time_sec / total_iters) * 1000.0 << " ms\n";
    write_line(oss.str()); oss.str("");
    
    oss << "  - Iterations per Second: " << std::fixed << std::setprecision(2)
        << (total_iters / ba_time_sec) << "\n";
    write_line(oss.str()); oss.str("");
    
    write_line("========================================\n");
    
    if (analysis_file) {
        analysis_file.close();
        std::cout << "\nAnalysis written to: " << output_filename << "\n";
    }

    // Write optimized poses and points
    std::string opt_poses_path = out + "/" + solver_type + "_poses_optimized.txt";
    std::string opt_points_path = out + "/" + solver_type + "_points_optimized.txt";
    
    std::ofstream poses_file(opt_poses_path);
    if (poses_file) {
        for (size_t i = 0; i < cameras.size(); ++i) {
            // Write in same format as input: frame_id tx ty tz qx qy qz qw
            // Camera array is: [qw, qx, qy, qz, tx, ty, tz]
            poses_file << i << " "
                      << std::setprecision(12) << cameras[i][4] << " "  // tx
                      << cameras[i][5] << " "  // ty
                      << cameras[i][6] << " "  // tz
                      << cameras[i][1] << " "  // qx
                      << cameras[i][2] << " "  // qy
                      << cameras[i][3] << " "  // qz
                      << cameras[i][0] << "\n"; // qw
        }
        poses_file.close();
        std::cout << "Optimized poses written to: " << opt_poses_path << "\n";
    } else {
        std::cerr << "Warning: Could not write optimized poses\n";
    }
    
    std::ofstream points_file(opt_points_path);
    if (points_file) {
        for (size_t i = 0; i < points.size(); ++i) {
            points_file << i;
            for (int j = 0; j < 3; ++j) {
                points_file << " " << std::setprecision(12) << points[i][j];
            }
            points_file << "\n";
        }
        points_file.close();
        std::cout << "Optimized points written to: " << opt_points_path << "\n";
    } else {
        std::cerr << "Warning: Could not write optimized points\n";
    }

    return 0;
}


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

    std::string root = std::string(std::getenv("HOME")) + "/slam";
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
    options.max_num_iterations = 100;  // Increased from 50

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n";

    return 0;
}


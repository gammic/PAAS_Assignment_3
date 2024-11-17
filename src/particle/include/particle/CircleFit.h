#include <ceres/ceres.h>


struct Circle{
    double x, y, r;
};

DEFINE_double(robust_threshold,
              0.0,
              "Robust loss parameter. Set to 0 for normal squared error (no "
              "robustification).");
              
class CircleFitting {
    public:
        CircleFitting(double aX, double aY, double aRadious) : mX(aX), mY(aY), mRadious(aRadious) {}
        template <typename T>
        bool operator()(const T* const aX, const T* const aY, T* aResidual) const {
            T xp = mX - *aX;
            T yp = mY - *aY;
            // sqrt() adds strong nonlinearities to the cost function, units distance^2 
            // produces more robust fits when there are outliers.
            aResidual[0] = mRadious * mRadious - xp * xp - yp * yp;
            return true;
        }
    private:
        double mX, mY, mRadious;
};

Circle 
CircleFitByCeres (std::vector<float> &aVecX, std::vector<float> &aVecY, float aRadious){

    double x = aVecX[0];
    double y = aVecY[0];

    ceres::Problem problem;
    for(size_t i = 0; i < aVecX.size(); i++){
        ceres::CostFunction* cost = 
            new ceres::AutoDiffCostFunction<CircleFitting, 1, 1, 1>(new CircleFitting(aVecX[i], aVecY[i], aRadious));
        problem.AddResidualBlock(cost, nullptr, &x, &y);
    }

    // Build and solve the problem.
    ceres::Solver::Options options;
    options.max_num_iterations = 500;
    options.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    return {x,y,aRadious};
}



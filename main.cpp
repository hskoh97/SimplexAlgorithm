#include <string>
#include <vector>
#include <Eigen/Dense>

#include "linear_program.h"

int main() {
    Eigen::VectorXd c(5);
    c << 3.0, 5.0, 0.0, 0.0, 0.0;

    double obj = 0;

    Eigen::MatrixXd A(3, 5);
    A << 1.0, 0.0, 1.0, 0.0, 0.0,
         0.0, 2.0, 0.0, 1.0, 0.0,
         3.0, 2.0, 0.0, 0.0, 1.0;

    Eigen::VectorXd b(3);
    b << 4.0, 12.0, 18.0;

    Eigen::Matrix<bool, 5, 1> BV;
    BV << false, false, true, true, true;

    double epsilon = 0.00001;
    int precision = 3;

    std::vector<std::string> variable_name = {"x1", "x2", "s1", "s2", "s3"};

    auto solution = LP::Simplex(c, A, b, BV, variable_name, epsilon, obj);

    LP::displayTableau(std::get<0>(solution), std::get<1>(solution),
            std::get<2>(solution), std::get<3>(solution),
            precision, variable_name);
}

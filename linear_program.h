//
// Created by James Koh on 05/02/2021.
//

#ifndef SIMPLEX_LINEAR_PROGRAM_H
#define SIMPLEX_LINEAR_PROGRAM_H

#endif //SIMPLEX_LINEAR_PROGRAM_H

namespace LP {
    using namespace Eigen;

    int findEBV(VectorXd& c, Matrix<bool, Dynamic, 1>& BV);
    int findPivotRowID(VectorXd& ev_col, VectorXd& b);
    int findLBV(RowVectorXd& lv_row, Matrix<bool, Dynamic, 1>& BV, double& epsilon);
    void pivotBV(int& ev_id, int& lv_id, Matrix<bool, Dynamic, 1>& BV);
    void gaussElimination(VectorXd& c, MatrixXd& A, VectorXd& b, double& obj, int& pivot_col_id, int& pivot_row_id);
    void displayTableau (VectorXd& c, double& obj, MatrixXd& A, VectorXd& b, int precision,
                         std::vector<std::string> variable_name);
    std::tuple<VectorXd, double, MatrixXd, VectorXd>
    Simplex(VectorXd c, MatrixXd A, VectorXd b, Matrix<bool, Dynamic, 1> BV,
            std::vector<std::string> variable_name, double epsilon = 0.00001, double obj = 0);
}
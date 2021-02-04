//
// Created by James Koh on 05/02/2021.
//
#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Dense>

namespace LP {
    using namespace Eigen;

    int findEBV(VectorXd& c, Matrix<bool, Dynamic, 1>& BV) {
        double max = -1;
        int ev_id = -1;

        for (int i = 0; i < c.size(); i++) {
            if (!BV(i) && max < c(i)) {
                max = c(i);
                ev_id = i;
            }
        }

        return ev_id;
    }

    int findPivotRowID(VectorXd& ev_col, VectorXd& b) {
        VectorXd r(b.size());

        for (int i = 0; i < ev_col.size(); i++) {
            if (ev_col(i) < 0)
                r(i) = -1;
            else
                r(i) = b(i) / ev_col(i);
        }

        bool init = false;
        int row = -1;
        double r_val = -1;

        for (int i = 0; i < r.size(); i++) {
            if (r(i) != -1) {
                if (!init) {
                    init = true;
                    row = i;
                    r_val = r(i);
                } else if (r(i) < r_val) {
                    row = i;
                    r_val = r(i);
                }
            }
        }

        return row;
    }

    int findLBV(RowVectorXd& lv_row, Matrix<bool, Dynamic, 1>& BV, double& epsilon) {
        for (int i = 0; i < lv_row.size(); i++) {
            if (BV(i) && lv_row(i) > (1 - epsilon) && lv_row(i) < (1 + epsilon)) {
                return i;
            }
        }

        throw std::logic_error("Leaving pivot is not 1 despite finding pivot row.");
    }

    void pivotBV(int& ev_id, int& lv_id, Matrix<bool, Dynamic, 1>& BV) {
        BV(ev_id) = true;
        BV(lv_id) = false;
    }

    void gaussElimination(VectorXd& c, MatrixXd& A, VectorXd& b, double& obj, int& pivot_col_id, int& pivot_row_id) {
        // updating A and b
        // pivot scaling transformation matrix E1
        MatrixXd E1 = MatrixXd::Identity(A.rows(),A.rows());
        E1(pivot_row_id, pivot_row_id) = 1 / A(pivot_row_id, pivot_col_id);

        // gaussian elimination transformation matrix E2
        MatrixXd E2 = MatrixXd::Identity(A.rows(),A.rows());
        for (int i = 0; i < E2.cols(); i++) {
            if (i != pivot_row_id)
                E2(i, pivot_row_id) = -A(i, pivot_col_id);
        }

        A = E2 * E1 * A;
        b = E2 * E1 * b;

        // updating c and obj
        double m = c(pivot_col_id);
        c = c - m * (A.row(pivot_row_id).transpose());
        obj = obj + m * b(pivot_row_id);
    }

    void displayTableau (VectorXd& c, double& obj, MatrixXd& A, VectorXd& b, int precision,
                         std::vector<std::string> variable_name) {
        std::cout << std::fixed;
        std::cout.precision(precision);
        std::string blank = "  ";

        for (auto it = variable_name.begin(); it < variable_name.end(); ++it) {
            std::cout << "|" << blank << *it << blank;
        }
        std::cout << "|" << blank << "obj" << std::endl;
        std::cout << "------------------------------------------------------" << std::endl;
        std::cout << c.transpose() << blank << obj << std::endl;

        for (auto it = variable_name.begin(); it < variable_name.end(); ++it) {
            std::cout << "|" << blank << *it << blank;
        }
        std::cout << "|" << blank << "b" << std::endl;
        std::cout << "------------------------------------------------------" << std::endl;
        MatrixXd Ab(A.rows(), A.cols() + 1);
        Ab << A, b;
        std::cout << Ab << std::endl;
    }

    std::tuple<VectorXd, double, MatrixXd, VectorXd>
    Simplex(VectorXd c, MatrixXd A, VectorXd b, Matrix<bool, Dynamic, 1> BV,
            std::vector<std::string> variable_name, double epsilon = 0.00001, double obj = 0) {
        int i = 1;

        while (true) {
            int ev_id = findEBV(c, BV);

            if (ev_id == -1) {
                std::cout << "########" << std::endl;
                std::cout << "Optimal solution found." << std::endl;
                std::cout << "Terminating ..." << std::endl;
                std::cout << "########" << std::endl;

                return {c, obj, A, b};
            }

            std::cout << "########" << std::endl;
            std::cout << "Iteration " << i << ":" << std::endl;
            std::cout << "########" << std::endl;

            VectorXd ev_column = A.col(ev_id);
            int pivot_row_id = findPivotRowID(ev_column, b);

            if (pivot_row_id == -1) {
                throw std::runtime_error("The linear program is unbounded.");
            }

            RowVectorXd lv_row = A.row(pivot_row_id);
            int lv_id = findLBV(lv_row, BV, epsilon);
            pivotBV(ev_id, lv_id, BV);

            gaussElimination(c, A, b, obj, ev_id, pivot_row_id);
            displayTableau(c, obj, A, b, 3, variable_name);

            i++;
        }
    }

}
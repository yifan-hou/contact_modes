#pragma once
#include <Eigen/Dense>


void get_rows(const Eigen::MatrixXd& A, const Eigen::VectorXi& idx, Eigen::MatrixXd& out);
void get_cols(const Eigen::MatrixXd& A, const Eigen::VectorXi& idx, Eigen::MatrixXd& out);
Eigen::MatrixXd get_rows(const Eigen::MatrixXd& A, const Eigen::VectorXi& idx);
Eigen::MatrixXd get_cols(const Eigen::MatrixXd& A, const Eigen::VectorXi& idx);

void get_rows(const Eigen::VectorXd& A, const Eigen::VectorXi& idx, Eigen::VectorXd& out);
Eigen::VectorXd get_rows(const Eigen::VectorXd& A, const Eigen::VectorXi& idx);

void image_basis(const Eigen::MatrixXd& A, Eigen::MatrixXd& image, double eps);
void kernel_basis(const Eigen::MatrixXd& A, Eigen::MatrixXd& null, double eps);
void image_and_kernel_bases(const Eigen::MatrixXd& A, Eigen::MatrixXd& image, 
                            Eigen::MatrixXd& kernel, double eps);

Eigen::MatrixXd image_basis(const Eigen::MatrixXd& A, double eps);
Eigen::MatrixXd kernel_basis(const Eigen::MatrixXd& A, double eps);
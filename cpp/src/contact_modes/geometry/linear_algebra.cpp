#include <contact_modes/geometry/linear_algebra.hpp>
#include <iostream>

static int DEBUG=0;

void get_rows(const Eigen::MatrixXd& A, 
              const Eigen::VectorXi& idx, 
              Eigen::MatrixXd& out)
{
    int n = idx.size();
    int c = A.cols();
    out.resize(n, c);
    for (int i = 0; i < n; i++) {
        out.row(i) = A.row(idx[i]);
    }
}

void get_cols(const Eigen::MatrixXd& A, 
              const Eigen::VectorXi& idx, 
              Eigen::MatrixXd& out)
{
    int n = idx.size();
    int r = A.rows();
    out.resize(r, n);
    for (int i = 0; i < n; i++) {
        out.col(i) = A.col(idx[i]);
    }
}

Eigen::MatrixXd get_rows(const Eigen::MatrixXd& A, const Eigen::VectorXi& idx) {
    Eigen::MatrixXd out;
    get_rows(A, idx, out);
    return out;
}

Eigen::MatrixXd get_cols(const Eigen::MatrixXd& A, const Eigen::VectorXi& idx) {
    Eigen::MatrixXd out;
    get_cols(A, idx, out);
    return out;
}

void get_rows(const Eigen::VectorXd& A, 
              const Eigen::VectorXi& idx, 
              Eigen::VectorXd& out)
{
    int n = idx.size();
    out.resize(n);
    for (int i = 0; i < n; i++) {
        out[i] = A[idx[i]];
    }
}

Eigen::VectorXd get_rows(const Eigen::VectorXd& A, const Eigen::VectorXi& idx) {
    Eigen::VectorXd out;
    get_rows(A, idx, out);
    return out;
}

void image_basis(const Eigen::MatrixXd& A, Eigen::MatrixXd& image, double eps) {
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod;
    cod.setThreshold(eps);
    cod.compute(A);
    image = cod.matrixQ();
}

void kernel_basis(const Eigen::MatrixXd& A, Eigen::MatrixXd& kernel, double eps) {
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod;
    cod.setThreshold(eps);
    cod.compute(A);
    Eigen::MatrixXd Z = cod.matrixZ();
    Eigen::MatrixXd P_inv = cod.colsPermutation().inverse();
    kernel = (Z * P_inv).bottomRows(cod.dimensionOfKernel()).transpose();
    if (DEBUG) {
        Eigen::MatrixXd Q = cod.matrixQ();
        Eigen::MatrixXd T = cod.matrixT().topLeftCorner(cod.rank(), cod.rank()).triangularView<Eigen::Upper>();
        std::cout << "A\n" << A << std::endl;
        std::cout << "Q\n" << Q << std::endl;
        std::cout << "T\n" << T << std::endl;
        std::cout << "ZP\n" << Z*P_inv << std::endl;
        std::cout << "TZP\n" << T*Z*P_inv << std::endl;
        std::cout << "QTZP\n" << Q*T*Z*P_inv << std::endl;
        std::cout << "kernel\n" << kernel << std::endl;
        std::cout << "A*(ZP)^T\n" << A*(Z*P_inv).transpose() << std::endl;
    }
}

void image_and_kernel_bases(const Eigen::MatrixXd& A, Eigen::MatrixXd& image,
                            Eigen::MatrixXd& kernel, double eps) {
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod;
    cod.setThreshold(eps);
    cod.compute(A);
    image = cod.matrixQ();
    Eigen::MatrixXd Z = cod.matrixZ();
    Eigen::MatrixXd P_inv = cod.colsPermutation().inverse();
    kernel = (Z * P_inv).bottomRows(cod.dimensionOfKernel()).transpose();
}

Eigen::MatrixXd image_basis(const Eigen::MatrixXd& A, double eps) {
    Eigen::MatrixXd image;
    image_basis(A, image, eps);
    return image;
}

Eigen::MatrixXd kernel_basis(const Eigen::MatrixXd& A, double eps) {
    Eigen::MatrixXd kernel;
    kernel_basis(A, kernel, eps);
    return kernel;
}
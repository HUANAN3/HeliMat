// =====================================================================================
// File:          HeliMat.h
// Project:       HeliMat Numerical Calculation Library
// Author:        Junbiao Shen
// Email:         junbiaoshen@nuaa.edu.cn
// Date:          2024-09-25
// Version:       1.0
// License:       MIT License
// Description:   This file contains the declaration and definition of the Matrix class, which provides
//                basic matrix operations.
// =====================================================================================

/*
 * MIT License
 * 
 * Copyright (c) 2024 Junbiao Shen
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include <iostream>
#include <vector>
#include <initializer_list>
#include <iomanip>
#include <typeinfo>
#include <chrono>

template <typename Ty>
class RowProxy;

template <typename Ty>
class ColumnProxy;

template <typename Ty>
class Matrix
{
public:
    Matrix() : rowSize(0), colSize(0) {};
    Matrix(size_t row, size_t col, std::initializer_list<Ty> iniList);
    Matrix(size_t row, size_t col);
    auto row(size_t indOfRow) -> RowProxy<Ty>;
    auto col(size_t indOfCol) -> ColumnProxy<Ty>;
    auto swapRows(size_t row1, size_t row2) -> void;
    auto swapCols(size_t col1, size_t col2) -> void;
    auto transpose() -> Matrix<Ty>;
    auto det() -> Ty;
    auto dot(const Matrix<Ty>& other) -> Ty;
    auto cross(const Matrix<Ty>& other) -> Matrix<Ty>;
    auto hadamard(const Matrix<Ty>& other) const -> Matrix<Ty>;
    auto inv() -> Matrix<Ty>;
    auto getRowSize() -> size_t;
    auto getColSize() -> size_t;
    auto numOfElements() const -> size_t;
    auto print() -> void;
    auto operator()(size_t indOfRow, size_t indOfCol) -> Ty&;
    auto operator()(size_t indOfRow, size_t indOfCol) const -> const Ty&;
    auto operator>>(Ty inputValue) -> Matrix<Ty>&;
    auto operator,(Ty inputValue) -> Matrix<Ty>&;
    auto operator=(const Matrix<Ty>& other) -> Matrix<Ty>&;
    auto operator+(const Matrix<Ty>& other) -> Matrix<Ty>;
    auto operator-(const Matrix<Ty>& other) -> Matrix<Ty>;
    auto operator*(const Matrix<Ty>& other) const -> Matrix<Ty>;
    auto operator*(Ty other) const -> Matrix<Ty>;
    auto operator/(Matrix<Ty> a) -> Matrix<Ty>;
    template <typename T>
    friend auto operator<<(std::ostream& os, const Matrix<T>& matrix) -> std::ostream&;
private:
    size_t rowSize, colSize;
    size_t indOfCur = 0;
    std::vector<Ty> mat;
}; 

template <typename Ty>
class RowProxy
{
public:
    RowProxy(Matrix<Ty>& matrix, size_t indOfRow);
    auto getRowSize() const -> size_t;
    auto operator=(const std::vector<Ty>& inputValue) -> RowProxy<Ty>&;
    auto operator=(const RowProxy<Ty>& otherRow) -> RowProxy<Ty>&;
    template <typename T>
    friend auto operator<<(std::ostream& os, const RowProxy<T>& matrixRow) -> std::ostream&;
private:
    Matrix<Ty>& matrix;
    size_t indOfRow;
};

template <typename Ty>
class ColumnProxy
{
public:
    ColumnProxy(Matrix<Ty>& matrix, size_t indOfCol);
    auto getColSize() const -> size_t;
    auto operator=(const std::vector<Ty>& inputValue) -> ColumnProxy<Ty>&;
    auto operator=(const ColumnProxy<Ty>& otherCol) -> ColumnProxy<Ty>&;
    template <typename T>
    friend auto operator<<(std::ostream& os, const ColumnProxy<T>& matrixCol) -> std::ostream&;
private:
    Matrix<Ty>& matrix;
    size_t indOfCol;
};

template <typename Ty>
auto eyes(size_t MatrixSize) -> Matrix<Ty>;

template <typename Ty> 
Matrix<Ty>::Matrix(size_t row, size_t col, std::initializer_list<Ty> inilist) 
 : rowSize(row), colSize(col), mat(inilist)
{
    if (rowSize * colSize != inilist.size())
        throw std::out_of_range("Initializer list size does no match matrix dimensions.");
}

template <typename Ty>
Matrix<Ty>::Matrix(size_t row, size_t col)
 : rowSize(row), colSize(col), mat(row * col)
{}

template <typename Ty>
auto Matrix<Ty>::row(size_t indOfRow) -> RowProxy<Ty>
{
    return RowProxy<Ty>(*this, indOfRow);
}

template <typename Ty>
auto Matrix<Ty>::col(size_t indOfCol) -> ColumnProxy<Ty>
{
    return ColumnProxy<Ty>(*this, indOfCol);
}

template <typename Ty>
auto Matrix<Ty>::swapRows(size_t row1, size_t row2) -> void
{
    if (row1 > rowSize || row2 > rowSize)
        throw std::out_of_range("矩阵选取行数越界！");
    if (row1 == row2)
        return;
    size_t row1Start = (row1 - 1) * colSize;
    size_t row2Start = (row2 - 1) * colSize; 
    std::swap_ranges(mat.begin() + row1Start, mat.begin() + row1Start + colSize, mat.begin() + row2Start);
}

template <typename Ty>
auto Matrix<Ty>::swapCols(size_t col1, size_t col2) -> void
{
    if (col1 > colSize || col2 > colSize)
        throw std::out_of_range("矩阵选取列数越界！");
    if (col1 == col2)
        return;
    for (size_t ind = 0; ind < colSize; ++ind) {
        std::iter_swap(mat.begin() + ind * colSize + (col1 - 1), mat.begin() + ind * colSize + (col2 - 1));
    }
}

template <typename Ty>
auto Matrix<Ty>::transpose() -> Matrix<Ty>
{
    Matrix<Ty> result(colSize, rowSize);
    for (size_t indOfRow = 1; indOfRow <= rowSize; ++indOfRow) {
        for (size_t indOfCol = 1; indOfCol <= colSize; ++indOfCol) {
            result(indOfCol, indOfRow) = (*this)(indOfRow, indOfCol);
        }
    }
    return result;
} 

template <typename Ty>
auto Matrix<Ty>::det() -> Ty
{
    if (colSize != rowSize || rowSize == 0)
        throw std::length_error("矩阵行数和列数不相等！");
    Matrix<Ty> U(rowSize, colSize);
    U = *this;
    for (size_t colInd = 1; colInd <= colSize - 1; ++colInd) {
        size_t rowPivot = colInd;
        for (size_t rowInd = colInd + 1; rowInd <= rowSize; ++rowInd) {
            if (U(rowPivot, colInd) < U(rowInd, colInd)) {
                rowPivot = rowInd;
            }
        }
        if (abs(U(rowPivot, colInd)) < 1.0e-9)
            throw std::logic_error("矩阵奇异，不可奇异值分解！");
        U.swapRows(rowPivot, colInd);
        for (size_t rowInd = colInd + 1; rowInd <= rowSize; ++rowInd) {
            Ty multiCoef = U(rowInd, colInd) / U(colInd, colInd);
            for (size_t colIndInRow = colInd; colIndInRow <= colSize; ++colIndInRow) {
                U(rowInd, colIndInRow) -= U(colInd, colIndInRow) * multiCoef; 
            }
            U(rowInd, colInd) = multiCoef;
        }
    }
    Ty result = 1;
    for (size_t ind = 1; ind <= U.rowSize; ++ind) {
        result *= U(ind, ind);
    }
    return result;
}

template <typename Ty>
auto Matrix<Ty>::dot(const Matrix<Ty>& other) -> Ty
{
    if (this->colSize != other.rowSize || (colSize != 1 && rowSize != 1) 
        || (other.colSize != 1 && other.rowSize != 1))
        throw std::logic_error("其中有一方不是向量或者向量形状不满足乘法要求！");
    Ty sum = 0;
    for (size_t ind = 0; ind < numOfElements(); ++ind) {
        sum += mat[ind] * other.mat[ind];
    }
    return sum;
}

template <typename Ty>
auto Matrix<Ty>::cross(const Matrix<Ty>& other) -> Matrix<Ty>
{
    if (colSize * rowSize != 3 || other.colSize * other.rowSize != 3)
        throw std::logic_error("叉积向量不是三维向量！");
    if (colSize != other.colSize)
        throw std::logic_error("叉积向量形状不一样！");
    Matrix<Ty> resultVector(3, 1);
    if (this->colSize == 3) {
        (*this) = (*this).transpose();
        other = other.transpose();
    }
        resultVector(1, 1) = (*this)(2, 1) * other(3, 1) - (*this)(3, 1) * other(2, 1);
        resultVector(2, 1) = (*this)(3, 1) * other(1, 1) - (*this)(1, 1) * other(3, 1);
        resultVector(3, 1) = (*this)(1, 1) * other(2, 1) - (*this)(2, 1) * other(1, 1);
    if (this->colSize == 3) {
        return resultVector.transpose();
    } 
    else {
        return resultVector;
    }
}

template <typename Ty>
auto Matrix<Ty>::hadamard(const Matrix<Ty>& other) const -> Matrix<Ty>
{
    if (rowSize != other.rowSize || colSize != other.colSize)
        throw std::logic_error("矩阵行数列数不相等！");
    Matrix<Ty> resultMatrix(rowSize, colSize);
    for (size_t ind = 0; ind < numOfElements(); ++ind) {
        resultMatrix.mat[ind] = this->mat[ind] * other.mat[ind];
    }
    return resultMatrix;
}

template <typename Ty>
auto Matrix<Ty>::inv() -> Matrix<Ty>
{
    if (colSize != rowSize || rowSize == 0)
        throw std::length_error("矩阵行数和列数不相等！");
    Matrix<Ty> P(rowSize, colSize);
    Matrix<Ty> U(rowSize, colSize);
    for (size_t ind = 1; ind <= rowSize; ++ind) {
        P(ind, ind) = 1.0;
    }
    U = *this;
    for (size_t colInd = 1; colInd <= colSize - 1; ++colInd) {
        size_t rowPivot = colInd;
        for (size_t rowInd = colInd + 1; rowInd <= rowSize; ++rowInd) {
            if (U(rowPivot, colInd) < U(rowInd, colInd)) {
                rowPivot = rowInd;
            }
        }
        if (abs(U(rowPivot, colInd)) < 1.0e-9)
            throw std::logic_error("矩阵奇异，不可奇异值分解！");
        U.swapRows(rowPivot, colInd);
        P.swapRows(rowPivot, colInd);
        for (size_t rowInd = colInd + 1; rowInd <= rowSize; ++rowInd) {
            Ty multiCoef = U(rowInd, colInd) / U(colInd, colInd);
            for (size_t colIndInRow = colInd; colIndInRow <= colSize; ++colIndInRow) {
                U(rowInd, colIndInRow) -= U(colInd, colIndInRow) * multiCoef; 
            }
            U(rowInd, colInd) = multiCoef;
        }
    }
    Matrix<Ty> invMatrix(rowSize, colSize);
    for (size_t indOfInv = 1; indOfInv <= colSize; ++indOfInv) {
        for (size_t indOfRow = 1; indOfRow <= rowSize; ++indOfRow) {
            invMatrix(indOfRow, indOfInv) = P(indOfRow, indOfInv);
            for (size_t indOfCol = 1; indOfCol < indOfRow; ++indOfCol) {
                invMatrix(indOfRow, indOfInv) -= U(indOfRow, indOfCol) * invMatrix(indOfCol, indOfInv);
            }
        }
    }
    for (size_t indOfInv = 1; indOfInv <= colSize; ++indOfInv) {
        for (size_t indOfRow = rowSize; indOfRow >= 1; --indOfRow) {
            for (size_t indOfCol = indOfRow + 1; indOfCol <= colSize; ++indOfCol) {
                invMatrix(indOfRow, indOfInv) -= U(indOfRow, indOfCol) * invMatrix(indOfCol, indOfInv);
            }
            invMatrix(indOfRow, indOfInv) /= U(indOfRow, indOfRow);
        }
    }
    return invMatrix;
}

template <typename Ty>
auto Matrix<Ty>::getRowSize() -> size_t
{
    return rowSize;
}

template <typename Ty>
auto Matrix<Ty>::getColSize() -> size_t
{
    return colSize;
}

template <typename Ty>
auto Matrix<Ty>::numOfElements() const -> size_t 
{
    return mat.size();
}

template <typename Ty>
auto Matrix<Ty>::print() -> void
{
    int ind = 0;
    for (auto &value : mat) {
        std::cout << std::right << std::fixed << std::setprecision(6) << std::setw(10) << value << " ";
        ++ind;
        if (ind % colSize == 0)
            std::cout << '\n'; 
    }
    std::cout << std::endl;
}

template <typename Ty>
auto Matrix<Ty>::operator()(size_t indOfRow, size_t indOfCol) -> Ty&
{
    return this->mat[colSize * (indOfRow - 1) + indOfCol - 1];
} 

template <typename Ty>
auto Matrix<Ty>::operator()(size_t indOfRow, size_t indOfCol) const -> const Ty&
{
    return this->mat[colSize * (indOfRow - 1) + indOfCol - 1];
} 

template <typename Ty>
auto Matrix<Ty>::operator>>(Ty inputValue) -> Matrix<Ty>&
{
    if (indOfCur > mat.size())
        throw std::out_of_range("输入元素个数超过矩阵容量");
    mat[indOfCur++] = inputValue;
    return *this;
}

template <typename Ty>
auto Matrix<Ty>::operator,(Ty inputValue) -> Matrix<Ty>&
{
    if (indOfCur > mat.size())
        throw std::out_of_range("输入元素个数超过矩阵容量");
    mat[indOfCur++] = inputValue;
    return *this;
}

template <typename Ty>
auto Matrix<Ty>::operator=(const Matrix<Ty>& other) -> Matrix<Ty>&
{
    this->mat = other.mat;
    this->colSize = other.colSize;
    this->rowSize = other.rowSize;
    return *this;
}

template <typename Ty>
auto Matrix<Ty>::operator+(const Matrix<Ty>& other) -> Matrix<Ty>
{
    if (rowSize != other.rowSize || colSize != other.colSize)
        throw std::logic_error("矩阵行数列数不相等！");
    Matrix<Ty> resultMatrix(rowSize, colSize);
    for (size_t ind = 0; ind < numOfElements(); ++ind) {
        resultMatrix.mat[ind] = this->mat[ind] + other.mat[ind];
    }
    return resultMatrix;
}

template <typename Ty>
auto Matrix<Ty>::operator-(const Matrix<Ty>& other) -> Matrix<Ty>
{
    if (rowSize != other.rowSize || colSize != other.colSize)
        throw std::logic_error("矩阵行数列数不相等！");
    Matrix<Ty> resultMatrix(rowSize, colSize);
    for (size_t ind = 0; ind < numOfElements(); ++ind) {
        resultMatrix.mat[ind] = this->mat[ind] - other.mat[ind];
    }
    return resultMatrix;
}

template <typename Ty>
auto operator<<(std::ostream& os, const Matrix<Ty>& matrix) -> std::ostream&
{
    size_t ind = 0;
    for (auto value : matrix.mat) {
        os << std::right << std::fixed << std::setprecision(6) << std::setw(10) << value << " ";
        ++ind;
        if (ind % matrix.colSize == 0)
            os << '\n';
    }
    return os;
}

template <typename Ty>
auto Matrix<Ty>::operator*(const Matrix<Ty>& other) const -> Matrix<Ty>
{
    if (colSize != other.rowSize)
        throw std::invalid_argument("矩阵相乘维数不匹配!");
    Matrix<Ty> C(rowSize, other.colSize);
    for (size_t i = 1; i <= rowSize; ++i) {
        for (size_t j = 1; j <= other.colSize; ++j) {
            for (size_t k = 1; k <= colSize; ++k) {
                C(i, j) = C(i, j) + (*this)(i, k) * other(k, j);  //索引从1开始！
            }
        }
    }
    return C;
}

template <typename Ty>
auto Matrix<Ty>::operator*(Ty other) const -> Matrix<Ty>
{
    Matrix<Ty> resultMatrix(rowSize, colSize);
    for (size_t ind = 0; ind < numOfElements(); ++ind) {
        resultMatrix.mat[ind] = mat[ind] * other;
    }
    return resultMatrix;
}

template <typename Ty>
auto operator*(Ty scalar, const Matrix<Ty>& matrix) -> Matrix<Ty> {
    return matrix * scalar; 
}

auto operator*(int scalar, const Matrix<double>& matrix) -> Matrix<double> {
    return matrix * scalar; 
}

template <typename Ty>
auto Matrix<Ty>::operator/(Matrix<Ty> other) -> Matrix<Ty>
{
    if (this->rowSize != other.colSize || other.rowSize != other.colSize)
        throw std::logic_error("方程组求解维数不匹配或者Ax=b中A矩阵不是方阵！");
    Matrix<Ty> B(rowSize, colSize);
    Matrix<Ty> A(other.rowSize, other.colSize);
    B = *this;
    A = other;
    for (size_t indOfCol = 1; indOfCol <= A.colSize - 1; ++indOfCol) {
        size_t rowPivot = indOfCol;
        for (size_t rowInd = indOfCol + 1; rowInd <= A.rowSize; ++rowInd) {
            if (A(rowPivot, indOfCol) < A(rowInd, indOfCol)) {
                rowPivot = rowInd;
            }
        }
        if (abs(A(rowPivot, indOfCol)) < 1.0e-6)
            throw std::logic_error("矩阵奇异，不可奇异值分解！");
        A.swapRows(rowPivot, indOfCol);
        B.swapRows(rowPivot, indOfCol);
        for (size_t indOfRow = indOfCol + 1; indOfRow <= A.rowSize; ++indOfRow) {
            Ty multiCoef = A(indOfRow, indOfCol) / A(indOfCol, indOfCol);
            for (size_t colOfRow = indOfCol; colOfRow <= A.colSize; ++colOfRow) {
                A(indOfRow, colOfRow) -= multiCoef * A(indOfCol, colOfRow);
            }
            for (size_t colOfRow = 1; colOfRow <= B.colSize; ++colOfRow) {
                B(indOfRow, colOfRow) -= multiCoef * B(indOfCol, colOfRow);
            }
            
        }
    }
    Matrix<Ty> invMatrix(rowSize, colSize);
    for (size_t indOfInv = 1; indOfInv <= B.colSize; ++indOfInv) {
        for (size_t indOfRow = A.rowSize; indOfRow >= 1; --indOfRow) {
            invMatrix(indOfRow, indOfInv) = B(indOfRow, indOfInv);
            for (size_t indOfCol = indOfRow + 1; indOfCol <= A.colSize; ++indOfCol) {
                invMatrix(indOfRow, indOfInv) -= A(indOfRow, indOfCol) * invMatrix(indOfCol, indOfInv);
            }
            invMatrix(indOfRow, indOfInv) /= A(indOfRow, indOfRow);
        }
    }
    return invMatrix;
}

template <typename Ty>
ColumnProxy<Ty>::ColumnProxy(Matrix<Ty>& matrix, size_t indOfCol)
 : matrix(matrix), indOfCol(indOfCol) 
{}

template <typename Ty>
auto ColumnProxy<Ty>::getColSize() const -> size_t
{
    return matrix.getRowSize();
}

template <typename Ty>
auto ColumnProxy<Ty>::operator=(const std::vector<Ty>& inputValue) -> ColumnProxy<Ty>&
{
    if (inputValue.size() != matrix.getRowSize())
        throw std::out_of_range("矩阵列元素个数赋值不匹配！");
    for (size_t ind = 0; ind < matrix.getRowSize(); ++ind) {
        matrix(ind + 1, indOfCol) = inputValue[ind];
    }
    return *this;
}

template <typename Ty>
auto ColumnProxy<Ty>::operator=(const ColumnProxy<Ty>& otherCol) -> ColumnProxy<Ty>&
{
    if (this->getColSize() != otherCol.getColSize()) 
        throw std::out_of_range("列向量大小不匹配！");
    for (size_t indOfRow = 1; indOfRow <= this->getColSize(); ++indOfRow) {
        matrix(indOfRow, indOfCol) = otherCol.matrix(indOfRow, otherCol.indOfCol);
    }
    return *this;
}

template <typename Ty>
auto operator<<(std::ostream& os, const ColumnProxy<Ty>& matrixCol) -> std::ostream&
{
    for (size_t indOfRow = 1; indOfRow <= matrixCol.getColSize(); ++indOfRow) {
        os << std::right << std::fixed << std::setprecision(6) << std::setw(10) << 
            matrixCol.matrix(indOfRow, matrixCol.indOfCol) << '\n';
    }
    os << std::endl;
    return os;
}

template <typename Ty>
RowProxy<Ty>::RowProxy(Matrix<Ty>& matrix, size_t indOfRow)
 : matrix(matrix), indOfRow(indOfRow) 
{}

template <typename Ty>
auto RowProxy<Ty>::getRowSize() const -> size_t
{
    return matrix.getColSize();
}

template <typename Ty>
auto RowProxy<Ty>::operator=(const std::vector<Ty>& inputValue) -> RowProxy<Ty>&
{
    if (inputValue.size() != matrix.getColSize())
        throw std::out_of_range("矩阵行元素个数赋值不匹配！");
    for (size_t ind = 0; ind < matrix.getColSize(); ++ind) {
        matrix(indOfRow, ind + 1) = inputValue[ind];
    }
    return *this;
}

template <typename Ty>
auto RowProxy<Ty>::operator=(const RowProxy<Ty>& otherRow) -> RowProxy<Ty>&
{
    if (this->getRowSize() != otherRow.getRowSize()) 
        throw std::out_of_range("行向量大小不匹配！");
    for (size_t indOfCol = 1; indOfCol <= this->getRowSize(); ++indOfCol) {
        matrix(indOfRow, indOfCol) = otherRow.matrix(otherRow.indOfRow, indOfCol);
    }
    return *this;
}

template <typename Ty>
auto operator<<(std::ostream& os, const RowProxy<Ty>& matrixRow) -> std::ostream&
{
    for (size_t indOfCol = 1; indOfCol <= matrixRow.getRowSize(); ++indOfCol) {
        os << std::right << std::fixed << std::setprecision(6) << std::setw(10) 
            << matrixRow.matrix(matrixRow.indOfRow, indOfCol);
    }
    os << std::endl;
    return os;
}

template <typename Ty> 
auto eyes(size_t MatrixSize) -> Matrix<Ty>
{
    Matrix<Ty> resultMatrix(MatrixSize, MatrixSize);
    for (size_t ind = 1; ind <= MatrixSize; ++ind) {
        resultMatrix(ind, ind) = 1;
    }
    return resultMatrix;
}
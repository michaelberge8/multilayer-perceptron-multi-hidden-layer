import random
import math

'''
    File name: main.py
    Author: Michael Berge
    Date created: 7/19/2018
    Date modified: 7/19/2018
    Python Version: 3.8.1
'''

class Matrix:
    def __init__(self, rows, cols):
        self.__rows = rows
        self.__cols = cols
        self.__data = [[0 for x in range(cols)] for y in range(rows)]

    # Matrix Product (a • b)
    # Return matrix product of matrix m1 and matrix m2
    @staticmethod
    def multiply(m1, m2):
        if m1.__cols == m2.__rows:
            result = Matrix(m1.__rows, m2.__cols)
            for i in range(result.__rows):
                for j in range(result.__cols):
                    total = 0
                    for k in range(m1.__cols):
                        total += m1.__data[i][k] * m2.__data[k][j]
                    result.__data[i][j] = total
            return result
        else:
            raise ValueError('dimension mismatch (multiply)')

    # Hadamard product (a • b)
    # Multiply matrix m or n by self
    def multiply_(self, m):
        if isinstance(m, Matrix):
            if self.__rows == m.__rows and self.__cols == m.__cols:
                for i in range(self.__rows):
                    for j in range(self.__cols):
                        self.__data[i][j] *= m.__data[i][j]
            else:
                raise ValueError('dimension mismatch (multiply_)')
        else:
            for i in range(self.__rows):
                for j in range(self.__cols):
                    self.__data[i][j] *= m

    # Hadamard product (c)
    # Add matrix m or n to self
    def add(self, m=None, n=None):
        if m is not None:
            if self.__rows == m.__rows and self.__cols == m.__cols:
                for i in range(self.__rows):
                    for j in range(self.__cols):
                        self.__data[i][j] += m.__data[i][j]
            else:
                raise ValueError('dimension mismatch (add)')
        else:
            for i in range(self.__rows):
                for j in range(self.__cols):
                    self.__data[i][j] += n

    @staticmethod
    def subtract(m1, m2):
        if m1.__rows == m2.__rows and m1.__cols == m2.__cols:
            result = Matrix(m1.__rows, m2.__cols)
            for i in range(result.__rows):
                for j in range(result.__cols):
                    result.__data[i][j] = m1.__data[i][j] - m2.__data[i][j]
            return result
        else:
            raise ValueError('dimension mismatch (subtract)')

    @staticmethod
    def to_array(m):
        count = 0
        result = []
        for i in range(m.__rows):
            for j in range(m.__cols):
                result.append(m.__data[i][j])
                count += 1
        return result

    @staticmethod
    def from_array(arr):
        result = Matrix(len(arr), 1)
        for i in range(result.__rows):
            result.__data[i][0] = arr[i]
        return result

    def randomize(self):
        for i in range(self.__rows):
            for j in range(self.__cols):
                self.__data[i][j] = (random.random()*2)-1

    @staticmethod
    def transpose(m):
        result = Matrix(m.__cols, m.__rows)
        for i in range(m.__rows):
            for j in range(m.__cols):
                result.__data[j][i] = m.__data[i][j]
        return result

    def map(self, func):
        for i in range(self.__rows):
            for j in range(self.__cols):
                val = self.__data[i][j]
                self.__data[i][j] = func(val)

    @staticmethod
    def map_(m, func):
        result = Matrix(m.__rows, m.__cols)
        for i in range(m.__rows):
            for j in range(m.__cols):
                val = m.__data[i][j]
                result.__data[i][j] = func(val)
        return result

    @staticmethod
    def sigmoid(n):
        return round(1/(1+(math.e**-n)), 6)

    @staticmethod
    def d_sigmoid(n):
        return n * (1 - n)

    @staticmethod
    def relu(n):
        threshold = 0
        return max(threshold, n)

    def get_data(self):
        return self.__data

    def get_rows(self):
        return self.__rows

    def get_cols(self):
        return self.__cols

    def get_size(self):
        return "rows:" + str(self.__rows) + " cols:" + str(self.__cols)
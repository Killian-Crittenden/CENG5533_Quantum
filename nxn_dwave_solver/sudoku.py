# Copyright 2019 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import dimod
import math
import sys
import copy

from dimod.generators.constraints import combinations
from hybrid.reference import KerberosSampler

import time

import neal

def get_label(row, col, digit):
    """Returns a string of the cell coordinates and the cell value in a
    standard format.
    """
    return "{row},{col}_{digit}".format(**locals())

def get_matrix(filename):
    """Return a list of lists containing the content of the input text file.

    Note: each line of the text file corresponds to a list. Each item in
    the list is from splitting the line of text by the whitespace ' '.
    """
    with open(filename, "r") as f:
        content = f.readlines()

    lines = []
    for line in content:
        new_line = line.rstrip()    # Strip any whitespace after last value

        if new_line:
            new_line = list(map(int, new_line.split(' ')))
            lines.append(new_line)

    return lines

def read_multi_boards(filename):
    puzzle_list = []
    puzzle = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if 'Grid' in line:
                if puzzle != []:
                    puzzle_list.append(puzzle)
                puzzle = []
            else:
                puzzle_line = []
                for number in line:
                    puzzle_line.append(int(number.strip()))
                puzzle.append(puzzle_line)
    return puzzle_list

def is_correct(matrix):
    """Verify that the matrix satisfies the Sudoku constraints.

    Args:
      matrix(list of lists): list contains 'n' lists, where each of the 'n'
        lists contains 'n' digits.
    """
    n = len(matrix)        # Number of rows/columns
    m = int(math.sqrt(n))  # Number of subsquare rows/columns
    unique_digits = set(range(1, n+1))  # Digits in a solution

    # Verifying rows
    for row in matrix:
        if set(row) != unique_digits:
            print("Error in row: ", row)
            return False

    # Verifying columns
    for j in range(n):
        col = [matrix[i][j] for i in range(n)]
        if set(col) != unique_digits:
            print("Error in col: ", col)
            return False

    # Verifying subsquares
    subsquare_coords = [(i, j) for i in range(m) for j in range(m)]
    for r_scalar in range(m):
        for c_scalar in range(m):
            subsquare = [matrix[i + r_scalar * m][j + c_scalar * m] for i, j
                         in subsquare_coords]
            if set(subsquare) != unique_digits:
                print("Error in sub-square: ", subsquare)
                return False

    return True

def build_bqm(matrix):
    """Build BQM using Sudoku constraints"""
    # Set up
    n = len(matrix)          # Number of rows/columns in sudoku
    m = int(math.sqrt(n))    # Number of rows/columns in sudoku subsquare
    digits = range(1, n+1)

    bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN)

    # Constraint: Each node can only select one digit
    for row in range(n):
        for col in range(n):
            node_digits = [get_label(row, col, digit) for digit in digits]
            one_digit_bqm = combinations(node_digits, 1)
            bqm.update(one_digit_bqm)

    # Constraint: Each row of nodes cannot have duplicate digits
    for row in range(n):
        for digit in digits:
            row_nodes = [get_label(row, col, digit) for col in range(n)]
            row_bqm = combinations(row_nodes, 1)
            bqm.update(row_bqm)

    # Constraint: Each column of nodes cannot have duplicate digits
    for col in range(n):
        for digit in digits:
            col_nodes = [get_label(row, col, digit) for row in range(n)]
            col_bqm = combinations(col_nodes, 1)
            bqm.update(col_bqm)

    # Constraint: Each sub-square cannot have duplicates
    # Build indices of a basic subsquare
    subsquare_indices = [(row, col) for row in range(m) for col in range(m)]

    # Build full sudoku array
    for r_scalar in range(m):
        for c_scalar in range(m):
            for digit in digits:
                # Shifts for moving subsquare inside sudoku matrix
                row_shift = r_scalar * m
                col_shift = c_scalar * m

                # Build the labels for a subsquare
                subsquare = [get_label(row + row_shift, col + col_shift, digit)
                             for row, col in subsquare_indices]
                subsquare_bqm = combinations(subsquare, 1)
                bqm.update(subsquare_bqm)

    # Constraint: Fix known values
    for row, line in enumerate(matrix):
        for col, value in enumerate(line):
            if value > 0:
                # Recall that in the "Each node can only select one digit"
                # constraint, for a given cell at row r and column c, we
                # produced 'n' labels. Namely,
                # ["r,c_1", "r,c_2", ..., "r,c_(n-1)", "r,c_n"]
                #
                # Due to this same constraint, we can only select one of these
                # 'n' labels (achieved by 'generators.combinations(..)').
                #
                # The 1 below indicates that we are selecting the label
                # produced by 'get_label(row, col, value)'. All other labels
                # with the same 'row' and 'col' will be discouraged from being
                # selected.
                bqm.fix_variable(get_label(row, col, value), 1)
                #Fixes all values in the row/col pair
                for k in range(1, len(matrix)+1):
                    if k != value:
                        label = get_label(row, col, k)
                        if label in bqm.variables:
                            bqm.fix_variable(label, -1)
                #Fix values in the row
                for k in range(0, len(matrix)):
                    if k != row:
                        label = get_label(k, col, value)
                        if label in bqm.variables:
                                bqm.fix_variable(label, -1)
                #Fix values in column
                for k in range(0, len(matrix)):
                    if k != col:
                        label = get_label(row, k, value)
                        if label in bqm.variables:
                                bqm.fix_variable(label, -1)
                #Fix values in subsquare
                n = len(matrix)
                box_size = int(n ** 0.5)

                start_row = (row // box_size) * box_size
                start_col = (col // box_size) * box_size

                for r in range(start_row, start_row + box_size):
                    for c in range(start_col, start_col + box_size):
                        if r != row or c != col:
                            label = get_label(r, c, value)
                            if label in bqm.variables:
                                bqm.fix_variable(label, -1)
    return bqm

def solve_sudoku(bqm, matrix):
    """Solve BQM and return matrix with solution."""
    sampler = neal.SimulatedAnnealingSampler()

    sampleset = sampler.sample(
        bqm,
        num_reads=500,        # number of independent runs
        num_sweeps=5000       # annealing length
    )

    best_solution = sampleset.first.sample
    solution_list = [k for k, v in best_solution.items() if v == 1]

    result = copy.deepcopy(matrix)

    for label in solution_list:
        coord, digit = label.split('_')
        row, col = map(int, coord.split(','))

        if result[row][col] > 0:
            # the returned solution is not optimal and either tried to
            # overwrite one of the starting values, or returned more than
            # one value for the position. In either case the solution is
            # likely incorrect.
            continue

        result[row][col] = int(digit)

    return result

if __name__ == "__main__":
    # Read user input
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "problem.txt"
        print("Warning: using default problem file, '{}'. Usage: python "
              "{} <sudoku filepath>".format(filename, sys.argv[0]))

    # Read sudoku problem as matrix
    #matrix = get_matrix(filename)
    multi_maxtrix = read_multi_boards('50_problems.txt')

    average_var_count = 0
    average_exe_time = 0

    # Solve BQM and update matrix
    for matrix in multi_maxtrix:
        start_time = time.perf_counter()

        bqm = build_bqm(matrix)
        print("Number of variables:", bqm.num_variables)
        average_var_count += bqm.num_variables
        result = solve_sudoku(bqm, matrix)

        end_time = time.perf_counter()

        average_exe_time += end_time - start_time
        print(f'Run time: {(end_time - start_time):.4f}')

        # Verify
        if is_correct(result):
            print("The solution is correct")
        else:
            print("The solution is incorrect")

    print('Average Variable Count After Reduction is:', average_var_count/50)
    print('Average Run time is:', average_exe_time/50)

    # Print solution
    # for line in result:
    #     print(*line, sep=" ")   # Print list without commas or brackets

    # Verify
    # if is_correct(result):
    #     print("The solution is correct")
    # else:
    #     print("The solution is incorrect")
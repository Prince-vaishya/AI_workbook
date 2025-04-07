from approvedimports import *

def exhaustive_search_4tumblers(puzzle: CombinationProblem) -> list:
    """simple brute-force search method that tries every combination until
    it finds the answer to a 4-digit combination lock puzzle.
    """

    # check that the lock has the expected number of digits
    assert puzzle.numdecisions == 4, "this code only works for 4 digits"

    # create an empty candidate solution
    my_attempt = CandidateSolution()

    # ====> insert your code below here
    for digit1 in puzzle.value_set:
        for digit2 in puzzle.value_set:
            for digit3 in puzzle.value_set:
                for digit4 in puzzle.value_set:
                    my_attempt.variable_values = [digit1, digit2, digit3, digit4]
                    try:
                        result = puzzle.evaluate(my_attempt.variable_values)
                        if result == 1:
                            return my_attempt.variable_values
                    except ValueError:
                        continue
    # <==== insert your code above here

    # should never get here
    return [-1, -1, -1, -1]

def get_names(namearray: np.ndarray) -> list:
    family_names = []
    # ====> insert your code below here
    for row in range(namearray.shape[0]):
        # Get last 6 characters from each row
        last_six = namearray[row, -6:]
        # Join the characters to form a string
        name = ''.join(last_six)
        # Add to our list of names
        family_names.append(name)
    # <==== insert your code above here
    return family_names

def check_sudoku_array(attempt: np.ndarray) -> int:
    tests_passed = 0
    slices = []  # this will be a list of numpy arrays

    # ====> insert your code below here
    # Check array dimensions
    assert len(attempt.shape) == 2, "Array must be 2D"
    assert attempt.shape[0] == 9 and attempt.shape[1] == 9, "Array must be 9x9"

    # Add all rows to slices
    for i in range(9):
        slices.append(attempt[i, :])

    # Add all columns to slices
    for j in range(9):
        slices.append(attempt[:, j])

    # Add all 3x3 sub-squares to slices
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            slices.append(attempt[i:i+3, j:j+3].flatten())

    # Check each slice
    for slice in slices:
        unique_values = np.unique(slice)
        if len(unique_values) == 9:
            tests_passed += 1

    # <==== insert your code above here
    # return count of tests passed
    return tests_passed

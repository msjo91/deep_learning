# Feel free to add any functions, import statements, and variables.


def predict(file):
    # Fill in this function. This function should return a list of length 52
    #   which is filled with floating point numbers. For example, the current
    #   implementation predicts all the instances in test.csv as 10.0.
    return list([10.0 for _ in range(52)])


def write_result(predictions):
    # You don't need to modify this function.
    with open('result.csv', 'w') as f:
        f.write('Value\n')
        for l in predictions:
            f.write('{}\n'.format(l))


def main():
    # You don't need to modify this function.
    predictions = predict('test.csv')
    write_result(predictions)


if __name__ == '__main__':
    # You don't need to modify this part.
    main()

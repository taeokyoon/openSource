# Get two numbers from user and add them together, save the result in ./result/file.txt

import os
import os.path


def main():
    if not os.path.exists('./result'):
        os.makedirs('./result')

    num1 = int(input('Enter the first number: '))
    num2 = int(input('Enter the second number: '))

    result = num1 + num2

    with open('./result/file.txt', 'w') as f:
        f.write("First number: " + str(num1) + '\n')
        f.write("Second number: " + str(num2) + '\n')
        f.write("Result : " + str(result))

    print(f'The result is: {result}')


if __name__ == '__main__':
    main()

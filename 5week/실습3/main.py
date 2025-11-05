# Get two numbers from user and add them together, save the result in ./result/file.txt

import os
import os.path


def main():
    num1 = int(input('Enter the number: '))

    num2 = int(os.environ.get('NUM1', 100))

    num3 = int(os.environ.get('NUM2', 100))

    result = num1 + num2 + num3

    print(f'The result is: {result}')


if __name__ == '__main__':
    main()

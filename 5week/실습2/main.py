# Get two numbers from user and add them together, save the result in ./result/file.txt

import os
import os.path


def main():
    if not os.path.exists('./result/file.txt'):
        with open('./result/file.txt', 'w') as f:
            f.write("2024")

    with open('./result/file.txt', 'r') as f:
        num = int(f.read())

    num1 = int(input('Enter the first number: '))
    num2 = int(input('Enter the second number: '))

    result = num1 + num2 + num

    with open('./result/file.txt', 'w') as f:
        f.write(str(num + 1))

    print(f'The result is: {result}')


if __name__ == '__main__':
    main()

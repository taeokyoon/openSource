import os
import os.path


def main():
    if os.path.exists('./result/file.txt'):
        print('File already exists.')
        with open('./result/file.txt', 'r') as f:
            for line in f:
                print(line.strip())
    else:
        print('File does not exist.')



if __name__ == '__main__':
    main()


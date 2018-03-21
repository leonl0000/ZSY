import sys, os
sys.stdout = open(os.devnull, 'w')
import zsyGame as zsy

def main():
    # print(sys.argv)
    zsy.runXGamesDeepQ(sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]))

if __name__ == "__main__":
    main()
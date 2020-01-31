import os

def bmatch():
    b_degree='/Users/siddharthashankardas/Purdue/Dataset/test_data/1test_data/uni_example_degrees.txt'
    weight_matrix='/Users/siddharthashankardas/Purdue/Dataset/test_data/1test_data/uni_example_weights.txt'

    os.system('Release/BMatchingSolver -w 1test_data/uni_example_weights.txt -d 1test_data/uni_example_degrees.txt -n 10 -o 1test_data/uni_example_solution.txt -c 5 -v 1')

    print("working")

if __name__ == '__main__':
   bmatch()
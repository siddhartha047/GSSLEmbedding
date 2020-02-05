import os
import sys
import subprocess

pc_name=(os.uname()[1]).split('-')[0]
executable=''

if(pc_name=="Siddharthas"):
    executable='Release_osx/BMatchingSolver'
elif(pc_name == "gilbreth"):
    executable='Release_linux/BMatchingSolver'
else:
    print(pc_name," not matched")
    sys.exit(0)

def bmatch_weight_matrix(weight_matrix,b_degree,output_edges,N,Cache,verbose=1):
    # Release/BMatchingSolver -w test/uni_example_weights.txt -d test/uni_example_degrees.txt -n 10 -o test/uni_example_ssolution.txt -c 5 -v 1
    args = (executable, "-w", weight_matrix, "-d", b_degree, "-n", str(N), "-o", output_edges,"-c",str(Cache),"-v",str(verbose))
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    output = popen.stdout.read()
    print(output.decode())

def bmatch_descriptor(feature_matrix,b_degree,output_edges,N,Cache,Dimension,verbose=1):
    # Release/BMatchingSolver -w test/uni_example_weights.txt -d test/uni_example_degrees.txt -n 10 -o test/uni_example_ssolution.txt -c 5 -v 1
    args = (executable, "-x", feature_matrix, "-d", b_degree, "-n", str(N), "-o", output_edges,"-c",str(Cache),"-v",str(verbose),"-t","1","-D",str(Dimension))
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    output = popen.stdout.read()
    print(output.decode())


def bmatch_test():
    b_degree='test/uni_example_degrees.txt'
    weight_matrix='test/uni_example_weights.txt'
    output_edges='test/uni_example_ssolution.txt'

    #Release/BMatchingSolver -w test/uni_example_weights.txt -d test/uni_example_degrees.txt -n 10 -o test/uni_example_ssolution.txt -c 5 -v 1
    #os.system('Release/BMatchingSolver -w 1test_data/uni_example_weights.txt -d 1test_data/uni_example_degrees.txt -n 10 -o 1test_data/uni_example_solution.txt -c 5 -v 1')


    args = (executable, "-w", weight_matrix, "-d", b_degree, "-n", "10", "-o", output_edges,"-c","5","-v","1")
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    output = popen.stdout.read()
    print(output.decode())

if __name__ == '__main__':

   # b_degree = 'test/uni_example_degrees.txt'
   # weight_matrix = 'test/uni_example_weights.txt'
   # output_edges = 'test/uni_example_ssolution.txt'
   # N=10
   # Cache=10 #keep it degree-1 (of course less than N)
   # bmatch_weight_matrix(weight_matrix,b_degree,output_edges,N,Cache)

   b_degree = 'test/test_degree.txt'
   feature_matrix = 'test/test_feature.txt'
   output_edges = 'test/test_solution.txt'
   N = 5
   Cache = 5  # keep it degree-1 (of course less than N)
   Dimension=2
   bmatch_descriptor(feature_matrix, b_degree, output_edges, N, Cache,Dimension)

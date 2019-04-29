#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// ref http://www.tsm-resources.com/alists/mers.html

// Set below to 100 for testing your code
// When ready to run set this to 1 million 1000000
// submit one code called prob1_mpi.c by modifying
//        the code to use MPI and run faster
//        All cores must do work
//        buffer tasks for each core
//        Use 2 nodes and all cores on the node.
// submit one called prob1_omp.c by modifying
//        the code to use openmp and run faster
//        Use 1 node and all cores on the node

// TODO: Change MAXPRIME to 1000000 before submission.
#define MAXPRIME 10000
#define MAXK MAXPRIME - 2

int main(int argc, char **argv);
bool is_prime(int n);
void make_prime_vector(int n, int *prime, int *k);
bool quick_is_prime(unsigned long long int j, int *prime, int k);

int main(int argc, char **argv)
{
    int rank, ncpu, globalNumPrimes = 0, globalIndex = 0;
    int globalPrimes[MAXPRIME], globalMersennePrimes[64];

#pragma omp parallel private(rank, ncpu) \
    shared(globalNumPrimes, globalPrimes, globalMersennePrimes, globalIndex)
    {
        rank = omp_get_thread_num();
        ncpu = omp_get_num_threads();

        clock_t begin = clock();

        int numTasks = MAXK / ncpu, leftOver = MAXK % ncpu;
        int min = rank * numTasks, max = (rank + 1) * numTasks - 1;

        if(rank == ncpu - 1)
        {
            max += leftOver;
        }

        int length = max - min + 1;
        int wPrimes[length];

        int i, index = 0;
#pragma omp private(index)
        {
            for(i = min; i <= max; ++i)
            {
                if(is_prime(i) == true)
                {
                    wPrimes[index++] = i;
                }
            }

            // Reduce number of primes to single value.
            // https://stackoverflow.com/questions/13290245/reduction-with-openmp
#pragma omp reduction(+ : globalNumPrimes)
            globalNumPrimes += index;

            // Collect all primes into single array.
#pragma omp barrier
            printf("[%d] Global number of primes is %d, local value is %d\n",
                   rank, globalNumPrimes, index);

#pragma omp shared(globalIndex) private(index) for
            if(rank == 0)
            {
                for(i = 0; i < ncpu; ++i)
                {
                    printf("[%d] Adding %d to globalIndex %d\n", rank, index,
                           globalIndex);
                    globalIndex += index;
                }
            }
        }  // End private(index)

#pragma omp barrier

        clock_t end = clock();
#pragma omp single
        {
            printf("global index is %d and global number of primes is %d\n",
                   globalIndex, globalNumPrimes);
            // printf("k=%d prime[k-1]=%d \n\n", k, prime[k - 1]);
            double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
            printf("time creating prime vector %f \n", time_spent);
        }

        //        n = 0;
        //        for(i = 2; i < 64; ++i)
        //        {
        //            j = (unsigned long long int)pow(2, i) - 1;
        //            if(quick_is_prime(j, prime, k))
        //            {
        //                mersenne[n] = i;
        //                ++n;
        //            }
        //        }
        //
        //#pragma omp single
        //        {
        //            clock_t end2 = clock();
        //            double time_spent2 = (double)(end2 - end) /
        //            CLOCKS_PER_SEC;
        //            // For mpi only core zero prints the timing
        //            printf("time creating mersenne primes %f \n",
        //            time_spent2);
        //        }
        //
        //        // output
        //#pragma omp single
        //        {
        //            for(i = 0; i < n; ++i)
        //            {
        //                j = (unsigned long long int)pow(2, mersenne[i]) - 1;
        //                printf("2^(%d)-1 = %llu \n", mersenne[i], j);
        //            }
        //        }
        //
        //        printf("prime[%d]=%d\n", k - 1, prime[k - 1]);
        //
        //        int isum;
        //        long int sum = 0;
        //        for(isum = 0; isum < k; ++isum)
        //        {
        //            if(sum > 1000000000)
        //                sum = sum - prime[isum];
        //            else
        //                sum = sum + prime[isum];
        //        }
        //
        //        printf("sum=%d\n", sum);
    }  // End global parallel
}

bool is_prime(int n)
{
    if(n <= 3) return (n > 1);
    if(n % 2 == 0 || n % 3 == 0) return (false);
    int i = 5;
    while(i * i <= n)
    {
        if(n % i == 0 || n % (i + 2) == 0) return (false);
        i = i + 6;
    }
    return (true);
}

void make_prime_vector(int n, int *prime, int *k)
{
    int i = 2;
    while(i < (*k))
    {
        if(n % prime[i] == 0) return;
        ++i;
    }
    prime[(*k)] = n;
    ++(*k);
    return;
}

bool quick_is_prime(unsigned long long int j, int *prime, int k)
{
    int i = 1;
    while(i < k)
    {
        if(j % (unsigned long long int)prime[i] == 0) return (false);
        if((unsigned long long int)prime[i] * (unsigned long long int)prime[i] >
           j)
        {
            return (true);
        }
        ++i;
    }

    unsigned long long int ii;
    ii = (unsigned long long int)(prime[k - 2] + 6);
    while(ii * ii <= j)
    {
        if(j % ii == 0 || j % (ii + 2) == 0) return (false);
        ii = ii + 6;
    }
    return (true);
}

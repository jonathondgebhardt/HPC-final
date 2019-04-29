#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
 * Leverage Slurm and OpenMP to find MAXK prime numbers and Mersenne prime
 * numbers between 0 and 64.
 *
 * https://en.wikipedia.org/wiki/Mersenne_prime
 *
 * Author: Jonathon Gebhardt
 * Class: CS4900-B90
 * Instructor: Dr. John Nehrbass
 * Assignment: Final
 * GitHub: https://github.com/jonathondgebhardt/HPC-final.git
 */

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

#define MAXPRIME 1000000
#define MAXK MAXPRIME - 2

int main(int argc, char **argv);
bool is_prime(int n);
void make_prime_vector(int n, int *prime, int *k);
bool quick_is_prime(unsigned long long int j, int *prime, int k);
int comp(const void *elem1, const void *elem2);

int main(int argc, char **argv)
{
    int rank, ncpu, globalNumPrimes = 0, globalNumMersennePrimes = 0, num = 0,
                    wNum, wIndex;
    int globalPrimes[MAXPRIME], globalMersennePrimes[64];

#pragma omp parallel num_threads(16) private(rank, ncpu, wNum, wIndex) \
    shared(globalNumPrimes, globalPrimes, globalMersennePrimes, num)
    {
        rank = omp_get_thread_num();
        ncpu = omp_get_num_threads();

        clock_t begin = clock();

        // It would be ideal to only use atomic structures instead of critical
        // structures. I tried this initially but this lead to many
        // complications. Making the entire operation critical did not impose
        // a large performance hit.
#pragma omp critical
        {
            while(globalNumPrimes < MAXK)
            {
                wNum = num++;
                if(is_prime(wNum) == true)
                {
#pragma omp atomic capture
                    wIndex = globalNumPrimes++;

                    globalPrimes[wIndex] = wNum;
                }
            }
        }

        clock_t end = clock();

#pragma omp single
        {
            printf("k=%d prime[k-1]=%d \n\n", globalNumPrimes,
                   globalPrimes[globalNumPrimes - 1]);
            double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
            printf("time creating prime vector %f \n", time_spent);
        }

#pragma omp barrier

        // Since we know the range of Mersenne primes we want to generate,
        // we can use the for structure. This vastly simplifies logic and
        // eliminates typical problems associated with multi-threaded
        // operations (e.g., race condition etc).
        int i;
        unsigned long long int j;
#pragma omp for
        for(i = 2; i < 64; ++i)
        {
            j = (unsigned long long int)pow(2, i) - 1;
            if(quick_is_prime(j, globalPrimes, globalNumPrimes))
            {
                globalMersennePrimes[globalNumMersennePrimes++] = i;
            }
        }

#pragma omp single
        {
            clock_t end2 = clock();
            double time_spent2 = (double)(end2 - end) / CLOCKS_PER_SEC;
            printf("time creating mersenne primes %f \n", time_spent2);

            // Because we used OpenMP's for construct, the list of mersenne
            // primes could be out of order. Sort them before showing them.
            // https://stackoverflow.com/questions/1787996/c-library-function-to-do-sort
            qsort(globalMersennePrimes, globalNumMersennePrimes, sizeof(int),
                  comp);

            for(i = 0; i < globalNumMersennePrimes; ++i)
            {
                j = (unsigned long long int)pow(2, globalMersennePrimes[i]) - 1;
                printf("2^(%d)-1 = %llu \n", globalMersennePrimes[i], j);
            }
        }

        printf("[%d] prime[%d]=%d\n", rank, globalNumPrimes - 1,
               globalPrimes[globalNumPrimes - 1]);

        int isum;
        long int sum = 0;
        for(isum = 0; isum < globalNumPrimes; ++isum)
        {
            if(sum > 1000000000)
            {
                sum = sum - globalPrimes[isum];
            }
            else
            {
                sum = sum + globalPrimes[isum];
            }
        }

        printf("[%d] sum=%d\n", rank, sum);
    }
    // End global parallel
}

//
//  \brief Check if the given number is prime.
//
//  \param n The number to check.
//
//  \return Whether the given number is prime.
//
bool is_prime(int n)
{
    if(n <= 3)
    {
        return n > 1;
    }

    if(n % 2 == 0 || n % 3 == 0)
    {
        return false;
    }

    int i = 5;
    while(i * i <= n)
    {
        if(n % i == 0 || n % (i + 2) == 0)
        {
            return false;
        }

        i = i + 6;
    }

    return true;
}

//
//  \brief Add the given n to the given index k in the given array prime if n
//  is not evenly divisible by the values leading up to k.
//
//  This function modifies the prime and k parameters.
//
//  \param n the number to add.
//  \param prime the array to add the number to.
//  \param k the index of the array in which to add the number.
//
void make_prime_vector(int n, int *prime, int *k)
{
    int i = 2;
    while(i < (*k))
    {
        if(n % prime[i] == 0)
        {
            return;
        }

        ++i;
    }

    prime[(*k)] = n;
    ++(*k);

    return;
}

//
//  \brief A quicker check for finding a number in an array.
//
//  This function is provided in an effort to shorten the time to find a prime
//  within an array. This function does not perform well, however, when the
//  length of prime increases (i.e., O(n) performance).
//
//  Unfortunately, for binary search to work, the number has to be in the
//  array. For mersenne primes, 2^31-1 and 2^61-1 isn't contained in the
//  prime array. This is where we take the major performance hit.
//
//  \param j The number to check.
//  \param prime The array to check in.
//  \param k The length of the given array.
//
//  \return True if the given j is prime, false otherwise.
//
bool quick_is_prime(unsigned long long int j, int *prime, int k)
{
    int i = 1;
    while(i < k)
    {
        if(j % (unsigned long long int)prime[i] == 0)
        {
            return false;
        }

        if((unsigned long long int)prime[i] * (unsigned long long int)prime[i] >
           j)
        {
            return true;
        }

        ++i;
    }

    unsigned long long int ii;
    ii = (unsigned long long int)(prime[k - 2] + 6);
    while(ii * ii <= j)
    {
        if(j % ii == 0 || j % (ii + 2) == 0)
        {
            return false;
        }

        ii = ii + 6;
    }

    return true;
}

//  \brief Compares the magnitude of two values.
//
//  A comparison function used to aid the standard C qsort function.
//  https://stackoverflow.com/questions/1787996/c-library-function-to-do-sort
//
//  \param elem1 The first element used to compare.
//  \param elem2 The second element used to compare.
//
//  \return -1 if the first element comes first, 0 if they are equal, and 1 if
//  first element comes second.
//
int comp(const void *elem1, const void *elem2)
{
    int f = *((int *)elem1);
    int s = *((int *)elem2);
    if(f > s) return 1;
    if(f < s) return -1;
    return 0;
}

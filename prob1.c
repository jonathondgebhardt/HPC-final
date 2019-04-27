#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
 * Leverage Slurm and MPI to find MAXK primes and 64 Mersenne primes.
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

//#define MAXPRIME 10000
#define MAXPRIME 100000
//#define MAXPRIME 1000000
#define MAXK MAXPRIME - 2

int main(int argc, char **argv);
bool is_prime(int n);
void make_prime_vector(int n, int *prime, int *k);
bool quick_is_prime(unsigned long long int j, int *prime, int k);
bool quick_is_prime_exclusive(unsigned long long int j, int *prime, int k);

int main(int argc, char **argv)
{
    int rank, ncpu;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ncpu);

    int mersenne[64], prime[MAXPRIME];

    // Initial prime values.
    if(rank == 0)
    {
        prime[0] = 2;
        prime[1] = 3;
        prime[2] = 5;
        prime[3] = 7;
        prime[4] = 11;
        prime[5] = 13;
    }

    int i, k = 6;
    unsigned long long int j;

    clock_t begin;
    if(rank == 0)
    {
        begin = clock();
    }

    int startTask = 1, stopTask = 0;

    if(rank == 0)
    {
        printf("[%d] Starting work\n", rank);

        // Create prime vector. The next prime after 13 is 17.
        int n = 17, incrementFlag = 0;
        int potentialPrimes[ncpu - 1];
        while(k < MAXK)
        {
            // Since each worker won't have a strictly current list of primes,
            // let them generate a number anyway and check afterwards.
            for(i = 0; i < ncpu - 1; ++i)
            {
                potentialPrimes[i] = -1;
            }

            // Send tasks to workers.
            for(i = 1; i < ncpu; ++i)
            {
                MPI_Send(&startTask, 1, MPI_INT, i, 1, MPI_COMM_WORLD);

                MPI_Send(&k, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
                MPI_Send(&prime, k, MPI_INT, i, 1, MPI_COMM_WORLD);
                MPI_Send(&n, 1, MPI_INT, i, 1, MPI_COMM_WORLD);

                // Skip factors of 6 and even numbers.
                if(incrementFlag == 0)
                {
                    n += 2;
                }
                else
                {
                    n += 4;
                }

                incrementFlag = (++incrementFlag) % 2;
            }

            // Receive answers from workers.
            for(i = 1; i < ncpu; ++i)
            {
                MPI_Recv(&potentialPrimes[i - 1], 1, MPI_INT, i, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            // Since the workers don't have the most current list of primes,
            // the master must double-check their work and add the number if
            // appropriate.
            for(i = 0; i < ncpu - 1; ++i)
            {
                if(potentialPrimes[i] != -1 &&
                   quick_is_prime_exclusive(potentialPrimes[i], potentialPrimes,
                                            ncpu - 1))
                {
                    // Use is_quick_prime on potential primes and add if true
                    // make_prime_vector(potentialPrimes[i], prime, &k);
                    prime[k] = potentialPrimes[i];
                    ++k;
                }
            }

            // The below many be helpful for debugging only
            //        if ((n-17)%10000==0)
            //          printf("n=%d k=%d\n",n,k);
        }

        // Tell workers to stop.
        for(i = 1; i < ncpu; ++i)
        {
            MPI_Send(&stopTask, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
        }
    }
    else
    {
        int msg;
        while(1)
        {
            // Continue while there's more work to do, otherwise stop.
            MPI_Recv(&msg, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(msg == stopTask)
            {
                printf("[%d] Stopping task\n", rank);
                break;
            }

            // Get the new list of primes, it's length, and the number to check.
            int length, num;
            MPI_Recv(&length, 1, MPI_INT, 0, 1, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            MPI_Recv(&prime, length, MPI_INT, 0, 1, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            MPI_Recv(&num, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            bool numIsPrime = quick_is_prime(num, prime, length);

            // Return numbers to master. If the number is not prime, return -1.
            int answer = -1;
            if(numIsPrime == true)
            {
                answer = num;
            }
            MPI_Send(&answer, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    printf("k=%d prime[k-1]=%d \n\n", k, prime[k - 1]);

    clock_t end;
    if(rank == 0)
    {
        end = clock();
        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        printf("time creating prime vector %f \n", time_spent);
    }

    //    // Build a list of Mersenne primes.
    //    // https://en.wikipedia.org/wiki/Mersenne_prime
    //    n = 0;
    //    for(i = 2; i < 64; ++i)
    //    {
    //        // If the potential Mersenne prime is contained with the generated
    //        // list of primes, add it to the list of Mersenne primes.
    //        // TODO: Parallelize this part.
    //        j = (unsigned long long int)pow(2, i) - 1;
    //        if(quick_is_prime(j, prime, k))
    //        {
    //            mersenne[n] = i;
    //            ++n;
    //        }
    //    }
    //
    //    if(rank == 0)
    //    {
    //        clock_t end2 = clock();
    //        double time_spent2 = (double)(end2 - end) / CLOCKS_PER_SEC;
    //
    //        // For mpi only core zero prints the timing.
    //        printf("time creating mersenne primes %f \n", time_spent2);
    //
    //
    //        // output
    //        // For mpi only core zero prints this output.
    //        for(i = 0; i < n; ++i)
    //        {
    //            j = (unsigned long long int)pow(2, mersenne[i]) - 1;
    //            printf("2^(%d)-1 = %llu \n", mersenne[i], j);
    //        }
    //    }
    //
    //    // Comment the below for mpi code.
    //    // printf("prime[%d]=%d\n", k - 1, prime[k - 1]);
    //
    //    int isum;
    //    long int sum = 0;
    //    for(isum = 0; isum < k; ++isum)
    //    {
    //        // 1,000,000,000
    //        if(sum > 1000000000)
    //        {
    //            sum = sum - prime[isum];
    //        }
    //        else
    //        {
    //            sum = sum + prime[isum];
    //        }
    //    }
    //
    //    // Comment the below for MPI code.
    //    // printf("sum=%d\n", sum);
    //
    //    // Uncomment the below for mpi code where rank is the rank for each
    //    core.
    //    // All cores must print this.
    //    printf("[%d] prime[%d]=%d\n", rank, k - 1, prime[k - 1]);
    //    printf("[%d] sum=%d\n", rank, sum);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}

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
//  \brief Adds the given n to the given index k in the given array prime if n
//  is not evenly divisible by the values leading up to k.
//
//  This function modifies the prime and k parameters.
//
//  \param n the number to add
//  \param prime the array to add the number to
//  \param k the index of the array in which to add the number
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
//  \brief A quick check if the given j is prime.
//
//  \param j the number to check
//  \param prime an array of prime numbers
//  \param k the length of the given array prime
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

bool quick_is_prime_exclusive(unsigned long long int j, int *prime, int k)
{
    int i = 1;
    while(i < k)
    {
        if(j % (unsigned long long int)prime[i] == 0 && prime[i] != j)
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

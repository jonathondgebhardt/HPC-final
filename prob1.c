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

#define MAXPRIME 100000
#define MAXK MAXPRIME - 2

int main(int argc, char **argv);
bool is_prime(int n);
void make_prime_vector(int n, int *prime, int *k);
bool quick_is_prime(unsigned long long int j, int *prime, int k);
int binarySearch(int arr[], int l, int r, unsigned long long int x);

int main(int argc, char **argv)
{
    int rank, ncpu;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ncpu);

    int mersenne[64], prime[MAXPRIME];

    // primeRange indicates a range of values each worker is responsible for
    // (i.e., worker 1 is responsible for 0-999, worker 2 is responsible
    // for 1,000-1,999 etc).
    int i, primeRange = 1000;

    clock_t begin;
    if(rank == 0)
    {
        begin = clock();
    }

    int startTask = 1, stopTask = 0;
    int primesGenerated = 0;
    if(rank == 0)
    {
        int tasksFulfilled = 0;
        do
        {
            // Send out tasks.
            int tasksIssued = 0;
            for(i = 1; i < ncpu; ++i)
            {
                int min = (tasksIssued + tasksFulfilled) * primeRange;
                int max = (tasksIssued + tasksFulfilled + 1) * primeRange - 1;

                MPI_Send(&startTask, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
                MPI_Send(&min, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
                MPI_Send(&max, 1, MPI_INT, i, 1, MPI_COMM_WORLD);

                ++tasksIssued;
            }

            // Receive solution.
            for(i = 1; i < ncpu; ++i)
            {
                int numWorkerPrimes;
                MPI_Recv(&numWorkerPrimes, 1, MPI_INT, i, 1, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);

                int workerPrimes[numWorkerPrimes];
                MPI_Recv(workerPrimes, numWorkerPrimes, MPI_INT, i, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Truncate the solution if the worker found more than our
                // desired maximum amount.
                if(numWorkerPrimes + primesGenerated > MAXK)
                {
                    numWorkerPrimes = MAXK - primesGenerated;
                }

                int k;
                for(k = 0; k < numWorkerPrimes; ++k)
                {
                    int dest = k + primesGenerated;
                    prime[dest] = workerPrimes[k];
                }

                primesGenerated += numWorkerPrimes;
                tasksFulfilled++;

                if(primesGenerated == MAXK)
                {
                    break;
                }
            }

        } while(primesGenerated != MAXK);

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
                break;
            }

            // Get the min and max values with which to generate primes
            // between.
            int min, max;
            MPI_Recv(&min, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&max, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Find all primes between min and max inclusive.
            int wNumPrimes = 0, wPrimes[primeRange];
            for(i = min; i <= max; ++i)
            {
                if(is_prime(i) == true)
                {
                    wPrimes[wNumPrimes++] = i;
                }
            }

            // Send generated primes back to master.
            MPI_Send(&wNumPrimes, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(wPrimes, wNumPrimes, MPI_INT, 0, 1, MPI_COMM_WORLD);
        }
    }

    clock_t end;
    if(rank == 0)
    {
        end = clock();
        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        printf("k=%d prime[k-1]=%d \n\n", primesGenerated,
               prime[primesGenerated - 1]);
        printf("time creating prime vector %f \n", time_spent);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Give primes to workers.
    MPI_Bcast(&primesGenerated, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(prime, primesGenerated, MPI_INT, 0, MPI_COMM_WORLD);

    int mersennesGenerated = 0, currentPower = 2, targetPower = 64;
    if(rank == 0)
    {
        do
        {
            // Send tasks to workers.
            for(i = 1; i < ncpu; ++i)
            {
                MPI_Send(&startTask, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
                MPI_Send(&currentPower, 1, MPI_INT, i, 1, MPI_COMM_WORLD);

                if(++currentPower > targetPower)
                {
                    break;
                }
            }

            // Receive solutions.
            for(i = 1; i < ncpu; ++i)
            {
                int answer;
                MPI_Recv(&answer, 1, MPI_INT, i, 1, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                //                printf("[%d] Received %d from %d\n", rank,
                //                answer, i);
                if(answer != 0)
                {
                    mersenne[mersennesGenerated++] = answer;
                }
            }
        } while(currentPower <= targetPower);

        // Tell workers to stop.
        for(i = 1; i < ncpu; ++i)
        {
            MPI_Send(&stopTask, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
        }
    }
    else
    {
        int msg, primesSent = 0;
        while(1)
        {
            // Continue while there's more work to do, otherwise stop.
            MPI_Recv(&msg, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if(msg == stopTask)
            {
                break;
            }

            int num;
            MPI_Recv(&num, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //            printf("[%d] Received %d to check\n", rank, num);

            int answer = 0;
            unsigned long long int j = (unsigned long long int)pow(2, num) - 1;
            if((j < (unsigned long long int)prime[primesGenerated - 1] &&
                binarySearch(prime, 0, primesGenerated, j) != -1) ||
               quick_is_prime(j, prime, primesGenerated) == true)
            {
                answer = num;
                ++primesSent;
            }

            MPI_Send(&answer, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            //            printf("[%d] %d is %d\n", rank, num, answer);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0)
    {
        clock_t end2 = clock();
        double time_spent2 = (double)(end2 - end) / CLOCKS_PER_SEC;
        printf("time creating mersenne primes %f \n", time_spent2);

        unsigned long long int j;
        for(i = 0; i < mersennesGenerated; ++i)
        {
            j = (unsigned long long int)pow(2, mersenne[i]) - 1;
            printf("2^(%d)-1 = %llu \n", mersenne[i], j);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int isum;
    long int sum = 0;
    for(isum = 0; isum < primesGenerated; ++isum)
    {
        if(sum > 1000000000)
            sum = sum - prime[isum];
        else
            sum = sum + prime[isum];
    }

    printf("[%d] prime[%d]=%d\n", rank, primesGenerated - 1,
           prime[primesGenerated - 1]);
    printf("[%d] sum=%d\n", rank, sum);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
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

//
//  \brief Get whether a number is in an array.
//
//  A recursive implementation of binary search found at:
//  https://www.geeksforgeeks.org/binary-search/
//
//  I included this function for better search performance (i.e., O(log(n)).
//
//  \param arr The array to search in.
//  \param l The lower bound at which to start.
//  \param r The upper bound at which to stop.
//  \param x The number to search for.
//
//  \return -1 if the number is not in the array or the the given x if x is
//  within the given arr.
//
int binarySearch(int arr[], int l, int r, unsigned long long int x)
{
    if(r >= l)
    {
        int mid = l + (r - l) / 2;

        // If the element is present at the middle itself.
        if((unsigned long long int)arr[mid] == x)
        {
            return mid;
        }

        // If element is smaller than mid, then it can only be present in left
        // subarray.
        if((unsigned long long int)arr[mid] > x)
        {
            return binarySearch(arr, l, mid - 1, x);
        }

        // Else the element can only be present in right subarray.
        return binarySearch(arr, mid + 1, r, x);
    }

    // We reach here when element is not present in array.
    return -1;
}

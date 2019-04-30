# HPC-final
The final project for HPC and Parallel Programming

- Class: CS4900-B90 -- HPC & Parallel Programming
- Semester: Spring 2019
- Instructor: Dr. John Nehrbass

## Problem 1
Find MAXK number of primes and all Mersenne primes between 0 and 64.

### MPI Obstacles
When finding MAXK number of primes, I initially tried dispatching one number at a time for worker nodes to try. The overhead incurred from message passing quickly proved this was not the optimal strategy. Passing ranges of numbers for worker nodes to try performed much better, taking only about one half second in it's entirety.

### OpenMP Obstacles
I had to deal with several race conditions when finding MAXK number of primes. I ultimately solved the problem by making the entire portion that finds a prime number a critical section. Surprisingly, this didn't incur a huge performance hit like I was expecting.

I initially tried to reduce the size of the critical section but this didn't improve things for several reasons. For instance, one thread might try the actual last prime to find but doesn't finish before another thread tries one past the actual last prime. This situation would result in the actual last prime not being added to the list because one past the actual last prime is added first.

I would have ultimately preferred to use the OpenMP for construct. This can't work since I don't know at runtime how many numbers I will have to try before finding MAXK primes.

### Mersenne Primes
The biggest time performance hit for this problem is finding the largest Mersenne prime between 0 and 64, namely 2<sup>61</sup>-1. This particular prime accounts for nearly all of the time spent finding all Mersenne primes in this range.

## Problem 2
Find the absolute minimum value in a multidimensional polynomial using steepest descent.

### Obstacles
I didn't encounter many obstacles for this problem since the algorithm is straight-forward and lends itself to multi-threaded solutions.
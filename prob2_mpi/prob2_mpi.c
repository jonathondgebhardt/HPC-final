#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

/*
 * Leverage Slurm and MPI to find an absolute minimum value in a 
 * multidimensional polynomial using steepest descent.
 *
 * https://en.wikipedia.org/wiki/Mersenne_prime
 *
 * Author: Jonathon Gebhardt
 * Class: CS4900-B90
 * Instructor: Dr. John Nehrbass
 * Assignment: Final
 * GitHub: https://github.com/jonathondgebhardt/HPC-final.git
 */

#define slow 2000
#define eps 0.00000001

int main(int argc, char **argv);
double local_minimum(double xguess, double yguess, double *x, double *y);
double Fxy(double x, double y);
double Dx(double x, double y, double dx);
double Dy(double x, double y, double dy);

// Global values
double r1 = -8.0, r2 = -2, r3 = 3, r4 = 7;
double r5 = -8, r6 = -2, r7 = 3, r8 = 7;
double X3, X2, X1, X0;
double Y3, Y2, Y1, Y0;
double xmin, xmax, ymin, ymax;

int main(int argc, char **argv)
{
    int rank, ncpu;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ncpu);

    // Global values
    X3 = (r1 + r2 + r3 + r4);
    X2 = (r1 * r2 + r1 * r3 + r1 * r4 + r2 * r3 + r2 * r4 + r3 * r4);
    X1 = (r1 * r2 * r3 + r1 * r2 * r4 + r1 * r3 * r4 + r2 * r3 * r4);
    X0 = r1 * r2 * r3 * r4;
    Y3 = (r5 + r6 + r7 + r8);
    Y2 = (r5 * r6 + r5 * r7 + r5 * r8 + r6 * r7 + r6 * r8 + r7 * r8);
    Y1 = (r5 * r6 * r7 + r5 * r6 * r8 + r5 * r7 * r8 + r6 * r7 * r8);
    Y0 = r5 * r6 * r7 * r8;

    // Range
    xmax = 10, xmin = -xmax;
    ymax = 10, ymin = -ymax;

    // Initial dense grid to search over
    int nx = 1000, ny = 1000;
    double xstep = (xmax - xmin) / (nx - 1);
    double ystep = (ymax - ymin) / (ny - 1);

    int ix, iy;
    double xguess, yguess, xbest, ybest, x, y;
    double flocal, fmin = 1e9;  // initialize at a big value

    struct timeval start, end;
    unsigned long secs_used, micros_used;

    // Divide up work. Each node is responsible for a range between min and max
    // inclusive.
    int numTasks = nx / ncpu, leftOver = nx % ncpu;
    int min = rank * numTasks, max = (rank + 1) * numTasks - 1;

    // Assign left over work to last worker.
    if(rank == ncpu - 1)
    {
        max += leftOver;
    }

    if(rank == 0)
    {
        gettimeofday(&start, NULL);
    }

    // brute force
    for(ix = min; ix <= max; ++ix)
    {
        xguess = xmin + xstep * ix;
        for(iy = 0; iy < ny; ++iy)
        {
            yguess = ymin + ystep * iy;
            flocal = Fxy(xguess, yguess);
            if(flocal < fmin)
            {
                fmin = flocal;
                xbest = xguess;
                ybest = yguess;
            }
        }
    }

    // Ideally, we would use reduce here, but because we don't simply want the
    // smallest x and y, we must use the x and y with the smallest flocal.
    double globalFMin[ncpu], globalXBest[ncpu], globalYBest[ncpu];
    MPI_Gather(&fmin, 1, MPI_DOUBLE, globalFMin, 1, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);
    MPI_Gather(&xbest, 1, MPI_DOUBLE, globalXBest, 1, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);
    MPI_Gather(&ybest, 1, MPI_DOUBLE, globalYBest, 1, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

    if(rank == 0)
    {
        fmin = 1e9;
        int i;
        for(i = 0; i < ncpu; ++i)
        {
            if(globalFMin[i] < fmin)
            {
                fmin = globalFMin[i];
                xbest = globalXBest[i];
                ybest = globalYBest[i];
            }
        }
    }

    // Give the other workers the same information.
    MPI_Bcast(&fmin, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&xbest, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ybest, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(rank == 0)
    {
        gettimeofday(&end, NULL);

        printf("Brute Force answer \n");
        printf("x=%f y=%f min=%f \n", xbest, ybest, fmin);

        secs_used =
            (end.tv_sec - start.tv_sec);  // avoid overflow by subtracting first
        micros_used = ((secs_used * 1000000) + end.tv_usec) - (start.tv_usec);
        printf("Time for Brute Force answer %f sec \n\n",
               micros_used / 1000000.0);
    }

    // Initial grid to guess
    nx = 4, ny = 4;
    xstep = (xmax - xmin) / (nx - 1);
    ystep = (ymax - ymin) / (ny - 1);
    fmin = 1e9;  // initialize at a big value

    // Divide up work. Each node is responsible for a range between min and max
    // inclusive. Since we only have 16 guesses to handle here, some nodes
    // won't have anything to do when running with full nodes and processes.
    bool doCompute = true;
    if(ncpu <= 4)
    {
        numTasks = nx / ncpu, leftOver = nx % ncpu;
        min = rank * numTasks, max = (rank + 1) * numTasks - 1;

        // Assign left over work to last worker.
        if(rank == ncpu - 1)
        {
            max += leftOver;
        }

        ix = min;
    }
    else
    {
        // If we have more than 4 nodes, assign an x for each worker according
        // to their rank.
        ix = rank;
        max = ix;

        if(rank > 4)
        {
            doCompute = false;
        }
    }

    if(rank == 0)
    {
        gettimeofday(&start, NULL);
    }

    if(doCompute == true)
    {
        for(; ix <= max; ++ix)
        {
            xguess = xmin + xstep * ix;
            for(iy = 0; iy < ny; ++iy)
            {
                yguess = ymin + ystep * iy;
                flocal = local_minimum(xguess, yguess, &x, &y);
                if(flocal < fmin)
                {
                    fmin = flocal;
                    xbest = x;
                    ybest = y;
                }
            }
        }
    }

    // Like we did for the brute force attempt, get the x and y with the best f.
    MPI_Gather(&fmin, 1, MPI_DOUBLE, globalFMin, 1, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);
    MPI_Gather(&xbest, 1, MPI_DOUBLE, globalXBest, 1, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);
    MPI_Gather(&ybest, 1, MPI_DOUBLE, globalYBest, 1, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

    if(rank == 0)
    {
        fmin = 1e9;
        int i;
        for(i = 0; i < ncpu; ++i)
        {
            if(globalFMin[i] < fmin)
            {
                fmin = globalFMin[i];
                xbest = globalXBest[i];
                ybest = globalYBest[i];
            }
        }

        gettimeofday(&end, NULL);

        printf("Smallest value %f\n", fmin);
        printf("Found at x=%f y=%f\n", xbest, ybest);
        printf("Within the range of x:%.2f %.2f\n", xmin, xmax);
        printf("                    y:%.2f %.2f\n", ymin, ymax);

        secs_used =
            (end.tv_sec - start.tv_sec);  // avoid overflow by subtracting first
        micros_used = ((secs_used * 1000000) + end.tv_usec) - (start.tv_usec);

        printf("Time for Steepest Descent Method %f sec \n",
               micros_used / 1000000.0);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}

double local_minimum(double xguess, double yguess, double *x, double *y)
{
    double xstep = 0.1;
    double ystep = 0.1;

    double dx = Dx(xguess, yguess, xstep);
    double dy = Dy(xguess, yguess, ystep);
    double xnew = xguess - dx * xstep;
    double ynew = yguess - dy * ystep;
    if(xnew < xmin) xnew = xmin;
    if(xnew > xmax) xnew = xmax;
    if(ynew < ymin) ynew = ymin;
    if(ynew > ymax) ynew = ymax;

    double dxnew = Dx(xnew, ynew, xstep);
    double dynew = Dy(xnew, ynew, ystep);
    int count = 0, itime = 1;

    while(fabs(xguess - xnew) > eps && fabs(yguess - ynew) > eps && count < 3)
    {
        // If slope changed we pased the local min
        if((dx * dxnew) < 0)
        {  // slope changed sign
            xstep = xstep / 2.0;
        }
        else
        {
            xguess = xnew;
        }
        if((dy * dynew) < 0)
        {  // slope changed sign
            ystep = ystep / 2.0;
        }
        else
        {
            yguess = ynew;
        }
        dx = Dx(xguess, yguess, xstep);
        dy = Dy(xguess, yguess, ystep);
        xnew = xguess - dx * xstep;
        ynew = yguess - dy * ystep;
        if(xnew < xmin) xnew = xmin;
        if(xnew > xmax) xnew = xmax;
        if(ynew < ymin) ynew = ymin;
        if(ynew > ymax) ynew = ymax;
        dxnew = Dx(xnew, ynew, xstep);
        dynew = Dy(xnew, ynew, ystep);
        ++itime;
        if(fabs(xguess - xnew) > eps || fabs(yguess - ynew) > eps)
        {
            count = 0;
        }
        else
        {
            ++count;
            xstep = xstep / 2.0;
            ystep = ystep / 2.0;
        }
    }
    //  printf("times %d \n",itime);
    // return location
    (*x) = xguess;
    (*y) = yguess;

    // return min vale at this location
    return (Fxy(xguess, yguess));
}

double Fxy(double x, double y)
{
    double fx, fy;
    usleep(slow);
    if(fabs(x) > xmax) return (0.0);
    if(fabs(y) > ymax) return (0.0);
    fx = cos(M_PI_2 * x / xmax) * (x - r1) * (x - r2) * (x - r3) * (x - r4);
    fy = cos(M_PI_2 * y / ymax) * (y - r5) * (y - r6) * (y - r7) * (y - r8);
    return (fx * fy);
}

double Dx(double x, double y, double dx)
{
    return (((Fxy(x + dx / 2.0, y) > Fxy(x - dx / 2.0, y)) ? 1.0 : -1.0));
}

double Dy(double x, double y, double dy)
{
    return (((Fxy(x, y + dy / 2.0) > Fxy(x, y - dy / 2.0)) ? 1.0 : -1.0));
}

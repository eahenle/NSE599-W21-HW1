#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include "common.h"
#include "omp.h"

int main(int argc, char **argv)
{
    int navg, nabsavg=0, numthreads;
    double dmin, absmin=1.0, davg, absavg=0.0;

    if(find_option(argc, argv, "-h") >= 0)
    {
        printf("Options:\n");
        printf("-h to see this help\n");
        printf("-n <int> to set number of particles\n");
        printf("-o <filename> to specify the output file name\n");
        printf("-s <filename> to specify a summary file name\n");
        printf("-no turns off all correctness checks and particle output\n");
        return 0;
    }

    int n = read_int(argc, argv, "-n", 1000);
    char *savename = read_string(argc, argv, "-o", NULL);
    char *sumname = read_string(argc, argv, "-s", NULL);

    FILE *fsave = savename ? fopen(savename, "w") : NULL;
    FILE *fsum = sumname ? fopen (sumname, "a") : NULL;

    particle_t *particles = (particle_t*)malloc(n * sizeof(particle_t));
    set_size(n);
    init_particles(n, particles);

    // bins is a vector of particle vectors
    // simulation space is a square grid of bins
    const int nb_bins_per_row = ceil(sqrt(density * n) / cutoff);
    const int nb_bins = nb_bins_per_row * nb_bins_per_row;
    std::vector<particle_t*> *bins = new std::vector<particle_t*>[nb_bins];

    //  simulate a number of time steps
    double simulation_time = read_timer();
    #pragma omp parallel private(dmin)
    {
        numthreads = omp_get_num_threads();
        for(int step = 0; step < 1000; step++)
        {
            navg = 0;
            davg = 0.0;
            dmin = 1.0;

            // bin particles
            #pragma omp for
            for (int i = 0; i < n; i++)
            {
                bins[int(floor(particles[i].x / cutoff) + nb_bins_per_row * floor(particles[i].y / cutoff))].push_back(&particles[i]);
            }

            //  compute all forces
            #pragma omp for reduction(+:navg) reduction(+:davg)
            for(int p = 0; p < n; p++)
            {
                particles[p].ax = particles[p].ay = 0;

                // apply nearby forces (only particles in same or adjacent bins can interact)
                int particle_bin = floor(particles[p].x / cutoff) + nb_bins_per_row * floor(particles[p].y / cutoff);
                for(int i = particle_bin % nb_bins_per_row == 0 ? 0 : -1; i <= (particle_bin % nb_bins_per_row == nb_bins_per_row - 1 ? 0 : 1); i++)
                {
                    for(int j = particle_bin < nb_bins_per_row ? 0 : -1; j <= (particle_bin >= nb_bins_per_row * (nb_bins_per_row - 1) ? 0 : 1); j++)
                    {
                        int bin_index = particle_bin + i + nb_bins_per_row * j;
                        for(int k = 0; k < bins[bin_index].size(); k++)
                        {
                            apply_force(particles[p], *bins[bin_index][k], &dmin, &davg, &navg);
                        }
                    }
                }
            }

            // clear bins for next iteration
            #pragma omp for
            for (int i = 0; i < nb_bins; i++)
            {
                bins[i].clear();
            }

            //  move particles
            #pragma omp for
            for(int i = 0; i < n; i++)
            {
                move(particles[i]);
            }

            if(find_option(argc, argv, "-no") == -1)
            {
                //  compute statistical data
                #pragma omp master
                if(navg)
                {
                    absavg += davg / navg;
                    nabsavg++;
                }

                #pragma omp critical
                if(dmin < absmin)
                {
                  absmin = dmin;
                }

                //  save if necessary
                #pragma omp master
                if(fsave && step % SAVEFREQ == 0)
                {
                  save(fsave, n, particles);
                }
            }
        }
    }
    simulation_time = read_timer() - simulation_time;

    printf("n = %d,threads = %d, simulation time = %g seconds", n, numthreads, simulation_time);

    if(find_option(argc, argv, "-no") == -1)
    {
        if(nabsavg)
        {
            absavg /= nabsavg;
        }
        printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
        if(absmin < 0.4)
        {
            printf("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
        }
        if(absavg < 0.8)
        {
            printf("\nThe average distance is below 0.8 meaning that most particles are not interacting");
        }
    }
    printf("\n");

    // Printing summary data
    if(fsum)
    {
        fprintf(fsum, "%d %d %g\n", n, numthreads, simulation_time);
        fclose(fsum);
    }
    if(fsave)
    {
        fclose(fsave);
    }
}

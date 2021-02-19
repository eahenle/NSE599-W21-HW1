#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <list>
#include <vector>
#include <cmath>
#include <algorithm>
#include <float.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "common.h"

using namespace std;

int nb_bins_per_row, nb_bins, nb_procs, mpi_rank, nb_rows_per_process;
double mpi_size;

MPI_Datatype PARTICLE;

// Encapsulated particle data type
class Indexed_particle
{
    public:
        particle_t p; // the particle
        int idx; // its index in the simulation
        int bin; // its bin number

        // reimplementation of move function
        // w/o this, program is VERY SLOW
        void move()
        {
            this->p.vx += this->p.ax * dt;
            this->p.vy += this->p.ay * dt;
            this->p.x  += this->p.vx * dt;
            this->p.y  += this->p.vy * dt;
            while(this->p.x < 0 || this->p.x > mpi_size)
            {
                this->p.x  = this->p.x < 0 ? -this->p.x : 2 * mpi_size - this->p.x;
                this->p.vx = -this->p.vx;
            }
            while(this->p.y < 0 || this->p.y > mpi_size)
            {
                this->p.y  = this->p.y < 0 ? -this->p.y : 2 * mpi_size - this->p.y;
                this->p.vy = -this->p.vy;
            }
        }

        // reimplementation of apply_force
        void apply_force(particle_t &neighbor, double *dmin, double *davg, int *navg)
        {
            double dx = neighbor.x - this->p.x;
            double dy = neighbor.y - this->p.y;
            double r2 = dx * dx + dy * dy;
            if(r2 > cutoff * cutoff)
            {
                return;
            }
        	if(r2 != 0){
    	       if(r2 / (cutoff * cutoff) < *dmin * (*dmin))
        	   {
                   *dmin = sqrt(r2) / cutoff;
               }
               (*davg) += sqrt(r2) / cutoff;
               (*navg)++;
            }
            r2 = fmax(r2, min_r * min_r);
            double coef = (1 - cutoff / sqrt(r2)) / r2 / mass;
            this->p.ax += coef * dx;
            this->p.ay += coef * dy;
        }
};

// Class encapsulation for spatial bins
class Bin
{
    public:
        list<Indexed_particle*> particles; // the bin's particles
        list<Indexed_particle*> incoming_particles; // staging ground for particles entering the bin during a move

        //move the particles in the bins for one time step
        void move_particles(vector<Bin> &bins, int bin_index)
        {
            auto it = this->particles.begin();
            while(it != this->particles.end())
            {
                Indexed_particle *p = *it;
                p->move();
                double bin_side_len = mpi_size / nb_bins_per_row;
                int row_b = floor(p->p.x / bin_side_len);
                int col_b = floor(p->p.y / bin_side_len);
                int new_bin_index =  row_b + col_b * nb_bins_per_row;
                if(new_bin_index != bin_index)
                { //if particle is not in the same position
                    p->bin = new_bin_index;
                    this->particles.erase(it++);
                    bins[new_bin_index].incoming_particles.push_back(p);
                }
                else
                {
                    it++;
                }
            }
        }
};

//initialize the position in the particles
void init_particles(int mpi_rank, int n, double size, Indexed_particle *ip)
{
    if(mpi_rank != 0)
    {
        return;
    }
    srand48(time(NULL));

    int sx = (int)ceil(sqrt((double)n));
    int sy = (n + sx - 1) / sx;

    int *shuffle = (int*)malloc(n * sizeof(int));
    for(int i = 0; i < n; i++)
    {
        shuffle[i] = i;
    }

    for(int i = 0; i < n; i++)
    {
        //  make sure particles are not spatially sorted
        int j = lrand48() % (n - i);
        int k = shuffle[j];
        shuffle[j] = shuffle[n - i - 1];

        //  distribute particles evenly to ensure proper spacing
        ip[i].p.x = size * (1 + (k % sx)) / (1 + sx);
        ip[i].p.y = size * (1 + (k / sx)) / (1 + sy);

        //  assign random velocities within a bound
        ip[i].p.vx = drand48() * 2 - 1;
        ip[i].p.vy = drand48() * 2 - 1;

        ip[i].idx = i;
    }
    free(shuffle);
}

//get the bin id for a particle
int particle_bin(double canvas_side_len, Indexed_particle &ip)
{
    int bin_row = ip.p.x * nb_bins_per_row / canvas_side_len;
    int bin_col = ip.p.y * nb_bins_per_row / canvas_side_len;
    return bin_col * nb_bins_per_row + bin_row;
}

//match the particles to the corresponding bin
void assign_particles_to_bins(int n, double canvas_side_len, Indexed_particle *particles, vector<Bin> &bins)
{
    for(int i = 0; i < n; ++i)
    {
        //Indexed_particle &p = particles[i];
        int b_idx = particle_bin(canvas_side_len, particles[i]);
        particles[i].bin = particle_bin(canvas_side_len, particles[i]);
        bins[b_idx].particles.push_back(&particles[i]);
    }
}

//initialize the bins in the canvas
void init_bins(int n, double size, Indexed_particle *particles, vector<Bin> &bins)
{
    for(int b_idx = 0; b_idx < nb_bins; b_idx++)
    {
        Bin b;
        bins.push_back(b);
    }
    assign_particles_to_bins(n, size, particles, bins);
}

//get the which process the bin is in
int mpi_rank_of_bin(int b_idx)
{
    int b_row = b_idx % nb_bins_per_row;
    return b_row / nb_rows_per_process;
}

//get bins in this processor
vector<int> bins_of_mpi_rank(int mpi_rank)
{
    vector<int> res;
    int row_s = mpi_rank * nb_rows_per_process, row_e = min(nb_bins_per_row, nb_rows_per_process * (mpi_rank + 1));
    for(int row = row_s; row < row_e; ++row)
    {
        for (int col = 0; col < nb_bins_per_row; ++col)
        {
            res.push_back(row + col * nb_bins_per_row);
        }
    }
    return res;
}

//get boerder particles around this processor
vector<Indexed_particle> get_mpi_rank_border_particles(int nei_mpi_rank, vector<Bin> &bins)
{
    int row;
    if(nei_mpi_rank < mpi_rank)
    {
        row = mpi_rank * nb_rows_per_process;
    }
    else
    {
        row = nb_rows_per_process * (mpi_rank + 1) - 1;
    }

    vector<Indexed_particle> res;
    if(row < 0 || row >= nb_bins_per_row)
    {
        return res;
    }
    for(int col = 0; col < nb_bins_per_row; ++col)
    {
        Bin &b = bins[row + col * nb_bins_per_row];
        for(auto &p : b.particles)
        {
            res.push_back(*p);
        }
    }
    return res;
}

int main(int argc, char **argv)
{
    int navg, nabsavg=0;
    double dmin, absmin=1.0, davg, absavg=0.0;
    double rdavg, rdmin;
    int rnavg;

    // Process command line parameters
    if(find_option(argc, argv, "-h") >= 0)
    {
        printf("Options:\n");
        printf("-h to see this help\n");
        printf("-n <int> to set the number of particles\n");
        printf("-o <filename> to specify the output file name\n");
        printf("-s <filename> to specify a summary file name\n");
        printf("-no turns off all correctness checks and particle output\n");
        return 0;
    }

    const int n = read_int(argc, argv, "-n", 1000);
    char *savename = read_string(argc, argv, "-o", NULL);
    char *sumname = read_string(argc, argv, "-s", NULL);

    // Set up MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nb_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    // Allocate generic resources
    FILE *fsave = savename && mpi_rank == 0 ? fopen(savename, "w") : NULL;
    FILE *fsum = sumname && mpi_rank == 0 ? fopen (sumname, "a") : NULL;

    // Allocate receieve & send buffer
    Indexed_particle *mpi_buff = new Indexed_particle[3 * n];
    MPI_Buffer_attach(mpi_buff, 10 * n * sizeof(Indexed_particle));

    // Allocate grobal particle buffer
    Indexed_particle *particles = (Indexed_particle*)malloc(n * sizeof(Indexed_particle));
    // Allocate local particle buffer
    Indexed_particle *local_particles = (Indexed_particle*)malloc(n * sizeof(Indexed_particle));

    // Allocate particle simulation coeffiency
    mpi_size = sqrt(density * n);
    double size = sqrt(density * n);

    nb_bins_per_row = max(1, size / (0.01 * 3));
    nb_bins = nb_bins_per_row * nb_bins_per_row;
    nb_rows_per_process = ceil(nb_bins_per_row / (float)nb_procs);

    init_particles(mpi_rank, n, size, particles);

    // initialize MPI PARTICLE type
    int n_local_particles, particle_size;
    int counter, cur_displs, counter_send;
    int lens[5];
    int counter_sends[nb_procs];
    int displs[nb_procs];

    MPI_Aint disp[5];
    MPI_Datatype temp;
    MPI_Datatype types[5];

    particle_size = sizeof(Indexed_particle);
    fill_n(lens, 5, 1);
    fill_n(types, 4, MPI_DOUBLE);
    types[4] = MPI_INT;
    disp[0] = (size_t)& (((Indexed_particle*)0)->p.x);
    disp[1] = (size_t)& (((Indexed_particle*)0)->p.y);
    disp[2] = (size_t)& (((Indexed_particle*)0)->p.vx);
    disp[3] = (size_t)& (((Indexed_particle*)0)->p.vy);
    disp[4] = (size_t)& (((Indexed_particle*)0)->idx);

    MPI_Type_create_struct(5, lens, disp, types, &temp);
    MPI_Type_create_resized(temp, 0, particle_size, &PARTICLE);
    MPI_Type_commit(&PARTICLE);

    //scatter the paritcles to each processors base on location
    Indexed_particle *particles_by_bin = new Indexed_particle[n];
    for(int pro = cur_displs = counter = 0; pro < nb_procs && mpi_rank == 0; cur_displs += counter_sends[pro], ++pro)
    {
        counter_send = 0;
        for(int i = 0; i < n; ++i)
        {
            if(mpi_rank_of_bin(particle_bin(size, particles[i])) != pro)
            {
                continue;
            }
            particles_by_bin[counter] = particles[i];
            counter_send++;
            counter++;
        }
        counter_sends[pro] = counter_send;
        displs[pro] = cur_displs;
    }

    // MPI initialize and send all other var in other processors
    MPI_Bcast(&counter_sends[0], nb_procs, MPI_INT, 0, MPI_COMM_WORLD);
    n_local_particles = counter_sends[mpi_rank];
    MPI_Bcast(&displs[0], nb_procs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(particles_by_bin, &counter_sends[0], &displs[0], PARTICLE, local_particles, n_local_particles, PARTICLE, 0, MPI_COMM_WORLD);

    // Initialize local bins
    vector<Bin> bins;
    vector<int> local_bin_idxs = bins_of_mpi_rank(mpi_rank);
    init_bins(n_local_particles, size, local_particles, bins);

    //  simulate a number of time steps
    double simulation_time = read_timer();
    for(int step = 0; step < NSTEPS; step++)
    {
        navg = 0;
        dmin = 1.0;
        davg = 0.0;

        // exchange neighbors particles for force computation
        vector<int> nei_mpi_ranks;
        if(mpi_rank > 0)
        {
            nei_mpi_ranks.push_back(mpi_rank - 1);
        }
        if(mpi_rank + 1 < nb_procs)
        {
            nei_mpi_ranks.push_back(mpi_rank + 1);
        }
        for(auto &nei_mpi_rank : nei_mpi_ranks)
        {
            vector<Indexed_particle> border_particles = get_mpi_rank_border_particles(nei_mpi_rank, bins);
            int n_b_particles = border_particles.size();
            const void *buf = n_b_particles == 0 ? 0 : &border_particles[0];
            MPI_Request request;
            MPI_Ibsend(buf, n_b_particles, PARTICLE, nei_mpi_rank, 0, MPI_COMM_WORLD, &request);
            MPI_Request_free(&request);
        }

        // neighbors collect border particles and assign to bins
        Indexed_particle *cur_pos = local_particles + n_local_particles;
        int n_particles_received = 0;
        for(auto &nei_mpi_rank : nei_mpi_ranks)
        {
            MPI_Status status;
            MPI_Recv(cur_pos, n, PARTICLE, nei_mpi_rank, 0, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, PARTICLE, &n_particles_received);
            assign_particles_to_bins(n_particles_received, size, cur_pos, bins);
            cur_pos += n_particles_received;
            n_local_particles += n_particles_received;
        }

        // Zero out the accelerations
        for(int i = 0; i < n_local_particles; ++i)
        {
            local_particles[i].p.ax = local_particles[i].p.ay = 0;
        }

        // Compute forces between each local bin and its neighbors
        for(auto &idx:local_bin_idxs)
        {
            int b1_row = idx % nb_bins_per_row;
            int b1_col = idx / nb_bins_per_row;
            for(int b2_row = max(0, b1_row - 1); b2_row <= min(nb_bins_per_row - 1, b1_row + 1); ++b2_row)
            {
                for(int b2_col = max(0, b1_col - 1); b2_col <= min(nb_bins_per_row - 1, b1_col + 1); ++b2_col)
                {
                    int b2 = b2_row + b2_col * nb_bins_per_row;
                    for(auto &it1 : bins[idx].particles)
                    {
                        for(auto &it2 : bins[b2].particles)
                        {
                             it1->apply_force(it2->p, &dmin, &davg, &navg);
                        }
                    }
                }
            }
        }

        if(find_option(argc, argv, "-no") == -1)
        {
            MPI_Reduce(&davg, &rdavg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&navg, &rnavg, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&dmin, &rdmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

            if(mpi_rank == 0)
            {
                if(rnavg)
                {
                    absavg += rdavg / rnavg;
                    nabsavg++;
                }
                if(rdmin < absmin)
                {
                    absmin = rdmin;
                }
            }
        }

        //  move particles in each bins
        for(auto &bin_index : local_bin_idxs)
        {
            bins[bin_index].move_particles(bins, bin_index);
        }

        // refresh the particles in bins
        for(auto &bin_index : local_bin_idxs)
        {
            bins[bin_index].particles.splice(bins[bin_index].particles.end(), bins[bin_index].incoming_particles);
            bins[bin_index].incoming_particles.clear();
        }

        // exchange particles after moveing
        vector<int> neighbor_mpi_ranks;
        if(mpi_rank > 0)
        {
            neighbor_mpi_ranks.push_back(mpi_rank - 1);
        }
        if(mpi_rank + 1 < nb_procs)
        {
            neighbor_mpi_ranks.push_back(mpi_rank + 1);
        }

        for(auto &nei_mpi_rank : neighbor_mpi_ranks)
        {
            vector<int> cur_bins = bins_of_mpi_rank(nei_mpi_rank);
            vector<Indexed_particle> moved_particles;
            for(auto b_idx : cur_bins)
            {
                for(auto &p : bins[b_idx].incoming_particles)
                {
                    moved_particles.push_back(*p);
                }
            }
            int n_moved_p = moved_particles.size();
            const void *buf = n_moved_p == 0 ? 0 : &moved_particles[0];
            MPI_Request request;
            MPI_Ibsend(buf, n_moved_p, PARTICLE, nei_mpi_rank, 0, MPI_COMM_WORLD, &request);
            MPI_Request_free(&request);
        }

        Indexed_particle *new_local_particles = new Indexed_particle[n];
        Indexed_particle *tmp_pos = new_local_particles;

        for(auto &nei_mpi_rank : neighbor_mpi_ranks)
        {
            MPI_Status status;
            MPI_Recv(tmp_pos, n, PARTICLE, nei_mpi_rank, 0, MPI_COMM_WORLD, &status);
            int n_particles_received;
            MPI_Get_count(&status, PARTICLE, &n_particles_received);
            tmp_pos += n_particles_received;
        }

        for(auto &b_idx : local_bin_idxs)
        {
            for(auto &p : bins[b_idx].particles)
            {
                *tmp_pos = *p;
                tmp_pos++;
            }
        }

        // Apply new_local_particles
        local_particles = new_local_particles;
        n_local_particles = tmp_pos - new_local_particles;
        // Rebin all particles
        bins.clear();
        init_bins(n_local_particles, size, new_local_particles, bins);
    }

    simulation_time = read_timer() - simulation_time;

    if(mpi_rank == 0)
    {
        printf("n = %d, simulation time = %g seconds", n, simulation_time);

        if(find_option(argc, argv, "-no") == -1)
        {
            if(nabsavg)
            {
                absavg /= nabsavg;
            }
            printf(", absmin = %lf, absavg = %lf", absmin, absavg);
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

        if(fsum){
            fprintf(fsum, "%d %d %g\n", n, nb_procs, simulation_time);
        }
    }

    if(fsum)
    {
        fclose(fsum);
    }
    free(particles);
    if(fsave)
    {
        fclose(fsave);
    }

    MPI_Finalize();

    return 0;
}

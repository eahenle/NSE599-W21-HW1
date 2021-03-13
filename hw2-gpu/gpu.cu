#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include "common.h"

#define NUM_THREADS 256

// get the bin ID for a particle
#define get_bin(p, bins_num, s) (int)(p.x/(double)(s/bins_num))+(int)(p.y/(double)(s/bins_num))*bins_num
// get the x-indices of a bin block
#define get_x(row) {row - 1, row - 1, row - 1, row, row, row, row + 1, row + 1, row + 1}
// get the y-indices of a bin block
#define get_y(col) {col - 1, col, col + 1, col - 1, col, col + 1, col - 1, col, col + 1}

extern double size;

// spatial bin for particles
class Bin
{
    public:
        // counters
        int nb_particles;
        int nb_particles_tPlusOne;
        int nb_particles_to_remove;
        // particle index array
        int particles[];
        // particle index buffers
        int particles_tPlusOne[16];
        int outgoing_particles[16];

        // constructor
        Bin()
        {
            this->nb_particles_tPlusOne = 0;
            this->nb_particles_to_remove = 0;
            this->nb_particles = 0;
        }

        // add a particle by index
        __host__ __device__ void add(int p_id)
        {
            this->particles[this->nb_particles] = p_id;
            this->nb_particles++;
        }

        // record the particle state in new step
        __host__ __device__ void update(int new_bin, int current_bin, int p_id)
        {
            if (new_bin != current_bin)
            {
                this->outgoing_particles[this->nb_particles_to_remove++] = p_id;
            }
            else
            {
                this->particles_tPlusOne[this->nb_particles_tPlusOne++] = p_id;
            }
        }

        // reset the counters
        __host__ __device__ void reset_counters()
        {
            this->nb_particles_tPlusOne = 0;
            this->nb_particles_to_remove = 0;
        }

        // exchange the particle from previous step to current step
        __host__ __device__ void exchange(int p_id)
        {
            this->particles[p_id] = this->particles_tPlusOne[p_id];
        }
};


// device-side acceleration calculation
__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor)
{
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if(r2 > cutoff * cutoff)
    {
        return;
    }
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}


// device-side position update
__device__ void move_particle_gpu(particle_t &p, double d_size)
{
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x  += p.vx * dt;
    p.y  += p.vy * dt;
    while(p.x < 0 || p.x > d_size)
    {
        p.x  = p.x < 0 ? -p.x : 2 * d_size - p.x;
        p.vx = -p.vx;
    }
    while(p.y < 0 || p.y > d_size)
    {
        p.y = p.y < 0 ? -p.y : 2 * d_size - p.y;
        p.vy = -p.vy;
    }
}


// loop over bins in blocks to calculate forces
__global__ void bin_block_forces(particle_t *particles, Bin *device_bins, int d_nb_bins, int d_nb_bins_per_row)
{
    // get bin ID
    int current_bin = threadIdx.x + blockIdx.x * blockDim.x;
    if(current_bin >= d_nb_bins)
    {
        return;
    }

    int b1_row = current_bin % d_nb_bins_per_row;
    int b1_col = current_bin / d_nb_bins_per_row;

    for(int p1 = 0; p1 < device_bins[current_bin].nb_particles; ++p1)
    {
        particles[device_bins[current_bin].particles[p1]].ax = particles[device_bins[current_bin].particles[p1]].ay = 0;
    }

    int x_idx[] = get_x(b1_row);
    int y_idx[] = get_y(b1_col);
    for(int i = 0; i < 9; ++i)
    {
        if(x_idx[i] >= 0 && x_idx[i] < d_nb_bins_per_row && y_idx[i] >= 0 && y_idx[i] < d_nb_bins_per_row)
        {
            int neighbor = x_idx[i] + y_idx[i] * d_nb_bins_per_row;
            for(int p1 = 0; p1 < device_bins[current_bin].nb_particles; ++p1)
            {
                for(int p2 = 0; p2 < device_bins[neighbor].nb_particles; ++p2)
                {
                    // compute force between particles of current/neighbor bins
                    apply_force_gpu(particles[device_bins[current_bin].particles[p1]], particles[device_bins[neighbor].particles[p2]]);
                }
            }
        }
    }
}


// update particle positions
__global__ void move_gpu(particle_t *particles, Bin *device_bins, double d_size, int d_nb_bins_per_row, int d_nb_bins)
{
    // Get bin ID
    int current_bin = threadIdx.x + blockIdx.x * blockDim.x;
    if(current_bin >= d_nb_bins)
    {
        return;
    }

    device_bins[current_bin].reset_counters();

    // assign particles as leaving or staying
    for(int p1 = 0; p1 < device_bins[current_bin].nb_particles; ++p1)
    {
        int p_new = device_bins[current_bin].particles[p1];
        particle_t &p = particles[p_new];
        move_particle_gpu(p, d_size);
        device_bins[current_bin].update(
            get_bin(p, d_nb_bins_per_row, d_size), current_bin, p_new);
    }
}


// bin/re-bin particles
__global__ void binning(particle_t *particles, Bin *device_bins, double d_size, int d_nb_bins_per_row, int d_nb_bins)
{
    // Get thread bin ID
    int current_bin = threadIdx.x + blockIdx.x * blockDim.x;
    if(current_bin >= d_nb_bins)
    {
        return;
    }

    // Saves the particle that stays in the bin
    device_bins[current_bin].nb_particles = device_bins[current_bin].nb_particles_tPlusOne;
    for(int p1 = 0; p1 < device_bins[current_bin].nb_particles; ++p1)
    {
        device_bins[current_bin].exchange(p1);
    }

    // accept the incoming particle to the bin
    int b_row = current_bin % d_nb_bins_per_row;
    int b_col = current_bin / d_nb_bins_per_row;
    int x_idx[] = get_x(b_row);
    int y_idx[] = get_y(b_col);

    for(int i = 0; i < 9; ++i)
    {
        if(x_idx[i] >= 0 && x_idx[i] < d_nb_bins_per_row && y_idx[i] >= 0 && y_idx[i] < d_nb_bins_per_row)
        {
            int b2 = x_idx[i] + y_idx[i] * d_nb_bins_per_row;
            for(int p2 = 0; p2 < device_bins[b2].nb_particles_to_remove; ++p2)
            {
                int incoming_particle = device_bins[b2].outgoing_particles[p2];
                particle_t &p = particles[incoming_particle];
                if(get_bin(p, d_nb_bins_per_row, d_size) == current_bin)
                {
                    device_bins[current_bin].add(incoming_particle);
                }
            }
        }
    }
}


int main(int argc, char **argv)
{
    cudaDeviceSynchronize();

    if(find_option(argc, argv, "-h") >= 0)
    {
        printf("Options:\n");
        printf("-h to see this help\n");
        printf("-n <int> to set the number of particles\n");
        printf("-o <filename> to specify the output file name\n");
        return 0;
    }

    const int n = read_int(argc, argv, "-n", 1000);

    char *savename = read_string(argc, argv, "-o", NULL);
    FILE *fsave = savename ? fopen(savename, "w") : NULL;

    // global memory particle array allocation
    particle_t *particles = (particle_t*)malloc(n * sizeof(particle_t));
    set_size(n);
    init_particles(n, particles);

    // GPU device memory particle array allocation
    particle_t *device_particles;
    cudaMalloc((void**)&device_particles, n * sizeof(particle_t));
    cudaDeviceSynchronize();

    // bins is an array of Bin objects (global memory allocation)
    const int nb_bins_per_row = ceil(sqrt(density * n) / cutoff);
    const int nb_bins = nb_bins_per_row * nb_bins_per_row;
    Bin *bins = new Bin[nb_bins];

    // GPU device memory bin allocation
    Bin *device_bins;
    cudaMalloc((void**)&device_bins, nb_bins * sizeof(Bin));
    cudaDeviceSynchronize();

    // assign initial particle bins in global memory
    for(int i = 0; i < n; ++i)
    {
        bins[get_bin(particles[i], nb_bins_per_row, size)].add(i);
    }

    // copy bins and particles to GPU device memory
    double copy_time = read_timer();
    cudaMemcpy(device_bins, bins, nb_bins * sizeof(Bin), cudaMemcpyHostToDevice);
    cudaMemcpy(device_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    copy_time = read_timer() - copy_time;

    // GPU thread-blocks
    const int nb_blocks = (nb_bins + NUM_THREADS - 1) / NUM_THREADS;

    double simulation_time = read_timer();
    for(int step = 0; step < NSTEPS; step++)
    {
        // calculate accelerations
        bin_block_forces <<< nb_blocks, NUM_THREADS >>> (device_particles, device_bins, nb_bins, nb_bins_per_row);

        // update positions
        move_gpu <<< nb_blocks, NUM_THREADS >>> (device_particles, device_bins, size, nb_bins_per_row, nb_bins);

        // re-bin
        binning <<< nb_blocks, NUM_THREADS >>> (device_particles, device_bins, size, nb_bins_per_row, nb_bins);

        if(fsave && (step % SAVEFREQ) == 0)
        {
            cudaMemcpy(particles, device_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            save(fsave, n, particles);
        }
    }

    simulation_time = read_timer() - simulation_time;
    printf("CPU-GPU copy time = %g seconds\n", copy_time);
    printf("n = %d, simulation time = %g seconds\n", n, simulation_time);
    if(fsave)
    {
        fclose(fsave);
    }
}

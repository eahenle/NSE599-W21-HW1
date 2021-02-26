#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include "common.h"

#define NUM_THREADS 256

/// change NUM_PARTICLE_BIN to dynamic?
#define NUM_PARTICLE_BIN 16

/// make into inline function?
#define get_bin(p, bins_num, s) (int)(p.x/(double)(s/bins_num))+(int)(p.y/(double)(s/bins_num))*bins_num

extern double size;

// spatial bin for particles
/// minimalize and annotate
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
        int particles_tPlusOne[NUM_PARTICLE_BIN];
        int outgoing_particles[NUM_PARTICLE_BIN];

        Bin(){
            this->nb_particles_tPlusOne = this->nb_particles_to_remove = this->nb_particles = 0;
        }

        //add new particle to the bin
        __host__ __device__ void add(int par_id){
            this->particles[this->nb_particles] = par_id;
            this->nb_particles++;
        }

        //record the particle state in new step
        __host__ __device__ void update(int new_bin, int cur_bin, int p_id){
            if (new_bin != cur_bin) {
                this->outgoing_particles[this->nb_particles_to_remove++] = p_id;
            } else {
                this->particles_tPlusOne[this->nb_particles_tPlusOne++] = p_id;
            }
        }

        //zero the counters
        __host__ __device__ void clear_counter(){
            this->nb_particles_tPlusOne = this->nb_particles_to_remove = 0;
        }

        //exchange the particle from previous step to current step
        __host__ __device__ void exchange(int p_id){
            this->particles[p_id] = this->particles_tPlusOne[p_id];
        }
};

/// write into compute_forces_gpu
__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor)
{
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if(r2 > cutoff*cutoff)
    {
        return;
    }
    r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
    double r = sqrt(r2);
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

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

// calculate particle accelerations
/// annotate and minimalize
__global__ void compute_forces_gpu(particle_t *particles, Bin *device_bins, int d_nb_bins, int d_nb_bins_per_row)
{
    // Get thread (bin) ID
    int cur_bin = threadIdx.x + blockIdx.x * blockDim.x;
    if(cur_bin >= d_nb_bins)
    {
        return;
    }

    int b1_row = cur_bin % d_nb_bins_per_row;
    int b1_col = cur_bin / d_nb_bins_per_row;

    for(int p1 = 0; p1 < device_bins[cur_bin].nb_particles; ++p1)
    {
        particles[device_bins[cur_bin].particles[p1]].ax = particles[device_bins[cur_bin].particles[p1]].ay = 0;
    }

    //nine bins, left to right, top do down
    int x_idx[] = {b1_row-1, b1_row-1, b1_row-1, b1_row,   b1_row, b1_row,   b1_row+1, b1_row+1, b1_row+1};
    int y_idx[] = {b1_col-1, b1_col,   b1_col+1, b1_col-1, b1_col, b1_col+1, b1_col-1,  b1_col,  b1_col+1};
    for(int i = 0; i < 9; ++i)
    {
        if(x_idx[i] >= 0 && x_idx[i] < d_nb_bins_per_row && y_idx[i] >= 0 && y_idx[i] < d_nb_bins_per_row)
        {
            int nei_bin = x_idx[i] + y_idx[i] * d_nb_bins_per_row; //get the neighbor bin
            for(int p1 = 0; p1 < device_bins[cur_bin].nb_particles; ++p1)
            {
                for(int p2 = 0; p2 < device_bins[nei_bin].nb_particles; ++p2)
                {
                    //compute force between cur bin and neighbor bin
                    /// inline (subsume apply_force_gpu)
                    apply_force_gpu(particles[device_bins[cur_bin].particles[p1]], particles[device_bins[nei_bin].particles[p2]]);
                }
            }
        }
    }
}

// update particle positions
__global__ void move_gpu(particle_t *particles, Bin *device_bins, double d_size, int d_nb_bins_per_row, int d_nb_bins)
{
    // Get thread (bin) ID
    int cur_bin = threadIdx.x + blockIdx.x * blockDim.x;
    if(cur_bin >= d_nb_bins)
    {
        return;
    }

    // initialize the exchange counter
    device_bins[cur_bin].clear_counter();

    // Move this bin's particles to either leaving or staying
    for(int p1 = 0; p1 < device_bins[cur_bin].nb_particles; ++p1)
    {
        //move the particle to new bin
        int p_new = device_bins[cur_bin].particles[p1];
        particle_t &p = particles[p_new];
        move_particle_gpu(p, d_size);
        // record the state of the partivle
        int new_b_idx = get_bin(p, d_nb_bins_per_row, d_size);
        device_bins[cur_bin].update(new_b_idx, cur_bin, p_new); // check whether the partivle move to a new bin
    }
}

//allocate the particle after moved, each particle is in new position. Assign the bins again for the particles.
__global__ void binning(particle_t *particles, Bin *device_bins, double d_size, int d_nb_bins_per_row, int d_nb_bins)
{
    // Get thread bin ID
    int cur_bin = threadIdx.x + blockIdx.x * blockDim.x;
    if(cur_bin >= d_nb_bins)
    {
        return;
    }

    // Saves the particle that stays in the bin
    device_bins[cur_bin].nb_particles = device_bins[cur_bin].nb_particles_tPlusOne;
    for(int p1 = 0; p1 < device_bins[cur_bin].nb_particles; ++p1)
    {
        device_bins[cur_bin].exchange(p1);
    }

    // accept the incoming particle to the bin
    int cur_b_row = cur_bin % d_nb_bins_per_row;
    int cur_b_col = cur_bin / d_nb_bins_per_row;
    int x_idx[] = {cur_b_row-1, cur_b_row-1, cur_b_row-1, cur_b_row,   cur_b_row, cur_b_row,   cur_b_row+1, cur_b_row+1, cur_b_row+1};
    int y_idx[] = {cur_b_col-1, cur_b_col,   cur_b_col+1, cur_b_col-1, cur_b_col, cur_b_col+1, cur_b_col-1,  cur_b_col,  cur_b_col+1};

    for(int i = 0; i < 9; ++i)
    {
      //check out of border
        if(x_idx[i] >= 0 && x_idx[i] < d_nb_bins_per_row && y_idx[i] >= 0 && y_idx[i] < d_nb_bins_per_row)
        {
            int b2 = x_idx[i] + y_idx[i] * d_nb_bins_per_row;
            for(int p2 = 0; p2 < device_bins[b2].nb_particles_to_remove; ++p2)
            {
                int par_comming = device_bins[b2].outgoing_particles[p2];
                particle_t &p = particles[par_comming];
                if(get_bin(p, d_nb_bins_per_row, d_size) == cur_bin)
                {//find the particle from the neighbor that arrives in the cur bin
                    device_bins[cur_bin].add(par_comming);
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

    double simulation_time = read_timer();
    for(int step = 0; step < NSTEPS; step++)
    {
        //  compute forces
        int blks = (nb_bins + NUM_THREADS - 1) / NUM_THREADS;
        compute_forces_gpu <<< blks, NUM_THREADS >>> (device_particles, device_bins, nb_bins, nb_bins_per_row);

        //  move particles
        move_gpu <<< blks, NUM_THREADS >>> (device_particles, device_bins, size, nb_bins_per_row, nb_bins);

        //  recalulate the particle in bins
        binning <<< blks, NUM_THREADS >>> (device_particles, device_bins, size, nb_bins_per_row, nb_bins);

        //  save if necessary
        if(fsave && (step % SAVEFREQ) == 0)
        {
            // Copy the particles back to the CPU
            cudaMemcpy(particles, device_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
            save(fsave, n, particles);
        }
    }

    cudaDeviceSynchronize();
    simulation_time = read_timer() - simulation_time;

    printf("CPU-GPU copy time = %g seconds\n", copy_time);
    printf("n = %d, simulation time = %g seconds\n", n, simulation_time);

    free(particles);
    cudaFree(device_particles);
    cudaFree(device_bins);
    if(fsave)
    {
        fclose(fsave);
    }
}

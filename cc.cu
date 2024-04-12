Copy/Compute Overlap Example Code

Below are two code examples for the techniques presented above, first for when the number of entries is evenly divided by the number of streams, and second, for when this is not so.
N is Evenly Divided by Number of Streams

// "Simple" version where number of entries is evenly divisible by number of streams.

// Set to a ridiculously low value to clarify mechanisms of the technique.
const uint64_t num_entries = 10;
const uint64_t num_iters = 1UL << 10;

// Allocate memory for all data entries. Make sure to pin host memory.
cudaMallocHost(&data_cpu, sizeof(uint64_t)*num_entries);
cudaMalloc    (&data_gpu, sizeof(uint64_t)*num_entries);

// Set the number of streams.
const uint64_t num_streams = 2;

// Create an array of streams containing number of streams
cudaStream_t streams[num_streams];
for (uint64_t stream = 0; stream < num_streams; stream++)
    cudaStreamCreate(&streams[stream]);

// Set number of entries for each "chunk". Assumes `num_entries % num_streams == 0`.
const uint64_t chunk_size = num_entries / num_streams;

// For each stream, calculate indices for its chunk of full dataset and then, HtoD copy, compute, DtoH copy.
for (uint64_t stream = 0; stream < num_streams; stream++) {

    // Get start index in full dataset for this stream's work.
    const uint64_t lower = chunk_size*stream;
    
    // Stream-indexed (`data+lower`) and chunk-sized HtoD copy in the non-default stream
    // `streams[stream]`.
    cudaMemcpyAsync(data_gpu+lower, data_cpu+lower, 
           sizeof(uint64_t)*chunk_size, cudaMemcpyHostToDevice, 
           streams[stream]);
    
    // Stream-indexed (`data_gpu+lower`) and chunk-sized compute in the non-default stream
    // `streams[stream]`.
    decrypt_gpu<<<80*32, 64, 0, streams[stream]>>>
        (data_gpu+lower, chunk_size, num_iters);
    
    // Stream-indexed (`data+lower`) and chunk-sized DtoH copy in the non-default stream
    // `streams[stream]`.
    cudaMemcpyAsync(data_cpu+lower, data_gpu+lower, 
           sizeof(uint64_t)*chunk_size, cudaMemcpyDeviceToHost, 
           streams[stream]);
}

// Destroy streams.
for (uint64_t stream = 0; stream < num_streams; stream++)
    cudaStreamDestroy(streams[stream]);

N is Not Evenly Divided by Number of Streams

// Able to handle when `num_entries % num_streams != 0`.

const uint64_t num_entries = 10;
const uint64_t num_iters = 1UL << 10;

cudaMallocHost(&data_cpu, sizeof(uint64_t)*num_entries);
cudaMalloc    (&data_gpu, sizeof(uint64_t)*num_entries);

// Set the number of streams to not evenly divide num_entries.
const uint64_t num_streams = 3;

cudaStream_t streams[num_streams];
for (uint64_t stream = 0; stream < num_streams; stream++)
    cudaStreamCreate(&streams[stream]);

// Use round-up division (`sdiv`, defined in helper.cu) so `num_streams*chunk_size`
// is never less than `num_entries`.
// This can result in `num_streams*chunk_size` being greater than `num_entries`, meaning
// we will need to guard against out-of-range errors in the final "tail" stream (see below).
const uint64_t chunk_size = sdiv(num_entries, num_streams);

for (uint64_t stream = 0; stream < num_streams; stream++) {

    const uint64_t lower = chunk_size*stream;
    // For tail stream `lower+chunk_size` could be out of range, so here we guard against that.
    const uint64_t upper = min(lower+chunk_size, num_entries);
    // Since the tail stream width may not be `chunk_size`,
    // we need to calculate a separate `width` value.
    const uint64_t width = upper-lower;

    // Use `width` instead of `chunk_size`.
    cudaMemcpyAsync(data_gpu+lower, data_cpu+lower, 
           sizeof(uint64_t)*width, cudaMemcpyHostToDevice, 
           streams[stream]);

    // Use `width` instead of `chunk_size`.
    decrypt_gpu<<<80*32, 64, 0, streams[stream]>>>
        (data_gpu+lower, width, num_iters);

    // Use `width` instead of `chunk_size`.
    cudaMemcpyAsync(data_cpu+lower, data_gpu+lower, 
           sizeof(uint64_t)*width, cudaMemcpyDeviceToHost, 
           streams[stream]);
}

// Destroy streams.
for (uint64_t stream = 0; stream < num_streams; stream++)
    cudaStreamDestroy(streams[stream]);

Check for Understanding
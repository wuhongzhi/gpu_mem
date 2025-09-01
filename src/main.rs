use clap::{Parser, Subcommand};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_GPU};
use opencl3::memory::{Buffer, ClMem, CL_MAP_READ, CL_MAP_WRITE, CL_MEM_READ_WRITE};
use opencl3::platform::get_platforms;
use opencl3::types::CL_BLOCKING;
use std::ptr;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "gpu_mem")]
#[command(about = "A GPU memory testing utility")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Verify basic read/write functionality on GPU memory
    Verify {
        /// Size in bytes (e.g., 1G, 512M, 64K)
        #[arg(long, default_value = "1G")]
        size: String,
        /// Use memory mapping
        #[arg(long)]
        mmap: bool,
        /// Device ID to use (default: 0)
        #[arg(long, default_value = "0")]
        device_id: usize,
    },
    /// Memory set operation benchmark
    Memset {
        /// Size in bytes (e.g., 1G, 512M, 64K)
        #[arg(long, default_value = "1G")]
        size: String,
        /// Chunk size for operations (e.g., 64K, 1M)
        #[arg(long, default_value = "64K")]
        chunk_size: String,
        /// Run test on GPU memory
        #[arg(long, conflicts_with = "host")]
        gpu: bool,
        /// Run test on host memory
        #[arg(long, conflicts_with = "gpu")]
        host: bool,
        /// Use memory mapping
        #[arg(long)]
        mmap: bool,
        /// Device ID to use (default: 0)
        #[arg(long, default_value = "0")]
        device_id: usize,
    },
    /// Memory copy operation benchmark
    Memcpy {
        /// Size in bytes (e.g., 1G, 512M, 64K)
        #[arg(long, default_value = "1G")]
        size: String,
        /// Chunk size for operations (e.g., 64K, 1M)
        #[arg(long, default_value = "64K")]
        chunk_size: String,
        /// Copy from GPU to GPU
        #[arg(long, group = "copy_type")]
        gpu_to_gpu: bool,
        /// Copy from host to host
        #[arg(long, group = "copy_type")]
        host_to_host: bool,
        /// Copy from GPU to host
        #[arg(long, group = "copy_type")]
        gpu_to_host: bool,
        /// Copy from host to GPU
        #[arg(long, group = "copy_type")]
        host_to_gpu: bool,
        /// Use memory mapping
        #[arg(long)]
        mmap: bool,
        /// Device ID to use (default: 0)
        #[arg(long, default_value = "0")]
        device_id: usize,
    },
}

fn parse_size(size_str: &str) -> Result<usize, Box<dyn std::error::Error>> {
    let size_str = size_str.trim().to_uppercase();
    let (number_part, suffix) = if size_str.ends_with('G') {
        (&size_str[..size_str.len() - 1], 1024 * 1024 * 1024)
    } else if size_str.ends_with('M') {
        (&size_str[..size_str.len() - 1], 1024 * 1024)
    } else if size_str.ends_with('K') {
        (&size_str[..size_str.len() - 1], 1024)
    } else {
        (size_str.as_str(), 1)
    };

    let number: usize = number_part.parse()?;
    Ok(number * suffix)
}

fn preload_memory(ptr: *mut u8, size: usize) -> Result<(), Box<dyn std::error::Error>> {
    // Write data to ensure pages are loaded into memory
    let page_size = 4096; // Common page size
    let mut offset = 0;

    unsafe {
        while offset < size {
            *(ptr.add(offset)) = 0; // Write to trigger page load
            offset += page_size;
        }
        // Write to the last byte to ensure the final page is loaded
        if size > 0 {
            *(ptr.add(size - 1)) = 0;
        }
    }
    Ok(())
}

fn chunked_memset(ptr: *mut u8, pattern: u8, size_bytes: usize, chunk_bytes: usize) {
    // Perform memset in chunks
    let mut offset = 0;
    while offset < size_bytes {
        let current_chunk = std::cmp::min(chunk_bytes, size_bytes - offset);
        unsafe {
            std::ptr::write_bytes(ptr.add(offset), pattern, current_chunk);
        }
        offset += current_chunk;
    }
}

fn chunked_memcpy(src_ptr: *const u8, dst_ptr: *mut u8, size_bytes: usize, chunk_bytes: usize) {
    // Perform memcpy in chunks
    let mut offset = 0;
    while offset < size_bytes {
        let current_chunk = std::cmp::min(chunk_bytes, size_bytes - offset);
        unsafe {
            std::ptr::copy_nonoverlapping(src_ptr.add(offset), dst_ptr.add(offset), current_chunk);
        }
        offset += current_chunk;
    }
}

struct OpenCLBuf<'a, T> {
    buf: Buffer<T>,
    mapped_ptr: *mut libc::c_void,
    ctx: &'a OpenCLContext,
}

impl<T> Drop for OpenCLBuf<'_, T> {
    fn drop(&mut self) {
        let addr = self.mapped_ptr;
        if addr != ptr::null_mut() {
            unsafe {
                let ctx = self.ctx;
                let buf = &self.buf;
                ctx.queue
                    .enqueue_unmap_mem_object(buf.get(), addr, &[])
                    .expect("unmap failure");
            }
        }
    }
}

struct OpenCLContext {
    context: Context,
    queue: CommandQueue,
}

impl OpenCLContext {
    fn new(device_id: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let platforms = get_platforms()?;
        if platforms.is_empty() {
            return Err("No OpenCL platforms found".into());
        }

        let device_ids =
            get_all_devices(CL_DEVICE_TYPE_GPU).or_else(|_| get_all_devices(CL_DEVICE_TYPE_ALL))?;

        if device_ids.is_empty() {
            return Err("No OpenCL devices found".into());
        }

        if device_id >= device_ids.len() {
            return Err(format!(
                "Device ID {} not found. Available devices: 0-{}",
                device_id,
                device_ids.len() - 1
            )
            .into());
        }

        let device = Device::new(device_ids[device_id]);
        let context = Context::from_device(&device)?;
        let queue =
            unsafe { CommandQueue::create_with_properties(&context, device_ids[device_id], 0, 0)? };

        println!("Available devices:");
        for (i, &dev_id) in device_ids.iter().enumerate() {
            let dev = Device::new(dev_id);
            let name = dev.name().unwrap_or_else(|_| "Unknown".to_string());
            let marker = if i == device_id { " (selected)" } else { "" };
            println!("  Device {}: {}{}", i, name, marker);
        }

        Ok(OpenCLContext { context, queue })
    }

    fn alloc_gpu_memory<'a, T>(
        & 'a self,
        element_count: usize,
        mmap: bool,
    ) -> Result<OpenCLBuf<'a, T>, Box<dyn std::error::Error>> {
        let mut mapped_ptr = ptr::null_mut();
        let mut buf = unsafe {
            Buffer::<T>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                element_count,
                ptr::null_mut(),
            )?
        };

        if mmap {
            unsafe {
                self.queue.enqueue_map_buffer::<T>(
                    &mut buf,
                    CL_BLOCKING,
                    CL_MAP_WRITE | CL_MAP_READ,
                    0,
                    element_count,
                    &mut mapped_ptr,
                    &[],
                )?;
            }
            preload_memory(
                mapped_ptr as *mut u8,
                element_count * std::mem::size_of::<T>(),
            )?;
        }

        Ok(OpenCLBuf {
            buf,
            mapped_ptr,
            ctx: self,
        })
    }
}

fn run_verify(
    size: String,
    use_mmap: bool,
    device_id: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let size_bytes = parse_size(&size)?;
    let num_elements = size_bytes / 4; // f32 elements

    println!(
        "Running verify test with {} bytes ({} elements), mmap: {}",
        size_bytes, num_elements, use_mmap
    );

    let cl_ctx = OpenCLContext::new(device_id)?;

    // Create test data
    let host_data: Vec<f32> = (0..num_elements).map(|i| i as f32).collect();
    // Read back and verify
    let mut result = vec![0.0f32; num_elements];

    preload_memory(result.as_mut_ptr() as *mut u8, size_bytes)?;

    // Create GPU buffer
    let mut buffer = cl_ctx.alloc_gpu_memory::<f32>(num_elements, use_mmap)?;

    // Mmap gpu buffer
    let start_time = Instant::now();
    if use_mmap {
        // Copy data to mapped memory
        unsafe {
            std::ptr::copy_nonoverlapping(
                host_data.as_ptr(),
                buffer.mapped_ptr as *mut f32,
                num_elements,
            );
        }
    } else {
        unsafe {
            cl_ctx
                .queue
                .enqueue_write_buffer(&mut buffer.buf, CL_BLOCKING, 0, &host_data, &[])?;
        }
    }

    if use_mmap {
        // Copy data from mapped memory
        unsafe {
            std::ptr::copy_nonoverlapping(
                buffer.mapped_ptr as *const f32,
                result.as_mut_ptr(),
                num_elements,
            );
        }
    } else {
        // Read using enqueue_read_buffer
        unsafe {
            cl_ctx
                .queue
                .enqueue_read_buffer(&mut buffer.buf, CL_BLOCKING, 0, &mut result, &[])?;
        }
    }

    let elapsed = start_time.elapsed();
    // Verify data
    let mut errors = 0;
    for (i, &value) in result.iter().enumerate() {
        if value != i as f32 {
            errors += 1;
            if errors <= 5 {
                println!("Error at [{}]: expected {}, got {}", i, i as f32, value);
            }
        }
    }

    if errors == 0 {
        println!("✓ Verification successful!");
    } else {
        println!("✗ {} verification errors found", errors);
    }

    println!("Time taken: {:.2} ms", elapsed.as_millis());

    Ok(())
}

fn run_memset(
    size: String,
    chunk_size: String,
    use_gpu: bool,
    use_host: bool,
    use_mmap: bool,
    device_id: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let size_bytes = parse_size(&size)?;
    let chunk_bytes = parse_size(&chunk_size)?;

    if !use_gpu && !use_host {
        return Err("Must specify either --gpu or --host for memset operation".into());
    }

    println!(
        "Running memset test with {} bytes, chunk size: {} bytes",
        size_bytes, chunk_bytes
    );
    println!(
        "Target: {}, MMap: {}",
        if use_gpu { "GPU" } else { "Host" },
        use_mmap
    );

    let start_time = Instant::now();
    let memset_elapsed = if use_host {
        // Host memory memset
        let mut host_buffer = vec![0u8; size_bytes];

        // Preload memory for better performance (don't count this time)
        preload_memory(host_buffer.as_mut_ptr(), size_bytes)?;

        let memset_start = Instant::now();
        // Perform memset in chunks
        chunked_memset(host_buffer.as_mut_ptr(), 0x42, size_bytes, chunk_bytes);
        memset_start.elapsed()
    } else {
        // GPU memory memset
        let cl_ctx = OpenCLContext::new(device_id)?;

        let mut buffer = cl_ctx.alloc_gpu_memory::<u8>(size_bytes, use_mmap)?;

        if use_mmap {
            let memset_start = Instant::now();
            // Perform memset in chunks on mapped memory
            chunked_memset(buffer.mapped_ptr as *mut u8, 0x42, size_bytes, chunk_bytes);
            cl_ctx.queue.finish()?;
            memset_start.elapsed()
        } else {
            // Memset using write buffer operations
            let pattern_data = vec![0x42u8; chunk_bytes];
            let memset_start = Instant::now();
            let mut offset = 0;

            while offset < size_bytes {
                let current_chunk = std::cmp::min(chunk_bytes, size_bytes - offset);
                let chunk_data = &pattern_data[..current_chunk];

                unsafe {
                    cl_ctx.queue.enqueue_write_buffer(
                        &mut buffer.buf,
                        CL_BLOCKING,
                        offset,
                        chunk_data,
                        &[],
                    )?;
                }
                offset += current_chunk;
            }
            cl_ctx.queue.finish()?;
            memset_start.elapsed()
        }
    };

    let throughput_mb_s = (size_bytes as f64 / (1024.0 * 1024.0)) / memset_elapsed.as_secs_f64();
    println!("memset completed:");
    println!("  Time: {:.2} ms", memset_elapsed.as_millis());
    println!("  Throughput: {:.2} MB/s", throughput_mb_s);

    let total_elapsed = start_time.elapsed();
    println!("Total time: {:.2} ms", total_elapsed.as_millis());

    Ok(())
}

fn run_memcpy(
    size: String,
    chunk_size: String,
    gpu_to_gpu: bool,
    host_to_host: bool,
    gpu_to_host: bool,
    host_to_gpu: bool,
    use_mmap: bool,
    device_id: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let size_bytes = parse_size(&size)?;
    let chunk_bytes = parse_size(&chunk_size)?;

    if ![gpu_to_gpu, host_to_host, gpu_to_host, host_to_gpu]
        .iter()
        .any(|&x| x)
    {
        return Err("Must specify one copy direction: --gpu-to-gpu, --host-to-host, --gpu-to-host, or --host-to-gpu".into());
    }

    let copy_type = if gpu_to_gpu {
        "GPU->GPU"
    } else if host_to_host {
        "Host->Host"
    } else if gpu_to_host {
        "GPU->Host"
    } else {
        "Host->GPU"
    };

    println!(
        "Running memcpy test: {} with {} bytes, chunk size: {} bytes",
        copy_type, size_bytes, chunk_bytes
    );
    println!("MMap: {}", use_mmap);

    let start_time = Instant::now();
    let mut memcpy_elapsed = start_time.elapsed();

    if host_to_host {
        // Host to Host memcpy
        let mut src_buffer = vec![0x42u8; size_bytes];
        let mut dst_buffer = vec![0u8; size_bytes];

        // Initialize source with pattern
        for (i, byte) in src_buffer.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }

        // Preload memory for not accounting page fault
        preload_memory(dst_buffer.as_mut_ptr(), size_bytes)?;

        let memcpy_start = Instant::now();
        // Perform memcpy in chunks
        chunked_memcpy(
            src_buffer.as_ptr(),
            dst_buffer.as_mut_ptr(),
            size_bytes,
            chunk_bytes,
        );
        memcpy_elapsed = memcpy_start.elapsed();
    } else {
        // GPU operations
        let cl_ctx = OpenCLContext::new(device_id)?;

        if gpu_to_gpu {
            // GPU to GPU memcpy
            let mut src_buffer = cl_ctx.alloc_gpu_memory::<u8>(size_bytes, use_mmap)?;
            let mut dst_buffer = cl_ctx.alloc_gpu_memory::<u8>(size_bytes, use_mmap)?;

            let memcpy_start = Instant::now();

            // Copy in chunks
            let mut offset = 0;
            while offset < size_bytes {
                let current_chunk = std::cmp::min(chunk_bytes, size_bytes - offset);
                unsafe {
                    if use_mmap {
                        std::ptr::copy_nonoverlapping(
                            src_buffer.mapped_ptr.add(offset),
                            dst_buffer.mapped_ptr.add(offset),
                            current_chunk,
                        );
                    } else {
                        cl_ctx.queue.enqueue_copy_buffer(
                            &mut src_buffer.buf,
                            &mut dst_buffer.buf,
                            offset,
                            offset,
                            current_chunk,
                            &[],
                        )?;
                    }
                }
                offset += current_chunk;
            }
            cl_ctx.queue.finish()?;
            memcpy_elapsed = memcpy_start.elapsed();
        } else if host_to_gpu {
            // Host to GPU memcpy
            let mut host_buffer = vec![0u8; size_bytes];
            for (i, byte) in host_buffer.iter_mut().enumerate() {
                *byte = (i % 256) as u8;
            }

            let mut gpu_buffer = cl_ctx.alloc_gpu_memory::<u8>(size_bytes, use_mmap)?;

            // Using write buffer
            let memcpy_start = Instant::now();
            let mut offset = 0;
            while offset < size_bytes {
                let current_chunk = std::cmp::min(chunk_bytes, size_bytes - offset);
                let chunk_data = &host_buffer[offset..offset + current_chunk];
                unsafe {
                    if use_mmap {
                        std::ptr::copy_nonoverlapping(
                            chunk_data.as_ptr(),
                            (gpu_buffer.mapped_ptr as *mut u8).add(offset),
                            current_chunk,
                        );
                    } else {
                        cl_ctx.queue.enqueue_write_buffer(
                            &mut gpu_buffer.buf,
                            CL_BLOCKING,
                            offset,
                            chunk_data,
                            &[],
                        )?;
                    }
                }
                offset += current_chunk;
            }
            cl_ctx.queue.finish()?;
            memcpy_elapsed = memcpy_start.elapsed();
        } else if gpu_to_host {
            let mut gpu_buffer = cl_ctx.alloc_gpu_memory::<u8>(size_bytes, use_mmap)?;

            let mut host_buffer = vec![0u8; size_bytes];
            // Preload host memory (don't count this time)
            preload_memory(host_buffer.as_mut_ptr(), size_bytes)?;

            // Using read buffer
            let memcpy_start = Instant::now();
            let mut offset = 0;
            while offset < size_bytes {
                let current_chunk = std::cmp::min(chunk_bytes, size_bytes - offset);
                let chunk_slice = &mut host_buffer[offset..offset + current_chunk];
                unsafe {
                    if use_mmap {
                        std::ptr::copy_nonoverlapping(
                            (gpu_buffer.mapped_ptr as *const u8).add(offset),
                            chunk_slice.as_mut_ptr(),
                            current_chunk,
                        );
                    } else {
                        cl_ctx.queue.enqueue_read_buffer(
                            &mut gpu_buffer.buf,
                            CL_BLOCKING,
                            offset,
                            chunk_slice,
                            &[],
                        )?;
                    }
                }
                offset += current_chunk;
            }
            cl_ctx.queue.finish()?;
            memcpy_elapsed = memcpy_start.elapsed();
        }
    }

    let throughput_mb_s = (size_bytes as f64 / (1024.0 * 1024.0)) / memcpy_elapsed.as_secs_f64();
    println!("{} memcpy completed:", copy_type);
    println!("  Time: {:.2} ms", memcpy_elapsed.as_millis());
    println!("  Throughput: {:.2} MB/s", throughput_mb_s);

    let total_elapsed = start_time.elapsed();
    println!("Total time: {:.2} ms", total_elapsed.as_millis());

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Verify {
            size,
            mmap,
            device_id,
        } => run_verify(size.clone(), *mmap, *device_id),
        Commands::Memset {
            size,
            chunk_size,
            gpu,
            host,
            mmap,
            device_id,
        } => run_memset(
            size.clone(),
            chunk_size.clone(),
            *gpu,
            *host,
            *mmap,
            *device_id,
        ),
        Commands::Memcpy {
            size,
            chunk_size,
            gpu_to_gpu,
            host_to_host,
            gpu_to_host,
            host_to_gpu,
            mmap,
            device_id,
        } => run_memcpy(
            size.clone(),
            chunk_size.clone(),
            *gpu_to_gpu,
            *host_to_host,
            *gpu_to_host,
            *host_to_gpu,
            *mmap,
            *device_id,
        ),
    }
}

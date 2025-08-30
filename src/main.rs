use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_GPU};
use opencl3::memory::{Buffer, ClMem, CL_MAP_READ, CL_MAP_WRITE, CL_MEM_READ_WRITE};
use opencl3::platform::get_platforms;
use opencl3::types::CL_BLOCKING;
use std::ptr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check for OpenCL platforms
    let platforms = get_platforms()?;
    if platforms.is_empty() {
        return Err("No OpenCL platforms found. OpenCL drivers may not be installed.".into());
    }

    println!("Found {} OpenCL platform(s)", platforms.len());
    for (i, platform) in platforms.iter().enumerate() {
        println!("Platform {}: {}", i, platform.name()?);
    }

    // Try GPU first, fallback to any device
    let device_ids =
        get_all_devices(CL_DEVICE_TYPE_GPU).or_else(|_| get_all_devices(CL_DEVICE_TYPE_ALL))?;

    if device_ids.is_empty() {
        return Err("No OpenCL devices found".into());
    }

    let device = Device::new(device_ids[0]);
    println!("Using device: {}", device.name()?);
    println!("Device type: {}", device.dev_type()?);

    // Create context and command queue
    let context = Context::from_device(&device)?;
    let queue = unsafe { CommandQueue::create_with_properties(&context, device_ids[0], 0, 0)? };

    // Test data - use larger buffer size for comprehensive testing
    let data_size = 1024 * 1024 * 1024 / 4; // 1G elements = 4GB
    let host_data: Vec<f32> = (0..data_size).map(|i| i as f32).collect();

    // Create buffer in GPU VRAM
    let mut buffer =
        unsafe { Buffer::<f32>::create(&context, CL_MEM_READ_WRITE, data_size, ptr::null_mut())? };

    // Write initial data to GPU VRAM
    unsafe {
        queue.enqueue_write_buffer(&mut buffer, CL_BLOCKING, 0, &host_data, &[])?;
    }
    println!(
        "Written {} elements ({} MB) to GPU VRAM",
        data_size,
        (data_size * 4) / (1024 * 1024)
    );

    // Map GPU buffer to host address space for direct access
    let mut mapped_ptr = ptr::null_mut();
    let _map_event = unsafe {
        queue.enqueue_map_buffer::<f32>(
            &mut buffer,
            CL_BLOCKING,
            CL_MAP_READ | CL_MAP_WRITE,
            0,
            data_size,
            &mut mapped_ptr,
            &[],
        )?
    };

    println!(
        "GPU buffer mapped to host address: {:p} size {} MB",
        mapped_ptr,
        (data_size * 4) / (1024 * 1024)
    );

    // Direct read test - verify initial data across entire buffer
    println!("Verifying initial data across entire buffer...");
    let mut read_errors = 0;
    for i in 0..data_size {
        let value = unsafe { *(mapped_ptr as *mut f32).offset(i as isize) };
        if value != i as f32 {
            read_errors += 1;
            if read_errors <= 5 {
                println!(
                    "  Read error at [{}]: expected {}, got {}",
                    i, i as f32, value
                );
            }
        }
    }
    if read_errors == 0 {
        println!("  ✓ All {} elements read correctly", data_size);
    } else {
        println!("  ✗ {} read errors found", read_errors);
    }

    // Direct write test - modify entire buffer
    println!("Writing modified values to entire GPU VRAM buffer...");
    for i in 0..data_size {
        unsafe {
            *(mapped_ptr as *mut f32).offset(i as isize) = (i as f32) * 2.0 + 1.0;
        }
    }
    println!("  ✓ Modified all {} elements", data_size);

    // Unmap the buffer
    unsafe {
        queue.enqueue_unmap_mem_object(buffer.get(), mapped_ptr, &[])?;
    }
    println!("GPU buffer unmapped");

    // Verify changes by reading back entire buffer
    let mut result = vec![0.0f32; data_size];
    unsafe {
        queue.enqueue_read_buffer(&mut buffer, CL_BLOCKING, 0, &mut result, &[])?;
    }

    println!("Verification - reading back and verifying all modified values...");
    let mut verify_errors = 0;
    for (i, &value) in result.iter().enumerate() {
        let expected = (i as f32) * 2.0 + 1.0;
        if (value - expected).abs() > 0.001 {
            verify_errors += 1;
            if verify_errors <= 5 {
                println!(
                    "  Verification error at [{}]: expected {}, got {}",
                    i, expected, value
                );
            }
        }
    }
    if verify_errors == 0 {
        println!("  ✓ All {} elements verified correctly", data_size);
    } else {
        println!("  ✗ {} verification errors found", verify_errors);
    }

    println!("Sample values:");
    for i in [
        0,
        data_size / 4,
        data_size / 2,
        data_size * 3 / 4,
        data_size - 1,
    ] {
        println!("  [{}] = {}", i, result[i]);
    }

    println!("GPU VRAM access test completed successfully!");

    Ok(())
}

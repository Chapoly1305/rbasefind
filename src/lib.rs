extern crate byteorder;
extern crate clap;
extern crate fnv;
extern crate num_cpus;
extern crate regex;
extern crate pbr;
extern crate ocl;

use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use clap::App;
use fnv::FnvHashSet;
use regex::bytes::Regex;
use std::collections::BinaryHeap;
use std::error::Error;
use std::fs::File;
use std::io::Cursor;
use std::io::prelude::*;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use pbr::MultiBar;
use ocl::{Context, Queue, Device, Program, Buffer, MemFlags, Kernel, SpatialDims, Platform};

pub struct Config {
    big_endian: bool,
    filename: String,
    min_str_len: usize,
    max_matches: usize,
    offset: u32,
    threads: usize,
    opencl: bool,
}

impl Config {
    pub fn new() -> Result<Config, &'static str> {
        let arg_matches = App::new("rbasefind")
            .version("0.1.2")
            .author("Scott G. <github.scott@gmail.com>")
            .about(
                "Scan a flat 32-bit binary and attempt to brute-force the base address via \
                 string/pointer comparison. Based on the excellent basefind.py by mncoppola.",
            )
            .args_from_usage(
                "<INPUT>                'The input binary to scan'
                -b, --bigendian         'Interpret as Big Endian (default is little)'
                -m, --minstrlen=[LEN]   'Minimum string search length (default is 10)'
                -n, --maxmatches=[LEN]   'Maximum matches to display (default is 10)'
                -o, --offset=[LEN]      'Scan every N (power of 2) addresses. (default is 0x1000)'
                -t  --threads=[NUM_THREADS] '# of threads to spawn. (default is # of cpu cores)'
                -c  --opencl                'Use OpenCL for the search'",
            )
            .get_matches();

        let config = Config {
            big_endian: arg_matches.is_present("bigendian"),
            filename: arg_matches.value_of("INPUT").unwrap().to_string(),
            max_matches: match arg_matches.value_of("maxmatches").unwrap_or("10").parse() {
                Ok(v) => v,
                Err(_) => return Err("failed to parse maxmatches"),
            },
            min_str_len: match arg_matches.value_of("minstrlen").unwrap_or("10").parse() {
                Ok(v) => v,
                Err(_) => return Err("failed to parse minstrlen"),
            },
            offset: {
                let offset_str = &arg_matches.value_of("offset").unwrap_or("0x1000");
                if offset_str.len() <= 2 {
                    return Err("offset format is invalid");
                }
                if &offset_str[0..2] != "0x" {
                    return Err("ensure offset parameter begins with 0x.");
                }
                let offset_num = match u32::from_str_radix(&offset_str[2..], 16) {
                    Ok(v) => v,
                    Err(_) => return Err("failed to parse offset"),
                };
                // This check also prevents offset_num from being zero.
                if offset_num.count_ones() != 1 {
                    return Err("Offset is not a power of 2");
                }
                offset_num
            },
            threads: match arg_matches.value_of("threads").unwrap_or("0").parse() {
                Ok(v) => if v == 0 {
                    num_cpus::get()
                } else {
                    v
                },
                Err(_) => return Err("failed to parse threads"),
            },
            opencl: arg_matches.is_present("opencl"),
        };

        Ok(config)
    }
}

pub struct Interval {
    start_addr: u32,
    end_addr: u32,
}

impl Interval {
    fn get_range(
        index: usize,
        max_threads: usize,
        offset: u32,
    ) -> Result<Interval, Box<dyn Error + Send + Sync>> {
        if index >= max_threads {
            return Err("Invalid index specified".into());
        }

        if offset.count_ones() != 1 {
            return Err("Invalid additive offset".into());
        }

        let mut start_addr = index as u64
            * ((u64::from(u32::max_value()) + max_threads as u64 - 1) / max_threads as u64);
        let mut end_addr = (index as u64 + 1)
            * ((u64::from(u32::max_value()) + max_threads as u64 - 1) / max_threads as u64);

        // Mask the address such that it's aligned to the 2^N offset.
        start_addr &= !(u64::from(offset) - 1);
        if end_addr >= u64::from(u32::max_value()) {
            end_addr = u64::from(u32::max_value());
        } else {
            end_addr &= !(u64::from(offset) - 1);
        }

        let interval = Interval {
            start_addr: start_addr as u32,
            end_addr: end_addr as u32,
        };

        Ok(interval)
    }
}

fn get_strings(config: &Config, buffer: &[u8]) -> Result<Vec<u32>, Box<dyn Error>> {
    let mut strings = Vec::<u32>::new();

    let reg_str = format!("[ -~\\t\\r\\n]{{{},}}\x00", config.min_str_len);
    for mat in Regex::new(&reg_str)?.find_iter(&buffer[..]) {
        strings.push(mat.start() as u32);
    }

    Ok(strings)
}

fn get_pointers(config: &Config, buffer: &[u8]) -> Result<FnvHashSet<u32>, Box<dyn Error>> {
    let mut pointers = FnvHashSet::default();
    let mut rdr = Cursor::new(&buffer);
    loop {
        let res = if config.big_endian {
            rdr.read_u32::<BigEndian>()
        } else {
            rdr.read_u32::<LittleEndian>()
        };
        match res {
            Ok(v) => pointers.insert(v),
            Err(_) => break,
        };
    }

    Ok(pointers)
}

fn find_matches(
    config: &Config,
    strings: &Vec<u32>,
    pointers: &FnvHashSet<u32>,
    scan_interval: usize,
    mut pb: pbr::ProgressBar<pbr::Pipe>,
) -> Result<BinaryHeap<(usize, u32)>, Box<dyn Error + Send + Sync>> {
    let interval = Interval::get_range(scan_interval, config.threads, config.offset)?;
    let mut current_addr = interval.start_addr;
    let mut heap = BinaryHeap::<(usize, u32)>::new();
    pb.total = ((interval.end_addr - interval.start_addr)/config.offset) as u64;
    while current_addr <= interval.end_addr {
        let mut news = FnvHashSet::default();
        for s in strings {
            match s.checked_add(current_addr) {
                Some(add) => news.insert(add),
                // strings are now ordered, if it overflows u32 we can just break instead of continuing,
                // it will overflow u32 also for the strings after.
                None => break,
            };
        }
        let intersection: FnvHashSet<_> = news.intersection(pointers).collect();
        if !intersection.is_empty() {
            heap.push((intersection.len(), current_addr));
        }
        match current_addr.checked_add(config.offset) {
            Some(_) => current_addr += config.offset,
            None => break,
        };
        pb.inc();
    }
    pb.finish();

    Ok(heap)
}

fn cpu_search(config: &Arc<Config>, strings: &Arc<Vec<u32>>, pointers: &Arc<FnvHashSet<u32>>) -> BinaryHeap::<(usize, u32)> {
    let mut children = vec![];

    let mb = MultiBar::new();
    mb.println(&format!("Scanning with {} threads...", config.threads));
    for i in 0..config.threads {
        let mut pb = mb.create_bar(100);
        pb.show_message = true;
        let child_config = Arc::clone(&config);
        let child_strings = Arc::clone(&strings);
        let child_pointers = Arc::clone(&pointers);
        children.push(thread::spawn(move || {
            find_matches(&child_config, &child_strings, &child_pointers, i, pb)
        }));
    }

    mb.listen();

    // Merge all of the heaps.
    let mut heap = BinaryHeap::<(usize, u32)>::new();
    for child in children {
        heap.append(&mut child.join().unwrap().unwrap());
    }

    heap
}

fn opencl_search(
    config: &Arc<Config>,
    strings: &[u32],
    pointers: &[u32],
) -> Result<BinaryHeap<(usize, u32)>, Box<dyn Error>> {
    const OPENCL_CHUNK_SIZE: usize = 0x100000;
    let profile_opencl = std::env::var("RBASEFIND_OPENCL_PROFILE").ok().as_deref() == Some("1");
    let opencl_total_start = Instant::now();
    let mut opencl_setup_time = Duration::new(0, 0);
    let mut opencl_kernel_time = Duration::new(0, 0);
    let mut opencl_read_time = Duration::new(0, 0);
    let mut opencl_heap_time = Duration::new(0, 0);
    let mut opencl_chunks = 0u64;
    let strings_in_count = strings.len();
    let pointers_in_count = pointers.len();
    let setup_start = Instant::now();

    let mut sorted_strings = strings.to_vec();
    sorted_strings.sort_unstable();
    sorted_strings.dedup();
    if sorted_strings.is_empty() {
        return Ok(BinaryHeap::<(usize, u32)>::new());
    }

    let mut filtered_pointers: Vec<u32> = if config.offset == 1 {
        pointers.to_vec()
    } else {
        let mask = config.offset - 1;
        let mut string_low_bits = FnvHashSet::default();
        for s in &sorted_strings {
            string_low_bits.insert(*s & mask);
        }
        pointers
            .iter()
            .cloned()
            .filter(|p| string_low_bits.contains(&(p & mask)))
            .collect()
    };
    filtered_pointers.sort_unstable();
    filtered_pointers.dedup();
    if filtered_pointers.is_empty() {
        if profile_opencl {
            eprintln!(
                "OpenCL profile: total={:?}, setup={:?}, kernel={:?}, readback={:?}, heap={:?}, chunks=0, candidates=0, strings_in={}, strings_unique={}, pointers_in={}, pointers_filtered=0",
                opencl_total_start.elapsed(),
                setup_start.elapsed(),
                Duration::new(0, 0),
                Duration::new(0, 0),
                Duration::new(0, 0),
                strings_in_count,
                sorted_strings.len(),
                pointers_in_count
            );
        }
        return Ok(BinaryHeap::<(usize, u32)>::new());
    }

    let ptr_min = *filtered_pointers.first().unwrap();
    let ptr_max = *filtered_pointers.last().unwrap();
    let compute_program = r#"
        __kernel void find(__global const uint* strings,
        uint str_count, 
        __global const uint* pointers,
        uint ptr_count,
        uint ptr_min,
        uint ptr_max,
        uint offset,
        uint base_start,
        uint candidate_count,
        __global uint* results) {
            uint gid = get_global_id(0);
            if (gid >= candidate_count) {
                return;
            }
            ulong current_addr = ((ulong)base_start) + (((ulong)gid) * ((ulong)offset));
            if (current_addr > 0xffffffffUL) {
                results[gid] = 0;
                return;
            }
            uint intersect_count = 0;
            for (uint i=0; i<str_count; i++) {
                ulong translated_string = ((ulong)strings[i]) + current_addr;
                if (translated_string > 0xffffffffUL) {
                    break;
                }
                if (translated_string < (ulong)ptr_min) {
                    continue;
                }
                if (translated_string > (ulong)ptr_max) {
                    break;
                }
                uint target = (uint)translated_string;
                uint lo = 0;
                uint hi = ptr_count;
                while (lo < hi) {
                    uint mid = lo + ((hi - lo) >> 1);
                    uint mid_val = pointers[mid];
                    if (mid_val < target) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                if (lo < ptr_count && pointers[lo] == target) {
                    intersect_count += 1;
                }
            }
            results[gid] = intersect_count;
        }
    "#;
    if sorted_strings.len() > u32::max_value() as usize || filtered_pointers.len() > u32::max_value() as usize {
        return Err("OpenCL path does not support > u32::MAX strings/pointers".into());
    }
    let platform = Platform::default();
    let device = match Device::list(platform, Some(ocl::flags::DEVICE_TYPE_GPU))? {
        gpu_devices if !gpu_devices.is_empty() => gpu_devices[0],
        _ => {
            let all_devices = Device::list(platform, None)?;
            if all_devices.is_empty() {
                return Err("no OpenCL devices found".into());
            }
            all_devices[0]
        }
    };

    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;
    let queue = Queue::new(&context, device, None)?;
    let program = Program::builder()
        .src(compute_program)
        .devices(device)
        .build(&context)
        ?;

    let string_buffer = Buffer::<u32>::builder()
        .queue(queue.clone())
        .flags(MemFlags::new().read_only())
        .len(sorted_strings.len())
        .copy_host_slice(&sorted_strings)
        .build()?;

    let pointer_buffer = Buffer::<u32>::builder()
        .queue(queue.clone())
        .flags(MemFlags::new().read_only())
        .len(filtered_pointers.len())
        .copy_host_slice(&filtered_pointers)
        .build()?;
    opencl_setup_time += setup_start.elapsed();

    let result_buffer = Buffer::<u32>::builder()
        .queue(queue.clone())
        .flags(MemFlags::new().write_only())
        .len(OPENCL_CHUNK_SIZE)
        .build()?;

    let kernel = Kernel::builder()
        .program(&program)
        .name("find")
        .queue(queue.clone())
        .global_work_size(SpatialDims::One(OPENCL_CHUNK_SIZE))
        .arg_named("strings", &string_buffer)
        .arg_named("str_count", string_buffer.len() as u32)
        .arg_named("pointers", &pointer_buffer)
        .arg_named("ptr_count", pointer_buffer.len() as u32)
        .arg_named("ptr_min", ptr_min)
        .arg_named("ptr_max", ptr_max)
        .arg_named("offset", config.offset)
        .arg_named("base_start", 0u32)
        .arg_named("candidate_count", 0u32)
        .arg_named("results", &result_buffer)
        .build()?;

    let mut vec_result = vec![0u32; OPENCL_CHUNK_SIZE];

    queue.finish()?;
    let mut heap = BinaryHeap::<(usize, u32)>::new();
    let total_candidates = (u64::from(u32::max_value()) / u64::from(config.offset)) + 1;
    let mut processed_candidates: u64 = 0;
    while processed_candidates < total_candidates {
        opencl_chunks += 1;
        let remaining = total_candidates - processed_candidates;
        let this_chunk = std::cmp::min(OPENCL_CHUNK_SIZE as u64, remaining) as usize;
        let base_start =
            (processed_candidates * u64::from(config.offset)) as u32;

        kernel.set_arg("base_start", base_start)?;
        kernel.set_arg("candidate_count", this_chunk as u32)?;
        if profile_opencl {
            let kernel_start = Instant::now();
            unsafe { kernel.enq()?; }
            queue.finish()?;
            opencl_kernel_time += kernel_start.elapsed();

            let read_start = Instant::now();
            result_buffer.read(&mut vec_result[..this_chunk]).enq()?;
            queue.finish()?;
            opencl_read_time += read_start.elapsed();
        } else {
            unsafe { kernel.enq()?; }
            result_buffer.read(&mut vec_result[..this_chunk]).enq()?;
        }

        let heap_start = Instant::now();
        for (idx, count) in vec_result[..this_chunk].iter().enumerate() {
            if *count > 0 {
                let addr = base_start.wrapping_add((idx as u32).wrapping_mul(config.offset));
                heap.push((*count as usize, addr));
            }
        }
        if profile_opencl {
            opencl_heap_time += heap_start.elapsed();
        }

        processed_candidates += this_chunk as u64;
    }
    queue.finish()?;
    if profile_opencl {
        let opencl_total_time = opencl_total_start.elapsed();
        eprintln!(
            "OpenCL profile: total={:?}, setup={:?}, kernel={:?}, readback={:?}, heap={:?}, chunks={}, candidates={}, strings_in={}, strings_unique={}, pointers_in={}, pointers_filtered={}",
            opencl_total_time,
            opencl_setup_time,
            opencl_kernel_time,
            opencl_read_time,
            opencl_heap_time,
            opencl_chunks,
            total_candidates,
            strings_in_count,
            sorted_strings.len(),
            pointers_in_count,
            filtered_pointers.len()
        );
    }
    Ok(heap)
}

pub fn run(config: Config) -> Result<(), Box<dyn Error>> {
    // Read in the input file. We jam it all into memory for now.
    let mut f = File::open(&config.filename)?;
    let mut buffer = Vec::new();
    f.read_to_end(&mut buffer)?;

    // Find indices of strings.
    let strings = get_strings(&config, &buffer)?;

    if strings.is_empty() {
        return Err("No strings found in target binary".into());
    }
    eprintln!("Located {} strings", strings.len());

    let pointers = get_pointers(&config, &buffer)?;
    eprintln!("Located {} pointers", pointers.len());

    let shared_config = Arc::new(config);
    let shared_strings = Arc::new(strings);
    let shared_pointers = Arc::new(pointers);

    let mut heap = if shared_config.opencl{
        let pointers_vec: Vec<u32> = shared_pointers.iter().cloned().collect();
        match opencl_search(&shared_config, &shared_strings, &pointers_vec) {
            Ok(opencl_heap) => opencl_heap,
            Err(err) => {
                eprintln!("OpenCL search failed: {}", err);
                eprintln!("Falling back to CPU search.");
                cpu_search(&shared_config, &shared_strings, &shared_pointers)
            }
        }
    } else {
        cpu_search(&shared_config, &shared_strings, &shared_pointers)
    };

    // Print (up to) top N results.
    for _ in 0..shared_config.max_matches {
        let (count, addr) = match heap.pop() {
            Some(v) => v,
            None => break,
        };
        println!("0x{:08x}: {}", addr, count);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::time::Instant;

    fn heap_to_map(mut heap: BinaryHeap<(usize, u32)>) -> HashMap<u32, usize> {
        let mut out = HashMap::new();
        while let Some((count, addr)) = heap.pop() {
            out.insert(addr, count);
        }
        out
    }

    fn cpu_reference_search(
        config: &Config,
        strings: &[u32],
        pointers: &FnvHashSet<u32>,
    ) -> BinaryHeap<(usize, u32)> {
        let mut heap = BinaryHeap::<(usize, u32)>::new();
        let mut current_addr: u32 = 0;
        loop {
            let mut count = 0usize;
            for s in strings {
                match s.checked_add(current_addr) {
                    Some(add) => {
                        if pointers.contains(&add) {
                            count += 1;
                        }
                    }
                    None => break,
                }
            }
            if count > 0 {
                heap.push((count, current_addr));
            }

            match current_addr.checked_add(config.offset) {
                Some(next) => current_addr = next,
                None => break,
            }
        }
        heap
    }

    #[test]
    #[should_panic]
    fn find_matches_invalid_interval() {
        let _ = Interval::get_range(1, 1, 0x1000).unwrap();
    }

    #[test]
    fn find_matches_single_cpu_interval_0() {
        let interval = Interval::get_range(0, 1, 0x1000).unwrap();
        assert_eq!(interval.start_addr, u32::min_value());
        assert_eq!(interval.end_addr, u32::max_value());
    }

    #[test]
    fn find_matches_double_cpu_interval_0() {
        let interval = Interval::get_range(0, 2, 0x1000).unwrap();
        assert_eq!(interval.start_addr, u32::min_value());
        assert_eq!(interval.end_addr, 0x80000000);
    }

    #[test]
    fn find_matches_double_cpu_interval_1() {
        let interval = Interval::get_range(1, 2, 0x1000).unwrap();
        assert_eq!(interval.start_addr, 0x80000000);
        assert_eq!(interval.end_addr, u32::max_value());
    }

    #[test]
    fn find_matches_triple_cpu_interval_0() {
        let interval = Interval::get_range(0, 3, 0x1000).unwrap();
        assert_eq!(interval.start_addr, u32::min_value());
        assert_eq!(interval.end_addr, 0x55555000);
    }

    #[test]
    fn find_matches_triple_cpu_interval_1() {
        let interval = Interval::get_range(1, 3, 0x1000).unwrap();
        assert_eq!(interval.start_addr, 0x55555000);
        assert_eq!(interval.end_addr, 0xAAAAA000);
    }

    #[test]
    fn find_matches_triple_cpu_interval_2() {
        let interval = Interval::get_range(2, 3, 0x1000).unwrap();
        assert_eq!(interval.start_addr, 0xAAAAA000);
        assert_eq!(interval.end_addr, u32::max_value());
    }

    #[test]
    fn opencl_matches_cpu_results() {
        let config = Arc::new(Config {
            big_endian: false,
            filename: String::new(),
            min_str_len: 0,
            max_matches: 10,
            offset: 0x80000000,
            threads: 2,
            opencl: true,
        });

        let strings = Arc::new(vec![0x1000u32, 0x2000u32, 0x3000u32]);
        let pointers_vec = vec![
            0x00001000u32,
            0x00002000u32,
            0x80001000u32,
            0x11111111u32,
        ];
        let mut pointers_set = FnvHashSet::default();
        for p in &pointers_vec {
            pointers_set.insert(*p);
        }
        let pointers_set = Arc::new(pointers_set);

        let cpu = cpu_reference_search(&config, &strings, &pointers_set);
        let opencl = match opencl_search(&config, &strings, &pointers_vec) {
            Ok(heap) => heap,
            Err(err) => {
                // Keep CI portable when OpenCL runtime/device is unavailable.
                eprintln!("Skipping OpenCL regression assertion: {}", err);
                return;
            }
        };

        assert_eq!(heap_to_map(cpu), heap_to_map(opencl));
    }

    #[test]
    fn benchmark_opencl_vs_cpu_reference() {
        let config = Arc::new(Config {
            big_endian: false,
            filename: String::new(),
            min_str_len: 0,
            max_matches: 10,
            offset: 0x00020000,
            threads: 4,
            opencl: true,
        });

        let strings: Vec<u32> = (0..4000).map(|i| (i as u32) * 0x20).collect();
        let mut pointers_seed: Vec<u32> = (0..400000)
            .map(|i| ((i as u32).wrapping_mul(0x1f123bb5)).rotate_left(7))
            .collect();
        // Force some deterministic hits for candidate base 0.
        for s in strings.iter().take(2000) {
            pointers_seed.push(*s);
        }

        let mut pointers_set = FnvHashSet::default();
        for p in &pointers_seed {
            pointers_set.insert(*p);
        }
        let pointers_vec: Vec<u32> = pointers_set.iter().cloned().collect();

        let cpu_start = Instant::now();
        let cpu = cpu_reference_search(&config, &strings, &pointers_set);
        let cpu_elapsed = cpu_start.elapsed();

        let opencl_start = Instant::now();
        let opencl = match opencl_search(&config, &strings, &pointers_vec) {
            Ok(heap) => heap,
            Err(err) => {
                eprintln!("OpenCL benchmark skipped: {}", err);
                return;
            }
        };
        let opencl_elapsed = opencl_start.elapsed();

        assert_eq!(heap_to_map(cpu), heap_to_map(opencl));
        eprintln!(
            "benchmark_opencl_vs_cpu_reference: cpu={:?}, opencl={:?}",
            cpu_elapsed, opencl_elapsed
        );
    }
}

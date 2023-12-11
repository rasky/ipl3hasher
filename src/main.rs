use emu_core::prelude::*;
use emu_glsl::*;
//use zerocopy::*;
use byteorder::{BigEndian, ByteOrder, LittleEndian, ReadBytesExt};
use rand::Rng;
use std::fs::File;
use std::io::prelude::*;
use std::io::Read;

use std::time::{Duration, Instant};

mod cpu;
use cpu::*;

use gumdrop::Options;

fn parse_hex_u32(s: &str) -> Result<u32, std::num::ParseIntError> {
    u32::from_str_radix(s, 16)
}

#[derive(Debug, Options)]
struct CSumOptions {
    #[options(help = "print help message")]
    help: bool,
    #[options(free, help = "The ROM to be modified")]
    source: String,
    #[options(
        short = "c",
        default = "6102",
        help = "The CIC for which a checksum must be calculated"
    )]
    cic: String,
    #[options(
        default = "0",
        help = "The GPU to use (0 for first, 1 for second, etc.)"
    )]
    device: i32,
    #[options(
        default = "400",
        help = "The number of threads to use",
        parse(try_from_str = "parse_hex_u32")
    )]
    threads: u32,
    #[options(
        default = "20000",
        help = "The number of groups to use",
        parse(try_from_str = "parse_hex_u32")
    )]
    groups: u32,
    #[options(default = "0", help = "The Y coordinate to start with")]
    init: u32,
    #[options(
        short = "v",
        default = "false",
        help = "Print each range of hashes as they're sent to the GPU"
    )]
    verbose: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opts = CSumOptions::parse_args_default_or_exit();    
    let seed: u8;
    let target_high: u32;
    let target_low: u32;
    (seed, target_high, target_low) = match opts.cic.as_str() {
        "6101" => (0x3f, 0x45cc, 0x73ee317a),
        "6102" | "7101" => (0x3f, 0xa536, 0xc0f1d859),
        "6103" | "7103" => (0x78, 0x586f, 0xd4709867),
        "6105" | "7105" => (0x91, 0x8618, 0xa45bc2d3),
        "6106" | "7106" => (0x85, 0x2bba, 0xd4e6eb74),
        "8303" => (0xdd, 0x32b2, 0x94e2ab90),
        "8401" => (0xdd, 0x6ee8, 0xd9e84970),
        "5167" => (0xdd, 0x083c, 0x6c77e0b1),
        "DDUS" => (0xde, 0x05ba, 0x2ef0a5f1),
        _ => panic!("Unknown CIC"),
    };

    println!("Target seed and checksum: {:#02X} {:#06X} {:#08X}", seed, target_high, target_low);

    let mut pre_csum: ChecksumInfo<BigEndian>;
    let device = opts.device as usize;

    if let Ok(mut file) = File::open(opts.source) {
        let mut rom: [u8; 4096] = [0; 4096];
        if let Ok(_) = file.read_exact(&mut rom) {
            pre_csum = ChecksumInfo::new(seed as u32, rom);
            pre_csum.checksum(0, 1005);
        } else {
            panic!();
        }
    } else {
        panic!();
    }

    // ensure that a device pool has been initialized
    // this should be called before every time when you assume you have devices to use
    // that goes for both library users and application users
    futures::executor::block_on(assert_device_pool_initialized());
    select(|idx, _info| {
        idx == device
    })?;

    println!("{:?}", take()?.lock().unwrap().info.as_ref().unwrap());

    // create some data on GPU
    // even mutate it once loaded to GPU
    //let mut state: DeviceBox<[u32]> = vec![0; 16].as_device_boxed_mut()?;
    let mut res: DeviceBox<[u32]> = vec![0u32, 0u32, 0u32].as_device_boxed_mut()?;
    let mut x_off_src = 0u64;
    let mut y_off_src = opts.init as u64;
    let mut x_off: DeviceBox<u32> = 0u32.into_device_boxed_mut()?;
    let mut y_off: DeviceBox<u32> = 0u32.into_device_boxed_mut()?;

    // compile GslKernel to SPIR-V
    // then, we can either inspect the SPIR-V or finish the compilation by generating a DeviceFnMut
    // then, run the DeviceFnMut
    let kernel = GlslKernel::new()
    .spawn(opts.threads)
    .param::<[u32], _>("uint[16] state_in")
    .param_mut::<u32, _>("uint x_offset")
    .param_mut::<u32, _>("uint y_offset")
    .param_mut::<[u32], _>("uint[3] result")
    .with_const("uint magic", "0x95DACFDC")
    .with_const("uint target_hi", format!("{}", target_high))
    .with_const("uint target_lo", format!("{}", target_low))
    .with_const("uint seed", format!("{}", seed as u32))
.with_helper_code(r#"
#extension GL_ARB_gpu_shader_int64 : enable

uint csum(uint op1, uint op2, uint op3) {
    uint hi;
    uint lo;
    if (op2 == 0) {
        op2 = op3;
    }

    uvec2 parts = unpackUint2x32(uint64_t(op1) * uint64_t(op2));
    if (parts.y - parts.x == 0)
        return op1;
    return parts.y - parts.x;
}

uint[16] round_x(uint[16] state, uint data_x, uint data_y) {
    uint y_top5bits = data_y >> 27;
    uint y_bottom5bits = data_y & 0x1f;

    // This is actually the second half of round 1007
    //uint tmp1 = csum(state[15], (data_y >> (0x20 - (data_last >> 27))) | (data_y << (data_last >> 27)), 1007);
    //state[15] = csum(tmp1, (data_x << (dataY >> 27)) | (data_x >> (0x20 - (data_y >> 27))), 1007);
    state[15] = csum(state[15], (data_x << y_top5bits) | (data_x >> (0x20 - y_top5bits)), 1007);

    state[14] = csum(state[14], (data_x >> (y_bottom5bits)) | (data_x << (0x20 - (y_bottom5bits))), 1007);

    state[13] += ((data_x >> (data_x & 0x1f)) | (data_x << (0x20 - (data_x & 0x1f))));
    state[10] = csum(state[10], data_x, 1007);
    state[11] = csum(state[11], data_x, 1007);

    // And now round 1008

    state[0] += csum(uint(0x3EF - 1008), data_x, 1008);
    state[1] = csum(state[1], data_x, 1008);
    state[2] ^= data_x;
    state[3] += csum(data_x + 5, 0x6c078965, 1008);

    if (data_y < data_x) {
        state[9] = csum(state[9], data_x, 1008);
    }
    else {
        state[9] += data_x;
    }

    state[4] += ((data_x << (0x20 - y_bottom5bits)) | (data_x >> y_bottom5bits));
    //state[7] = csum(state[7], ((data_x >> (0x20 - (data_y & 0x1f))) | (data << (data_y & 0x1f))), 1008);
    state[7] = csum(state[7], ((data_x >> (0x20 - y_bottom5bits)) | (data_x << y_bottom5bits)), 1008);

    if (data_x < state[6]) {
        state[6] = (data_x + 1008) ^ (state[3] + state[6]);
    }
    else {
        state[6] = (state[4] + data_x) ^ state[6];
    }

    state[5] += (data_x >> (0x20 - y_top5bits)) | (data_x << y_top5bits);
    state[8] = csum(state[8], (data_x << (0x20 - y_top5bits)) | (data_x >> y_top5bits), 1008);

    return state;
}

uint finalize_hi(uint[16] state) {
    uint buf[2];

    for (int i = 0; i < 2; i++) {
        buf[i] = state[0];
    }

    for (uint i = 0; i < 16; i++) {
        uint data = state[i];
        uint shift = data & 0x1f;
        uint data_shifted_left = data << (32 - shift);
        uint data_shifted_right = data >> shift;
        uint tmp = buf[0] + (data_shifted_right | data_shifted_left);
        buf[0] = tmp;

        if (data < tmp) {
            buf[1] += data;
        }
        else {
            buf[1] = csum(buf[1], data, i);
        }
    }

    return csum(buf[0], buf[1], 16) & 0xFFFF;
}

uint finalize_lo(uint[16] state) {
    uint buf[2];

    for (int i = 0; i < 2; i++) {
        buf[i] = state[0];
    }

    for (uint i = 0; i < 16; i++) {
        uint data = state[i];

        uint tmp = (data & 0x02) >> 1;
        uint tmp2 = data & 0x01;

        if (tmp == tmp2) {
            buf[0] += data;
        }
        else {
            buf[0] = csum(buf[0], data, i);
        }

        if (tmp2 == 1) {
            buf[1] ^= data;
        }
        else {
            buf[1] = csum(buf[1], data, i);
        }
    }

    return buf[0] ^ buf[1];
}

"#)
.with_kernel_code(
r#"
    uint y = y_offset;
    uint x = x_offset + gl_GlobalInvocationID.x;

    uint state[16] = round_x(state_in, x, y);
    if (finalize_hi(state) == target_hi) {
        if (finalize_lo(state) == target_lo) {
            if (atomicOr(result[2], 1) == 0) {
                result[0] = x;
                result[1] = y;
            }
        }
    }
"#,
);
    let c = compile::<GlslKernel, GlslKernelCompile, Vec<u32>, GlobalCache>(kernel)?.finish()?;
    //return Ok(());
    let mut finished_src;
    let unlocked = std::io::stdout();
    let mut stdout = unlocked.lock();
    loop {
        x_off_src = 0;
        x_off.set(x_off_src as u32)?;
        let mut y_csum = pre_csum.clone();
        y_csum.rom[4088] = (y_off_src >> 24) as u8;
        y_csum.rom[4089] = (y_off_src >> 16) as u8;
        y_csum.rom[4090] = (y_off_src >> 8) as u8;
        y_csum.rom[4091] = (y_off_src >> 0) as u8;
        y_csum.checksum(1005, 1006);
        y_csum.round_y();
        let state_vec: Vec<u32> = y_csum.buffer.iter().cloned().collect();
        let state_in: DeviceBox<[u32]> = state_vec.as_device_boxed()?;
        let start = Instant::now();
        loop {
            let bump = (opts.threads as u64) * (opts.groups as u64);
            if opts.verbose {
                stdout.write_fmt(format_args!(
                    "should calc from ({}, {}) to ({}, {}) in {} threads on {} workgroups\n",
                    x_off_src,
                    y_off_src,
                    x_off_src + bump,
                    y_off_src,
                    opts.threads,
                    opts.groups
                ))?;
            }
            unsafe {
                spawn(opts.groups).launch(call!(
                    c.clone(),
                    &state_in,
                    &mut x_off,
                    &mut y_off,
                    &mut res
                ))?;
            }
            finished_src = futures::executor::block_on(res.get())?[2] != 0;
            if finished_src {
                break;
            }

            x_off_src += bump;

            if x_off_src >= std::u32::MAX as u64 {
                break;
            }

            x_off.set(x_off_src as u32)?;
        }
        let duration = start.elapsed();
        write!(stdout, "Inner loop Y=={} took {:?}\n", y_off_src, duration)?;
        stdout.flush()?;
        //return Ok(());

        if finished_src {
            break;
        }

        y_off_src += 1;

        if y_off_src >= std::u32::MAX as u64 {
            break;
        }

        y_off.set(y_off_src as u32)?;
    }

    // download from GPU and print out
    if finished_src {
        let res2 = futures::executor::block_on(res.get())?;
        println!("match found with code: {:08X?} {:08X?}", res2[1], res2[0]);
    } else {
        println!("sorry, no dice");
    }
    Ok(())
}

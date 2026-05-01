// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2026 Gyroflow contributors
//
// gyroflow-bench: terminal-based benchmark harness for the stabilization core.
//
// Usage:
//   gyroflow-bench run --name <NAME> --project <PATH> [--resolution WxH]
//                                                     [--frames N]
//                                                     [--pixel-format FMT]
//                                                     [--device IDX]
//   gyroflow-bench stop --name <NAME>
//   gyroflow-bench list
//   gyroflow-bench compare <NAME1> <NAME2> [NAME3...]

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use gyroflow_core::StabilizationManager;
use gyroflow_core::gpu;

static INTERRUPTED: AtomicBool = AtomicBool::new(false);

// ---------- Data model ----------

#[derive(Serialize, Deserialize, Debug, Clone)]
struct BenchmarkResult {
    name: String,
    timestamp: String,
    git_commit: String,
    backend: String,

    project_file: String,
    resolution: (usize, usize),
    pixel_format: String,
    device_index: Option<i32>,
    frames_requested: usize,
    frames_completed: usize,

    frame_times_us: Vec<u64>,
    total_time_ms: f64,
    warmup_time_ms: f64,
    mean_frame_us: f64,
    median_frame_us: f64,
    p95_frame_us: f64,
    p99_frame_us: f64,
    min_frame_us: u64,
    max_frame_us: u64,
    fps: f64,

    peak_rss_bytes: u64,

    machine: MachineInfo,
    completed: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct MachineInfo {
    hostname: String,
    os: String,
    arch: String,
    cpu_count: usize,
    gpu_devices: Vec<String>,
}

struct RunConfig {
    name: String,
    project: String,
    resolution: Option<(usize, usize)>,
    frames: Option<usize>,
    pixel_format: String,
    device_index: Option<i32>,
}

// ---------- Pixel format dispatch ----------

macro_rules! dispatch_process {
    ($fmt:expr, $mgr:expr, $ts:expr, $frame:expr, $buf:expr) => {
        match $fmt {
            "rgba8"   => $mgr.process_pixels::<gyroflow_core::stabilization::RGBA8>($ts, $frame, $buf),
            "rgba16"  => $mgr.process_pixels::<gyroflow_core::stabilization::RGBA16>($ts, $frame, $buf),
            "rgbaf16" => $mgr.process_pixels::<gyroflow_core::stabilization::RGBAf16>($ts, $frame, $buf),
            "rgbaf"   => $mgr.process_pixels::<gyroflow_core::stabilization::RGBAf>($ts, $frame, $buf),
            other => panic!("unsupported pixel format: {other}"),
        }
    };
}

fn bytes_per_pixel(format: &str) -> usize {
    match format {
        "rgba8" => 4,
        "rgba16" | "rgbaf16" => 8,
        "rgbaf" => 16,
        _ => 4,
    }
}

// ---------- main ----------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    let result: Result<(), String> = match args[1].as_str() {
        "run" => cmd_run(&args[2..]),
        "stop" => cmd_stop(&args[2..]),
        "list" => cmd_list(),
        "compare" => cmd_compare(&args[2..]),
        "-h" | "--help" | "help" => { print_usage(); Ok(()) }
        other => {
            eprintln!("Unknown command: {other}");
            print_usage();
            std::process::exit(1);
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn print_usage() {
    eprintln!("gyroflow-bench — Gyroflow stabilization benchmark harness");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  gyroflow-bench run --name <NAME> --project <PATH> [options]");
    eprintln!("  gyroflow-bench stop --name <NAME>");
    eprintln!("  gyroflow-bench list");
    eprintln!("  gyroflow-bench compare <NAME1> <NAME2> [NAME3...]");
    eprintln!();
    eprintln!("Run options:");
    eprintln!("  --name <NAME>           Run name (required, must be unique)");
    eprintln!("  --project <PATH>        Path to .gyroflow project (required)");
    eprintln!("  --resolution <WxH>      Override resolution (default: project native)");
    eprintln!("  --frames <N>            Number of frames (default: project length)");
    eprintln!("  --pixel-format <FMT>    rgba8 | rgba16 | rgbaf16 | rgbaf (default: rgba8)");
    eprintln!("  --device <INDEX>        GPU device index, -1 for CPU (default: auto)");
    eprintln!();
    eprintln!("Results saved to <gyroflow_data_dir>/benchmarks/<name>.bench.json");
}

// ---------- Argument parsing ----------

fn parse_run_args(args: &[String]) -> Result<RunConfig, String> {
    let mut name = None;
    let mut project = None;
    let mut resolution = None;
    let mut frames = None;
    let mut pixel_format = "rgba8".to_string();
    let mut device_index = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--name" => {
                i += 1;
                name = Some(args.get(i).ok_or("--name requires a value")?.clone());
            }
            "--project" => {
                i += 1;
                project = Some(args.get(i).ok_or("--project requires a value")?.clone());
            }
            "--resolution" => {
                i += 1;
                let val = args.get(i).ok_or("--resolution requires a value")?;
                let parts: Vec<&str> = val.split('x').collect();
                if parts.len() != 2 {
                    return Err(format!("Invalid resolution format: {val} (expected WxH)"));
                }
                let w = parts[0].parse::<usize>().map_err(|_| format!("Invalid width: {}", parts[0]))?;
                let h = parts[1].parse::<usize>().map_err(|_| format!("Invalid height: {}", parts[1]))?;
                resolution = Some((w, h));
            }
            "--frames" => {
                i += 1;
                let val = args.get(i).ok_or("--frames requires a value")?;
                frames = Some(val.parse::<usize>().map_err(|_| format!("Invalid frame count: {val}"))?);
            }
            "--pixel-format" => {
                i += 1;
                let val = args.get(i).ok_or("--pixel-format requires a value")?;
                match val.as_str() {
                    "rgba8" | "rgba16" | "rgbaf16" | "rgbaf" => pixel_format = val.clone(),
                    _ => return Err(format!("Unknown pixel format: {val}")),
                }
            }
            "--device" => {
                i += 1;
                let val = args.get(i).ok_or("--device requires a value")?;
                device_index = Some(val.parse::<i32>().map_err(|_| format!("Invalid device index: {val}"))?);
            }
            other => return Err(format!("Unknown option: {other}")),
        }
        i += 1;
    }

    Ok(RunConfig {
        name: name.ok_or("--name is required")?,
        project: project.ok_or("--project is required")?,
        resolution,
        frames,
        pixel_format,
        device_index,
    })
}

// ---------- run ----------

fn cmd_run(args: &[String]) -> Result<(), String> {
    let config = parse_run_args(args)?;

    // Refuse to overwrite an existing run.
    let result_path = result_path(&config.name);
    if result_path.exists() {
        return Err(format!(
            "Run '{}' already exists at {}. Choose a different name or delete the file.",
            config.name,
            result_path.display()
        ));
    }

    // Clean up any stale stop sentinel.
    let stop_file = stop_path(&config.name);
    let _ = std::fs::remove_file(&stop_file);

    install_signal_handler();

    eprintln!("Initializing GPU context...");
    let gpu_name = gpu::initialize_contexts();
    if let Some((adapter, backend)) = &gpu_name {
        eprintln!("  GPU: {adapter} ({backend})");
    } else {
        eprintln!("  No GPU available, will fall back to CPU");
    }

    eprintln!("Loading project: {}", config.project);
    let project_url = path_to_url(&config.project);
    let manager = StabilizationManager::default();
    let cancel = Arc::new(AtomicBool::new(false));
    manager
        .import_gyroflow_file(&project_url, true, |_| {}, cancel, true)
        .map_err(|e| format!("Failed to load project: {e:?}"))?;

    // Determine resolution: explicit override, otherwise project's native size.
    let (w, h) = config.resolution.unwrap_or_else(|| {
        let p = manager.params.read();
        p.size
    });
    if w == 0 || h == 0 {
        return Err(format!(
            "Invalid resolution {w}x{h} (project may not have video size; pass --resolution)"
        ));
    }
    eprintln!("Resolution: {w}x{h}");

    // Apply size if it differs, then recompute.
    let needs_recompute = {
        let p = manager.params.read();
        p.size != (w, h) || p.output_size != (w, h)
    };
    if needs_recompute {
        manager.set_size(w, h);
        manager.set_output_size(w, h);
    }
    if let Some(dev) = config.device_index {
        manager.stabilization.write().set_device(dev as isize);
    }
    manager.recompute_blocking();

    let fps = {
        let p = manager.params.read();
        let f = p.get_scaled_fps();
        if f <= 0.0 { 30.0 } else { f }
    };
    let total_in_project = manager.params.read().frame_count.max(1);
    let num_frames = config.frames.unwrap_or(total_in_project).min(total_in_project);
    eprintln!("Frames: {num_frames} @ {fps:.3} fps  pixel format: {}", config.pixel_format);

    // Allocate input/output buffers once; reuse across all iterations.
    let bpp = bytes_per_pixel(&config.pixel_format);
    let stride = w * bpp;
    let mut input_buf = vec![0u8; stride * h];
    let mut output_buf = vec![0u8; stride * h];

    let mut buffers = gpu::Buffers {
        input: gpu::BufferDescription {
            size: (w, h, stride),
            rect: None,
            rotation: None,
            data: gpu::BufferSource::Cpu { buffer: &mut input_buf },
            texture_copy: false,
        },
        output: gpu::BufferDescription {
            size: (w, h, stride),
            rect: None,
            rotation: None,
            data: gpu::BufferSource::Cpu { buffer: &mut output_buf },
            texture_copy: false,
        },
    };

    // Warmup: first frame triggers GPU shader compilation and pipeline init.
    eprintln!("Warmup...");
    let warmup_start = Instant::now();
    let warmup_info = dispatch_process!(config.pixel_format.as_str(), &manager, 0i64, Some(0), &mut buffers)
        .map_err(|e| format!("Warmup frame failed: {e:?}"))?;
    let warmup_ms = warmup_start.elapsed().as_secs_f64() * 1000.0;
    let mut detected_backend = warmup_info.backend.to_string();
    eprintln!("  warmup: {warmup_ms:.1}ms  backend: {detected_backend}");

    // Main timed loop.
    let mut frame_times_us = Vec::with_capacity(num_frames);
    let total_start = Instant::now();
    let mut interrupted = false;

    for frame_idx in 0..num_frames {
        if INTERRUPTED.load(Ordering::Relaxed) || stop_file.exists() {
            interrupted = true;
            let _ = std::fs::remove_file(&stop_file);
            eprintln!("\nStop signal received, ending early at frame {frame_idx}");
            break;
        }

        let timestamp_ms = gyroflow_core::timestamp_at_frame(frame_idx as i32, fps);
        let timestamp_us = (timestamp_ms * 1000.0).round() as i64;

        let t0 = Instant::now();
        let info = dispatch_process!(config.pixel_format.as_str(), &manager, timestamp_us, Some(frame_idx), &mut buffers)
            .map_err(|e| format!("Frame {frame_idx} failed: {e:?}"))?;
        frame_times_us.push(t0.elapsed().as_micros() as u64);

        if detected_backend.is_empty() {
            detected_backend = info.backend.to_string();
        }

        if frame_idx % 10 == 0 {
            let elapsed_s = total_start.elapsed().as_secs_f64();
            let cur_fps = (frame_idx + 1) as f64 / elapsed_s.max(0.001);
            eprint!("\r  Frame {}/{num_frames} ({cur_fps:.1} fps)   ", frame_idx + 1);
        }
    }
    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!();

    // Compute statistics.
    let frames_completed = frame_times_us.len();
    if frames_completed == 0 {
        return Err("No frames were processed".into());
    }
    let mut sorted = frame_times_us.clone();
    sorted.sort_unstable();
    let mean = sorted.iter().sum::<u64>() as f64 / frames_completed as f64;
    let median = sorted[frames_completed / 2] as f64;
    let p95 = sorted[((frames_completed as f64 * 0.95) as usize).min(frames_completed - 1)] as f64;
    let p99 = sorted[((frames_completed as f64 * 0.99) as usize).min(frames_completed - 1)] as f64;
    let min_us = *sorted.first().unwrap();
    let max_us = *sorted.last().unwrap();
    let achieved_fps = frames_completed as f64 / (total_ms / 1000.0);

    let machine = MachineInfo {
        hostname: get_hostname(),
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        cpu_count: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        gpu_devices: manager.stabilization.read().list_devices(),
    };

    let result = BenchmarkResult {
        name: config.name.clone(),
        timestamp: iso_now(),
        git_commit: git_hash(),
        backend: detected_backend,
        project_file: config.project,
        resolution: (w, h),
        pixel_format: config.pixel_format.to_uppercase(),
        device_index: config.device_index,
        frames_requested: num_frames,
        frames_completed,
        frame_times_us,
        total_time_ms: total_ms,
        warmup_time_ms: warmup_ms,
        mean_frame_us: mean,
        median_frame_us: median,
        p95_frame_us: p95,
        p99_frame_us: p99,
        min_frame_us: min_us,
        max_frame_us: max_us,
        fps: achieved_fps,
        peak_rss_bytes: peak_rss_bytes(),
        machine,
        completed: !interrupted,
    };

    let json = serde_json::to_string_pretty(&result)
        .map_err(|e| format!("Failed to serialize result: {e}"))?;
    std::fs::write(&result_path, json)
        .map_err(|e| format!("Failed to write result to {}: {e}", result_path.display()))?;

    println!();
    println!("Results saved to {}", result_path.display());
    print_single_summary(&result);
    Ok(())
}

// ---------- stop ----------

fn cmd_stop(args: &[String]) -> Result<(), String> {
    let mut name = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--name" => {
                i += 1;
                name = Some(args.get(i).ok_or("--name requires a value")?.clone());
            }
            other => return Err(format!("Unknown option: {other}")),
        }
        i += 1;
    }
    let name = name.ok_or("--name is required")?;
    let path = stop_path(&name);
    std::fs::write(&path, b"")
        .map_err(|e| format!("Failed to write stop sentinel {}: {e}", path.display()))?;
    eprintln!("Stop signal sent to run '{name}' (sentinel: {})", path.display());
    Ok(())
}

// ---------- list ----------

fn cmd_list() -> Result<(), String> {
    let results = all_results();
    if results.is_empty() {
        println!("No benchmark runs found in {}", bench_dir().display());
        return Ok(());
    }

    println!();
    println!(
        "{:<24} {:<11} {:<8} {:<10} {:>8} {:>10} {:>10} {:>10}",
        "Name", "Resolution", "Format", "Backend", "Frames", "FPS", "Mean (µs)", "Peak RSS"
    );
    println!("{:-<104}", "");
    for r in &results {
        let res_str = format!("{}x{}", r.resolution.0, r.resolution.1);
        let status = if r.completed { "" } else { " *" };
        println!(
            "{:<24} {:<11} {:<8} {:<10} {:>8} {:>10.2} {:>10.0} {:>10}{}",
            truncate(&r.name, 24),
            res_str,
            r.pixel_format,
            truncate(&r.backend, 10),
            r.frames_completed,
            r.fps,
            r.mean_frame_us,
            format_bytes(r.peak_rss_bytes),
            status,
        );
    }
    println!();
    println!("(* = run was stopped before completion)");
    println!();
    println!("Storage: {}", bench_dir().display());
    Ok(())
}

// ---------- compare ----------

fn cmd_compare(args: &[String]) -> Result<(), String> {
    if args.len() < 2 {
        return Err("compare requires at least 2 run names".into());
    }
    let results: Vec<BenchmarkResult> = args
        .iter()
        .map(|name| load_result(name).map_err(|e| format!("Run '{name}': {e}")))
        .collect::<Result<Vec<_>, _>>()?;

    print_compare_table(&results);
    Ok(())
}

// ---------- Output formatting ----------

fn print_single_summary(r: &BenchmarkResult) {
    println!();
    println!("Summary for run '{}':", r.name);
    println!("  Backend:      {}", r.backend);
    println!("  Resolution:   {}x{} ({})", r.resolution.0, r.resolution.1, r.pixel_format);
    println!("  Frames:       {} / {} ({})", r.frames_completed, r.frames_requested,
        if r.completed { "complete" } else { "stopped early" });
    println!("  Total time:   {:.2} ms", r.total_time_ms);
    println!("  Warmup:       {:.2} ms", r.warmup_time_ms);
    println!("  FPS:          {:.2}", r.fps);
    println!("  Mean frame:   {:.0} µs", r.mean_frame_us);
    println!("  Median:       {:.0} µs", r.median_frame_us);
    println!("  P95:          {:.0} µs", r.p95_frame_us);
    println!("  P99:          {:.0} µs", r.p99_frame_us);
    println!("  Min/Max:      {} / {} µs", r.min_frame_us, r.max_frame_us);
    println!("  Peak RSS:     {}", format_bytes(r.peak_rss_bytes));
    println!("  Git commit:   {}", r.git_commit);
    println!("  Machine:      {} ({}/{}, {} CPUs)", r.machine.hostname, r.machine.os, r.machine.arch, r.machine.cpu_count);
}

const COL_LABEL: usize = 16;
const COL_DATA: usize = 26;

fn print_compare_table(results: &[BenchmarkResult]) {
    let baseline = &results[0];

    // Header
    print_h_separator(results.len(), '┌', '┬', '┐');
    print_row("Metric", &results.iter().map(|r| truncate(&r.name, COL_DATA - 2)).collect::<Vec<_>>(), &vec![None; results.len()]);
    print_h_separator(results.len(), '├', '┼', '┤');

    // String rows
    print_string_row("Resolution", results, |r| format!("{}x{}", r.resolution.0, r.resolution.1));
    print_string_row("Pixel Format", results, |r| r.pixel_format.clone());
    print_string_row("Backend", results, |r| r.backend.clone());

    // Numeric rows with delta vs baseline
    print_numeric_row("Frames",      results, baseline, |r| r.frames_completed as f64, "%-6.0f", false);
    print_numeric_row("Total time",  results, baseline, |r| r.total_time_ms,           "%-.2f ms", false);
    print_numeric_row("Warmup",      results, baseline, |r| r.warmup_time_ms,          "%-.2f ms", false);
    print_numeric_row("FPS",         results, baseline, |r| r.fps,                     "%-.2f",   true);
    print_numeric_row("Mean frame",  results, baseline, |r| r.mean_frame_us,           "%-.0f µs", false);
    print_numeric_row("Median",      results, baseline, |r| r.median_frame_us,         "%-.0f µs", false);
    print_numeric_row("P95",         results, baseline, |r| r.p95_frame_us,            "%-.0f µs", false);
    print_numeric_row("P99",         results, baseline, |r| r.p99_frame_us,            "%-.0f µs", false);
    print_numeric_row("Max frame",   results, baseline, |r| r.max_frame_us as f64,     "%-.0f µs", false);
    print_numeric_row("Peak RSS",    results, baseline, |r| r.peak_rss_bytes as f64,   "bytes",   false);

    // Identity rows
    print_string_row("Git commit", results, |r| r.git_commit.clone());
    print_string_row("Timestamp",  results, |r| r.timestamp.clone());
    print_string_row("Machine",    results, |r| r.machine.hostname.clone());

    print_h_separator(results.len(), '└', '┴', '┘');
}

fn print_h_separator(n_data_cols: usize, left: char, mid: char, right: char) {
    let mut s = String::new();
    s.push(left);
    s.push_str(&"─".repeat(COL_LABEL + 2));
    for _ in 0..n_data_cols {
        s.push(mid);
        s.push_str(&"─".repeat(COL_DATA + 2));
    }
    s.push(right);
    println!("{s}");
}

fn print_row(label: &str, values: &[String], deltas: &[Option<String>]) {
    let mut s = String::new();
    s.push('│');
    let label_trunc = truncate(label, COL_LABEL);
    s.push_str(&format!(" {:<width$} ", label_trunc, width = COL_LABEL));
    for (val, delta) in values.iter().zip(deltas.iter()) {
        s.push('│');
        let val_trunc = truncate(val, COL_DATA);
        let combined = match delta {
            Some(d) => {
                let val_w = COL_DATA.saturating_sub(7);
                format!("{:<vw$} {:>5}", val_trunc, d, vw = val_w)
            }
            None => format!("{:<w$}", val_trunc, w = COL_DATA),
        };
        let cell = pad_right(&combined, COL_DATA);
        s.push_str(&format!(" {} ", cell));
    }
    s.push('│');
    println!("{s}");
}

fn pad_right(s: &str, width: usize) -> String {
    let len = s.chars().count();
    if len >= width {
        s.chars().take(width).collect()
    } else {
        let mut out = s.to_string();
        out.extend(std::iter::repeat(' ').take(width - len));
        out
    }
}

fn print_string_row<F: Fn(&BenchmarkResult) -> String>(label: &str, results: &[BenchmarkResult], f: F) {
    let values: Vec<String> = results.iter().map(&f).collect();
    let deltas: Vec<Option<String>> = vec![None; results.len()];
    print_row(label, &values, &deltas);
}

fn print_numeric_row<F: Fn(&BenchmarkResult) -> f64>(
    label: &str,
    results: &[BenchmarkResult],
    baseline: &BenchmarkResult,
    f: F,
    fmt_kind: &str,
    higher_is_better: bool,
) {
    let baseline_val = f(baseline);
    let mut values = Vec::with_capacity(results.len());
    let mut deltas = Vec::with_capacity(results.len());
    for (i, r) in results.iter().enumerate() {
        let v = f(r);
        let formatted = format_numeric(v, fmt_kind);
        values.push(formatted);
        if i == 0 || baseline_val == 0.0 {
            deltas.push(None);
        } else {
            let pct = (v - baseline_val) / baseline_val * 100.0;
            let arrow = if (pct > 0.0) == higher_is_better { "↑" } else { "↓" };
            let sign = if pct >= 0.0 { "+" } else { "" };
            deltas.push(Some(format!("{sign}{pct:.0}%{arrow}")));
        }
    }
    print_row(label, &values, &deltas);
}

fn format_numeric(v: f64, fmt_kind: &str) -> String {
    if fmt_kind == "bytes" {
        return format_bytes(v as u64);
    }
    if fmt_kind == "%-6.0f" {
        return format!("{v:.0}");
    }
    if fmt_kind == "%-.2f" {
        return format!("{v:.2}");
    }
    if fmt_kind == "%-.2f ms" {
        return format!("{v:.2} ms");
    }
    if fmt_kind == "%-.0f µs" {
        return format!("{v:.0} µs");
    }
    format!("{v}")
}

fn truncate(s: &str, max: usize) -> String {
    let count = s.chars().count();
    if count <= max {
        s.to_string()
    } else {
        s.chars().take(max.saturating_sub(1)).collect::<String>() + "…"
    }
}

fn format_bytes(b: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    let f = b as f64;
    if f >= GB { format!("{:.2} GB", f / GB) }
    else if f >= MB { format!("{:.1} MB", f / MB) }
    else if f >= KB { format!("{:.1} KB", f / KB) }
    else { format!("{b} B") }
}

// ---------- Storage / persistence ----------

fn bench_dir() -> PathBuf {
    let dir = gyroflow_core::settings::data_dir().join("benchmarks");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

fn result_path(name: &str) -> PathBuf {
    bench_dir().join(format!("{name}.bench.json"))
}

fn stop_path(name: &str) -> PathBuf {
    bench_dir().join(format!("{name}.stop"))
}

fn load_result(name: &str) -> Result<BenchmarkResult, String> {
    let path = result_path(name);
    let data = std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_str(&data).map_err(|e| format!("parse {}: {e}", path.display()))
}

fn all_results() -> Vec<BenchmarkResult> {
    let dir = bench_dir();
    let mut results = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json")
                && path.file_name().and_then(|n| n.to_str()).map(|s| s.ends_with(".bench.json")).unwrap_or(false)
            {
                if let Ok(data) = std::fs::read_to_string(&path) {
                    if let Ok(r) = serde_json::from_str::<BenchmarkResult>(&data) {
                        results.push(r);
                    }
                }
            }
        }
    }
    results.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
    results
}

// ---------- Environment helpers ----------

fn git_hash() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

fn get_hostname() -> String {
    std::process::Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

fn iso_now() -> String {
    let now = time::OffsetDateTime::now_utc();
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        now.year(),
        u8::from(now.month()),
        now.day(),
        now.hour(),
        now.minute(),
        now.second()
    )
}

fn peak_rss_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(rest) = line.strip_prefix("VmHWM:") {
                    let kb = rest.split_whitespace()
                        .next()
                        .and_then(|s| s.parse::<u64>().ok())
                        .unwrap_or(0);
                    return kb * 1024;
                }
            }
        }
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        // SAFETY: getrusage is async-signal-safe and rusage is plain old data.
        let mut info: libc::rusage = unsafe { std::mem::zeroed() };
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut info) } == 0 {
            // ru_maxrss is bytes on macOS, kilobytes on Linux.
            return info.ru_maxrss as u64;
        }
    }
    0
}

fn path_to_url(path: &str) -> String {
    if path.contains("://") {
        return path.to_string();
    }
    let abs = std::fs::canonicalize(path).unwrap_or_else(|_| PathBuf::from(path));
    let s = abs.to_string_lossy().replace('\\', "/");
    if s.starts_with('/') {
        format!("file://{s}")
    } else {
        format!("file:///{s}")
    }
}

// ---------- Signal handling ----------

extern "C" fn handle_sigint(_sig: libc::c_int) {
    INTERRUPTED.store(true, Ordering::SeqCst);
}

fn install_signal_handler() {
    // SAFETY: handle_sigint only performs an atomic store, which is async-signal-safe.
    // Cast the fn item through a fn pointer to satisfy strict-provenance rules.
    let handler: extern "C" fn(libc::c_int) = handle_sigint;
    unsafe {
        libc::signal(libc::SIGINT, handler as usize);
    }
}

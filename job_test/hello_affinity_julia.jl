using Sockets

sleep(5)  # Simulate some work being done

hostname = gethostname()

# Detect GPUs
gpu_id = get(ENV, "CUDA_VISIBLE_DEVICES",
         get(ENV, "ZE_AFFINITY_MASK", "No GPUs assigned"))

# Detect CPU affinity (Linux: read /proc/self/status)
cpu_affinity_str = "unknown"
ncpu = 0
try
    open("/proc/self/status", "r") do f
        for line in eachline(f)
            if startswith(line, "Cpus_allowed_list:")
                cpu_affinity_str = strip(split(line, ":")[2])
                # Parse "0-3,8,12-15" → count of CPUs
                ncpu = sum(
                    let p = split(rng, "-")
                        length(p) == 1 ? 1 : parse(Int, p[2]) - parse(Int, p[1]) + 1
                    end
                    for rng in split(cpu_affinity_str, ",")
                )
                break
            end
        end
    end
    println("Detected CPU affinity from /proc/self/status: $cpu_affinity_str")
catch e
    cpu_affinity_str = "all"
    ncpu = Sys.CPU_THREADS
    println("Fallback CPU affinity using Sys.CPU_THREADS: $ncpu")
end

println()
println("Hello from host $hostname:")
println("  GPU ID(s): $gpu_id")
println("  CPU affinity: $ncpu CPUs available to this process")
println("  CPU IDs: $cpu_affinity_str")
println()

# Report status to PBX
open("PBX_JOB_STATUS_REPORT", "w") do f
    write(f, "Done")
end

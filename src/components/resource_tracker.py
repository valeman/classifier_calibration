import threading
import time
import psutil
import os
import logging

lg = logging.getLogger(__name__)

class ResourceTracker:
    """
    Context manager to measure:
    - CPU usage (user, system, total) from cgroup v2
    - Peak resident memory (RAM) usage
    - Peak total swap and zswap usage
    - Total disk I/O (read/write in MiB)

    Requires the script to run in its own cgroup (e.g. via systemd-run --user --scope).
    """


    def __init__(self, sample_interval=0.1):
        self.sample_interval = sample_interval #Secounds
        self._stop_event = threading.Event()
        

    def _get_cgroup_path(self):
        # Parse /proc/self/cgroup to find the unified cgroupv2 path
        with open("/proc/self/cgroup") as f:
            for line in f:
                if line.startswith("0::"):
                    _, _, path = line.strip().partition("0::")
                    path = path.lstrip("/")
                    return os.path.join("/sys/fs/cgroup/", path)
        raise RuntimeError("Could not determine cgroup path")

    def _read_cpu_usage(self):
        path = os.path.join(self.cgroup_path, "cpu.stat")
        cpu_secs = [0,0,0] #Total, User, System
        with open(path) as f:
            for line in f:
                if line.startswith("usage_usec"):
                    # Total CPU time in microseconds
                    cpu_secs[0] = int(line.split()[1])
                
                if line.startswith("user_usec"):
                    # User CPU time in microseconds
                    cpu_secs[1] = int(line.split()[1])
        
                if line.startswith("system_usec"):
                    # System CPU time in microseconds
                    cpu_secs[2] = int(line.split()[1])
        return cpu_secs

    def _snapshot_process_io(self):
        pids = []
        with open(os.path.join(self.cgroup_path, "cgroup.procs")) as f:
            pids = [int(pid) for pid in f.read().split()]
        total_read, total_write = 0, 0
        for pid in pids:
            try:
                proc = psutil.Process(pid)
                io = proc.io_counters()
                total_read += io.read_bytes
                total_write += io.write_bytes
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total_read, total_write
    
    def _read_memory(self, prev_peak, file_name):
        mem_path = os.path.join(self.cgroup_path, file_name)
        new_peak = 0
        with open(mem_path) as f:
            current = int(f.read().strip())
            if current > prev_peak:
                new_peak = current
            else:
                new_peak = prev_peak
        return new_peak

    def _sample_memory(self):
        while not self._stop_event.is_set():
            try:
                self.max_r_memory = self._read_memory(self.max_r_memory, "memory.current")
                self.max_zs_memory = self._read_memory(self.max_zs_memory, "memory.zswap.current")
                self.max_s_memory = self._read_memory(self.max_s_memory, "memory.swap.current")
            except Exception:
                pass
            time.sleep(self.sample_interval)

    def __enter__(self):
        # Determine cgroup path and initial snapshots
        self.cgroup_path = self._get_cgroup_path()
        lg.info(f"Current cgroup path: {self.cgroup_path}")
        # CPU usage in microseconds
        self.t_cpu_start, self.u_cpu_start, self.s_cpu_start = self._read_cpu_usage()
        # IO  counters
        self.io_read_start, self.io_write_start = self._snapshot_process_io()
        # Start memory sampling thread
        self._stop_event.clear()
        self.max_r_memory = 0
        self.max_zs_memory = 0
        self.max_s_memory = 0
        self._thread = threading.Thread(target=self._sample_memory, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        # Stop sampling thread
        self._stop_event.set()
        self._thread.join()
        # Final snapshots
        t_cpu_end, u_cpu_end, s_cpu_end = self._read_cpu_usage()
        io_read_end, io_write_end = self._snapshot_process_io()

        # Collect results
        b_to_mib = (1024**2)
        self.total_cpu_time_sec = (t_cpu_end - self.t_cpu_start) / 1_000_000  # convert usec to sec
        self.user_cpu_time_sec = (u_cpu_end - self.u_cpu_start) / 1_000_000
        self.system_cpu_time_sec = (s_cpu_end - self.s_cpu_start) / 1_000_000

        self.peak_memory_mib = self.max_r_memory / b_to_mib
        self.peak_swap_mib = self.max_s_memory / b_to_mib
        self.peak_zswap_mib = self.max_zs_memory / b_to_mib
        
        self.total_io_read_mib = (io_read_end - self.io_read_start) / b_to_mib
        self.total_io_write_mib = (io_write_end - self.io_write_start) / b_to_mib
        
        

# Example usage:
if __name__ == "__main__":

    def simulate_cpu(duration=2):
        start = time.time()
        while time.time() - start < duration:
            _ = sum(i*i for i in range(1000))

    def simulate_memory(allocation_mib=100):
        data = [' ' * 1024 * 1024 for _ in range(allocation_mib)]
        time.sleep(1)
        return data  

    def simulate_disk_io(size_mb=10, filepath="/tmp/test_read_file"):
        """Simulates disk read by reading from an uncached file."""
        # Step 1: Write a file to disk
        with open(filepath, "wb") as f:
            f.write(os.urandom(size_mb * 1024 * 1024))  

        # Step 2: Drop it from page cache (non-privileged hint)
        os.sync()  
        
        # Step 3: Read file to trigger actual disk IO
        with open(filepath, "rb") as f:
            # Hint to kernel that we won't need cached pages
            try:
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            except AttributeError:
                pass  # posix_fadvise not available on all platforms
            f.read()  # Trigger read

        os.remove(filepath)

    def test_resource_tracking():
        with ResourceTracker(sample_interval=0.05) as rt:
            simulate_cpu(2)
            simulate_memory(100)
            simulate_disk_io(10)

        print(f"CPU time: {rt.total_cpu_time_sec:.2f} s (Expected: ~2s)")
        print(f"Peak memory: {rt.peak_memory_mib:.2f} MiB (Expected: >= 100 MiB)")
        print(f"Disk IO read: {rt.total_io_read_mib:.2f} MiB (Expected: >= 10 MiB)")
        print(f"Disk IO write: {rt.total_io_write_mib:.2f} MiB (Expected: >= 10 MiB)")

    test_resource_tracking()
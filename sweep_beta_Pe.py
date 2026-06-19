"""
Parameter sweep for beta and Pe.

Runs coalescence simulations and postprocessing across a parameter grid.

Usage:
    python sweep_beta_Pe.py
    python sweep_beta_Pe.py --process-only
    python sweep_beta_Pe.py --continue-from-batch 3
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


BETA_VALUES = [0.1, 0.4, 0.8]
PE_VALUES = [1, 10,100, 1000, 10000]
BATCH_SIZE = 1


def print_header(script_dir, total_cases, process_only):
    print("==============================================")
    print("Parameter Sweep: beta and Pe (ROBUST VERSION)")
    print(f"beta values: {', '.join(str(v) for v in BETA_VALUES)}")
    print(f"Pe values: {', '.join(str(v) for v in PE_VALUES)}")
    print(f"Total cases: {total_cases}")
    print(f"Script directory: {script_dir}")
    print(f"Batch size: {BATCH_SIZE} (serialized to avoid DLL locking)")
    if process_only:
        print("Mode: Post-processing only")
    print("==============================================")
    print()


def clean_compiled_files(outdir):
    """Remove compiled files that can cause locking issues."""
    ccode_dir = outdir / "_ccode"
    if not ccode_dir.exists():
        return

    for pattern in ("*.dll", "*.o"):
        for path in ccode_dir.glob(pattern):
            try:
                path.unlink()
            except OSError:
                pass


def clean_large_outputs(outdir):
    """Remove bulky files that are not needed by the text-data postprocessing."""
    for dirname in ("_states", "_ccode"):
        shutil.rmtree(outdir / dirname, ignore_errors=True)

    for pattern in ("*.pvd", "*.vtu"):
        for path in outdir.glob(pattern):
            try:
                path.unlink()
            except OSError:
                pass

    domain_dir = outdir / "domain"
    if domain_dir.exists():
        for path in domain_dir.iterdir():
            if path.is_file() and path.suffix.lower() != ".txt":
                try:
                    path.unlink()
                except OSError:
                    pass


def is_simulation_complete(outdir):
    """Check if a simulation has already completed successfully.
    
    A simulation is considered complete if it has produced a substantial number
    of domain output files (domain_*.txt). These files are only created during
    the simulation and are NOT deleted by cleanup operations, making them the
    most reliable indicator of whether the simulation actually ran to completion.
    """
    if not outdir.exists():
        return False
    
    # The key indicator: count domain output files
    # Each represents one timestep of the simulation
    # If there are 100+, the simulation definitely ran to meaningful completion
    domain_dir = outdir / "domain"
    if not domain_dir.exists():
        return False
    
    domain_files = list(domain_dir.glob("domain_*.txt"))
    
    # If we have 100+ timestep files, simulation ran substantially
    if len(domain_files) >= 100:
        return True
    
    return False


def run_simulation(script_dir, python_exe, beta, pe, outdir, logfile):
    """Run one coalescence simulation and append all output to its log file."""
    command = [
        str(python_exe),
        "coalescence.py",
        "--beta",
        str(beta),
        "--Pe",
        str(pe),
        "--output-dir",
        str(outdir),
    ]

    outdir.mkdir(parents=True, exist_ok=True)
    with logfile.open("a", encoding="utf-8", errors="replace") as log:
        result = subprocess.run(
            command,
            cwd=script_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    clean_large_outputs(outdir)
    return result.returncode == 0



def run_simulation_grid(script_dir, base_dir, continue_from_batch, total_cases):
    base_dir.mkdir(parents=True, exist_ok=True)

    python_exe = script_dir / ".venv" / "Scripts" / "python.exe"
    failed_cases = []

    print()
    print("Launching simulations...")
    print("----------------------------------------------")

    case_num = 0
    batch_num = 0

    for beta in BETA_VALUES:
        for pe in PE_VALUES:
            case_num += 1

            if case_num < (continue_from_batch * BATCH_SIZE + 1):
                continue

            outdir = base_dir / f"beta_{beta}_Pe_{pe}"
            logfile = base_dir / f"beta_{beta}_Pe_{pe}_log.txt"

            if is_simulation_complete(outdir):
                print(f"Case {case_num}/{total_cases}: beta={beta}, Pe={pe}")
                print(f"  [SKIPPED] Simulation already complete")
                print()
                continue

            clean_compiled_files(outdir)
            outdir.mkdir(parents=True, exist_ok=True)

            print(f"Case {case_num}/{total_cases}: beta={beta}, Pe={pe}")
            print(f"  Output: {outdir}")
            print(f"  Log: {logfile}")

            batch_num += 1
            print(f"  Waiting for batch {batch_num} (1 jobs) to complete...")

            ok = run_simulation(script_dir, python_exe, beta, pe, outdir, logfile)
            if ok:
                print(f"    [OK] Case {case_num} completed (beta={beta}, Pe={pe})")
            else:
                print(f"    [RETRY] Case {case_num} needs retry (beta={beta}, Pe={pe})")
                failed_cases.append({
                    "case_num": case_num,
                    "beta": beta,
                    "pe": pe,
                    "outdir": outdir,
                    "logfile": logfile,
                })
            print()

    if failed_cases:
        print()
        print(f"Retrying {len(failed_cases)} failed case(s)...")
        print("----------------------------------------------")

        for case in failed_cases:
            print(f"Retry Case {case['case_num']}: beta={case['beta']}, Pe={case['pe']}")

            shutil.rmtree(case["outdir"], ignore_errors=True)
            try:
                case["logfile"].unlink()
            except FileNotFoundError:
                pass
            case["outdir"].mkdir(parents=True, exist_ok=True)

            ok = run_simulation(
                script_dir,
                python_exe,
                case["beta"],
                case["pe"],
                case["outdir"],
                case["logfile"],
            )

            if ok:
                print("  [OK] Retry succeeded")
            else:
                print(f"  [FAILED] Retry failed - check log at {case['logfile']}")

    print()
    print("All simulations completed!")


def get_postprocess_python(script_dir):
    python_exe = script_dir / ".venv" / "Scripts" / "python.exe"
    if python_exe.exists():
        return python_exe

    print(f"Warning: Python venv not found at {python_exe}, using system python")
    return Path("python")


def count_png_files(path):
    if not path.exists():
        return 0
    return sum(1 for item in path.glob("*.png") if item.is_file())


def run_postprocessing(script_dir, base_dir, total_cases):
    print()
    print("Post-processing results...")
    print("----------------------------------------------")

    python_exe = get_postprocess_python(script_dir)
    postproc_count = 0
    postproc_failed = []

    for beta in BETA_VALUES:
        for pe in PE_VALUES:
            simdir = base_dir / f"beta_{beta}_Pe_{pe}"

            if not simdir.exists():
                print(f"Warning: {simdir} not found, skipping post-processing")
                continue

            postproc_count += 1
            plotsdir = Path(f"{simdir}_plots")

            if plotsdir.exists() and count_png_files(plotsdir) >= 4:
                print(f"[{postproc_count}/{total_cases}] SKIPPED (plots exist): beta={beta}, Pe={pe}")
                continue

            print(f"[{postproc_count}/{total_cases}] Processing: beta={beta}, Pe={pe}")
            print(f"  Input: {simdir}")

            try:
                result = subprocess.run(
                    [str(python_exe), "postprocess.py", str(simdir)],
                    cwd=script_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

                plotcount = count_png_files(plotsdir)
                if result.returncode == 0 and plotsdir.exists() and plotcount >= 4:
                    print(f"  [OK] Plots saved to: {plotsdir} ({plotcount} PNG files)")
                elif plotsdir.exists():
                    print(f"  [ERROR] Only {plotcount} plots generated (expected >= 4)")
                    postproc_failed.append(f"beta={beta}, Pe={pe}")
                else:
                    print("  [ERROR] Plots directory not created")
                    postproc_failed.append(f"beta={beta}, Pe={pe}")
            except Exception as error:
                print(f"  [ERROR] Post-processing failed: {error}")
                postproc_failed.append(f"beta={beta}, Pe={pe}")

    return postproc_count, postproc_failed


def print_summary(total_cases, postproc_count, postproc_failed, base_dir):
    print()
    print("==============================================")
    print("FINAL SUMMARY")
    print("==============================================")
    print(f"Total simulations: {total_cases}")
    print(f"Post-processed: {postproc_count}")

    if postproc_failed:
        print(f"Failed post-processing: {len(postproc_failed)}")
        for failure in postproc_failed:
            print(f"  - {failure}")
    else:
        print("All post-processing successful!")

    print(f"Results location: {base_dir}")
    print("==============================================")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run coalescence parameter sweep for beta and Pe."
    )
    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Only run postprocessing on existing simulation results.",
    )
    parser.add_argument(
        "--continue-from-batch",
        type=int,
        default=0,
        help="Skip cases before this batch number.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir / "sweep_beta_Pe_results"
    total_cases = len(BETA_VALUES) * len(PE_VALUES)

    print_header(script_dir, total_cases, args.process_only)

    if not args.process_only:
        run_simulation_grid(script_dir, base_dir, args.continue_from_batch, total_cases)
    elif not base_dir.exists():
        print(f"Error: {base_dir} not found. Run simulations first without --process-only flag.")
        sys.exit(1)

    postproc_count, postproc_failed = run_postprocessing(script_dir, base_dir, total_cases)
    print_summary(total_cases, postproc_count, postproc_failed, base_dir)


if __name__ == "__main__":
    main()

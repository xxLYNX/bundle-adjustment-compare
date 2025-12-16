#!/usr/bin/env python3
"""
Benchmark runner for Bundle Adjustment solvers.
Runs each solver multiple times and computes statistical metrics.
"""

import subprocess
import os
import sys
import json
import re
import numpy as np
from datetime import datetime
import argparse


def parse_ba_output(filepath):
    """Parse BA analysis file to extract metrics."""
    metrics = {}
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
            # Extract timing
            match = re.search(r'BA Optimization Time: ([\d.]+) seconds', content)
            if match:
                metrics['ba_time'] = float(match.group(1))
            
            # Extract iterations
            match = re.search(r'Total Iterations: (\d+)', content)
            if match:
                metrics['total_iterations'] = int(match.group(1))
                
            match = re.search(r'Successful Steps: (\d+)', content)
            if match:
                metrics['successful_steps'] = int(match.group(1))
                
            match = re.search(r'Unsuccessful Steps: (\d+)', content)
            if match:
                metrics['unsuccessful_steps'] = int(match.group(1))
            
            # Extract costs
            match = re.search(r'Initial Cost: ([\d.e+-]+)', content)
            if match:
                metrics['initial_cost'] = float(match.group(1))
                
            match = re.search(r'Final Cost: ([\d.e+-]+)', content)
            if match:
                metrics['final_cost'] = float(match.group(1))
            
            # Extract RMS error
            match = re.search(r'RMS Reprojection Error: ([\d.]+) pixels', content)
            if match:
                metrics['rms_error'] = float(match.group(1))
            
            # Extract time per iteration
            match = re.search(r'Time per Iteration: ([\d.]+) ms', content)
            if match:
                metrics['time_per_iter'] = float(match.group(1))
                
    except Exception as e:
        print(f"Warning: Failed to parse {filepath}: {e}")
    
    return metrics


def compute_statistics(values):
    """Compute statistical metrics for a list of values."""
    if not values:
        return {}
    
    arr = np.array(values)
    stats = {
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'count': len(arr)
    }
    
    # 95% confidence interval (only if there's variance)
    if len(arr) > 1 and stats['std'] > 0:
        from scipy import stats as scipy_stats
        sem = scipy_stats.sem(arr)
        ci = scipy_stats.t.interval(0.95, len(arr)-1, 
                                     loc=stats['mean'], 
                                     scale=sem)
        stats['ci_95_lower'] = float(ci[0])
        stats['ci_95_upper'] = float(ci[1])
        stats['ci_95_margin'] = float(ci[1] - stats['mean'])
    
    return stats


def run_benchmark(solver, num_runs, ba_module_path, output_dir):
    """Run BA solver multiple times and collect metrics."""
    print(f"\n{'='*60}")
    print(f"Benchmarking {solver.upper()} solver with {num_runs} runs")
    print(f"{'='*60}")
    
    all_runs = []
    
    for run_idx in range(num_runs):
        print(f"\nRun {run_idx + 1}/{num_runs}...", end=' ', flush=True)
        
        # Run BA module
        try:
            result = subprocess.run(
                [ba_module_path, solver],
                cwd=os.path.dirname(ba_module_path),
                capture_output=True,
                text=True,
                check=True,
                env={**os.environ, 'BA_ROOT': os.path.dirname(os.path.dirname(ba_module_path))}
            )
            
            # Parse output file
            output_file = os.path.join(output_dir, f'{solver}_BA.txt')
            if os.path.exists(output_file):
                metrics = parse_ba_output(output_file)
                if metrics:
                    all_runs.append(metrics)
                    print(f"✓ (Time: {metrics.get('ba_time', 0):.3f}s, Cost: {metrics.get('final_cost', 0):.2e})")
                else:
                    print("✗ Failed to parse output")
            else:
                print(f"✗ Output file not found: {output_file}")
                
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed: {e}")
            continue
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    if not all_runs:
        print(f"\nError: No successful runs for {solver}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Completed {len(all_runs)}/{num_runs} successful runs")
    print(f"{'='*60}")
    
    # Aggregate statistics
    metrics_keys = ['ba_time', 'total_iterations', 'successful_steps', 
                   'unsuccessful_steps', 'initial_cost', 'final_cost', 
                   'rms_error', 'time_per_iter']
    
    aggregated = {}
    for key in metrics_keys:
        values = [run[key] for run in all_runs if key in run]
        if values:
            aggregated[key] = compute_statistics(values)
    
    # Add raw data
    aggregated['raw_runs'] = all_runs
    aggregated['solver'] = solver
    aggregated['num_runs'] = len(all_runs)
    
    return aggregated


def print_summary(results):
    """Print summary of benchmark results."""
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}\n")
    
    for solver_name, data in results.items():
        if not data:
            continue
            
        print(f"{solver_name.upper()} Solver ({data['num_runs']} runs):")
        print(f"{'─'*60}")
        
        # Timing
        if 'ba_time' in data:
            bt = data['ba_time']
            print(f"  BA Time:       {bt['mean']:.3f} ± {bt['std']:.3f} s  "
                  f"[{bt['min']:.3f}, {bt['max']:.3f}]")
        
        # Iterations
        if 'total_iterations' in data:
            it = data['total_iterations']
            print(f"  Iterations:    {it['mean']:.1f} ± {it['std']:.1f}  "
                  f"[{int(it['min'])}, {int(it['max'])}]")
        
        # Final cost
        if 'final_cost' in data:
            fc = data['final_cost']
            print(f"  Final Cost:    {fc['mean']:.2e} ± {fc['std']:.2e}  "
                  f"[{fc['min']:.2e}, {fc['max']:.2e}]")
        
        # RMS error
        if 'rms_error' in data:
            rms = data['rms_error']
            print(f"  RMS Error:     {rms['mean']:.3f} ± {rms['std']:.3f} px  "
                  f"[{rms['min']:.3f}, {rms['max']:.3f}]")
        
        # Success rate
        if 'successful_steps' in data and 'total_iterations' in data:
            ss = data['successful_steps']
            ti = data['total_iterations']
            success_rate = (ss['mean'] / ti['mean']) * 100 if ti['mean'] > 0 else 0
            print(f"  Success Rate:  {success_rate:.1f}%")
        
        print()


def compare_solvers(results):
    """Compare two solvers if both present."""
    if 'lm' not in results or 'dogleg' not in results:
        return
    
    print(f"{'='*60}")
    print("SOLVER COMPARISON")
    print(f"{'='*60}\n")
    
    lm = results['lm']
    dog = results['dogleg']
    
    # Time comparison
    if 'ba_time' in lm and 'ba_time' in dog:
        lm_time = lm['ba_time']['mean']
        dog_time = dog['ba_time']['mean']
        speedup = dog_time / lm_time
        faster = "LM" if speedup > 1 else "Dogleg"
        percent = abs(speedup - 1) * 100
        print(f"Speed: {faster} is {percent:.1f}% faster")
    
    # Iteration comparison
    if 'total_iterations' in lm and 'total_iterations' in dog:
        lm_iter = lm['total_iterations']['mean']
        dog_iter = dog['total_iterations']['mean']
        diff = abs(lm_iter - dog_iter)
        fewer = "LM" if lm_iter < dog_iter else "Dogleg"
        print(f"Iterations: {fewer} uses {diff:.1f} fewer iterations on average")
    
    # Cost comparison
    if 'final_cost' in lm and 'final_cost' in dog:
        lm_cost = lm['final_cost']['mean']
        dog_cost = dog['final_cost']['mean']
        better = "LM" if lm_cost < dog_cost else "Dogleg"
        percent = abs((lm_cost - dog_cost) / max(lm_cost, dog_cost)) * 100
        print(f"Final Cost: {better} achieves {percent:.1f}% better cost")
    
    print()


def main():
    # Detect project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Parent of scripts/
    
    parser = argparse.ArgumentParser(description='Benchmark Bundle Adjustment solvers')
    parser.add_argument('--runs', type=int, default=10,
                       help='Number of runs per solver (default: 10)')
    parser.add_argument('--solvers', nargs='+', default=['lm', 'dogleg'],
                       choices=['lm', 'dogleg'],
                       help='Solvers to benchmark (default: both)')
    parser.add_argument('--ba_module', type=str, 
                       default=os.path.join(project_root, 'build', 'ba_module'),
                       help='Path to ba_module executable')
    parser.add_argument('--output_dir', type=str, 
                       default=os.path.join(project_root, 'output'),
                       help='Output directory for results')
    args = parser.parse_args()
    
    # Check if ba_module exists
    if not os.path.exists(args.ba_module):
        print(f"Error: ba_module not found at {args.ba_module}")
        print("Build it first with:")
        print(f"  cd {project_root}")
        print(f"  mkdir -p build && cd build")
        print(f"  cmake .. && make")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("BUNDLE ADJUSTMENT BENCHMARK")
    print(f"{'='*60}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Runs per solver: {args.runs}")
    print(f"Solvers: {', '.join(args.solvers)}")
    
    # Run benchmarks
    results = {}
    for solver in args.solvers:
        result = run_benchmark(solver, args.runs, args.ba_module, args.output_dir)
        if result:
            results[solver] = result
    
    if not results:
        print("\nError: No successful benchmark runs")
        sys.exit(1)
    
    # Print summary
    print_summary(results)
    
    # Compare solvers
    if len(results) > 1:
        compare_solvers(results)
    
    # Save results
    output_file = os.path.join(args.output_dir, 'benchmark_results.json')
    
    # Remove raw_runs from saved JSON (too verbose)
    results_to_save = {}
    for solver, data in results.items():
        results_to_save[solver] = {k: v for k, v in data.items() if k != 'raw_runs'}
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'num_runs': args.runs,
            'results': results_to_save
        }, f, indent=2)
    
    print(f"{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

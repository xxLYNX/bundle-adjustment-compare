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
            
            match = re.search(r'Total Solver Time: ([\d.]+) seconds', content)
            if match:
                metrics['total_solver_time'] = float(match.group(1))
            
            # Extract problem size
            match = re.search(r'Camera Poses: (\d+)', content)
            if match:
                metrics['num_cameras'] = int(match.group(1))
                
            match = re.search(r'3D Points: (\d+)', content)
            if match:
                metrics['num_points'] = int(match.group(1))
                
            match = re.search(r'Observations: (\d+)', content)
            if match:
                metrics['num_observations'] = int(match.group(1))
                
            match = re.search(r'Parameters: (\d+)', content)
            if match:
                metrics['num_parameters'] = int(match.group(1))
            
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
                
            # Extract cost reduction
            match = re.search(r'Cost Reduction: ([\d.e+-]+) \(([\d.]+)%\)', content)
            if match:
                metrics['cost_reduction'] = float(match.group(1))
                metrics['cost_reduction_percent'] = float(match.group(2))
            
            # Extract memory usage
            match = re.search(r'Peak Memory \(BA\): ([\d.]+) MB', content)
            if match:
                metrics['peak_memory_mb'] = float(match.group(1))
                
            match = re.search(r'Total Process Memory: ([\d.]+) MB', content)
            if match:
                metrics['total_memory_mb'] = float(match.group(1))
            
            # Extract accuracy
            match = re.search(r'RMS Reprojection Error: ([\d.]+) pixels', content)
            if match:
                metrics['rms_error'] = float(match.group(1))
                
            match = re.search(r'Average Error per Observation: ([\d.]+) pixels', content)
            if match:
                metrics['avg_error_per_obs'] = float(match.group(1))
            # Try with ² if plain format didn't match
            if 'avg_error_per_obs' not in metrics:
                match = re.search(r'Average Error per Observation: ([\d.]+)', content)
                if match:
                    metrics['avg_error_per_obs'] = float(match.group(1))
            
            # Extract performance
            match = re.search(r'Time per Iteration: ([\d.]+) ms', content)
            if match:
                metrics['time_per_iter'] = float(match.group(1))
                
            match = re.search(r'Iterations per Second: ([\d.]+)', content)
            if match:
                metrics['iter_per_sec'] = float(match.group(1))
            
            # Extract termination reason
            match = re.search(r'Termination: (.+?)(?:\n|$)', content)
            if match:
                metrics['termination'] = match.group(1).strip()
                
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
    
    # Aggregate statistics for all metrics
    metrics_keys = [
        'ba_time', 'total_solver_time', 'total_iterations', 
        'successful_steps', 'unsuccessful_steps', 
        'initial_cost', 'final_cost', 'cost_reduction', 'cost_reduction_percent',
        'peak_memory_mb', 'total_memory_mb',
        'rms_error', 'avg_error_per_obs',
        'time_per_iter', 'iter_per_sec'
    ]
    
    aggregated = {}
    for key in metrics_keys:
        values = [run[key] for run in all_runs if key in run]
        if values:
            aggregated[key] = compute_statistics(values)
    
    # Add problem size (constant across runs)
    if all_runs:
        for key in ['num_cameras', 'num_points', 'num_observations', 'num_parameters']:
            if key in all_runs[0]:
                aggregated[key] = all_runs[0][key]
        
        # Add termination reason (should be same for all runs)
        if 'termination' in all_runs[0]:
            aggregated['termination'] = all_runs[0]['termination']
    
    # Calculate success rate
    if 'successful_steps' in aggregated and 'total_iterations' in aggregated:
        success_rate = (aggregated['successful_steps']['mean'] / 
                       aggregated['total_iterations']['mean']) * 100
        aggregated['success_rate'] = success_rate
    
    # Add raw data
    aggregated['raw_runs'] = all_runs
    aggregated['solver'] = solver
    aggregated['num_runs'] = len(all_runs)
    
    return aggregated


def print_summary(results):
    """Print comprehensive summary of benchmark results."""
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}\n")
    
    for solver_name, data in results.items():
        if not data:
            continue
            
        print(f"{solver_name.upper()} Solver ({data['num_runs']} runs):")
        print(f"{'─'*60}")
        
        # Problem size
        print(f"\nProblem Size:")
        if 'num_cameras' in data:
            print(f"  Cameras:       {data['num_cameras']}")
        if 'num_points' in data:
            print(f"  3D Points:     {data['num_points']}")
        if 'num_observations' in data:
            print(f"  Observations:  {data['num_observations']}")
        if 'num_parameters' in data:
            print(f"  Parameters:    {data['num_parameters']}")
        
        # Timing
        print(f"\nTiming:")
        if 'ba_time' in data:
            bt = data['ba_time']
            print(f"  BA Time:       {bt['mean']:.3f} ± {bt['std']:.3f} s  "
                  f"[{bt['min']:.3f}, {bt['max']:.3f}]")
        if 'time_per_iter' in data:
            tpi = data['time_per_iter']
            print(f"  Time/Iter:     {tpi['mean']:.2f} ± {tpi['std']:.2f} ms")
        if 'iter_per_sec' in data:
            ips = data['iter_per_sec']
            print(f"  Iter/Sec:      {ips['mean']:.2f}")
        
        # Convergence
        print(f"\nConvergence:")
        if 'total_iterations' in data:
            it = data['total_iterations']
            print(f"  Total Iters:   {it['mean']:.1f} ± {it['std']:.1f}  "
                  f"[{int(it['min'])}, {int(it['max'])}]")
        if 'successful_steps' in data:
            ss = data['successful_steps']
            print(f"  Successful:    {ss['mean']:.1f} ± {ss['std']:.1f}")
        if 'unsuccessful_steps' in data:
            us = data['unsuccessful_steps']
            print(f"  Unsuccessful:  {us['mean']:.1f} ± {us['std']:.1f}")
        if 'success_rate' in data:
            print(f"  Success Rate:  {data['success_rate']:.1f}%")
        if 'termination' in data:
            print(f"  Termination:   {data['termination']}")
        
        # Cost function
        print(f"\nCost Function:")
        if 'initial_cost' in data:
            ic = data['initial_cost']
            print(f"  Initial Cost:  {ic['mean']:.2e}")
        if 'final_cost' in data:
            fc = data['final_cost']
            print(f"  Final Cost:    {fc['mean']:.2e} ± {fc['std']:.2e}  "
                  f"[{fc['min']:.2e}, {fc['max']:.2e}]")
        if 'cost_reduction_percent' in data:
            cr = data['cost_reduction_percent']
            print(f"  Cost Reduction: {cr['mean']:.2f}%")
        
        # Accuracy
        print(f"\nAccuracy:")
        if 'rms_error' in data:
            rms = data['rms_error']
            print(f"  RMS Error:     {rms['mean']:.3f} ± {rms['std']:.3f} px  "
                  f"[{rms['min']:.3f}, {rms['max']:.3f}]")
        if 'avg_error_per_obs' in data:
            avg = data['avg_error_per_obs']
            print(f"  Avg Error/Obs: {avg['mean']:.3f} ± {avg['std']:.3f} px²")
        
        # Memory
        print(f"\nMemory Usage:")
        if 'peak_memory_mb' in data:
            pm = data['peak_memory_mb']
            print(f"  Peak (BA):     {pm['mean']:.2f} MB")
        if 'total_memory_mb' in data:
            tm = data['total_memory_mb']
            print(f"  Total Process: {tm['mean']:.2f} ± {tm['std']:.2f} MB  "
                  f"[{tm['min']:.2f}, {tm['max']:.2f}]")
        
        print()


def compare_solvers(results):
    """Comprehensive comparison between two solvers."""
    if 'lm' not in results or 'dogleg' not in results:
        return
    
    print(f"{'='*60}")
    print("SOLVER COMPARISON")
    print(f"{'='*60}\n")
    
    lm = results['lm']
    dog = results['dogleg']
    
    # Performance comparison
    print("Performance:")
    if 'ba_time' in lm and 'ba_time' in dog:
        lm_time = lm['ba_time']['mean']
        dog_time = dog['ba_time']['mean']
        speedup = dog_time / lm_time
        faster = "LM" if speedup > 1 else "Dogleg"
        percent = abs(speedup - 1) * 100
        print(f"  Speed:         {faster} is {percent:.1f}% faster ({lm_time:.2f}s vs {dog_time:.2f}s)")
    
    if 'time_per_iter' in lm and 'time_per_iter' in dog:
        lm_tpi = lm['time_per_iter']['mean']
        dog_tpi = dog['time_per_iter']['mean']
        faster = "LM" if lm_tpi < dog_tpi else "Dogleg"
        percent = abs((lm_tpi - dog_tpi) / max(lm_tpi, dog_tpi)) * 100
        print(f"  Time/Iter:     {faster} is {percent:.1f}% faster ({lm_tpi:.2f}ms vs {dog_tpi:.2f}ms)")
    
    # Convergence comparison
    print(f"\nConvergence:")
    if 'total_iterations' in lm and 'total_iterations' in dog:
        lm_iter = lm['total_iterations']['mean']
        dog_iter = dog['total_iterations']['mean']
        diff = abs(lm_iter - dog_iter)
        fewer = "LM" if lm_iter < dog_iter else "Dogleg"
        percent = (diff / max(lm_iter, dog_iter)) * 100
        print(f"  Iterations:    {fewer} uses {diff:.0f} ({percent:.1f}%) fewer ({int(lm_iter)} vs {int(dog_iter)})")
    
    if 'success_rate' in lm and 'success_rate' in dog:
        lm_sr = lm['success_rate']
        dog_sr = dog['success_rate']
        better = "LM" if lm_sr > dog_sr else "Dogleg"
        print(f"  Success Rate:  {better} accepts more steps ({lm_sr:.1f}% vs {dog_sr:.1f}%)")
    
    # Quality comparison
    print(f"\nSolution Quality:")
    if 'final_cost' in lm and 'final_cost' in dog:
        lm_cost = lm['final_cost']['mean']
        dog_cost = dog['final_cost']['mean']
        better = "LM" if lm_cost < dog_cost else "Dogleg"
        percent = abs((lm_cost - dog_cost) / max(lm_cost, dog_cost)) * 100
        print(f"  Final Cost:    {better} achieves {percent:.1f}% better ({lm_cost:.2e} vs {dog_cost:.2e})")
    
    if 'rms_error' in lm and 'rms_error' in dog:
        lm_rms = lm['rms_error']['mean']
        dog_rms = dog['rms_error']['mean']
        better = "LM" if lm_rms < dog_rms else "Dogleg"
        percent = abs((lm_rms - dog_rms) / max(lm_rms, dog_rms)) * 100
        print(f"  RMS Error:     {better} is {percent:.1f}% better ({lm_rms:.3f}px vs {dog_rms:.3f}px)")
    
    # Resource comparison
    print(f"\nResource Usage:")
    if 'total_memory_mb' in lm and 'total_memory_mb' in dog:
        lm_mem = lm['total_memory_mb']['mean']
        dog_mem = dog['total_memory_mb']['mean']
        efficient = "LM" if lm_mem < dog_mem else "Dogleg"
        ratio = max(lm_mem, dog_mem) / min(lm_mem, dog_mem)
        print(f"  Memory:        {efficient} uses {ratio:.1f}x less ({lm_mem:.1f}MB vs {dog_mem:.1f}MB)")
    
    # Overall recommendation
    print(f"\nOverall Assessment:")
    
    # Count wins
    lm_wins = 0
    dog_wins = 0
    
    if 'ba_time' in lm and 'ba_time' in dog:
        if lm['ba_time']['mean'] < dog['ba_time']['mean']:
            lm_wins += 1
        else:
            dog_wins += 1
    
    if 'final_cost' in lm and 'final_cost' in dog:
        if lm['final_cost']['mean'] < dog['final_cost']['mean']:
            lm_wins += 2  # Weight quality higher
        else:
            dog_wins += 2
    
    if 'total_iterations' in lm and 'total_iterations' in dog:
        if lm['total_iterations']['mean'] < dog['total_iterations']['mean']:
            lm_wins += 1
        else:
            dog_wins += 1
    
    winner = "Levenberg-Marquardt" if lm_wins > dog_wins else "Dogleg"
    print(f"  Recommended:   {winner} for this problem configuration")
    
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

#!/usr/bin/env python3
"""
Brute Force Bivariate Sparse Square Search over F_3
=====================================================
For degree d, enumerates ALL complete bivariate polynomials
of total degree <= d over F_3 and checks if f² is sparser.

SPLITTING STRATEGY (2048 processes per degree):
  All polynomials are indexed 0 .. 3^n - 1  (base-3 encoding)
  Process p handles indices:
      start = p * chunk_size
      end   = (p+1) * chunk_size   (exclusive)
  where chunk_size = 3^n // 2048

USAGE (single process):
  python3 brute_force_parallel.py --degree 4 --process_id 0

USAGE (launch all 2048 processes for degree 4):
  for i in $(seq 0 2047); do
    python3 brute_force_parallel.py --degree 4 --process_id $i > logs/deg4_proc$i.log 2>&1 &
  done
  wait
  echo "Degree 4 done"

USAGE (loop over degrees, 2048 processes each):
  for d in $(seq 2 10); do
    mkdir -p logs/deg$d results/deg$d
    for i in $(seq 0 2047); do
      python3 brute_force_parallel.py --degree $d --process_id $i \
          > logs/deg$d/proc$i.log 2>&1 &
    done
    wait
    echo "Degree $d complete"
  done

MONOMIAL COUNT per degree:
  deg 2  →   6 monomials →  3^6  =       729 polynomials  (tiny)
  deg 3  →  10 monomials →  3^10 =    59,049 polynomials  (fast)
  deg 4  →  15 monomials →  3^15 = 14,348,907 polynomials (minutes)
  deg 5  →  21 monomials →  3^21 ≈     10 billion         (hours)
  deg 6+ → grows as 3^((d+1)(d+2)/2) — quickly infeasible

Output files per process:
  results/deg{d}/solutions_proc{p}.txt  — all f with S2 < S1
  results/deg{d}/summary_proc{p}.txt    — count checked, hits found
"""

import argparse
import math
import os
import time


# ─────────────────────────────────────────────────────────────────────
#  F_3 ARITHMETIC
# ─────────────────────────────────────────────────────────────────────

def poly_square_f3(a):
    """
    Compute f² over F_3.
    f = dict {(i,j): c}  where c in {1,2}
    Returns dict of same form.

    f² = sum_k c_k² x^{2ik} y^{2jk}          [diagonal]
       + sum_{k<l} 2 c_k c_l x^{ik+il} y^{jk+jl}  [cross]
    All mod 3.
    """
    result = {}
    items  = list(a.items())
    n      = len(items)

    # Diagonal terms: c^2 mod 3. Note 1^2=1, 2^2=4=1 mod 3 → all = 1
    for (i, j), c in items:
        key = (2*i, 2*j)
        val = (result.get(key, 0) + c * c) % 3
        if val:
            result[key] = val
        elif key in result:
            del result[key]

    # Cross terms: 2 * c1 * c2 mod 3
    for idx in range(n):
        (i1, j1), c1 = items[idx]
        for jdx in range(idx + 1, n):
            (i2, j2), c2 = items[jdx]
            key = (i1 + i2, j1 + j2)
            val = (result.get(key, 0) + 2 * c1 * c2) % 3
            if val:
                result[key] = val
            elif key in result:
                del result[key]

    return result


def is_genuinely_bivariate(poly):
    """
    Check that f truly depends on both x and y.
    Returns False if f is constant in x or y or disguised univariate.
    """
    if not poly or len(poly) < 2:
        return False
    keys = list(poly.keys())
    xs = {i for i, j in keys}
    ys = {j for i, j in keys}
    if len(xs) < 2 or len(ys) < 2:
        return False

    # Check not secretly univariate in x^a * y^b
    i0, j0 = keys[0]
    diffs = [(i - i0, j - j0) for i, j in keys[1:]]
    all_di = [abs(di) for di, dj in diffs if di != 0]
    all_dj = [abs(dj) for di, dj in diffs if dj != 0]
    if not all_di or not all_dj:
        return False

    gx = all_di[0]
    for v in all_di[1:]:
        gx = math.gcd(gx, v)
    gy = all_dj[0]
    for v in all_dj[1:]:
        gy = math.gcd(gy, v)

    for di, dj in diffs:
        if di % gx != 0:
            return True
        k = di // gx
        if dj != k * gy:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────
#  MONOMIAL ENUMERATION
# ─────────────────────────────────────────────────────────────────────

def get_monomials(degree):
    """
    Return sorted list of all monomials (i, j) with i >= 0, j >= 0, i+j <= degree.
    Sorted by total degree then lex order.
    Count = (degree+1)(degree+2)/2
    
    Example degree=2:
        (0,0), (1,0), (0,1), (2,0), (1,1), (0,2)
    """
    monomials = []
    for total in range(degree + 1):         # total degree 0, 1, ..., d
        for i in range(total + 1):          # i from 0 to total
            j = total - i
            monomials.append((i, j))
    return monomials


# ─────────────────────────────────────────────────────────────────────
#  INDEX ↔ POLYNOMIAL CONVERSION
# ─────────────────────────────────────────────────────────────────────

def index_to_coeffs(idx, n_monomials):
    """
    Convert integer index (base 10) to coefficient vector (base 3).
    Returns list of n_monomials integers, each in {0, 1, 2}.

    Index 0       → [0, 0, 0, ..., 0]   (zero polynomial)
    Index 1       → [1, 0, 0, ..., 0]
    Index 2       → [2, 0, 0, ..., 0]
    Index 3       → [0, 1, 0, ..., 0]
    ...

    This is just writing idx in base 3 with n_monomials digits.
    """
    coeffs = []
    for _ in range(n_monomials):
        coeffs.append(idx % 3)
        idx //= 3
    return coeffs   # coeffs[k] is the coefficient for monomials[k]


def coeffs_to_poly(coeffs, monomials):
    """
    Convert coefficient list to polynomial dict.
    Only includes nonzero coefficients.
    """
    poly = {}
    for k, c in enumerate(coeffs):
        if c != 0:
            poly[monomials[k]] = c
    return poly


def poly_to_str(poly):
    """Pretty print polynomial."""
    if not poly:
        return "0"
    terms = sorted(poly.items(), key=lambda kv: (kv[0][0]+kv[0][1], kv[0][0]))
    parts = []
    for (i, j), c in terms:
        coef  = "" if c == 1 else "2*"
        xp    = "" if i == 0 else ("x" if i == 1 else f"x^{i}")
        yp    = "" if j == 0 else ("y" if j == 1 else f"y^{j}")
        sep   = "*" if xp and yp else ""
        if not xp and not yp:
            parts.append(str(c))
        else:
            parts.append(f"{coef}{xp}{sep}{yp}")
    return " + ".join(parts)


def metrics_str(s1, s2):
    r  = s2 / s1
    lr = math.log(s2) / math.log(s1) if s1 > 1 and s2 > 0 else float('nan')
    return f"S1={s1}  S2={s2}  S2/S1={r:.4f}  log_S1(S2)={lr:.4f}"


# ─────────────────────────────────────────────────────────────────────
#  PROCESS RANGE CALCULATION
# ─────────────────────────────────────────────────────────────────────

def get_process_range(process_id, total_processes, total_polynomials):
    """
    Divide total_polynomials among total_processes.
    Returns (start_idx, end_idx) for this process (end is exclusive).

    Process 0    : [0,          chunk)
    Process 1    : [chunk,      2*chunk)
    ...
    Process 2047 : [2047*chunk, total_polynomials)   ← last gets remainder
    """
    chunk = total_polynomials // total_processes
    start = process_id * chunk
    if process_id == total_processes - 1:
        end = total_polynomials          # last process takes the remainder
    else:
        end = start + chunk
    return start, end


# ─────────────────────────────────────────────────────────────────────
#  MAIN BRUTE FORCE LOOP
# ─────────────────────────────────────────────────────────────────────

def run_brute_force(degree, process_id, total_processes, output_dir):

    monomials   = get_monomials(degree)
    n_monomials = len(monomials)
    total_polys = 3 ** n_monomials         # includes zero polynomial

    start_idx, end_idx = get_process_range(process_id, total_processes, total_polys)
    my_count = end_idx - start_idx

    os.makedirs(output_dir, exist_ok=True)
    sol_path = os.path.join(output_dir, f"solutions_proc{process_id}.txt")
    sum_path = os.path.join(output_dir, f"summary_proc{process_id}.txt")

    # ── Print startup info ────────────────────────────────────────────
    print("=" * 65)
    print(f"  BRUTE FORCE — Degree {degree}  |  Process {process_id}/{total_processes}")
    print("=" * 65)
    print(f"  Monomials     : {n_monomials}  →  {monomials}")
    print(f"  Total polys   : 3^{n_monomials} = {total_polys:,}")
    print(f"  My range      : [{start_idx:,} , {end_idx:,})  →  {my_count:,} polynomials")
    print(f"  Output dir    : {output_dir}")
    print(f"  Started       : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # ── Iterate ───────────────────────────────────────────────────────
    hits        = 0
    checked     = 0
    t0          = time.time()
    LOG_EVERY   = max(1, my_count // 20)    # print progress 20 times

    with open(sol_path, 'w') as sol_f:
        sol_f.write(f"# Degree={degree}  Process={process_id}  Range=[{start_idx},{end_idx})\n\n")

        for idx in range(start_idx, end_idx):

            # Skip zero polynomial
            if idx == 0:
                checked += 1
                continue

            coeffs = index_to_coeffs(idx, n_monomials)
            poly   = coeffs_to_poly(coeffs, monomials)

            # Must have at least 2 terms
            if len(poly) < 2:
                checked += 1
                continue

            # Must be genuinely bivariate
            if not is_genuinely_bivariate(poly):
                checked += 1
                continue

            # Compute f²
            sq = poly_square_f3(poly)
            s1 = len(poly)
            s2 = len(sq)

            # Check sparse square condition
            if s2 < s1:
                hits += 1
                line = (f"*** SPARSE SQUARE ***\n"
                        f"  idx     = {idx}\n"
                        f"  f       = {poly_to_str(poly)}\n"
                        f"  f²      = {poly_to_str(sq)}\n"
                        f"  {metrics_str(s1, s2)}\n\n")
                sol_f.write(line)
                print(line, end="")

            checked += 1

            # Progress log
            if checked % LOG_EVERY == 0:
                elapsed = time.time() - t0
                pct     = 100 * checked / my_count
                rate    = checked / elapsed if elapsed > 0 else 0
                eta     = (my_count - checked) / rate if rate > 0 else 0
                print(f"  [{process_id}]  {pct:5.1f}%  checked={checked:,}  "
                      f"hits={hits}  rate={rate:.0f}/s  ETA={eta:.0f}s")

    elapsed = time.time() - t0

    # ── Write summary ─────────────────────────────────────────────────
    with open(sum_path, 'w') as sf:
        sf.write(f"degree={degree}\n")
        sf.write(f"process_id={process_id}\n")
        sf.write(f"total_processes={total_processes}\n")
        sf.write(f"start_idx={start_idx}\n")
        sf.write(f"end_idx={end_idx}\n")
        sf.write(f"checked={checked}\n")
        sf.write(f"hits={hits}\n")
        sf.write(f"elapsed_sec={elapsed:.2f}\n")

    print(f"\n  [{process_id}] DONE  checked={checked:,}  hits={hits}  "
          f"time={elapsed:.1f}s")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────
#  COLLECT RESULTS (run after all processes finish)
# ─────────────────────────────────────────────────────────────────────

def collect_results(degree, total_processes, base_dir):
    """
    Merge all solution files for a degree into one final file.
    Call this after all 2048 processes have finished.
    """
    output_dir = os.path.join(base_dir, f"deg{degree}")
    final_path = os.path.join(base_dir, f"degree_{degree}_all_solutions.txt")

    total_hits    = 0
    total_checked = 0

    with open(final_path, 'w') as out:
        out.write(f"# All sparse squares — degree {degree}\n\n")

        for p in range(total_processes):
            sol_path = os.path.join(output_dir, f"solutions_proc{p}.txt")
            sum_path = os.path.join(output_dir, f"summary_proc{p}.txt")

            # Read summary
            if os.path.exists(sum_path):
                with open(sum_path) as sf:
                    for line in sf:
                        if line.startswith("checked="):
                            total_checked += int(line.split("=")[1])
                        elif line.startswith("hits="):
                            total_hits += int(line.split("=")[1])

            # Copy solutions
            if os.path.exists(sol_path):
                with open(sol_path) as sf:
                    content = sf.read().strip()
                    if "SPARSE SQUARE" in content:
                        out.write(content + "\n")

    print(f"\n  Degree {degree} — MERGED")
    print(f"  Total checked : {total_checked:,}")
    print(f"  Total hits    : {total_hits}")
    print(f"  Output        : {final_path}")


# ─────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Brute force sparse bivariate squares over F_3 — parallel version"
    )
    parser.add_argument("--degree",           type=int, required=True,
                        help="Total degree of polynomials to check")
    parser.add_argument("--process_id",       type=int, default=0,
                        help="This process ID (0 to total_processes-1)")
    parser.add_argument("--total_processes",  type=int, default=2048,
                        help="Total number of parallel processes")
    parser.add_argument("--output_dir",       type=str, default=None,
                        help="Output directory (default: results/deg{d})")
    parser.add_argument("--collect",          action="store_true",
                        help="Merge results after all processes finish")
    parser.add_argument("--base_dir",         type=str, default="results",
                        help="Base directory for collect mode")

    args = parser.parse_args()

    if args.collect:
        collect_results(args.degree, args.total_processes, args.base_dir)
    else:
        out_dir = args.output_dir or os.path.join("results", f"deg{args.degree}")

        # Sanity check
        n_mono = len(get_monomials(args.degree))
        total  = 3 ** n_mono
        print(f"\n  Degree {args.degree}: {n_mono} monomials → "
              f"3^{n_mono} = {total:,} total polynomials")
        print(f"  Each process handles ≈ {total // args.total_processes:,} polynomials\n")

        run_brute_force(
            degree          = args.degree,
            process_id      = args.process_id,
            total_processes = args.total_processes,
            output_dir      = out_dir,
        )

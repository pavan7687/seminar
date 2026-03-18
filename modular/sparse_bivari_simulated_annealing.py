#!/usr/bin/env python3
"""
Q2 — Bivariate Sparse Square Search over F_3
==============================================
Question: Does there exist a SPARSE (incomplete) bivariate polynomial
f(x,y) over F_3 such that f^2 is even sparser?  i.e.  S2 < S1
where  S1 = |f| = number of nonzero terms in f
       S2 = |f^2| = number of nonzero terms in f^2

Why sparse/incomplete polynomials?
  - Complete polynomials are already dense, squaring keeps them dense.
  - Sparse polynomials can have massive cancellations mod 3 when squared.
  - This is the interesting and open case.

Two metrics tracked (from the whiteboard):
    S2/S1          — direct ratio,  want < 1
    log_{S1}(S2)   — log ratio,     want < 1

Strategy: Simulated Annealing with multiple random restarts per degree.
  Each degree runs RESTARTS independent SA chains and keeps the global best.
  This covers the search space much better than a single long chain.

ZERO external dependencies — runs with plain  python3  (no pip needed).

Usage on IIT-B server:
    nohup python3 sparse_sa_overnight.py > run.log 2>&1 &
    tail -f run.log
    grep "SPARSE SQUARE" run.log      # watch for hits

Output files (written continuously, safe to kill and resume):
    results_summary.txt  — full log
    sparse_solutions.txt — confirmed hits where S2 < S1
    best_ratios.txt      — one row per degree showing best S2/S1
    checkpoint.txt       — last completed degree for auto-resume
"""

import random
import math
import time
import os
import gc

# =============================================================================
#  CONFIGURATION — only edit this block
# =============================================================================
START_DEG   = 10          # first degree to search
END_DEG     = 100         # last  degree to search (inclusive)

# Each degree runs RESTARTS independent SA chains of ITER_PER_RESTART steps.
# Total iterations per degree = RESTARTS * ITER_PER_RESTART
RESTARTS          = 10          # independent random restarts per degree
ITER_PER_RESTART  = 500_000     # SA steps per restart  (total = 5 million)

T_START     = 5.0         # initial SA temperature
T_END       = 0.005       # final   SA temperature
SEED_BASE   = 1234        # seed for restart r at degree d = SEED_BASE + d*100 + r

MIN_TERMS   = 6           # min nonzero terms  (need enough for cancellations)
MAX_TERMS   = 60          # max nonzero terms  (cap prevents O(n^2) blowup)
LOG_EVERY   = 100_000     # print a line every this many iterations

SUMMARY_FILE   = "results_summary.txt"
SOLUTIONS_FILE = "sparse_solutions.txt"
RATIOS_FILE    = "best_ratios.txt"
CHECKPOINT     = "checkpoint.txt"
# =============================================================================


# -----------------------------------------------------------------------------
#  F_3 polynomial arithmetic
#  A polynomial is a dict  {(i, j): c}  with c in {1, 2}.
#  Zero polynomial = empty dict.  Missing keys are implicitly 0.
# -----------------------------------------------------------------------------

def poly_square(a):
    """
    Compute f^2 over F_3 using the identity:
        (sum c_k x^ik y^jk)^2
        = sum c_k^2 x^{2ik} y^{2jk}          [diagonal terms]
        + sum_{k<l} 2 c_k c_l x^{ik+il} y^{jk+jl}  [cross terms]
    All coefficients reduced mod 3.
    O(|a|^2 / 2) — fast as long as |a| <= MAX_TERMS.
    """
    result = {}
    items  = list(a.items())
    n      = len(items)

    # diagonal: c^2 mod 3.  Note 1^2=1, 2^2=4=1 mod 3, so all diagonal coeffs = 1
    for (i, j), c in items:
        key = (2*i, 2*j)
        val = (result.get(key, 0) + c * c) % 3
        if val:
            result[key] = val
        elif key in result:
            del result[key]

    # cross terms: coefficient is 2*c_k*c_l mod 3
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
    f is TRULY bivariate iff it cannot be written as x^r * y^s * g(x^a * y^b)
    for any a,b (i.e. it is not univariate in any single monomial substitution).

    This rejects ALL disguised univariates:
      - y^14 * g(x)       : step = (1,0), same y-exponent
      - x^5  * g(y)       : step = (0,1), same x-exponent
      - x^4  * g(x^3*y^2) : step = (3,2), the degree-19 fake hit
      - any lattice line  : step = (a,b) for any a,b > 0

    Algorithm: support points lie on a single arithmetic line
      (i0,j0), (i0+gx,j0+gy), (i0+2gx,j0+2gy), ...
    iff every difference from the first point equals k*(gx,gy).
    If so, f is univariate in x^gx * y^gy — reject it.
    """
    if not poly or len(poly) < 2:
        return False

    keys = list(poly.keys())
    i0, j0 = keys[0]
    diffs = [(i - i0, j - j0) for i, j in keys[1:]]

    # Fast-path axis checks
    xs = {i for i, j in keys}
    ys = {j for i, j in keys}
    if len(xs) < 2 or len(ys) < 2:
        return False

    # GCD of all nonzero x-differences and y-differences
    all_di = [abs(di) for di, dj in diffs if di != 0]
    all_dj = [abs(dj) for di, dj in diffs if dj != 0]

    if not all_di or not all_dj:
        return False   # all on one axis line

    gx = all_di[0]
    for v in all_di[1:]:
        gx = math.gcd(gx, v)
    gy = all_dj[0]
    for v in all_dj[1:]:
        gy = math.gcd(gy, v)

    # Check: is every point on the line i0+k*gx, j0+k*gy ?
    for di, dj in diffs:
        if di % gx != 0:
            return True          # off the lattice line -> truly bivariate
        k = di // gx
        if dj != k * gy:
            return True          # y-coord doesn't match -> truly bivariate

    return False  # all on one arithmetic line -> univariate in x^gx * y^gy


# -----------------------------------------------------------------------------
#  SA move operators
# -----------------------------------------------------------------------------

def random_sparse_bivariate(max_deg):
    """
    Sample a random SPARSE bivariate polynomial:
      - monomials drawn from the full grid  {0..max_deg} x {0..max_deg}
      - number of terms between MIN_TERMS and MAX_TERMS
      - guaranteed to be genuinely bivariate (depends on both x and y)
    """
    all_keys = [(i, j) for i in range(max_deg + 1)
                        for j in range(max_deg + 1)]
    cap = min(MAX_TERMS, max(MIN_TERMS + 1, len(all_keys) // 4))
    n   = random.randint(MIN_TERMS, cap)
    while True:
        chosen = random.sample(all_keys, min(n, len(all_keys)))
        poly   = {k: random.choice([1, 2]) for k in chosen}
        if is_genuinely_bivariate(poly):
            return poly


def neighbor(poly, max_deg):
    """
    Perturb poly by one random move:
      flip   (50%) — change one coefficient:  1 <-> 2
      add    (25%) — insert one new monomial
      remove (25%) — delete one monomial  (never go below MIN_TERMS)

    Any move that makes the polynomial non-bivariate is silently rejected
    and the original is returned unchanged.
    """
    d    = dict(poly)
    move = random.choice(('flip', 'flip', 'add', 'remove'))

    if move == 'flip':
        key    = random.choice(list(d.keys()))
        d[key] = 1 if d[key] == 2 else 2

    elif move == 'add' and len(d) < MAX_TERMS:
        all_keys = [(i, j) for i in range(max_deg + 1)
                            for j in range(max_deg + 1)]
        missing  = [k for k in all_keys if k not in d]
        if missing:
            k    = random.choice(missing)
            d[k] = random.choice([1, 2])

    elif move == 'remove' and len(d) > MIN_TERMS:
        del d[random.choice(list(d.keys()))]

    return d if is_genuinely_bivariate(d) else poly


# -----------------------------------------------------------------------------
#  Metrics
# -----------------------------------------------------------------------------

def ratio(s1, s2):
    """S2 / S1 — want < 1"""
    return s2 / s1 if s1 else float('nan')


def log_ratio(s1, s2):
    """log_{S1}(S2) = log(S2)/log(S1) — want < 1"""
    if s1 <= 1 or s2 <= 0:
        return float('nan')
    return math.log(s2) / math.log(s1)


def metrics_str(s1, s2):
    r  = ratio(s1, s2)
    lr = log_ratio(s1, s2)
    rs  = f"{r:.4f}"  if not math.isnan(r)  else " nan "
    lrs = f"{lr:.4f}" if not math.isnan(lr) else " nan "
    return (f"S1={s1:>4}  S2={s2:>4}  "
            f"S2/S1={rs}  log_{{S1}}(S2)={lrs}")


# -----------------------------------------------------------------------------
#  Polynomial pretty-printer
# -----------------------------------------------------------------------------

def poly_str(poly, max_show=15):
    if not poly:
        return "0"
    terms = sorted(poly.items(), key=lambda kv: (-kv[0][0], -kv[0][1]))
    parts = []
    for (i, j), c in terms[:max_show]:
        coef  = "" if c == 1 else "2*"
        xpart = ("" if i == 0 else ("x" if i == 1 else f"x^{i}"))
        ypart = ("" if j == 0 else ("y" if j == 1 else f"y^{j}"))
        sep   = "*" if xpart and ypart else ""
        if not xpart and not ypart:
            parts.append(str(c))
        else:
            parts.append(f"{coef}{xpart}{sep}{ypart}")
    tail = f" + ...({len(poly)} terms total)" if len(poly) > max_show else ""
    return " + ".join(parts) + tail


# -----------------------------------------------------------------------------
#  File helpers
# -----------------------------------------------------------------------------

def fappend(path, text):
    with open(path, 'a') as fh:
        fh.write(text + '\n')


def log_solution(deg, f, sq, s1, s2):
    block = (f"\n{'='*65}\n"
             f"  *** SPARSE SQUARE FOUND — degree {deg} ***\n"
             f"  {metrics_str(s1, s2)}\n"
             f"  f(x,y)   = {poly_str(f)}\n"
             f"  f(x,y)^2 = {poly_str(sq)}\n"
             f"{'='*65}")
    fappend(SOLUTIONS_FILE, block)


def log_ratio_row(deg, s1, s2, hits):
    r  = ratio(s1, s2)
    lr = log_ratio(s1, s2)
    fappend(RATIOS_FILE,
            f"deg={deg:>3}  S1={s1:>4}  S2={s2:>4}  "
            f"S2/S1={r:.4f}  log_{{S1}}(S2)={lr:.4f}  "
            f"sparse_hits={hits}")


def resume_from():
    if os.path.exists(CHECKPOINT):
        val = open(CHECKPOINT).read().strip()
        if val.isdigit():
            return int(val) + 1
    return START_DEG


# -----------------------------------------------------------------------------
#  Single SA chain
# -----------------------------------------------------------------------------

def sa_chain(max_deg, n_iter, T_start, T_end, seed):
    """
    Run one SA chain of n_iter steps.
    Returns (best_poly, best_sq, best_s1, best_s2, sparse_dict)
    where sparse_dict maps poly-key -> (f, sq, s1, s2) for all hits with s2 < s1.
    """
    random.seed(seed)

    f      = random_sparse_bivariate(max_deg)
    sq     = poly_square(f)
    s1, s2 = len(f), len(sq)
    score  = s2 - s1

    best_f, best_sq  = f, sq
    best_s1, best_s2 = s1, s2
    best_score       = score

    sparse_found = {}   # deduplicated on the fly

    T       = T_start
    cooling = (T_end / T_start) ** (1.0 / n_iter)

    for it in range(n_iter):
        f_new     = neighbor(f, max_deg)
        sq_new    = poly_square(f_new)
        s1_new    = len(f_new)
        s2_new    = len(sq_new)
        new_score = s2_new - s1_new

        delta = new_score - score
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
            f, sq, s1, s2, score = f_new, sq_new, s1_new, s2_new, new_score

        if score < best_score:
            best_score           = score
            best_f, best_sq      = f, sq
            best_s1, best_s2     = s1, s2

        if new_score < 0:
            key = str(sorted(f_new.items()))
            if key not in sparse_found:
                sparse_found[key] = (f_new, sq_new, s1_new, s2_new)

        T *= cooling

    return best_f, best_sq, best_s1, best_s2, sparse_found


# -----------------------------------------------------------------------------
#  One degree: run RESTARTS independent chains, collect global best
# -----------------------------------------------------------------------------

def run_degree(degree):
    header = ("\n" + "="*65 + "\n"
              f"  Q2. Sparse bivariate square — degree {degree}\n"
              f"  Searching incomplete (sparse) polynomials over F_3\n"
              f"  Monomials: full grid {{0..{degree}}} x {{0..{degree}}}\n"
              f"  Terms per poly: {MIN_TERMS} .. {MAX_TERMS}   "
              f"(sparse, not complete)\n"
              f"  Restarts: {RESTARTS}  x  {ITER_PER_RESTART:,} iters  "
              f"= {RESTARTS*ITER_PER_RESTART:,} total\n"
              f"  Metrics: S2/S1  and  log_{{S1}}(S2)   (want < 1)\n"
              f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
              + "="*65)
    print(header, flush=True)
    fappend(SUMMARY_FILE, header)

    global_best_f  = None
    global_best_sq = None
    global_best_s1 = 10**9
    global_best_s2 = 10**9
    global_best_score = 10**9
    all_sparse = {}   # merged across all restarts

    t_degree = time.time()

    for r in range(RESTARTS):
        seed = SEED_BASE + degree * 100 + r
        t0   = time.time()

        bf, bsq, bs1, bs2, found = sa_chain(
            max_deg  = degree,
            n_iter   = ITER_PER_RESTART,
            T_start  = T_START,
            T_end    = T_END,
            seed     = seed,
        )
        elapsed = time.time() - t0
        sc      = bs2 - bs1

        # merge sparse finds
        all_sparse.update(found)

        # update global best
        if sc < global_best_score:
            global_best_score = sc
            global_best_f     = bf
            global_best_sq    = bsq
            global_best_s1    = bs1
            global_best_s2    = bs2

        hit_marker = "  *** HIT ***" if found else ""
        line = (f"  restart {r+1:>2}/{RESTARTS}  seed={seed}  "
                f"{metrics_str(bs1, bs2)}  "
                f"score={sc:>4}  t={elapsed:.0f}s"
                f"{hit_marker}")
        print(line, flush=True)
        fappend(SUMMARY_FILE, line)

    # sort all sparse finds by S2
    unique = sorted(all_sparse.values(), key=lambda t: t[3])

    elapsed_deg = time.time() - t_degree
    footer = ("\n" + "-"*65 + "\n"
              f"  DONE degree={degree}  total_time={elapsed_deg:.0f}s\n"
              f"  Global best: {metrics_str(global_best_s1, global_best_s2)}\n"
              f"  score (S2-S1) = {global_best_score}\n"
              f"  Unique sparse solutions found: {len(unique)}\n"
              + "-"*65)
    print(footer, flush=True)
    fappend(SUMMARY_FILE, footer)

    if global_best_score < 0:
        alert = (f"\n  *** SPARSE SQUARE FOUND at degree {degree}! ***\n"
                 f"  {metrics_str(global_best_s1, global_best_s2)}\n"
                 f"  f = {poly_str(global_best_f)}\n")
        print(alert, flush=True)
        fappend(SUMMARY_FILE, alert)

    # write all unique solutions to solutions file
    for bf, bsq, bs1, bs2 in unique:
        log_solution(degree, bf, bsq, bs1, bs2)

    # print top 5 to stdout
    if unique:
        print(f"  Top finds:", flush=True)
        for bf, bsq, bs1, bs2 in unique[:5]:
            print(f"    {metrics_str(bs1, bs2)}", flush=True)
            print(f"    f = {poly_str(bf)}", flush=True)

    log_ratio_row(degree, global_best_s1, global_best_s2, len(unique))
    return global_best_score, len(unique)


# -----------------------------------------------------------------------------
#  Main sweep
# -----------------------------------------------------------------------------

def main():
    resume = resume_from()

    if resume == START_DEG:
        fappend(RATIOS_FILE,
                "# Q2 — Sparse bivariate square over F_3\n"
                "# Per-degree best S2/S1 and log_{S1}(S2)\n"
                "# deg   S1    S2    S2/S1    log_{S1}(S2)   sparse_hits")

    banner = ("\n" + "#"*65 + "\n"
              f"#  Q2. Sparse bivariate f over F_3  s.t.  |f^2| < |f|\n"
              f"#\n"
              f"#  Polynomial type : SPARSE / INCOMPLETE\n"
              f"#    (not complete — only {MIN_TERMS}..{MAX_TERMS} of the possible monomials)\n"
              f"#  Why sparse: complete polys are already dense, no room\n"
              f"#    for cancellations.  Sparse polys can cancel heavily mod 3.\n"
              f"#\n"
              f"#  Sweep : degrees {START_DEG} .. {END_DEG}\n"
              f"#  Per degree: {RESTARTS} restarts x {ITER_PER_RESTART:,} iters"
              f" = {RESTARTS*ITER_PER_RESTART:,} total\n"
              f"#  Metrics: S2/S1  and  log_{{S1}}(S2)\n"
              f"#  Resuming from degree: {resume}\n"
              f"#\n"
              f"#  Output files:\n"
              f"#    {SUMMARY_FILE}\n"
              f"#    {SOLUTIONS_FILE}\n"
              f"#    {RATIOS_FILE}\n"
              f"#\n"
              f"#  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
              + "#"*65)
    print(banner, flush=True)
    fappend(SUMMARY_FILE, banner)

    total_sparse = 0
    for deg in range(resume, END_DEG + 1):
        score, n_sparse = run_degree(deg)
        total_sparse   += n_sparse
        open(CHECKPOINT, 'w').write(str(deg))
        gc.collect()

    final = ("\n" + "#"*65 + "\n"
             f"#  SWEEP COMPLETE  degrees {START_DEG}..{END_DEG}\n"
             f"#  Total sparse solutions (S2 < S1): {total_sparse}\n"
             f"#  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
             + "#"*65)
    print(final, flush=True)
    fappend(SUMMARY_FILE, final)


if __name__ == "__main__":
    main()


    

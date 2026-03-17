#!/usr/bin/env python3


import random
import math
import time
import os

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit only this block
# ══════════════════════════════════════════════════════════════════════════════
START_DEG      = 10           # first degree
END_DEG        = 50           # last degree (inclusive)
N_ITER         = 5_000_000    # SA iterations per degree
T_START        = 5.0          # initial temperature
T_END          = 0.005        # final temperature
SEED_BASE      = 1234         # seed for degree d  =  SEED_BASE + d
MIN_TERMS      = 5            # min nonzero terms in starting polynomial
LOG_EVERY      = 500_000      # progress line every this many iterations

SUMMARY_FILE   = "results_summary.txt"
SOLUTIONS_FILE = "sparse_solutions.txt"
RATIOS_FILE    = "best_ratios.txt"
CHECKPOINT     = "checkpoint.txt"
# ══════════════════════════════════════════════════════════════════════════════


# ── F_3 arithmetic ────────────────────────────────────────────────────────────

def f3(x):
    """Reduce an integer to F_3."""
    return x % 3


# ── Polynomial representation ─────────────────────────────────────────────────
# A polynomial is a plain dict  {(i, j): c}  where c in {1, 2}.
# The zero polynomial is the empty dict {}.

def poly_mul(a, b):
    """Multiply two F_3 bivariate polynomials (as dicts). O(|a|*|b|)."""
    result = {}
    for (ai, aj), ac in a.items():
        for (bi, bj), bc in b.items():
            key = (ai + bi, aj + bj)
            val = f3(result.get(key, 0) + ac * bc)
            if val:
                result[key] = val
            elif key in result:
                del result[key]
    return result

def poly_square(a):
    """Square a polynomial over F_3. Uses symmetry: faster than poly_mul(a,a)."""
    result = {}
    items  = list(a.items())
    n      = len(items)
    # diagonal terms:  c_k^2 * x^{2i} y^{2j}
    for (i, j), c in items:
        key = (2*i, 2*j)
        val = f3(result.get(key, 0) + c * c)
        if val:
            result[key] = val
        elif key in result:
            del result[key]
    # cross terms:  2 * c_k * c_l * x^{i1+i2} y^{j1+j2}
    for idx in range(n):
        (i1, j1), c1 = items[idx]
        for jdx in range(idx + 1, n):
            (i2, j2), c2 = items[jdx]
            key = (i1 + i2, j1 + j2)
            val = f3(result.get(key, 0) + 2 * c1 * c2)
            if val:
                result[key] = val
            elif key in result:
                del result[key]
    return result

def sparsity(poly):
    return len(poly)

def poly_to_str(poly, max_terms=12):
    """Human-readable string for a polynomial dict."""
    if not poly:
        return "0"
    terms = sorted(poly.items(), key=lambda kv: (-kv[0][0], -kv[0][1]))
    parts = []
    for (i, j), c in terms[:max_terms]:
        coef  = "" if c == 1 else "2*"
        xpart = f"x^{i}" if i > 1 else ("x" if i == 1 else "")
        ypart = f"y^{j}" if j > 1 else ("y" if j == 1 else "")
        if not xpart and not ypart:
            parts.append(str(c))
        else:
            parts.append(f"{coef}{xpart}{'*' if xpart and ypart else ''}{ypart}")
    suffix = f" + ... ({len(poly)} terms)" if len(poly) > max_terms else ""
    return " + ".join(parts) + suffix


# ── Metrics ───────────────────────────────────────────────────────────────────

def ratio(s1, s2):
    """S_2 / S_1"""
    return s2 / s1 if s1 else float('nan')

def log_s1_s2(s1, s2):
    """log_{S_1}(S_2) = log(S_2) / log(S_1)"""
    if s1 <= 1 or s2 <= 0:
        return float('nan')
    return math.log(s2) / math.log(s1)

def metrics_str(s1, s2):
    r  = ratio(s1, s2)
    lr = log_s1_s2(s1, s2)
    rs  = f"{r:.4f}"  if not math.isnan(r)  else "nan  "
    lrs = f"{lr:.4f}" if not math.isnan(lr) else "nan  "
    return f"S1={s1:>4}  S2={s2:>4}  S2/S1={rs}  log_{{S1}}(S2)={lrs}"


# ── SA moves ──────────────────────────────────────────────────────────────────

def random_bivariate(max_dx, max_dy):
    all_keys = [(i, j) for i in range(max_dx + 1) for j in range(max_dy + 1)]
    n        = random.randint(MIN_TERMS, max(MIN_TERMS + 1, len(all_keys) // 2))
    chosen   = random.sample(all_keys, min(n, len(all_keys)))
    return {k: random.choice([1, 2]) for k in chosen}

def neighbor(poly, max_dx, max_dy):
    """
    One-step perturbation (flip weighted 2x — effective at all degrees):
      flip   — change a coefficient 1↔2          (50 %)
      add    — insert a new monomial              (25 %)
      remove — delete a monomial (keep ≥ 4 terms) (25 %)
    """
    d    = dict(poly)
    move = random.choice(('flip', 'add', 'remove', 'flip'))

    if move == 'flip' and d:
        key    = random.choice(list(d.keys()))
        d[key] = 1 if d[key] == 2 else 2

    elif move == 'add':
        all_keys = [(i, j) for i in range(max_dx + 1)
                            for j in range(max_dy + 1)]
        missing  = [k for k in all_keys if k not in d]
        if missing:
            k    = random.choice(missing)
            d[k] = random.choice([1, 2])

    elif move == 'remove' and len(d) > 4:
        del d[random.choice(list(d.keys()))]

    return d if d else poly


# ── File I/O ──────────────────────────────────────────────────────────────────

def fappend(path, text):
    with open(path, 'a') as fh:
        fh.write(text + '\n')

def write_solution(deg, f, sq, s1, s2):
    block = (f"\n[DEG {deg}]  *** S2 < S1  —  SPARSE SQUARE ***\n"
             f"  {metrics_str(s1, s2)}\n"
             f"  f(x,y)   = {poly_to_str(f)}\n"
             f"  f(x,y)^2 = {poly_to_str(sq)}\n"
             f"  {'─'*60}")
    fappend(SOLUTIONS_FILE, block)

def write_ratio_row(deg, s1, s2, n_sparse):
    r  = ratio(s1, s2)
    lr = log_s1_s2(s1, s2)
    fappend(RATIOS_FILE,
            f"deg={deg:>3}  S1={s1:>4}  S2={s2:>4}  "
            f"S2/S1={r:.4f}  log_{{S1}}(S2)={lr:.4f}  "
            f"sparse_hits={n_sparse}")

def resume_from():
    if os.path.exists(CHECKPOINT):
        val = open(CHECKPOINT).read().strip()
        if val.isdigit():
            return int(val) + 1
    return START_DEG


# ── Simulated Annealing (one degree) ─────────────────────────────────────────

def run_sa(degree, n_iter=N_ITER, T_start=T_START, T_end=T_END, seed=None):
    if seed is None:
        seed = SEED_BASE + degree
    random.seed(seed)

    max_dx = max_dy = degree

    header = ("\n" + "="*70 + "\n"
              f"  Q2. Bivariate sparse square — degree {degree}\n"
              f"  max_dx = max_dy = {degree}   n_iter = {n_iter:,}   seed = {seed}\n"
              f"  Tracking:  S2/S1  and  log_{{S1}}(S2)   (both want < 1)\n"
              f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
              + "="*70)
    print(header, flush=True)
    fappend(SUMMARY_FILE, header)

    # ── initialise ──
    f      = random_bivariate(max_dx, max_dy)
    sq     = poly_square(f)
    s1, s2 = sparsity(f), sparsity(sq)
    score  = s2 - s1

    best_f, best_sq  = f, sq
    best_s1, best_s2 = s1, s2
    best_score       = score

    # key = str(f_new) → (f, sq, s1, s2); deduped on the fly, O(unique hits)
    sparse_found = {}

    T       = T_start
    cooling = (T_end / T_start) ** (1.0 / n_iter)
    t0      = time.time()

    # ── main loop ──
    for it in range(n_iter):
        f_new     = neighbor(f, max_dx, max_dy)
        sq_new    = poly_square(f_new)
        s1_new    = sparsity(f_new)
        s2_new    = sparsity(sq_new)
        new_score = s2_new - s1_new

        delta = new_score - score
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
            f, sq, s1, s2, score = f_new, sq_new, s1_new, s2_new, new_score

        if score < best_score:
            best_score           = score
            best_f, best_sq      = f, sq
            best_s1, best_s2     = s1, s2

        # deduplicate in the hot loop — prevents memory blowup on long runs
        if new_score < 0:
            key = str(sorted(f_new.items()))
            if key not in sparse_found:
                sparse_found[key] = (f_new, sq_new, s1_new, s2_new)

        T *= cooling

        if (it + 1) % LOG_EVERY == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (it + 1) * (n_iter - it - 1)
            line    = (f"  iter {it+1:>8,}  T={T:.5f}  "
                       f"best: {metrics_str(best_s1, best_s2)}  "
                       f"score={best_score:>4}  "
                       f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")
            print(line, flush=True)

    # ── sort unique hits by S2 ──
    unique = sorted(sparse_found.values(), key=lambda t: t[3])

    # ── per-degree footer ──
    elapsed = time.time() - t0
    footer  = ("\n" + "─"*70 + "\n"
               f"  DONE  degree={degree}  time={elapsed:.1f}s\n"
               f"  Best: {metrics_str(best_s1, best_s2)}\n"
               f"  score (S2-S1) = {best_score}\n"
               f"  Unique sparse solutions (S2 < S1): {len(unique)}\n"
               + "─"*70)
    print(footer, flush=True)
    fappend(SUMMARY_FILE, footer)

    if best_score < 0:
        alert = (f"\n  *** SPARSE SQUARE FOUND — degree {degree} ***\n"
                 f"  {metrics_str(best_s1, best_s2)}\n"
                 f"  f = {poly_to_str(best_f)}\n")
        print(alert, flush=True)
        fappend(SUMMARY_FILE, alert)

    for bf, bsq, bft, bst in unique:
        write_solution(degree, bf, bsq, bft, bst)

    if unique:
        print("  Top sparse finds this degree:", flush=True)
        for bf, bsq, bft, bst in unique[:5]:
            print(f"    {metrics_str(bft, bst)}", flush=True)
            print(f"    f = {poly_to_str(bf)}", flush=True)

    write_ratio_row(degree, best_s1, best_s2, len(unique))
    return best_score, len(unique)


# ── Main sweep ────────────────────────────────────────────────────────────────

def main():
    import gc
    resume = resume_from()

    if resume == START_DEG:
        fappend(RATIOS_FILE,
                "# Per-degree best S2/S1 and log_{S1}(S2)  —  Q2 Bivariate over F_3\n"
                "# deg   S1    S2    S2/S1    log_{S1}(S2)   sparse_hits")

    banner = ("\n" + "#"*70 + "\n"
              f"#  Q2. Bivariate sparse square over F_3\n"
              f"#  Sweep: degrees {START_DEG} .. {END_DEG}\n"
              f"#  n_iter = {N_ITER:,}   T: {T_START} -> {T_END}\n"
              f"#  Metrics: S2/S1  and  log_{{S1}}(S2)\n"
              f"#  Resuming from degree {resume}\n"
              f"#  No external dependencies — pure python3\n"
              f"#  Summary   -> {SUMMARY_FILE}\n"
              f"#  Solutions -> {SOLUTIONS_FILE}\n"
              f"#  Ratios    -> {RATIOS_FILE}\n"
              f"#  Started   : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
              + "#"*70)
    print(banner, flush=True)
    fappend(SUMMARY_FILE, banner)

    total_sparse = 0
    for deg in range(resume, END_DEG + 1):
        score, n_sparse = run_sa(deg)
        total_sparse   += n_sparse
        open(CHECKPOINT, 'w').write(str(deg))
        gc.collect()

    final = ("\n" + "#"*70 + "\n"
             f"#  SWEEP COMPLETE  degrees {START_DEG}..{END_DEG}\n"
             f"#  Total sparse solutions (S2 < S1): {total_sparse}\n"
             f"#  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
             + "#"*70)
    print(final, flush=True)
    fappend(SUMMARY_FILE, final)


if __name__ == "__main__":
    main()

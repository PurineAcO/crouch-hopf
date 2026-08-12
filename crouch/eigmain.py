import time
import csv
import numpy as np
import scipy.sparse.linalg as spla
import console


def solve_eig(S, T, k: int = 20, sigma: complex = 0.0 + 0.0j,
              ncv: int | None = None, v0_seed: int = 42,
              maxiter: int = 3000, tol: float = 1e-8, save: bool = True):
    """求解广义特征值问题,使用ARPACK.其中,`k`是要计算的特征值个数,默认`20`\n
    `sigma`为移位点序号,默认0+0j,`ncv`为使用的 Krylov 子空间维度，若为`None`则自动设为`min(3*k, n)`\n
    `v0_seed`随机种子,用于生成初始迭代向量`v0`，保证结果可重现\n
    `maxiter`最大迭代次数,默认3000,`tol`收敛容差,默认1e-8,`save`决定是否将结果保存为 .npy 和 .csv """

    n = S.shape[0]
    if ncv is None:
        ncv = min(3 * k, n)
    rng = np.random.default_rng(v0_seed)
    v0 = rng.standard_normal(n)

    t0 = time.time()
    console.info(f"shift-invert ARPACK: sigma={sigma}, k={k}, ncv={ncv}, v0_seed={v0_seed} ...")
    vals, vecs = spla.eigs(S, k=k, M=T, sigma=sigma, which="LM",
                           ncv=ncv, v0=v0, maxiter=maxiter, tol=tol)
    order = np.argsort(-vals.real)

    # 每个模态的归一化残差 (纯后处理)
    res = np.zeros(k)
    for i in range(k):
        lhs = S @ vecs[:, i]
        rhs = vals[i] * (T @ vecs[:, i])
        res[i] = np.linalg.norm(lhs - rhs) / (np.linalg.norm(lhs) + np.linalg.norm(rhs) + 1e-300)

    console.ok(f"Eigen solve done in {time.time()-t0:.1f}s, top {k} growth rates Re(lambda):")
    console.info(f"{'Re(lambda)':>14s} {'Im(lambda)':>14s} {'growth':>10s} {'freq':>10s} {'resid':>10s}")
    rows = []
    for idx, i in enumerate(order, 1):
        v = vals[i]
        line = f"{v.real:14.6e} {v.imag:14.6e} {v.real:10.4e} {v.imag/2/np.pi:10.4e} {res[i]:10.2e}"
        if res[i] >= 100 * tol:          # 未收敛: 橙色警告
            console.warn_raw(line + "   <-- not converged?")
        elif v.real > 0:                 # 不稳定模态: 绿色
            console.ok_raw(line)
        else:                            # 稳定模态: 默认黑色
            print(line)
        rows.append((idx, v.real, v.imag, v.real, v.imag/2/np.pi, res[i]))
    if save:
        np.save("eigvals.npy", vals)
        np.save("eigvecs.npy", vecs)
        with open("result.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["mode", "Re(lambda)", "Im(lambda)", "growth", "freq", "resid"])
            w.writerows(rows)
        console.ok("Saved eigvals.npy / eigvecs.npy / result.csv")
    return vals, vecs

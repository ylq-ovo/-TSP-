// =============================================================================
//  tsp_experiment.cpp
//  旅行商问题(TSP)四种启发式算法对比实验 + 状态压缩动态规划精确对拍器
//
//  精确算法 : Held-Karp 状态压缩 DP, O(n^2 * 2^n)
//  启发式   : 蚁群算法(MMAS) / 禁忌搜索(2-opt Tabu) / 模拟退火(2-opt SA)
//             遗传算法(OX 交叉 + 逆转变异)
//
//  编译:  g++ -O2 -std=c++17 -o tsp_experiment src/tsp_experiment.cpp
//  运行:  ./tsp_experiment            (结果写入 results/ 目录)
//
//  测试数据完全可复现: 城市坐标由固定种子的 std::mt19937_64 生成,
//  距离为欧氏距离(对称矩阵), 与 TSPLIB 的 EUC_2D 定义一致。
//
//  实验设计
//  --------
//  规模 n = 10 / 15 / 20 : 每种规模 5 个随机实例, 每个实例每个算法重复 30 轮,
//                          以 Held-Karp 精确最优解为参照, 统计误差与命中率。
//  规模 n = 30           : 3 个实例 x 10 轮, DP 已不可行, 以"10 倍预算多算法
//                          多次运行的最好结果"作为 best-known 参照。
//  四种算法的参数按"单轮墙钟耗时处于同一量级(1~5 ms)"整定, 详见实验报告。
// =============================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
// windows.h 默认把 min/max 定义成宏, 会把 std::max(...) / std::min(...) 里的
// "max(" / "min(" 当成宏调用展开, 导致 MSVC 报 C2589 "非法的标记"。
// 必须在包含 windows.h 之前关闭这两个宏。
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

using std::size_t;
using std::vector;

// 兜底: 某些工程会在命令行/预编译头里抢先引入 windows.h, 这里再主动清理一次
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

// -----------------------------------------------------------------------------
// 0. 通用工具
// -----------------------------------------------------------------------------
static const double INF = std::numeric_limits<double>::infinity();

using Clock = std::chrono::steady_clock;

static double elapsedMs(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// 小型 RNG 封装: 全部随机行为都走这里, 保证 -- 给定种子 -- 完全可复现
struct Rng {
    std::mt19937_64 g;
    explicit Rng(uint64_t seed) : g(seed) {}
    inline uint64_t raw() { return g(); }
    // [0,1) 均匀分布
    inline double u01() { return static_cast<double>(g() >> 11) * (1.0 / 9007199254740992.0); }
    // [0,m) 均匀整数
    inline int nextInt(int m) { return static_cast<int>(g() % static_cast<uint64_t>(m)); }
};

// -----------------------------------------------------------------------------
// 1. 实例与基本操作
// -----------------------------------------------------------------------------
struct Instance {
    int n = 0;
    vector<double> x, y;      // 坐标
    vector<double> d;         // 距离矩阵 n*n
    std::string name;

    inline double dist(int a, int b) const { return d[static_cast<size_t>(a) * n + b]; }
};

// 固定种子生成 n 个 [0,1000]^2 整数坐标的城市
static Instance makeInstance(int n, uint64_t seed, const std::string& name) {
    Rng rng(seed);
    Instance inst;
    inst.n = n;
    inst.name = name;
    inst.x.resize(n);
    inst.y.resize(n);
    for (int i = 0; i < n; ++i) {
        inst.x[i] = static_cast<double>(rng.raw() % 1001);
        inst.y[i] = static_cast<double>(rng.raw() % 1001);
    }
    inst.d.assign(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double dx = inst.x[i] - inst.x[j];
            double dy = inst.y[i] - inst.y[j];
            inst.d[static_cast<size_t>(i) * n + j] = std::sqrt(dx * dx + dy * dy);
        }
    }
    return inst;
}

// 回路长度
static double tourLen(const Instance& in, const vector<int>& t) {
    const int n = in.n;
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += in.dist(t[i], t[(i + 1) % n]);
    return s;
}

// 2-opt 增量: 反转 t[i+1 .. j] 后的长度变化量
// 旧边 (t_i, t_{i+1}) (t_j, t_{j+1})  ->  新边 (t_i, t_j) (t_{i+1}, t_{j+1})
static inline double delta2opt(const Instance& in, const vector<int>& t, int i, int j) {
    const int n = in.n;
    const int a = t[i], b = t[(i + 1) % n], c = t[j], e = t[(j + 1) % n];
    return in.dist(a, c) + in.dist(b, e) - in.dist(a, b) - in.dist(c, e);
}

static inline void apply2opt(vector<int>& t, int i, int j) {
    // 反转 [i+1, j]; i==0 && j==n-1 时为整体反转, 增量恒为 0, 无副作用
    std::reverse(t.begin() + i + 1, t.begin() + j + 1);
}

// 最近邻构造 (作为部分算法的初始解 / 蚁群 tau0 估计)
static vector<int> nearestNeighborTour(const Instance& in, int start) {
    const int n = in.n;
    vector<int> t;
    t.reserve(n);
    vector<char> vis(n, 0);
    int cur = start;
    t.push_back(cur);
    vis[cur] = 1;
    for (int step = 1; step < n; ++step) {
        int best = -1;
        double bd = INF;
        for (int c = 0; c < n; ++c) {
            if (!vis[c] && in.dist(cur, c) < bd) { bd = in.dist(cur, c); best = c; }
        }
        t.push_back(best);
        vis[best] = 1;
        cur = best;
    }
    return t;
}

// 纯 2-opt 局部搜索 (best improvement), 供参考/后续实验扩展使用
[[maybe_unused]] static double twoOptLocalSearch(const Instance& in, vector<int>& t) {
    const int n = in.n;
    double len = tourLen(in, t);
    bool improved = true;
    while (improved) {
        improved = false;
        double bestDelta = -1e-12;
        int bi = -1, bj = -1;
        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (i == 0 && j == n - 1) continue;
                double dl = delta2opt(in, t, i, j);
                if (dl < bestDelta) { bestDelta = dl; bi = i; bj = j; }
            }
        }
        if (bi >= 0) { apply2opt(t, bi, bj); len += bestDelta; improved = true; }
    }
    return len;
}

struct Solution {
    double length = INF;
    vector<int> tour;
    long long evals = 0;   // 评价过的解的个数(用于"同等计算预算"讨论)
    bool ok = false;
};

// -----------------------------------------------------------------------------
// 2. 精确对拍器: Held-Karp 状态压缩 DP
//    dp[mask][j] = 从起点 0 出发, 恰好访问集合 mask 中所有点, 当前停在 j 的最短长度
//    时间 O(n^2 2^n), 空间 O(n 2^n)
// -----------------------------------------------------------------------------
static Solution heldKarp(const Instance& in) {
    Solution res;
    const int n = in.n;
    if (n > 22) return res;                       // 内存/时间保护
    const size_t full = (size_t(1) << n) - 1;
    const size_t total = (size_t(1) << n) * size_t(n);

    vector<double> dp(total, INF);
    vector<uint8_t> par(total, 255);

    dp[size_t(1) * n + 0] = 0.0;                  // mask={0}, 停在 0

    for (size_t mask = 1; mask <= full; mask += 2) {          // 只枚举含点 0 的集合
        const size_t base = mask * n;
        for (int j = 0; j < n; ++j) {
            if (!(mask & (size_t(1) << j))) continue;
            const double cur = dp[base + j];
            if (cur == INF) continue;
            for (int k = 1; k < n; ++k) {
                if (mask & (size_t(1) << k)) continue;
                const size_t nmask = mask | (size_t(1) << k);
                const double nd = cur + in.dist(j, k);
                const size_t idx = nmask * n + k;
                if (nd < dp[idx] - 1e-12) {
                    dp[idx] = nd;
                    par[idx] = static_cast<uint8_t>(j);
                }
            }
        }
    }

    // 收尾: 回到起点
    double best = INF;
    int bestJ = -1;
    for (int j = 1; j < n; ++j) {
        const double c = dp[full * n + j];
        if (c == INF) continue;
        const double tot = c + in.dist(j, 0);
        if (tot < best) { best = tot; bestJ = j; }
    }
    if (bestJ < 0) return res;

    // 回溯路径
    vector<int> path;
    size_t mask = full;
    int j = bestJ;
    while (true) {
        path.push_back(j);
        uint8_t p = par[mask * n + j];
        if (p == 255) break;
        mask ^= (size_t(1) << j);
        j = static_cast<int>(p);
    }
    std::reverse(path.begin(), path.end());       // path[0] == 0
    res.length = best;
    res.tour = path;
    res.evals = static_cast<long long>(1) << n;
    res.ok = true;
    return res;
}

// 附加校验: n <= 11 时用全排列暴力枚举(true optimum)交叉验证状压 DP 的正确性
static Solution bruteForce(const Instance& in) {
    Solution res;
    const int n = in.n;
    if (n > 11) return res;
    vector<int> perm(n - 1);
    std::iota(perm.begin(), perm.end(), 1);
    double best = INF;
    vector<int> bestTour;
    do {
        double len = in.dist(0, perm[0]);
        for (int i = 0; i + 1 < n - 1; ++i) len += in.dist(perm[i], perm[i + 1]);
        len += in.dist(perm[n - 2], 0);
        if (len < best) {
            best = len;
            bestTour.clear();
            bestTour.push_back(0);
            bestTour.insert(bestTour.end(), perm.begin(), perm.end());
        }
    } while (std::next_permutation(perm.begin(), perm.end()));
    res.length = best;
    res.tour = bestTour;
    res.ok = true;
    return res;
}

// -----------------------------------------------------------------------------
// 3. 蚁群算法 MMAS (Max-Min Ant System)
//    - 每只蚂蚁按 tau^alpha * eta^beta 轮盘赌构造回路
//    - 仅最优蚂蚁更新信息素, 信息素被限制在 [tauMin, tauMax] 内, 防止早熟停滞
// -----------------------------------------------------------------------------
struct AcoParams {
    int ants = 15;
    int iterations = 150;
    double alpha = 1.0;    // 信息素重要度
    double beta = 4.0;     // 启发式(1/d)重要度
    double rho = 0.15;     // 挥发率
};

static Solution antColony(const Instance& in, const AcoParams& p, Rng& rng) {
    Solution res;
    const int n = in.n;
    const size_t nn = static_cast<size_t>(n) * n;

    // eta^beta 预计算, 避免重复 pow
    vector<double> eta(nn, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) eta[static_cast<size_t>(i) * n + j] = std::pow(1.0 / in.dist(i, j), p.beta);

    // 初始信息素: tau0 = 1/(rho * L_nn)
    double lnn = tourLen(in, nearestNeighborTour(in, 0));
    for (int i = 0; i < n; ++i) lnn = std::min(lnn, tourLen(in, nearestNeighborTour(in, i)));
    double tauMax = 1.0 / (p.rho * std::max(lnn, 1e-9));
    double tauMin = tauMax / (2.0 * n);
    vector<double> tau(nn, tauMax);

    vector<int> bestTour;
    double bestLen = INF;
    long long evals = 0;

    vector<int> tour;
    tour.reserve(n);
    vector<char> vis(n, 0);
    vector<double> w(n, 0.0);

    for (int it = 0; it < p.iterations; ++it) {
        vector<int> iterBestTour;
        double iterBestLen = INF;

        for (int a = 0; a < p.ants; ++a) {
            std::fill(vis.begin(), vis.end(), 0);
            tour.clear();
            int cur = rng.nextInt(n);
            tour.push_back(cur);
            vis[cur] = 1;

            for (int step = 1; step < n; ++step) {
                double sum = 0.0;
                for (int c = 0; c < n; ++c) {
                    if (vis[c]) { w[c] = 0.0; continue; }
                    double t = (p.alpha == 1.0) ? tau[static_cast<size_t>(cur) * n + c]
                                                : std::pow(tau[static_cast<size_t>(cur) * n + c], p.alpha);
                    w[c] = t * eta[static_cast<size_t>(cur) * n + c];
                    sum += w[c];
                }
                int nxt = -1;
                if (sum > 0.0) {
                    double r = rng.u01() * sum;
                    for (int c = 0; c < n; ++c) {
                        if (vis[c]) continue;
                        r -= w[c];
                        if (r <= 0.0) { nxt = c; break; }
                    }
                }
                if (nxt < 0) {                       // 兜底: 取第一个未访问点
                    for (int c = 0; c < n; ++c) if (!vis[c]) { nxt = c; break; }
                }
                tour.push_back(nxt);
                vis[nxt] = 1;
                cur = nxt;
            }
            double len = tourLen(in, tour);
            ++evals;
            if (len < iterBestLen) { iterBestLen = len; iterBestTour = tour; }
            if (len < bestLen) { bestLen = len; bestTour = tour; }
        }

        // 信息素挥发
        for (size_t i = 0; i < nn; ++i) tau[i] *= (1.0 - p.rho);
        // 全局最优蚂蚁释放信息素
        double dep = 1.0 / bestLen;
        for (int i = 0; i < n; ++i) {
            int a = bestTour[i], b = bestTour[(i + 1) % n];
            tau[static_cast<size_t>(a) * n + b] += dep;
            tau[static_cast<size_t>(b) * n + a] += dep;
        }
        // MMAS 上下界钳制
        tauMax = 1.0 / (p.rho * bestLen);
        tauMin = tauMax / (2.0 * n);
        for (size_t i = 0; i < nn; ++i) tau[i] = std::min(std::max(tau[i], tauMin), tauMax);
    }

    res.length = bestLen;
    res.tour = bestTour;
    res.evals = evals;
    res.ok = true;
    return res;
}

// -----------------------------------------------------------------------------
// 4. 禁忌搜索 (Tabu Search)
//    邻域: 全部 2-opt 移动; 每步取最优非禁忌移动(或满足特赦准则的移动)
//    禁忌对象: 被删除的边, 禁忌长度 tenure(随 n 调整并带随机抖动)
//    停滞时用 double-bridge 扰动重启, 保证多样性
// -----------------------------------------------------------------------------
struct TabuParams {
    int iterations = 2000;
    int tenureBase = 7;        // 实际禁忌长度 = tenureBase + n/4 + [0,4)
    int stagnation = 50;       // 连续多少步无改进则扰动
};

static vector<int> doubleBridge(vector<int> t, Rng& rng) {
    const int n = t.size();
    if (n < 8) return t;
    int a = 1 + rng.nextInt(n - 4);
    int b = a + 1 + rng.nextInt(n - a - 3);
    int c = b + 1 + rng.nextInt(n - b - 2);
    vector<int> r;
    r.reserve(n);
    r.insert(r.end(), t.begin(), t.begin() + a);
    r.insert(r.end(), t.begin() + c, t.end());
    r.insert(r.end(), t.begin() + b, t.begin() + c);
    r.insert(r.end(), t.begin() + a, t.begin() + b);
    return r;
}

static Solution tabuSearch(const Instance& in, const TabuParams& p, Rng& rng) {
    Solution res;
    const int n = in.n;

    vector<int> cur = nearestNeighborTour(in, rng.nextInt(n));
    double curLen = tourLen(in, cur);
    vector<int> best = cur;
    double bestLen = curLen;
    long long evals = 1;

    // tabu[u][v] = 该边被禁止重新加入的迭代截止序号
    vector<int> tabu(static_cast<size_t>(n) * n, 0);
    const int tenure = p.tenureBase + n / 4;

    int lastImprove = 0;
    for (int it = 1; it <= p.iterations; ++it) {
        double bestDelta = INF;
        int bi = -1, bj = -1;

        // best-improvement 扫描整个 2-opt 邻域
        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (i == 0 && j == n - 1) continue;
                const double dl = delta2opt(in, cur, i, j);
                ++evals;
                if (dl >= bestDelta - 1e-12) continue;

                const int a = cur[i], b = cur[(i + 1) % n];
                const int c = cur[j], e = cur[(j + 1) % n];
                const bool isTabu = (tabu[static_cast<size_t>(a) * n + c] > it) ||
                                    (tabu[static_cast<size_t>(b) * n + e] > it);
                // 特赦准则: 若该移动能刷新历史最优, 则无视禁忌
                const bool aspiration = (curLen + dl < bestLen - 1e-9);
                if (isTabu && !aspiration) continue;

                bestDelta = dl;
                bi = i; bj = j;
            }
        }
        if (bi < 0) break;   // 整个邻域被禁忌

        const int a = cur[bi], b = cur[(bi + 1) % n];
        const int c = cur[bj], e = cur[(bj + 1) % n];
        const int tl = tenure + rng.nextInt(5);
        tabu[static_cast<size_t>(a) * n + b] = it + tl;      // 禁止旧边复活
        tabu[static_cast<size_t>(b) * n + a] = it + tl;
        tabu[static_cast<size_t>(c) * n + e] = it + tl;
        tabu[static_cast<size_t>(e) * n + c] = it + tl;

        apply2opt(cur, bi, bj);
        curLen += bestDelta;
        if (curLen < 1e-12) curLen = tourLen(in, cur);       // 数值保险

        if (curLen < bestLen - 1e-9) {
            bestLen = curLen;
            best = cur;
            lastImprove = it;
        }

        if (it - lastImprove > p.stagnation) {               // 扰动重启
            cur = doubleBridge(best, rng);
            curLen = tourLen(in, cur);
            ++evals;
            lastImprove = it;
        }
    }

    res.length = bestLen;
    res.tour = best;
    res.evals = evals;
    res.ok = true;
    return res;
}

// -----------------------------------------------------------------------------
// 5. 模拟退火 (Simulated Annealing)
//    邻域: 随机 2-opt 移动(对称 TSP 下等价于随机片段反转)
//    初温: 采样估计, 使"平均劣化移动"的初始接受率约 80%; 几何降温
// -----------------------------------------------------------------------------
struct SaParams {
    long long iterations = 100000;
    double tf = 0.01;
    double targetAccept = 0.80;
};

static Solution simulatedAnnealing(const Instance& in, const SaParams& p, Rng& rng) {
    Solution res;
    const int n = in.n;

    vector<int> cur = nearestNeighborTour(in, rng.nextInt(n));
    double curLen = tourLen(in, cur);
    vector<int> best = cur;
    double bestLen = curLen;
    long long evals = 1;

    // --- 初温估计: 采样 2000 个随机 2-opt 移动的增量 ---
    double posSum = 0.0;
    int posCnt = 0;
    const int SAMPLE = 2000;
    for (int s = 0; s < SAMPLE; ++s) {
        int i = rng.nextInt(n), j = rng.nextInt(n);
        if (i > j) std::swap(i, j);
        if (i == j) continue;
        double dl = delta2opt(in, cur, i, j);
        if (dl > 0) { posSum += dl; ++posCnt; }
    }
    double avgPos = (posCnt > 0) ? posSum / posCnt : 1.0;
    double t0 = avgPos / (-std::log(p.targetAccept));        // Metropolis 反解
    double t = t0;
    const double cooling = std::pow(p.tf / t0, 1.0 / static_cast<double>(p.iterations));

    long long acc = 0;
    for (long long it = 0; it < p.iterations; ++it) {
        int i = rng.nextInt(n), j = rng.nextInt(n);
        if (i > j) std::swap(i, j);
        if (i == j) { t *= cooling; continue; }
        double dl = delta2opt(in, cur, i, j);
        ++evals;
        if (dl <= 0.0 || rng.u01() < std::exp(-dl / t)) {
            apply2opt(cur, i, j);
            curLen += dl;
            ++acc;
            if (curLen < bestLen - 1e-9) {
                bestLen = curLen;
                best = cur;
            }
        }
        t *= cooling;
    }
    (void)acc;

    res.length = bestLen;
    res.tour = best;
    res.evals = evals;
    res.ok = true;
    return res;
}

// -----------------------------------------------------------------------------
// 6. 遗传算法 (Genetic Algorithm)
//    编码: 城市序列排列; 选择: 锦标赛(k=3); 交叉: OX(order crossover);
//    变异: 片段逆转(2-opt 同构); 保留精英
// -----------------------------------------------------------------------------
struct GaParams {
    int pop = 80;
    int gens = 150;
    double pc = 0.90;
    double pm = 0.25;
    int tournament = 3;
    int elite = 2;
};

static vector<int> oxCrossover(const vector<int>& p1, const vector<int>& p2, Rng& rng) {
    const int n = p1.size();
    int a = rng.nextInt(n), b = rng.nextInt(n);
    if (a > b) std::swap(a, b);
    vector<int> child(n, -1);
    vector<char> used(n, 0);
    for (int i = a; i <= b; ++i) { child[i] = p1[i]; used[p1[i]] = 1; }
    int k = (b + 1) % n;
    for (int i = 0; i < n; ++i) {
        int city = p2[(b + 1 + i) % n];
        if (!used[city]) {
            child[k] = city;
            used[city] = 1;
            k = (k + 1) % n;
        }
    }
    return child;
}

static void invertMutate(vector<int>& t, Rng& rng) {
    const int n = t.size();
    int i = rng.nextInt(n), j = rng.nextInt(n);
    if (i > j) std::swap(i, j);
    if (i == j) return;
    std::reverse(t.begin() + i, t.begin() + j + 1);
}

static Solution geneticAlgorithm(const Instance& in, const GaParams& p, Rng& rng) {
    Solution res;
    const int n = in.n;

    auto randomTour = [&]() {
        vector<int> t(n);
        std::iota(t.begin(), t.end(), 0);
        for (int i = n - 1; i > 0; --i) std::swap(t[i], t[rng.nextInt(i + 1)]);
        return t;
    };

    vector<vector<int>> pop(p.pop);
    vector<double> len(p.pop);
    long long evals = 0;
    for (int i = 0; i < p.pop; ++i) { pop[i] = randomTour(); len[i] = tourLen(in, pop[i]); ++evals; }
    // 其中一半用最近邻解打底(带随机起点), 兼顾随机性与质量
    for (int i = 0; i < p.pop / 2; ++i) {
        pop[i] = nearestNeighborTour(in, rng.nextInt(n));
        len[i] = tourLen(in, pop[i]);
        ++evals;
    }

    vector<int> order(p.pop);
    vector<int> best;
    double bestLen = INF;

    for (int g = 0; g < p.gens; ++g) {
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return len[a] < len[b]; });

        if (len[order[0]] < bestLen - 1e-9) {
            bestLen = len[order[0]];
            best = pop[order[0]];
        }

        // 锦标赛选择
        auto select = [&]() -> int {
            int bIdx = rng.nextInt(p.pop);
            for (int k = 1; k < p.tournament; ++k) {
                int c = rng.nextInt(p.pop);
                if (len[c] < len[bIdx]) bIdx = c;
            }
            return bIdx;
        };

        vector<vector<int>> nextPop;
        nextPop.reserve(p.pop);
        for (int i = 0; i < p.elite && i < p.pop; ++i) nextPop.push_back(pop[order[i]]);

        while (static_cast<int>(nextPop.size()) < p.pop) {
            int a = select(), b = select();
            vector<int> child;
            if (rng.u01() < p.pc) child = oxCrossover(pop[a], pop[b], rng);
            else child = pop[a];
            if (rng.u01() < p.pm) invertMutate(child, rng);
            nextPop.push_back(std::move(child));
        }

        pop.swap(nextPop);
        for (int i = 0; i < p.pop; ++i) { len[i] = tourLen(in, pop[i]); ++evals; }
    }

    // 收尾检查
    for (int i = 0; i < p.pop; ++i) {
        if (len[i] < bestLen) { bestLen = len[i]; best = pop[i]; }
    }

    res.length = bestLen;
    res.tour = best;
    res.evals = evals;
    res.ok = true;
    return res;
}

// -----------------------------------------------------------------------------
// 7. 实验框架
// -----------------------------------------------------------------------------
struct RunRecord {
    std::string instance;
    int n = 0;
    std::string algo;
    int run = 0;
    double timeMs = 0.0;
    double length = 0.0;
    double gap = 0.0;      // 绝对误差 (相对参考解)
    double gapPct = 0.0;   // 相对误差 %
    bool hit = false;      // 是否达到参考最优
    long long evals = 0;
};

struct InstResult {
    Instance inst;
    Solution ref;          // 参考解: 精确最优 或 best-known
    bool exactRef = true;
    double exactMs = 0.0;
    vector<RunRecord> recs;
};

struct Agg {
    int samples = 0;
    double timeMean = 0, timeMin = 0, timeMax = 0;
    double lenMean = 0, lenMin = 0, lenSd = 0;
    double gapMean = 0, gapSd = 0, gapMax = 0;
    double pctMean = 0, pctMin = 0, pctMax = 0, pctSd = 0;
    double hitRate = 0;
    double evalsMean = 0;
};

static Agg aggregate(const vector<const RunRecord*>& v) {
    Agg a;
    a.samples = static_cast<int>(v.size());
    if (v.empty()) return a;
    double sT = 0, sL = 0, sG = 0, sP = 0;
    a.timeMin = a.lenMin = a.pctMin = 1e18;
    a.timeMax = a.pctMax = -1e18;
    a.gapMax = -1e18;
    int hits = 0;
    long long sE = 0;
    for (auto* p : v) {
        sT += p->timeMs;  a.timeMin = std::min(a.timeMin, p->timeMs); a.timeMax = std::max(a.timeMax, p->timeMs);
        sL += p->length;  a.lenMin = std::min(a.lenMin, p->length);
        sG += p->gap;     a.gapMax = std::max(a.gapMax, p->gap);
        sP += p->gapPct;  a.pctMin = std::min(a.pctMin, p->gapPct); a.pctMax = std::max(a.pctMax, p->gapPct);
        if (p->hit) ++hits;
        sE += p->evals;
    }
    const double m = static_cast<double>(v.size());
    a.timeMean = sT / m;
    a.lenMean = sL / m;
    a.gapMean = sG / m;
    a.pctMean = sP / m;
    double vL = 0, vG = 0, vP = 0;
    for (auto* p : v) {
        vL += (p->length - a.lenMean) * (p->length - a.lenMean);
        vG += (p->gap - a.gapMean) * (p->gap - a.gapMean);
        vP += (p->gapPct - a.pctMean) * (p->gapPct - a.pctMean);
    }
    a.lenSd = std::sqrt(vL / m);
    a.gapSd = std::sqrt(vG / m);
    a.pctSd = std::sqrt(vP / m);
    a.hitRate = static_cast<double>(hits) / m;
    a.evalsMean = static_cast<double>(sE) / m;
    return a;
}

static std::string tourToString(const vector<int>& t) {
    std::string s;
    for (size_t i = 0; i < t.size(); ++i) {
        if (i) s += "-";
        s += std::to_string(t[i]);
    }
    return s;
}

static const char* kAlgoName[4] = {"ACO", "Tabu", "SA", "GA"};

// 大规模实例(状压 DP 不可行)的参考解: 用 10 倍预算、多个种子把四种算法各跑一遍取最好,
// 得到一个与本次测量无关的强参考值(best-known, 仍是可行解长度, 故必 >= 真实最优)
static double bestKnownReference(const Instance& in, const AcoParams& ap, const TabuParams& tp,
                                 const SaParams& sp, const GaParams& gp, uint64_t seed) {
    AcoParams a = ap;  a.iterations *= 10;
    TabuParams t = tp; t.iterations *= 10;
    SaParams s = sp;   s.iterations *= 10;
    GaParams g = gp;   g.gens *= 10;
    double best = INF;
    for (int k = 0; k < 2; ++k) {
        Rng r1(seed + 11 * k), r2(seed + 11 * k + 1), r3(seed + 11 * k + 2), r4(seed + 11 * k + 3);
        best = std::min(best, antColony(in, a, r1).length);
        best = std::min(best, tabuSearch(in, t, r2).length);
        best = std::min(best, simulatedAnnealing(in, s, r3).length);
        best = std::min(best, geneticAlgorithm(in, g, r4).length);
    }
    return best;
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    const uint64_t BASE_SEED = 20261124ULL;   // 固定主种子: 整个实验可复现
    int runsOverride = 0;
    if (argc > 1) runsOverride = std::max(1, std::atoi(argv[1]));

    // (城市数, 实例个数, 每实例重复轮数, 是否有精确参考解)
    struct Group { int n, nInst, runs; bool exact; };
    vector<Group> groups = {
        {10, 5, 30, true},
        {15, 5, 30, true},
        {20, 5, 30, true},
        {30, 3, 10, false},   // 扩展规模: 状压 DP 已不可行, 以 best-known 作参考
    };
    if (runsOverride > 0)
        for (auto& g : groups) g.runs = runsOverride;

    AcoParams acoP;
    TabuParams tabuP;
    SaParams saP;
    GaParams gaP;

    vector<InstResult> data;

    std::cout << "======================================================================\n";
    std::cout << " TSP experiment: ACO / Tabu / SA / GA  vs.  Held-Karp exact DP\n";
    std::cout << " master seed = " << BASE_SEED << "\n";
    std::cout << "======================================================================\n\n";

    // ---------------- 阶段 1: 生成实例 + 精确求解 ----------------
    for (const auto& g : groups) {
        for (int k = 0; k < g.nInst; ++k) {
            uint64_t seed = BASE_SEED + static_cast<uint64_t>(g.n) * 7919ULL + static_cast<uint64_t>(k) * 104729ULL;
            InstResult ir;
            ir.inst = makeInstance(g.n, seed, "R" + std::to_string(g.n) + "_" + std::to_string(k + 1));
            ir.exactRef = g.exact;
            if (g.exact) {
                std::cout << "[exact] " << ir.inst.name << " (n=" << g.n << ", seed=" << seed << ") ... " << std::flush;
                auto t0 = Clock::now();
                ir.ref = heldKarp(ir.inst);
                ir.exactMs = elapsedMs(t0);
                std::cout << "opt = " << std::fixed << std::setprecision(4) << ir.ref.length
                          << "  (" << std::setprecision(2) << ir.exactMs << " ms)";
                if (g.n <= 11) {   // 小规模再用暴力枚举交叉验证 DP
                    Solution bf = bruteForce(ir.inst);
                    bool same = std::fabs(bf.length - ir.ref.length) < 1e-6;
                    std::cout << "  [brute-force check " << (same ? "PASS" : "FAIL") << "]";
                }
                std::cout << "\n";
            } else {
                std::cout << "[bknwn] " << ir.inst.name << " (n=" << g.n << ", seed=" << seed << ") ... " << std::flush;
                auto t0 = Clock::now();
                ir.ref.length = bestKnownReference(ir.inst, acoP, tabuP, saP, gaP, seed ^ 0x9E3779B97F4A7C15ULL);
                ir.ref.ok = true;
                ir.exactMs = elapsedMs(t0);
                std::cout << "best-known = " << std::fixed << std::setprecision(4) << ir.ref.length
                          << "  (" << std::setprecision(2) << ir.exactMs << " ms)\n";
            }
            data.push_back(ir);
        }
    }
    std::cout << "\n";

    // ---------------- 阶段 2: 四种启发式算法多轮运行 ----------------
    for (auto& ir : data) {
        const Instance& in = ir.inst;
        std::cout << "---- " << in.name << " (n=" << in.n << ") ----\n";
        const Group* gp = nullptr;
        for (const auto& g : groups) if (g.n == in.n) gp = &g;
        const int runs = gp ? gp->runs : 10;

        for (int ai = 0; ai < 4; ++ai) {
            for (int r = 0; r < runs; ++r) {
                uint64_t seed = BASE_SEED * 1000003ULL
                              + static_cast<uint64_t>(in.n) * 7919ULL
                              + static_cast<uint64_t>(in.name.back() - '0') * 65537ULL
                              + static_cast<uint64_t>(ai) * 104729ULL
                              + static_cast<uint64_t>(r) * 2654435761ULL;
                Rng rng(seed);
                Solution s;
                auto t0 = Clock::now();
                switch (ai) {
                    case 0: s = antColony(in, acoP, rng); break;
                    case 1: s = tabuSearch(in, tabuP, rng); break;
                    case 2: s = simulatedAnnealing(in, saP, rng); break;
                    default: s = geneticAlgorithm(in, gaP, rng); break;
                }
                const double ms = elapsedMs(t0);
                RunRecord rec;
                rec.instance = in.name;
                rec.n = in.n;
                rec.algo = kAlgoName[ai];
                rec.run = r;
                rec.timeMs = ms;
                rec.length = s.length;
                rec.evals = s.evals;
                ir.recs.push_back(rec);
            }
        }

        // 非精确实例的参考值已在阶段 1 由"重预算 best-known"给出, 这里不再改动
        // 回填误差
        for (auto& r : ir.recs) {
            r.gap = r.length - ir.ref.length;
            if (std::fabs(r.gap) < 1e-6) r.gap = 0.0;       // 消除浮点求和顺序带来的 -0.0
            r.gapPct = (ir.ref.length > 0) ? r.gap / ir.ref.length * 100.0 : 0.0;
            if (std::fabs(r.gapPct) < 1e-9) r.gapPct = 0.0;
            r.hit = (r.gap == 0.0);
        }

        for (int ai = 0; ai < 4; ++ai) {
            vector<const RunRecord*> v;
            for (const auto& r : ir.recs) if (r.algo == kAlgoName[ai]) v.push_back(&r);
            Agg a = aggregate(v);
            std::cout << "  " << std::left << std::setw(6) << kAlgoName[ai]
                      << std::right << std::fixed
                      << " time " << std::setw(8) << std::setprecision(2) << a.timeMean << " ms"
                      << " | best " << std::setw(10) << std::setprecision(2) << a.lenMin
                      << " | mean " << std::setw(10) << std::setprecision(2) << a.lenMean
                      << " | meanErr " << std::setw(7) << std::setprecision(3) << a.gapMean
                      << " | hit " << std::setw(5) << std::setprecision(1) << a.hitRate * 100.0 << "%\n";
        }
        std::cout << "\n";
    }

    // ---------------- 阶段 3: 写 CSV ----------------
    {
        std::ofstream f("results/runs_raw.csv");
        f << "instance,n,algo,run,time_ms,length,gap_abs,gap_pct,is_optimal,evals\n";
        f << std::setprecision(10);
        for (const auto& ir : data)
            for (const auto& r : ir.recs)
                f << r.instance << "," << r.n << "," << r.algo << "," << r.run << ","
                  << r.timeMs << "," << r.length << "," << r.gap << "," << r.gapPct << ","
                  << (r.hit ? 1 : 0) << "," << r.evals << "\n";
    }
    {
        std::ofstream f("results/cities.csv");
        f << "instance,n,city,x,y\n";
        for (const auto& ir : data)
            for (int i = 0; i < ir.inst.n; ++i)
                f << ir.inst.name << "," << ir.inst.n << "," << i << ","
                  << ir.inst.x[i] << "," << ir.inst.y[i] << "\n";
    }
    {
        std::ofstream f("results/reference.csv");
        f << "instance,n,ref_type,ref_length,exact_dp_ms,tour\n";
        f << std::setprecision(10);
        for (const auto& ir : data)
            f << ir.inst.name << "," << ir.inst.n << "," << (ir.exactRef ? "EXACT" : "BEST_KNOWN") << ","
              << ir.ref.length << "," << ir.exactMs << ","
              << (ir.ref.ok ? tourToString(ir.ref.tour) : std::string("-")) << "\n";
    }
    {
        std::ofstream f("results/summary_by_instance.csv");
        f << "instance,n,ref_length,algo,runs,time_mean_ms,time_min_ms,time_max_ms,"
             "len_mean,len_min,len_sd,gap_mean,gap_sd,gap_max,gap_pct_mean,gap_pct_min,"
             "gap_pct_max,hit_rate,evals_mean\n";
        f << std::setprecision(10);
        for (const auto& ir : data) {
            for (int ai = 0; ai < 4; ++ai) {
                vector<const RunRecord*> v;
                for (const auto& r : ir.recs) if (r.algo == kAlgoName[ai]) v.push_back(&r);
                Agg a = aggregate(v);
                f << ir.inst.name << "," << ir.inst.n << "," << ir.ref.length << "," << kAlgoName[ai] << ","
                  << a.samples << "," << a.timeMean << "," << a.timeMin << "," << a.timeMax << ","
                  << a.lenMean << "," << a.lenMin << "," << a.lenSd << ","
                  << a.gapMean << "," << a.gapSd << "," << a.gapMax << ","
                  << a.pctMean << "," << a.pctMin << "," << a.pctMax << ","
                  << a.hitRate << "," << a.evalsMean << "\n";
            }
        }
    }
    {
        std::ofstream f("results/summary_by_size.csv");
        f << "n,algo,instances,runs_per_instance,samples,ref_type,time_mean_ms,time_min_ms,time_max_ms,"
             "len_mean,len_min,len_sd,gap_mean,gap_sd,gap_max,gap_pct_mean,gap_pct_min,"
             "gap_pct_max,gap_pct_sd,hit_rate,evals_mean\n";
        f << std::setprecision(10);
        for (const auto& g : groups) {
            for (int ai = 0; ai < 4; ++ai) {
                vector<const RunRecord*> v;
                for (const auto& ir : data)
                    if (ir.inst.n == g.n)
                        for (const auto& r : ir.recs) if (r.algo == kAlgoName[ai]) v.push_back(&r);
                Agg a = aggregate(v);
                f << g.n << "," << kAlgoName[ai] << "," << g.nInst << "," << g.runs << "," << a.samples << ","
                  << (g.exact ? "EXACT" : "BEST_KNOWN") << ","
                  << a.timeMean << "," << a.timeMin << "," << a.timeMax << ","
                  << a.lenMean << "," << a.lenMin << "," << a.lenSd << ","
                  << a.gapMean << "," << a.gapSd << "," << a.gapMax << ","
                  << a.pctMean << "," << a.pctMin << "," << a.pctMax << "," << a.pctSd << ","
                  << a.hitRate << "," << a.evalsMean << "\n";
            }
        }
    }

    // ---------------- 阶段 4: 控制台汇总(按规模) ----------------
    std::cout << "======================================================================\n";
    std::cout << " SUMMARY by size (mean over instances x runs)\n";
    std::cout << "======================================================================\n";
    std::cout << std::left << std::setw(5) << "n" << std::setw(7) << "algo"
              << std::right << std::setw(11) << "time(ms)" << std::setw(12) << "errPct_avg"
              << std::setw(11) << "errPct_max" << std::setw(9) << "sd"
              << std::setw(12) << "errAbs_avg" << std::setw(9) << "hit%"
              << std::setw(10) << "samples" << "\n";
    std::cout << std::string(85, '-') << "\n";
    for (const auto& g : groups) {
        for (int ai = 0; ai < 4; ++ai) {
            vector<const RunRecord*> v;
            for (const auto& ir : data)
                if (ir.inst.n == g.n)
                    for (const auto& r : ir.recs) if (r.algo == kAlgoName[ai]) v.push_back(&r);
            Agg a = aggregate(v);
            std::cout << std::left << std::setw(5) << g.n << std::setw(7) << kAlgoName[ai]
                      << std::right << std::fixed
                      << std::setw(11) << std::setprecision(2) << a.timeMean
                      << std::setw(12) << std::setprecision(4) << a.pctMean
                      << std::setw(11) << std::setprecision(4) << a.pctMax
                      << std::setw(9) << std::setprecision(3) << a.pctSd
                      << std::setw(12) << std::setprecision(3) << a.gapMean
                      << std::setw(9) << std::setprecision(1) << a.hitRate * 100.0
                      << std::setw(10) << a.samples << "\n";
        }
        std::cout << std::string(85, '-') << "\n";
    }
    std::cout << "\nCSV files written to ./results/\n";
    return 0;
}

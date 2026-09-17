#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <map>

namespace hnnx {

// Multicast optimizer: grdep_mcast_optimizer.cc
// Uses HiGHS LP/MIP solver to optimize multicast channel (MCID) scheduling
// Minimizes total MCID usage, can merge McSends into supercasts

struct McSend {
    uint32_t tag;           // mcsend tag
    uint32_t sender_nsp;    // sending NSP
    uint32_t num_mcids;     // number of multicast channel IDs
    uint64_t payload_size;  // payload size in bytes
    std::vector<uint32_t> mcids;        // multicast channel IDs
    std::vector<uint32_t> receivers;   // receiving NSPs
};

struct SuperCast {
    std::vector<uint32_t> mcsend_tags;  // merged McSend tags
    std::vector<uint32_t> mcids;        // combined MCIDs
    uint32_t sender_nsp;
    uint64_t total_payload;
};

// ILP variable: ~(num_mcsends * 2) + graph_multicast_count
struct MCastLPInput {
    std::vector<McSend> mcsends;
    std::vector<SuperCast> supercasts;
    uint32_t max_mcsend_tag;
    uint32_t graph_multicast_count;
    uint64_t max_mcast_buffer_size = 0;  // Phase 4.3: NSP multicast buffer cap
};

// Multicast ILP optimizer
// Top-level: 0x1075FF0 (720 bytes)
// LP builder: 0x1176340 (12614 bytes, 98KB decompiled)
// Result extractor: 0x10758E0
class McastOptimizer {
public:
    McastOptimizer();
    ~McastOptimizer();

    // Optimize multicast schedule
    // Returns optimized McSend list with merged supercasts
    std::vector<McSend> optimize(
        const std::vector<McSend>& mcsends,
        uint32_t graph_multicast_count);

    // Build LP input from McSend list
    MCastLPInput build_lp_input(
        const std::vector<McSend>& mcsends,
        uint32_t graph_multicast_count);

    // Solve the ILP using HiGHS
    // Returns true if optimal solution found
    bool solve_ilp(const MCastLPInput& input);

    // Apply optimization results
    std::vector<McSend> apply_results(
        const std::vector<McSend>& original,
        const std::vector<SuperCast>& supercasts);

    // Dump LP to MPS file format
    void dump_mps(const std::string& filename) const;

    // Configuration
    void set_run_crossover(bool enable) { run_crossover_ = enable; }
    void set_dump_mps(bool enable) { dump_mps_ = enable; }
    // Phase 4.3: set NSP multicast buffer size limit (bytes).
    // Supercasts exceeding this are rejected by the solver.
    void set_max_mcast_buffer_size(uint64_t size) { max_mcast_buffer_size_ = size; }

    // Statistics (Phase 4.3)
    size_t supercast_count() const { return last_supercast_count_; }
    size_t original_mcsend_count() const { return last_original_count_; }
    size_t result_mcsend_count() const { return last_result_count_; }

private:
    bool run_crossover_ = false;
    bool dump_mps_ = false;
    uint64_t max_mcast_buffer_size_ = 0;  // 0 = no limit (Phase 4.3)

    // Statistics from last optimize() call
    size_t last_supercast_count_ = 0;
    size_t last_original_count_ = 0;
    size_t last_result_count_ = 0;

    // HiGHS SimplexSolver wrapper
    // Source: clp_simplex.cc
    struct SimplexSolver {
        bool create_and_populate(const MCastLPInput& input);
        bool solve();
        void get_solution(std::vector<double>& primal, std::vector<double>& dual,
                         int& status, int& iterations, double& objective) const;

        // Phase 4.3: greedy-improvement solver state
        // Instead of requiring HiGHS, we implement a simple local-search
        // improvement over the greedy solution: try un-merging supercasts
        // that violate capacity or receiver-conflict constraints.
        std::vector<double> primal_solution_;
        int status_ = 0;       // 0=not solved, 1=optimal, 2=feasible
        int iterations_ = 0;
        double objective_ = 0.0;
    };

    SimplexSolver solver_;

    // LP model
    int num_variables_ = 0;
    int num_constraints_ = 0;
    std::vector<double> objective_coeffs_;
    std::vector<std::vector<double>> constraint_matrix_;
    std::vector<double> constraint_lb_;
    std::vector<double> constraint_ub_;
    std::vector<double> var_lb_;
    std::vector<double> var_ub_;

    // Phase 4.3: check if two McSends can be merged (capacity + receiver conflict)
    bool can_merge(const McSend& a, const McSend& b, uint64_t merged_payload) const;
    // Phase 4.3: check receiver-set disjointness (partial overlap = conflict)
    static bool receivers_conflict(const std::vector<uint32_t>& ra,
                                   const std::vector<uint32_t>& rb);
};

} // namespace hnnx

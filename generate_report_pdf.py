"""
Generate cs23b1019_report.pdf - A professional 2-page project report
for the Hamiltonian Hyperparameter Dynamics (HHD-ABBO) framework.

Uses fpdf2 with Unicode TTF fonts. Tightly formatted for exactly 2 pages.
All 28 citations from ieee_paper.tex included.
"""

from fpdf import FPDF
import os

FONTS_DIR = os.path.join(os.environ["WINDIR"], "Fonts")

class ReportPDF(FPDF):
    def __init__(self):
        super().__init__(format="A4")
        self.set_auto_page_break(auto=True, margin=14)
        self.set_margins(12, 10, 12)

        self.add_font("Cal", "", os.path.join(FONTS_DIR, "calibri.ttf"))
        self.add_font("Cal", "B", os.path.join(FONTS_DIR, "calibrib.ttf"))
        self.add_font("Cal", "I", os.path.join(FONTS_DIR, "calibrii.ttf"))
        self.add_font("Cal", "BI", os.path.join(FONTS_DIR, "calibriz.ttf"))
        self.add_font("Ar", "", os.path.join(FONTS_DIR, "arial.ttf"))
        self.add_font("Ar", "B", os.path.join(FONTS_DIR, "arialbd.ttf"))

    def header(self):
        if self.page_no() > 1:
            self.set_font("Cal", "I", 7)
            self.set_text_color(130, 130, 130)
            self.cell(0, 5, "Desai | CS23B1019 | HHD-ABBO: Self-Tuning Hyperparameter Optimization via Hamiltonian Dynamics", align="C")
            self.ln(2)
            self.set_draw_color(190, 190, 190)
            self.line(12, self.get_y(), 198, self.get_y())
            self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font("Cal", "I", 7)
        self.set_text_color(130, 130, 130)
        self.cell(0, 8, f"Page {self.page_no()}/{{nb}}", align="C")

    def sec(self, num, title):
        self.set_font("Cal", "B", 11)
        self.set_text_color(18, 50, 110)
        self.cell(0, 5.0, f"{num}. {title}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(18, 50, 110)
        self.line(12, self.get_y(), 198, self.get_y())
        self.ln(1)

    def sub(self, title):
        self.set_font("Cal", "B", 9)
        self.set_text_color(40, 40, 40)
        self.cell(0, 4.5, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(0.2)

    def p(self, text):
        self.set_font("Cal", "", 8.5)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 3.6, text)
        self.ln(0.5)

    def bp(self, bold_part, text):
        self.set_font("Cal", "B", 8.8)
        self.set_text_color(30, 30, 30)
        w = self.get_string_width(bold_part) + 0.5
        self.cell(w, 3.8, bold_part)
        self.set_font("Cal", "", 8.8)
        self.multi_cell(0, 3.8, text)
        self.ln(0.5)

    def bl(self, bold_part, text):
        self.set_font("Cal", "", 8.5)
        self.set_text_color(30, 30, 30)
        self.cell(4, 3.6, "\u2022")
        if bold_part:
            self.set_font("Cal", "B", 8.5)
            w = self.get_string_width(bold_part) + 0.5
            self.cell(w, 3.6, bold_part)
            self.set_font("Cal", "", 8.5)
        self.multi_cell(0, 3.6, text)
        self.ln(0.2)

    def eq(self, text):
        self.set_font("Ar", "", 8.5)
        self.set_text_color(50, 50, 50)
        self.set_fill_color(244, 244, 250)
        self.cell(0, 5.5, text, align="C", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def ref(self, num, text):
        self.set_font("Cal", "", 6.8)
        self.set_text_color(45, 45, 45)
        tag = f"[{num}] "
        tw = self.get_string_width(tag) + 0.5
        self.cell(tw, 3.0, tag)
        self.multi_cell(0, 3.0, text)
        self.ln(0.1)


def build_report():
    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # ── Title ──────────────────────────────────────────────────────────────
    pdf.set_font("Cal", "B", 15)
    pdf.set_text_color(14, 42, 100)
    pdf.multi_cell(0, 6.5, "Self-Tuning Hyperparameter Optimization\nvia Hamiltonian Dynamics", align="C")
    pdf.ln(1)
    pdf.set_font("Cal", "I", 9)
    pdf.set_text_color(65, 65, 65)
    pdf.cell(0, 4.5, "A Comparative Study of HHD, Hybrid Optimisation, and a Unified Framework", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1.5)
    pdf.set_font("Cal", "", 9)
    pdf.set_text_color(85, 85, 85)
    pdf.cell(0, 4.5, "Kartik Desai  |  CS23B1019  |  IIIT Raichur  |  May 2026", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    pdf.set_draw_color(18, 50, 110)
    pdf.set_line_width(0.4)
    pdf.line(12, pdf.get_y(), 198, pdf.get_y())
    pdf.ln(3)

    # ── 1. Abstract ────────────────────────────────────────────────────────
    pdf.sec("1", "Abstract")
    pdf.p(
        "We present and compare three hyperparameter optimisation strategies "
        "for neural networks trained on the harmonic oscillator energy surface "
        "H(q,p)=p\u00b2/(2m)+\u00bdkq\u00b2 and on MNIST CNN classification:"
    )
    pdf.bl("(A) Hamiltonian Hyperparameter Dynamics (HHD), ", "which uses symplectic Hamiltonian Monte Carlo (HMC) [8, 9] to co-evolve network weights and hyperparameters;")
    pdf.bl("(B) a Hybrid Adam + L-BFGS + Bayesian Optimisation (ABBO) ", "method [1, 2, 6]; and")
    pdf.bl("(C) a novel Unified HHD-ABBO framework ", "that integrates all three optimisers into a single three-phase curriculum.")
    pdf.ln(0.2)
    pdf.p(
        "On the harmonic oscillator, the Unified method (Method C) achieves the lowest landscape MSE (0.0033) "
        "and highest R\u00b2 (0.9999), outperforming ABBO (Method B) by 30\u00d7 in error magnitude. On MNIST, "
        "Method C achieves 97.70% validation accuracy, completing in 3.3\u00d7 less wall time than Method B "
        "(214.8s vs 698.7s). We provide a comprehensive analysis including dataset visualisation, loss convergence, "
        "energy surface reconstruction, Hamiltonian conservation plots, and multi-metric radar comparisons."
    )

    # ── 2. HP Tuning ──────────────────────────────────────────────────────
    pdf.sec("2", "Hyperparameter Tuning Approach")
    pdf.p(
        "The central innovation formulates HPO as a physics problem [10, 15, 16]. The training loss "
        "L(\u03b8, \u03bb) acts as potential energy V in an extended Hamiltonian where weights (\u03b8) and "
        "hyperparameters (\u03bb) are particles with mass and momentum [9, 17]:"
    )
    pdf.eq("H(\u03b8, p\u03b8, \u03bb, p\u03bb)  =  T(p\u03b8)/m\u03b8  +  T(p\u03bb)/m\u03bb  +  L(\u03b8, \u03bb)")
    pdf.p(
        "The system evolves via symplectic leapfrog integration [11, 12], ensuring bounded energy "
        "error O(\u03b5\u00b2). Key hyperparameters include learning rate (log-scaled), dropout, depth, "
        "width, and batch size - all continuously relaxed [18, 24]. Structural HPs (e.g., layer count) "
        "use weight-transfer architecture rebuilding [20]: building a temporary model with "
        "the proposed layer count, copying compatible weights, and computing a surrogate gradient."
    )
    pdf.p(
        "HP gradients use central finite differences (\u00b110% perturbation); weight gradients use "
        "autodifferentiation. An adaptive step-size controller monitors acceptance rate and "
        "gradient norm for stability [25, 27]. Metropolis\u2013Hastings correction guarantees detailed "
        "balance and ergodicity [9, 10]."
    )

    # ── 3. Algorithms ─────────────────────────────────────────────────────
    pdf.sec("3", "Proposed Algorithms and Workflow")

    pdf.sub("3.1  Method A - Pure Hamiltonian Hyperparameter Dynamics (HHD)")
    pdf.p(
        "Method A treats hyperparameter optimization as a purely physical simulation operating in two continuous phases. "
        "First, a 20-epoch Adam warm-up [1] descends rapidly into a stable loss basin, preventing chaotic initial dynamics. "
        "Second, HMC co-evolution [8, 9] begins: weights and hyperparameters are assigned artificial momenta, and the entire "
        "system evolves through the loss landscape using symplectic leapfrog integration over 60 epochs. A Metropolis\u2013Hastings "
        "step [10] ensures detailed balance. The hyperparameters glide through smooth, physically interpretable trajectories "
        "without discrete restarts. Strength: Guaranteed symplectic energy conservation [11] and elegant physical continuity. "
        "Weakness: Lacks second-order curvature information to escape narrow plateaus."
    )

    pdf.sub("3.2  Method B - Hybrid Adam + L-BFGS + Bayesian Optimization (ABBO)")
    pdf.p(
        "Method B represents a powerful, decoupled black-box approach that wraps a conventional optimizer sequence inside a "
        "probabilistic outer loop. The outer loop uses a Gaussian Process (GP) [7] surrogate model and the Expected Improvement "
        "(EI) acquisition function [5, 19] to propose static hyperparameter configurations across 15 independent trials (5 random, "
        "10 GP-guided). Inside each trial, the network is reinitialized and trained from scratch using Adam for 15 epochs, followed "
        "by 10 full-batch steps of L-BFGS [2, 4] for tight convergence. Strength: Unmatched reconstruction accuracy due to "
        "extensive parallel exploration and L-BFGS precision. Weakness: Extremely computationally expensive due to full "
        "network reinitialization and discontinuous hyperparameter jumps [6]."
    )

    pdf.sub("3.3  Method C - Unified HHD-ABBO (Novel)")
    pdf.p(
        "Method C is the principal novel contribution, integrating the physical elegance of Method A with the second-order precision "
        "of Method B into a cohesive three-phase epoch curriculum. There is no outer BO loop; hyperparameters evolve intra-epoch:"
    )
    pdf.bl("Phase 1 - Adam Micro-Epochs: ", "Executes 3 micro-epochs per outer epoch with a cosine-annealed learning rate and gradient clipping [1]. This rapidly navigates the local basin while keeping HPs momentarily frozen.")
    pdf.bl("Phase 2 - HMC Co-evolution: ", "Performs a 6-step leapfrog integration [11] to jointly update \u03b8 and \u03bb. An adaptive step-size controller targets a ~65% acceptance rate [10], preserving symplectic energy globally.")
    pdf.bl("Phase 3 - Plateau-Triggered L-BFGS: ", "A continuous monitor [23] checks for local stagnation (tolerance \u03c4=5e-5). If triggered, a strong Wolfe line search [3] executes 30 steps of L-BFGS to exploit second-order curvature [2].")
    pdf.ln(0.2)

    pdf.sub("3.4  Execution Workflow")
    pdf.p(
        "The execution pipeline sequentially runs: config.py \u2192 main.py (dynamically instantiating Methods "
        "A, B, and C) \u2192 results/ serialization \u2192 evaluate.py (computing MAE, RMSE, R\u00b2) \u2192 generating "
        "figures in the plots/ directory."
    )

    # ── 4. Technologies & Frameworks ──────────────────────────────────────────
    pdf.sec("4", "Technologies and Frameworks Used")
    pdf.p(
        "This project leverages PyTorch [30] for autodifferentiation, dynamic computational graphs, and constructing "
        "the neural networks. SciPy [31] handles Gaussian Process surrogate modeling and L-BFGS-B acquisition function "
        "maximization for Bayesian Optimization. Custom Python scripts manage the symplectic Hamiltonian integration, "
        "Metropolis-Hastings sampling, and the cohesive evaluation pipeline."
    )

    # ── 5. Validation Methodology ──────────────────────────────────────────
    pdf.add_page()
    pdf.sec("5", "Validation Methodology")
    pdf.p(
        "To rigorously quantify theoretical and empirical properties, evaluation routines continuously compute "
        "reconstruction residuals and accuracy metrics. The models are tested on two distinct paradigms:"
    )
    pdf.set_font("Cal", "B", 8)
    pdf.set_fill_color(220, 230, 245)
    pdf.cell(38, 5, "Benchmark", border=1, fill=True)
    pdf.cell(30, 5, "Metric", border=1, fill=True)
    pdf.cell(118, 5, "Calculation & Purpose", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Cal", "", 8)
    pdf.cell(38, 5, "Harmonic Osc.", border=1)
    pdf.cell(30, 5, "Landscape MAE", border=1)
    pdf.cell(118, 5, "Mean Absolute Error: avg(|H_pred - H_true|) across the 6,400-point 2D energy grid.", border=1, new_x="LMARGIN", new_y="NEXT")

    pdf.cell(38, 5, "Harmonic Osc.", border=1)
    pdf.cell(30, 5, "Landscape RMSE", border=1)
    pdf.cell(118, 5, "Root Mean Squared Error: Highlights severe local deviations in energy prediction.", border=1, new_x="LMARGIN", new_y="NEXT")

    pdf.cell(38, 5, "Harmonic Osc.", border=1)
    pdf.cell(30, 5, "R\u00b2 Score", border=1)
    pdf.cell(118, 5, "Coefficient of Determination: Quantifies the overall structural fit of the manifold.", border=1, new_x="LMARGIN", new_y="NEXT")

    pdf.cell(38, 4.5, "MNIST CNN", border=1)
    pdf.cell(30, 4.5, "Val. Accuracy", border=1)
    pdf.cell(118, 4.5, "Percentage of correctly classified unseen images, testing effective regularization.", border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    # ── 6. Results ─────────────────────────────────────────────────────────
    pdf.sec("6", "Results")

    pdf.sub("6.1  Harmonic Oscillator Benchmark")
    pdf.p(
        "Background & Setup: The physics-driven testbed is the Harmonic Oscillator energy surface, utilizing a dense "
        "6,400-point 2D grid. The network aims to reconstruct a quadratic potential V(q)=\u00bdkq\u00b2. "
        "This rigorously tests the algorithm's capability for continuous landscape learning."
    )
    pdf.p(
        "Performance: Method C achieved the lowest best validation loss (0.0033 MSE) vs. Method B (0.099) and "
        "Method A (0.151). Testbed: Method C - R\u00b2=0.9999 in 81.4s (1.8\u00d7 faster than B); Method B - "
        "R\u00b2=0.9984 in 147.1s. Method A was fastest (17.8s) but least precise (R\u00b2=0.9498). "
        "Crucially, only Methods A and C preserved Hamiltonian energy conservation [11, 12], "
        "detailed balance [9], and generated smooth continuous HP trajectories."
    )

    pdf.sub("6.2  CNN MNIST Classification Benchmark")
    pdf.p(
        "Background & Setup: The secondary testbed is a traditional Convolutional Neural Network (CNN) classification "
        "task on the MNIST dataset, explicitly configured with a restricted 5,000 training samples and 1,000 validation "
        "samples to purposefully enforce overfitting pressure and demand effective regularization."
    )
    pdf.p(
        "Performance: Method A achieved 97.80% validation accuracy, Method C achieved 97.70%, and "
        "Method B reached 97.40%. Method C completed in 214.8s - 3.3\u00d7 less than Method B's 698.7s "
        "(which required 15 BO trials). Method A completed in 122.7s. The final dropout selections "
        "for Methods A, B, and C converged to 0.21, 0.22, and 0.24 respectively, showing stable "
        "regularisation behavior across all models."
    )

    pdf.sub("6.3  Overall Assessment")
    pdf.p(
        "Method C consistently balanced accuracy, theoretical guarantees (symplectic conservation "
        "[11], detailed balance [9, 10]), second-order convergence [2], and efficiency - validating "
        "it as a principled alternative to standalone HMC and traditional BO [6, 13]. While Method B "
        "remains the best for pure, unconstrained landscape reconstruction, Method C proves to be the "
        "superior framework in realistic, compute-constrained deep learning scenarios."
    )

    # ── 7. Future Applications ────────────────────────────────────────────
    pdf.sec("7", "Future Applications of Method C")

    pdf.bl("Automated NAS: ",
        "Continuous relaxations would let the Hamiltonian system dynamically grow/prune layers "
        "in a single physics simulation, eliminating thousands of separate architecture trials [18].")
    pdf.bl("LLM Pre-training: ",
        "Method C enables on-the-fly self-correction of LR, weight decay, and masking "
        "probabilities via energy conservation [11], saving thousands of GPU hours.")
    pdf.bl("Physics-Informed Neural Networks: ",
        "Addressing gradient pathologies [21, 22, 27] and optimizer selection [23, 28] in PINNs "
        "by letting the Hamiltonian system co-evolve PDE-specific hyperparameters adaptively [20, 25].")
    pdf.bl("Continual Learning: ",
        "HP kinetic energy naturally increases on distribution shift, re-initiating exploration "
        "without catastrophic forgetting [26].")
    pdf.bl("HPOBench Integration: ",
        "Evaluating the Unified HHD-ABBO framework against comprehensive, multi-family hyperparameter "
        "optimization benchmark suites like HPOBench [29] to rigorously quantify generalization bounds "
        "across diverse dataset modalities. Frameworks like PyTorch [30] and SciPy [31] will enable "
        "scaling these implementations.")

    # ── 8. References ─────────────────────────────────────────────────────
    pdf.sec("8", "References")

    refs = [
        'D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in Proc. ICLR, 2015.',
        'D. C. Liu and J. Nocedal, "On the Limited Memory BFGS Method for Large Scale Optimization," Math. Prog., vol. 45, no. 1\u20133, pp. 503\u2013528, 1989.',
        'P. Wolfe, "Convergence Conditions for Ascent Methods," SIAM Review, vol. 11, no. 2, pp. 226\u2013235, 1969.',
        'R. H. Byrd, P. Lu, J. Nocedal, and C. Zhu, "A Limited Memory Algorithm for Bound Constrained Optimization," SIAM J. Sci. Comput., vol. 16, no. 5, pp. 1190\u20131208, 1995.',
        'J. Mockus, "On Bayesian Methods for Seeking the Extremum," in Optimization Techniques IFIP Technical Conference, 1975.',
        'J. Snoek, H. Larochelle, and R. P. Adams, "Practical Bayesian Optimization of ML Algorithms," in NeurIPS, vol. 25, 2012.',
        'C. E. Rasmussen and C. K. I. Williams, Gaussian Processes for Machine Learning. MIT Press, 2006.',
        'S. Duane, A. D. Kennedy, B. J. Pendleton, and D. Roweth, "Hybrid Monte Carlo," Physics Letters B, vol. 195, no. 2, pp. 216\u2013222, 1987.',
        'R. M. Neal, "MCMC Using Hamiltonian Dynamics," in Handbook of Markov Chain Monte Carlo, CRC Press, 2011, ch. 5, pp. 113\u2013162.',
        'M. Betancourt, "A Conceptual Introduction to Hamiltonian Monte Carlo," arXiv:1701.02434, 2017.',
        'B. Leimkuhler and S. Reich, Simulating Hamiltonian Dynamics. Cambridge University Press, 2004.',
        'E. Hairer, C. Lubich, and G. Wanner, Geometric Numerical Integration, 2nd ed. Springer, 2006.',
        'J. Bergstra and Y. Bengio, "Random Search for Hyper-Parameter Optimization," JMLR, vol. 13, pp. 281\u2013305, 2012.',
        'Y. LeCun, Y. Bengio, and G. Hinton, "Deep Learning," Nature, vol. 521, no. 7553, pp. 436\u2013444, 2015.',
        'S. Greydanus, M. Dzamba, and J. Yosinski, "Hamiltonian Neural Networks," in NeurIPS, vol. 32, 2019.',
        'Z. Zhang, S. Bai, Y. Liang, and Z. Bao, "Neural Networks Under Hamiltonian Constraints: A Comprehensive Review," IEEE Access, 2025.',
        'H. Goldstein, C. Poole, and J. Safko, Classical Mechanics, 3rd ed. Addison-Wesley, 2002.',
        'M. A. K. Raiaan et al., "A Systematic Review of HP Optimization Techniques in CNNs," Decision Analytics J., vol. 11, p. 100470, 2024.',
        'P. I. Frazier, "A Tutorial on Bayesian Optimization," arXiv:1807.02811, 2018.',
        'P. Rathore, W. Lei, Z. Frangella, L. Lu, and M. Udell, "Challenges in Training PINNs: A Loss Landscape Perspective," in Proc. 41st ICML, PMLR 235, 2024.',
        'A. Krishnapriyan et al., "Characterizing Possible Failure Modes in Physics-Informed Neural Networks," in NeurIPS, vol. 34, pp. 26548\u201326560, 2021.',
        'S. Wang, S. Sankaran, H. Wang, and P. Perdikaris, "An Expert\'s Guide to Training PINNs," arXiv:2308.08468, 2023.',
        'E. Kiyani et al., "Optimizing the Optimizer for PINNs and Kolmogorov-Arnold Networks," arXiv:2501.16371, 2025.',
        'M. Jin et al., "Hyperparameter Tuning of ANNs for Well Production Estimation," ACS Omega, vol. 7, no. 28, pp. 24145\u201324156, 2022.',
        'W. Chen, A. A. Howard, and P. Stinis, "Self-Adaptive Weights Based on Balanced Residual Decay Rate for PINNs," arXiv:2404.18055, 2024.',
        'L. Newhouse, Mathematical Techniques for Tuning Hyperparameters of RNNs. Research Monograph, 2024.',
        'S. Wang, Y. Teng, and P. Perdikaris, "Understanding and Mitigating Gradient Pathologies in PINNs," SIAM J. Sci. Comput., 2021.',
        'J. C. Wong et al., "Evolutionary Optimization of PINNs: Evo-PINN Frontiers," arXiv:2501.06572, 2025.',
        'K. Eggensperger et al., "HPOBench: A Collection of Reproducible Multi-Fidelity Benchmark Problems for HPO," in NeurIPS Datasets and Benchmarks, 2021.',
        'A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," in NeurIPS, vol. 32, 2019.',
        'P. Virtanen et al., "SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python," Nature Methods, vol. 17, pp. 261\u2013272, 2020.',
    ]

    for i, r in enumerate(refs, 1):
        pdf.ref(i, r)

    # ── Output ─────────────────────────────────────────────────────────────
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cs23b1019_report.pdf")
    pdf.output(out)
    print(f"\nReport generated: {out}")
    print(f"Total pages: {pdf.pages_count}")


if __name__ == "__main__":
    build_report()

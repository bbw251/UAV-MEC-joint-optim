"""
UAV-Aided Mobile Edge Computing Networks: Joint Optimization Simulation

This module implements the optimization algorithms for UAV deployment and user association
in mobile edge computing networks, reproducing results from academic research.

Author: Research Implementation
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
import pandas as pd

# Configure plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
class Parameters:
    """System parameters for UAV-aided MEC network simulation."""
    
    def __init__(self):
        # Environment parameters
        self.AREA_SIZE = 1000  # Coverage area: 1km x 1km
        self.N_USERS = 100
        self.N_UAV = 3
        self.N_TBS = 1
        self.N_BS = self.N_UAV + self.N_TBS
        
        # User distribution (bi-modal Gaussian)
        self.MEAN_1 = [330, 330]
        self.MEAN_2 = [660, 660]
        self.COV = [[20000, 0], [0, 20000]]
        
        # Task and computation parameters
        self.M_TASK = 1e6      # Task size: 1 Mbit
        self.L_TASK = 60e9     # Computational load: 60 GFLOP
        self.ALPHA_RATE = 0.015 # Task arrival rate (tasks/sec)             #0.015
        self.C_TBS = 5500e9    # TBS computational capacity (GFLOP/s)     #5500e9
        self.C_UAV = 4500e9    # UAV computational capacity (GFLOP/s)     #4500e9

        # Communication parameters
        self.FC = 2e9          # Carrier frequency: 2 GHz
        self.BW_TBS = 12e6     # TBS bandwidth: 12 MHz                             #12e6
        self.BW_UAV = 10e6     # UAV bandwidth: 10 MHz                              #10e6
        self.P_USER = 2        # User transmission power: 2 W
        self.NOISE_DENSITY = 10**(-170/10) / 1000  # Noise density: -170 dBm/Hz
        
        # Path loss model parameters (dense urban environment)
        self.PL_EXP_TBS = 3.0
        self.A_LOS = 8.96
        self.B_LOS = 0.04
        self.MU_LOS = 10**(3/10)    # 3 dB
        self.MU_NLOS = 10**(23/10)  # 23 dB
        self.K0 = (4 * np.pi * self.FC / (3e8))**2
        
        # Base station locations
        self.TBS_LOCS = np.array([[200, 200, 50]])
        
        # PSO optimization parameters
        self.PSO_ITER = 100
        self.PSO_PARTICLES = 30
        self.PSO_W = 0.5      # Inertia weight
        self.PSO_C1 = 1.5     # Cognitive coefficient
        self.PSO_C2 = 1.5     # Social coefficient
        

def generate_users(p):
    """Generate user locations based on bi-modal Gaussian distribution."""
    num_users1 = p.N_USERS // 2
    num_users2 = p.N_USERS - num_users1
    
    dist1 = multivariate_normal(mean=p.MEAN_1, cov=p.COV)
    dist2 = multivariate_normal(mean=p.MEAN_2, cov=p.COV)
    
    users1 = dist1.rvs(size=num_users1)
    users2 = dist2.rvs(size=num_users2)
    
    users = np.vstack([users1, users2])
    users = np.clip(users, 0, p.AREA_SIZE)  # Ensure users are within area
    return np.hstack([users, np.zeros((p.N_USERS, 1))])  # Add z-coordinate

# Delay calculation models

def path_loss_tbs(user_loc, tbs_loc, p):
    """Calculate path loss for terrestrial base station."""
    dist_3d = np.linalg.norm(user_loc - tbs_loc)
    return p.K0 * (dist_3d**p.PL_EXP_TBS)

def path_loss_uav(user_loc, uav_loc, p):
    """Calculate probabilistic path loss for UAV with LoS/NLoS consideration."""
    dist_2d = np.linalg.norm(user_loc[:2] - uav_loc[:2])
    h = uav_loc[2]
    elevation_angle = np.arctan(h / dist_2d) * (180/np.pi)
    
    prob_los = 1 / (1 + p.A_LOS * np.exp(-p.B_LOS * (elevation_angle - p.A_LOS)))
    
    dist_3d_sq = dist_2d**2 + h**2
    pl_los = p.MU_LOS * p.K0 * dist_3d_sq
    pl_nlos = p.MU_NLOS * p.K0 * dist_3d_sq
    
    return pl_los * prob_los + pl_nlos * (1 - prob_los)

def calculate_delay(user_loc, bs_loc, is_uav, p, n_k):
    """Calculate total delay (communication + processing + queuing) for a user."""
    # Communication delay
    if is_uav:
        pl = path_loss_uav(user_loc, bs_loc, p)
        bw = p.BW_UAV
        comp_cap = p.C_UAV
    else:
        pl = path_loss_tbs(user_loc, bs_loc, p)
        bw = p.BW_TBS
        comp_cap = p.C_TBS
        
    noise_power = p.NOISE_DENSITY * bw
    rate = bw * np.log2(1 + p.P_USER / (pl * noise_power))
    tc = p.M_TASK / rate
    
    # Processing and queuing delay (M/D/1 model)
    if p.ALPHA_RATE * n_k * p.L_TASK >= comp_cap:
        return np.inf  # Unstable queue
    
    tp_numerator = 2 * p.L_TASK * comp_cap - p.ALPHA_RATE * (p.L_TASK**2) * n_k
    tp_denominator = 2 * comp_cap * (comp_cap - p.ALPHA_RATE * n_k * p.L_TASK)
    tp = tp_numerator / tp_denominator
    
    return tc + tp

# Association and optimization algorithms

def ott_association(users, bs_locs, p):
    """Optimal Transport Theory-based user association (Algorithm 1)."""
    n_users = len(users)
    
    # Initial association based on closest base station
    dists = cdist(users, bs_locs)
    association = np.argmin(dists, axis=1)
    
    # Iterative optimization
    for _ in range(10):  # Typically converges quickly
        old_association = np.copy(association)
        n_k = np.bincount(association, minlength=p.N_BS)
        delays = np.zeros((n_users, p.N_BS))
        
        for i in range(n_users):
            current_bs_idx = old_association[i]
            for k in range(p.N_BS):
                is_uav = k >= p.N_TBS
                # Calculate prospective load
                if k == current_bs_idx:
                    prospective_n_k = n_k[k]
                else:
                    prospective_n_k = n_k[k] + 1
                
                delays[i, k] = calculate_delay(users[i], bs_locs[k], is_uav, p, prospective_n_k)
        
        # Update associations to minimize delay
        new_association = np.argmin(delays, axis=1)
        
        if np.array_equal(association, new_association):
            break
        association = new_association
    
    return association

def snr_association(users, bs_locs, p):
    """SNR-based user association (minimum path loss)."""
    n_users = len(users)
    path_losses = np.zeros((n_users, p.N_BS))

    for i in range(n_users):
        for k in range(p.N_BS):
            is_uav = k >= p.N_TBS
            if is_uav:
                path_losses[i, k] = path_loss_uav(users[i], bs_locs[k], p)
            else:
                path_losses[i, k] = path_loss_tbs(users[i], bs_locs[k], p)
                
    return np.argmin(path_losses, axis=1)

def get_fitness(uav_deployment, users, assoc_func, p):
    """Calculate fitness function: average delay for given UAV deployment."""
    uav_locs = uav_deployment.reshape(p.N_UAV, 3)
    bs_locs = np.vstack([p.TBS_LOCS, uav_locs])
    
    association = assoc_func(users, bs_locs, p)
    n_k = np.bincount(association, minlength=p.N_BS)
    
    total_delay = 0
    for i in range(len(users)):
        k = association[i]
        is_uav = k >= p.N_TBS
        total_delay += calculate_delay(users[i], bs_locs[k], is_uav, p, n_k[k])
        
    return total_delay / len(users)

def run_pso(users, assoc_func, p):
    """Particle Swarm Optimization for UAV deployment."""
    # Initialize particles
    particles_pos = np.random.rand(p.PSO_PARTICLES, p.N_UAV * 3)
    particles_pos[:, 0::3] *= p.AREA_SIZE  # x-coordinates
    particles_pos[:, 1::3] *= p.AREA_SIZE  # y-coordinates
    particles_pos[:, 2::3] = 100           # Fixed height
    
    particles_vel = np.zeros_like(particles_pos)
    pbest_pos = np.copy(particles_pos)
    pbest_val = np.array([get_fitness(pos, users, assoc_func, p) for pos in particles_pos])
    
    gbest_idx = np.argmin(pbest_val)
    gbest_pos = pbest_pos[gbest_idx]
    gbest_val = pbest_val[gbest_idx]
    
    # PSO iterations
    for _ in range(p.PSO_ITER):
        r1, r2 = np.random.rand(2)
        particles_vel = p.PSO_W * particles_vel + \
                        p.PSO_C1 * r1 * (pbest_pos - particles_pos) + \
                        p.PSO_C2 * r2 * (gbest_pos - particles_pos)
        particles_pos += particles_vel
        
        # Constrain positions
        particles_pos[:, 0::3] = np.clip(particles_pos[:, 0::3], 0, p.AREA_SIZE)
        particles_pos[:, 1::3] = np.clip(particles_pos[:, 1::3], 0, p.AREA_SIZE)
        particles_pos[:, 2::3] = 100  # Keep height fixed
        
        new_vals = np.array([get_fitness(pos, users, assoc_func, p) for pos in particles_pos])
        
        # Update personal and global bests
        update_indices = new_vals < pbest_val
        pbest_pos[update_indices] = particles_pos[update_indices]
        pbest_val[update_indices] = new_vals[update_indices]
        
        gbest_idx = np.argmin(pbest_val)
        gbest_pos = pbest_pos[gbest_idx]
        gbest_val = pbest_val[gbest_idx]
        
    return gbest_pos.reshape(p.N_UAV, 3), gbest_val

# Scientific plotting functions

def plot_convergence_analysis(users, p):
    """Generate convergence analysis plots for OTT and PSO algorithms."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Algorithm 1 convergence (OTT)
    iterations_ott = []
    convergence_ott = []
    
    for max_iter in range(1, 21):
        # Run OTT with limited iterations
        uniform_locs = np.array([[250, 500, 100], [500, 250, 100], [750, 500, 100]])
        bs_locs = np.vstack([p.TBS_LOCS, uniform_locs])
        
        n_users = len(users)
        association = np.zeros(n_users, dtype=int)
        dists = cdist(users, bs_locs)
        association = np.argmin(dists, axis=1)
        
        for iter_count in range(max_iter):
            n_k = np.bincount(association, minlength=p.N_BS)
            delays = np.zeros((n_users, p.N_BS))
            for i in range(n_users):
                for k in range(p.N_BS):
                    is_uav = k >= p.N_TBS
                    delays[i, k] = calculate_delay(users[i], bs_locs[k], is_uav, p, n_k[k])
            
            new_association = np.argmin(delays, axis=1)
            if np.array_equal(association, new_association):
                break
            association = new_association
        
        fitness = get_fitness(uniform_locs, users, lambda u, b, p: association, p)
        iterations_ott.append(max_iter)
        convergence_ott.append(fitness)
    
    # Algorithm 2 convergence (PSO)
    iterations_pso = []
    convergence_pso = []
    
    for max_iter in range(10, 101, 10):
        # Run PSO with limited iterations
        original_iter = p.PSO_ITER
        p.PSO_ITER = max_iter
        _, fitness = run_pso(users, ott_association, p)
        p.PSO_ITER = original_iter
        
        iterations_pso.append(max_iter)
        convergence_pso.append(fitness)
    
    # Plot Algorithm 1 convergence
    ax1.plot(iterations_ott, convergence_ott, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Number of Iterations')
    ax1.set_ylabel('Average Task Delay (s)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=min(convergence_ott) * 0.95)
    
    # Plot Algorithm 2 convergence  
    ax2.plot(iterations_pso, convergence_pso, 'r-s', linewidth=2, markersize=6)
    ax2.set_xlabel('Number of Iterations')
    ax2.set_ylabel('Average Task Delay (s)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=min(convergence_pso) * 0.95)
    
    plt.tight_layout()
    
    # # Add figure caption
    # fig.text(0.5, 0.02, 'Convergence analysis of the proposed algorithms.', 
    #          ha='center', fontsize=14, fontweight='bold')
    
    plt.savefig('convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_user_association_comparison(users, p):
    """Generate comprehensive user association comparison plots."""
    # SNR association with uniform UAV locations
    uniform_locs = np.array([[250, 500, 100], [500, 250, 100], [750, 500, 100]])
    bs_locs_uniform = np.vstack([p.TBS_LOCS, uniform_locs])
    snr_assoc = snr_association(users, bs_locs_uniform, p)

    # OTT association with PSO-optimized UAV locations
    pso_ott_locs, _ = run_pso(users, ott_association, p)
    bs_locs_ott = np.vstack([p.TBS_LOCS, pso_ott_locs])
    ott_assoc = ott_association(users, bs_locs_ott, p)

    fig = plt.figure(figsize=(22, 26))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    colors = sns.color_palette("bright", p.N_BS)
    
    # SNR-based association plot
    for k in range(p.N_BS):
        user_subset = users[snr_assoc == k]
        if len(user_subset) > 0:
            ax1.scatter(user_subset[:, 0], user_subset[:, 1],
                        c=[colors[k]], s=100, alpha=0.9, edgecolors='k', linewidths=1,
                        label=f'Users for {"TBS" if k==0 else f"UAV {k}"} ({len(user_subset)})')
    ax1.scatter(p.TBS_LOCS[:, 0], p.TBS_LOCS[:, 1], c='black', marker='s', s=400,
                label='TBS 1', edgecolors='white', linewidths=2, zorder=5)
    ax1.scatter(uniform_locs[:, 0], uniform_locs[:, 1], c='black', marker='^', s=400,
                label='UAVs (Uniform)', edgecolors='white', linewidths=2, zorder=5)
    ax1.set_xlabel('X-coordinate (m)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Y-coordinate (m)', fontweight='bold', fontsize=12)
    ax1.set_xlim(0, p.AREA_SIZE)
    ax1.set_ylim(0, p.AREA_SIZE)
    ax1.grid(True, alpha=0.4)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    ax1.set_aspect('equal', adjustable='box')

    # OTT-based association plot
    for k in range(p.N_BS):
        user_subset = users[ott_assoc == k]
        if len(user_subset) > 0:
            ax2.scatter(user_subset[:, 0], user_subset[:, 1],
                        c=[colors[k]], s=100, alpha=0.9, edgecolors='k', linewidths=1,
                        label=f'Users for {"TBS" if k==0 else f"UAV {k}"} ({len(user_subset)})')
    ax2.scatter(p.TBS_LOCS[:, 0], p.TBS_LOCS[:, 1], c='black', marker='s', s=400,
                label='TBS 1', edgecolors='white', linewidths=2, zorder=5)
    ax2.scatter(pso_ott_locs[:, 0], pso_ott_locs[:, 1], c='black', marker='^', s=400,
                label='UAVs (Optimal)', edgecolors='white', linewidths=2, zorder=5)
    ax2.set_xlabel('X-coordinate (m)', fontweight='bold', fontsize=12)
    ax2.set_xlim(0, p.AREA_SIZE)
    ax2.set_ylim(0, p.AREA_SIZE)
    ax2.grid(True, alpha=0.4)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    ax2.set_aspect('equal', adjustable='box')

    # (c) Jain's fairness index and Delay comparison
    sigma_values = [0.5, 1.0, 1.5, 2.0]
    jains_ott, jains_snr, delays_ott, delays_snr = [], [], [], []
    for sigma in sigma_values:
        np.random.seed(42)
        temp_p = p.__class__()
        temp_p.COV = [[sigma * 10000, 0], [0, sigma * 10000]]
        temp_users = generate_users(temp_p)
        temp_bs_locs = np.vstack([temp_p.TBS_LOCS, uniform_locs])
        ott_assoc_temp = ott_association(temp_users, temp_bs_locs, temp_p)
        snr_assoc_temp = snr_association(temp_users, temp_bs_locs, temp_p)
        n_k_ott = np.bincount(ott_assoc_temp, minlength=p.N_BS)
        n_k_snr = np.bincount(snr_assoc_temp, minlength=p.N_BS)
        delays_per_user_ott = np.array([
            calculate_delay(temp_users[i], temp_bs_locs[ott_assoc_temp[i]], ott_assoc_temp[i] >= p.N_TBS, temp_p, n_k_ott[ott_assoc_temp[i]])
            for i in range(len(temp_users))])
        delays_per_user_snr = np.array([
            calculate_delay(temp_users[i], temp_bs_locs[snr_assoc_temp[i]], snr_assoc_temp[i] >= p.N_TBS, temp_p, n_k_snr[snr_assoc_temp[i]])
            for i in range(len(temp_users))])
        jains_fairness = lambda x: (np.sum(x)**2) / (len(x) * np.sum(x**2)) if len(x) > 0 and np.sum(x**2) > 0 else 0
        jains_ott.append(jains_fairness(delays_per_user_ott))
        jains_snr.append(jains_fairness(delays_per_user_snr))
        delays_ott.append(np.mean(delays_per_user_ott))
        delays_snr.append(np.mean(delays_per_user_snr))
    
    x_pos = np.arange(len(sigma_values))
    width = 0.35
    
    bars1 = ax3.bar(x_pos - width/2, jains_ott, width, label='OTT Fairness', color='skyblue', edgecolor='black', linewidth=1.5)
    bars2 = ax3.bar(x_pos + width/2, jains_snr, width, label='SNR Fairness', color='salmon', edgecolor='black', linewidth=1.5)
    
    ax3.set_xlabel('User Distribution Parameter σ²', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Jain\'s Fairness Index', fontweight='bold', color='blue', fontsize=12)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'{s}×10⁴' for s in sigma_values], fontsize=11)
    ax3.tick_params(axis='y', labelcolor='blue', labelsize=11)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.4, axis='y')

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax4 = ax3.twinx()
    ax4.plot(x_pos, delays_ott, 'o-', color='blue', label='OTT Delay', linewidth=2.5, markersize=8)
    ax4.plot(x_pos, delays_snr, 's-', color='red', label='SNR Delay', linewidth=2.5, markersize=8)
    ax4.set_ylabel('Average Task Delay (s)', fontweight='bold', color='red', fontsize=12)
    ax4.tick_params(axis='y', labelcolor='red', labelsize=11)
    
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax4.get_legend_handles_labels()
    ax4.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=12)

    plt.tight_layout()
    
    # # Add figure caption at the bottom
    # fig.text(0.5, 0.02, 'Comparison of different user association schemes.', 
    #          ha='center', fontsize=14, fontweight='bold')
    
    plt.savefig('user_association_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_performance_comparison(p):
    """Generate performance comparison across different optimization schemes."""
    sigma_values = [0.5, 1.0, 1.5, 2.0]
    results_data = []
    
    print("   - Running simulations for different user distributions...")
    
    for sigma in sigma_values:
        print(f"     - Simulating with σ² = {sigma}×10⁴...")
        temp_p = p.__class__()
        temp_p.COV = [[sigma * 10000, 0], [0, sigma * 10000]]
        temp_users = generate_users(temp_p)
        
        # 1. PSO+OTT
        _, pso_ott_delay = run_pso(temp_users, ott_association, temp_p)
        
        # 2. PSO+SNR
        _, pso_snr_delay = run_pso(temp_users, snr_association, temp_p)
        
        # 3. Uniform+OTT
        uniform_locs = np.array([[250, 500, 100], [500, 250, 100], [750, 500, 100]])
        uniform_ott_delay = get_fitness(uniform_locs.flatten(), temp_users, ott_association, temp_p)
        
        # 4. Uniform+SNR
        uniform_snr_delay = get_fitness(uniform_locs.flatten(), temp_users, snr_association, temp_p)
        
        results_data.extend([
            {'Method': 'PSO+OTT', 'User Distribution σ²': f'{sigma}×10⁴', 'Average Task Delay (s)': pso_ott_delay, 'σ²': sigma},
            {'Method': 'PSO+SNR', 'User Distribution σ²': f'{sigma}×10⁴', 'Average Task Delay (s)': pso_snr_delay, 'σ²': sigma},
            {'Method': 'Uniform+OTT', 'User Distribution σ²': f'{sigma}×10⁴', 'Average Task Delay (s)': uniform_ott_delay, 'σ²': sigma},
            {'Method': 'Uniform+SNR', 'User Distribution σ²': f'{sigma}×10⁴', 'Average Task Delay (s)': uniform_snr_delay, 'σ²': sigma}
        ])

    df = pd.DataFrame(results_data)
    
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=df, x='σ²', y='Average Task Delay (s)', hue='Method', 
                     palette='bright')
    
    ax.set_xlabel('User Distribution Parameter σ²', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Task Delay (s)', fontsize=12, fontweight='bold')
    ax.set_xticklabels([f'{s}×10⁴' for s in sigma_values])
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=10, weight='bold')
    
    ax.legend(title='Optimization Scheme', title_fontsize=12, fontsize=11, 
             loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')
    
    y_min = df['Average Task Delay (s)'].min() * 0.9
    ax.set_ylim(bottom=y_min)
    
    plt.tight_layout()
    
    # # Add figure caption at the bottom
    # plt.figtext(0.5, 0.02, 'Performance comparison of different optimization schemes.', 
    #             ha='center', fontsize=14, fontweight='bold')
    
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_optimal_deployment(users, pso_ott_locs, p):
    """Visualize optimal UAV deployment with user associations."""
    final_bs_locs = np.vstack([p.TBS_LOCS, pso_ott_locs])
    final_association = ott_association(users, final_bs_locs, p)
    
    plt.figure(figsize=(12, 10))
    
    colors = sns.color_palette("bright", p.N_BS)
    
    for k in range(p.N_BS):
        user_subset = users[final_association == k]
        if len(user_subset) > 0:
            label = f'Users for {"TBS" if k == 0 else f"UAV {k}"} ({len(user_subset)})'
            plt.scatter(user_subset[:, 0], user_subset[:, 1], 
                       c=[colors[k]], s=100, alpha=0.9, edgecolors='k', linewidths=2,
                       label=label)
    
    plt.scatter(p.TBS_LOCS[:, 0], p.TBS_LOCS[:, 1], 
               c='black', marker='s', s=400, label='TBS 1', 
               edgecolors='white', linewidths=3, zorder=5)
    
    for i, uav_loc in enumerate(pso_ott_locs):
        plt.scatter(uav_loc[0], uav_loc[1], 
                   c='black', marker='^', s=400, 
                   label='UAVs (Optimal)' if i == 0 else "", 
                   edgecolors='white', linewidths=3, zorder=5)
        
        plt.annotate(f'UAV {i+1}', (uav_loc[0], uav_loc[1]), 
                    xytext=(15, 15), textcoords='offset points',
                    fontsize=14, fontweight='bold', color='darkred',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.7))
    
    plt.annotate('TBS 1', (p.TBS_LOCS[0, 0], p.TBS_LOCS[0, 1]), 
                xytext=(15, 15), textcoords='offset points',
                fontsize=14, fontweight='bold', color='black',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.7))
    
    plt.xlim(0, p.AREA_SIZE)
    plt.ylim(0, p.AREA_SIZE)
    plt.xlabel('X-coordinate (m)', fontsize=12, fontweight='bold')
    plt.ylabel('Y-coordinate (m)', fontsize=12, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.4)
    plt.gca().set_aspect('equal', adjustable='box')
    
    n_k = np.bincount(final_association, minlength=p.N_BS)
    avg_delay = get_fitness(pso_ott_locs.flatten(), users, ott_association, p)
    
    info_text = f'Average Delay: {avg_delay:.4f} s\nUsers per BS: {n_k}'
    plt.text(0.02, 0.98, info_text, 
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 0.85, 1])
    
    # # Add figure caption at the bottom
    # plt.figtext(0.5, 0.02, 'Optimal UAV deployment and user association.', 
    #             ha='center', fontsize=14, fontweight='bold')
    
    plt.savefig('optimal_deployment.png', dpi=300, bbox_inches='tight')
    plt.show()

# --- SECTION 5: Main Simulation and Enhanced Results ---

def create_publication_quality_plot(users, p):
    """Create comprehensive comparison plot of OTT vs SNR association schemes."""
    strategic_locs = np.array([[350, 350, 100], [650, 650, 100], [500, 750, 100]])
    bs_locs = np.vstack([p.TBS_LOCS, strategic_locs])
    
    ott_assoc = ott_association(users, bs_locs, p)
    snr_assoc = snr_association(users, bs_locs, p)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    colors = sns.color_palette("bright", p.N_BS)
    
    # Plot 1: SNR-based association
    for k in range(p.N_BS):
        user_subset = users[snr_assoc == k]
        if len(user_subset) > 0:
            label = f'{"TBS" if k==0 else f"UAV {k}"} ({len(user_subset)} users)'
            ax1.scatter(user_subset[:, 0], user_subset[:, 1], c=[colors[k]], 
                       s=80, alpha=0.8, edgecolors='white', linewidths=1, label=label)
    
    # Add base stations
    ax1.scatter(p.TBS_LOCS[:, 0], p.TBS_LOCS[:, 1], c='black', marker='s', s=400,
                label='TBS', edgecolors='white', linewidths=3, zorder=5)
    ax1.scatter(strategic_locs[:, 0], strategic_locs[:, 1], c='black', marker='^', s=400,
                label='UAVs', edgecolors='white', linewidths=3, zorder=5)
    
    ax1.set_xlabel('X-coordinate (m)', fontweight='bold')
    ax1.set_ylabel('Y-coordinate (m)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, p.AREA_SIZE)
    ax1.set_ylim(0, p.AREA_SIZE)
    
    # Plot 2: OTT-based association  
    for k in range(p.N_BS):
        user_subset = users[ott_assoc == k]
        if len(user_subset) > 0:
            label = f'{"TBS" if k==0 else f"UAV {k}"} ({len(user_subset)} users)'
            ax2.scatter(user_subset[:, 0], user_subset[:, 1], c=[colors[k]], 
                       s=80, alpha=0.8, edgecolors='white', linewidths=1, label=label)
    
    ax2.scatter(p.TBS_LOCS[:, 0], p.TBS_LOCS[:, 1], c='black', marker='s', s=400,
                label='TBS', edgecolors='white', linewidths=3, zorder=5)
    ax2.scatter(strategic_locs[:, 0], strategic_locs[:, 1], c='black', marker='^', s=400,
                label='UAVs', edgecolors='white', linewidths=3, zorder=5)
    
    ax2.set_xlabel('X-coordinate (m)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xlim(0, p.AREA_SIZE)
    ax2.set_ylim(0, p.AREA_SIZE)
    
    # Plot 3: Load distribution comparison
    snr_n_k = np.bincount(snr_assoc, minlength=p.N_BS)
    ott_n_k = np.bincount(ott_assoc, minlength=p.N_BS)
    
    x_pos = np.arange(p.N_BS)
    width = 0.35
    
    bars1 = ax3.bar(x_pos - width/2, snr_n_k, width, label='SNR Association', 
                    color='lightcoral', edgecolor='black', linewidth=1)
    bars2 = ax3.bar(x_pos + width/2, ott_n_k, width, label='OTT Association', 
                    color='skyblue', edgecolor='black', linewidth=1)
    
    ax3.set_xlabel('Base Station', fontweight='bold')
    ax3.set_ylabel('Number of Users', fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(['TBS'] + [f'UAV{i+1}' for i in range(p.N_UAV)])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                        fontweight='bold')
    
    # Plot 4: Performance metrics
    snr_delay = get_fitness(strategic_locs.flatten(), users, snr_association, p)
    ott_delay = get_fitness(strategic_locs.flatten(), users, ott_association, p)
    
    # Calculate fairness (Jain's index)
    def jains_fairness(delays):
        return (np.sum(delays)**2) / (len(delays) * np.sum(delays**2)) if len(delays) > 0 and np.sum(delays**2) > 0 else 0
    
    snr_delays_per_user = np.array([
        calculate_delay(users[i], bs_locs[snr_assoc[i]], snr_assoc[i] >= p.N_TBS, p, snr_n_k[snr_assoc[i]])
        for i in range(len(users))])
    ott_delays_per_user = np.array([
        calculate_delay(users[i], bs_locs[ott_assoc[i]], ott_assoc[i] >= p.N_TBS, p, ott_n_k[ott_assoc[i]])
        for i in range(len(users))])
    
    snr_fairness = jains_fairness(snr_delays_per_user)
    ott_fairness = jains_fairness(ott_delays_per_user)
    
    metrics = ['Average Delay (s)', 'Fairness Index', 'Load Balance\n(1/Std Dev)']
    snr_values = [snr_delay, snr_fairness, 1/np.std(snr_n_k)]
    ott_values = [ott_delay, ott_fairness, 1/np.std(ott_n_k)]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax4.bar(x_pos - width/2, snr_values, width, label='SNR Association', 
                    color='lightcoral', edgecolor='black', linewidth=1)
    bars2 = ax4.bar(x_pos + width/2, ott_values, width, label='OTT Association', 
                    color='skyblue', edgecolor='black', linewidth=1)
    
    ax4.set_xlabel('Performance Metrics', fontweight='bold')
    ax4.set_ylabel('Normalized Values', fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add improvement percentages
    improvements = [(ott_values[i] - snr_values[i])/snr_values[i]*100 for i in range(len(metrics))]
    for i, improvement in enumerate(improvements):
        ax4.text(i, max(snr_values[i], ott_values[i]) + 0.01, f'{improvement:+.1f}%', 
                ha='center', va='bottom', fontweight='bold', 
                color='green' if improvement > 0 else 'red')
    
    plt.tight_layout()
    
    # Add figure caption at the bottom
    fig.text(0.5, 0.02, 'Comprehensive comparison of OTT vs SNR association schemes.', 
             ha='center', fontsize=14, fontweight='bold')
    
    plt.savefig('ott_vs_snr_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("OTT vs SNR ASSOCIATION SUMMARY")
    print("="*60)
    print(f"Average Delay:")
    print(f"  SNR: {snr_delay:.4f} s")
    print(f"  OTT: {ott_delay:.4f} s")
    print(f"  OTT Improvement: {(snr_delay-ott_delay)/snr_delay*100:.2f}%")
    print(f"\nFairness Index:")
    print(f"  SNR: {snr_fairness:.3f}")
    print(f"  OTT: {ott_fairness:.3f}")
    print(f"  OTT Improvement: {(ott_fairness-snr_fairness)/snr_fairness*100:.2f}%")
    print("="*60)

if __name__ == '__main__':
    """Main simulation execution."""
    np.random.seed(42)  # Ensure reproducible results
    
    p = Parameters()
    users = generate_users(p)

    print("=" * 60)
    print("UAV-AIDED MEC NETWORKS: JOINT OPTIMIZATION SIMULATION")
    print("=" * 60)
    print(f"Simulation Parameters:")
    print(f"  - Coverage Area: {p.AREA_SIZE}m × {p.AREA_SIZE}m")
    print(f"  - Users: {p.N_USERS}")
    print(f"  - UAVs: {p.N_UAV}")
    print(f"  - Terrestrial BS: {p.N_TBS}")
    print(f"  - PSO Iterations: {p.PSO_ITER}")
    print("=" * 60)

    # Run optimization schemes
    print("\n1. PSO + OTT (Proposed)...")
    pso_ott_locs, pso_ott_delay = run_pso(users, ott_association, p)
    print(f"   Optimal delay: {pso_ott_delay:.4f} s")

    print("\n2. PSO + SNR...")
    pso_snr_locs, pso_snr_delay = run_pso(users, snr_association, p)
    print(f"   Optimal delay: {pso_snr_delay:.4f} s")
    
    print("\n3. Uniform + OTT...")
    uniform_locs = np.array([[250, 500, 100], [500, 250, 100], [750, 500, 100]])
    uniform_ott_delay = get_fitness(uniform_locs.flatten(), users, ott_association, p)
    print(f"   Average delay: {uniform_ott_delay:.4f} s")

    print("\n4. Uniform + SNR...")
    uniform_snr_delay = get_fitness(uniform_locs.flatten(), users, snr_association, p)
    print(f"   Average delay: {uniform_snr_delay:.4f} s")

    # Performance analysis
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    results_df = pd.DataFrame({
        'Scheme': ['PSO+OTT (Proposed)', 'PSO+SNR', 'Uniform+OTT', 'Uniform+SNR'],
        'Average Delay (s)': [pso_ott_delay, pso_snr_delay, uniform_ott_delay, uniform_snr_delay]
    })
    
    print(results_df.to_string(index=False, float_format='%.4f'))
    
    improvement = ((uniform_snr_delay - pso_ott_delay) / uniform_snr_delay) * 100
    print(f"\nMaximum improvement: {improvement:.2f}%")

    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    plot_convergence_analysis(users, p)
    plot_user_association_comparison(users, p)
    plot_performance_comparison(p)
    plot_optimal_deployment(users, pso_ott_locs, p)
    
    print("\n✓ All simulations completed successfully!")
    print("✓ Plots saved as high-resolution PNG files.")

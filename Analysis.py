import numpy as np
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_sac_progress():
    # Load SAC data
    sac_train = np.load("results/SAC_v2_Train.npy")[:7]     # (13, 20)
    sac_test = np.load("results/SAC_v2_Test.npy")[:7]       # (13, 20)
    sacl0_train = np.load("results/SACL0_v2_Train.npy")[:7]
    sacl0_test = np.load("results/SACL0_v2_Test.npy")[:7]

    assert sac_train.shape == sac_test.shape == sacl0_train.shape == sacl0_test.shape, "All arrays must have the same shape"
    num_saves, num_maps = sac_test.shape

    # Compute average completion across maps
    sac_train_avg = sac_train.mean(axis=1)
    sac_test_avg = np.sum(sac_test, axis=1) / 19.0
    sacl0_train_avg = sacl0_train.mean(axis=1)
    sacl0_test_avg = np.sum(sacl0_test, axis=1) / 19.0

    # X-axis: number of transitions trained on
    save_indices = np.arange(num_saves)
    transitions = save_indices * 6 * 512 * 500

    # --- Overall Average Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(transitions, sac_train_avg * 100, label="SAC Train", color="blue", linestyle="-")
    plt.plot(transitions, sac_test_avg * 100, label="SAC Test", color="blue", linestyle="--")
    plt.plot(transitions, sacl0_train_avg * 100, label="SACL0 Train", color="orange", linestyle="-")
    plt.plot(transitions, sacl0_test_avg * 100, label="SACL0 Test", color="orange", linestyle="--")
    plt.xlabel("Number of Transitions Trained On")
    plt.ylabel("Average Completion Rate (%)")
    plt.title("SAC vs SACL0 - Average Performance - lambda = 0.001")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Per-Test-Map Subplots ---
    fig, axs = plt.subplots(5, 4, figsize=(20, 16))  # 20 subplots in a 5x4 grid
    axs = axs.flatten()

    for map_idx in range(num_maps):
        ax = axs[map_idx]
        ax.plot(transitions, sac_test[:, map_idx] * 100, label="SAC", color="blue")
        ax.plot(transitions, sacl0_test[:, map_idx] * 100, label="SACL0", color="orange")
        ax.set_title(f"Map {map_idx}")
        ax.set_xlabel("Transitions")
        ax.set_ylabel("Completion (%)")
        ax.grid(True)

        if map_idx == 0:
            ax.legend()

    plt.suptitle("SAC vs SACL0 - Performance on Individual Test Maps - lambda = 0.001", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def plot_test_300():
    # Load SAC data
    sac_train = np.load("results/SAC_v2_Train.npy")
    sacl0_train = np.load("results/SACL0_v2_Train.npy")
    sac_test = np.load("results/SAC_v2_300_Test.npy")
    sacl0_test = np.load("results/SACL0_v2_300_Test.npy")
    sacl0_003_train = np.load("results/SACL0_003_300_Train.npy")
    sacl0_003_test = np.load("results/SACL0_003_300_Test.npy")
    sacl0_010_train = np.load("results/SACL0_010_300_Train.npy")
    sacl0_010_test = np.load("results/SACL0_010_300_Test.npy")

    #assert sac_train.shape == sac_test.shape == sacl0_train.shape == sacl0_test.shape, "All arrays must have the same shape"
    num_saves, num_maps = sac_test.shape

    # Compute average completion across maps
    sac_test_avg = sac_test.mean(axis=1)
    sacl0_test_avg = sacl0_test.mean(axis=1)
    sac_train_avg = sac_train.mean(axis=1)
    sacl0_train_avg = sacl0_train.mean(axis=1)
    sacl0_003_train_avg = sacl0_003_train.mean(axis=1)
    sacl0_003_test_avg = sacl0_003_test.mean(axis=1)
    sacl0_010_train_avg = sacl0_010_train.mean(axis=1)
    sacl0_010_test_avg = sacl0_010_test.mean(axis=1)

    # X-axis: number of transitions trained on
    save_indices = np.arange(num_saves)
    transitions = save_indices * 6 * 512 * 500

    # --- Overall Average Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(transitions, sac_train_avg * 100, label="SAC Train", color="blue", linestyle="-")
    plt.plot(transitions, sac_test_avg * 100, label="SAC Test", color="blue", linestyle="--")
    plt.plot(transitions, sacl0_train_avg * 100, label="SACL0 1 Train", color="orange", linestyle="-")
    plt.plot(transitions, sacl0_test_avg * 100, label="SACL0 1 Test", color="orange", linestyle="--")
    plt.plot(transitions, sacl0_003_train_avg * 100, label="SACL0 3 Train", color="red", linestyle="-")
    plt.plot(transitions, sacl0_003_test_avg * 100, label="SACL0 3 Test", color="red", linestyle="--")
    plt.plot(transitions, sacl0_010_train_avg * 100, label="SACL0 10 Train", color="green", linestyle="-")
    plt.plot(transitions, sacl0_010_test_avg * 100, label="SACL0 10 Test", color="green", linestyle="--")
    plt.xlabel("Number of Transitions Trained On")
    plt.ylabel("Average Completion Rate (%)")
    plt.title("SAC vs SACL0 - Test avg across 300 new maps - lambda = 0.001")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



plot_test_300()
while True:
    pass
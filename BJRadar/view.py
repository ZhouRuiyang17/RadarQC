from typing import List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pcolors
import reader


REF_CMAP = pcolors.ListedColormap([[255 / 255, 255 / 255, 255 / 255], [41 / 255, 237 / 255, 238 / 255], [29 / 255, 175 / 255, 243 / 255],
                                   [10 / 255, 35 / 255, 244 / 255], [41 / 255, 253 / 255, 47 / 255], [30 / 255, 199 / 255, 34 / 255],
                                   [19 / 255, 144 / 255, 22 / 255], [254 / 255, 253 / 255, 56 / 255], [230 / 255, 191 / 255, 43 / 255],
                                   [251 / 255, 144 / 255, 37 / 255], [249 / 255, 14 / 255, 28 / 255], [209 / 255, 11 / 255, 21 / 255],
                                   [189 / 255, 8 / 255, 19 / 255], [219 / 255, 102 / 255, 252 / 255], [186 / 255, 36 / 255, 235 / 255]])
REF_NORM = pcolors.BoundaryNorm(np.linspace(0.0, 75.0, 16), REF_CMAP.N)

plt.rcParams['font.sans-serif'] = 'Arial'


def plot_radar_ppi(elevs: np.ndarray, refs: np.ndarray, img_path: str) -> None:
    thetas = np.arange(reader.AZIMUTH_RANGE) / 180 * np.pi
    rhos = np.arange(reader.MAX_NUM_REF_RANGE_BIN)
    thetas, rhos = np.meshgrid(thetas, rhos)

    fig = plt.figure(figsize=(24, 18), dpi=600)

    for i in range(len(elevs)):
        ax = fig.add_subplot(3, 3, i + 1, projection='polar')
        # ax, pm = wradlib.vis.plot_ppi(refs, r=rhos, az=thetas, elev=0.5, proj='cg', 
                                    #   ax=111, fig=fig, cmap=REF_CMAP, norm=REF_NORM)
        ax.grid(False)
        pm = ax.pcolormesh(thetas, rhos, refs[i].T, cmap=REF_CMAP, norm=REF_NORM)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction('clockwise')
        ax.grid(True, linewidth=1)
        ax.tick_params(labelsize=14)
        ax.set_title('\n$e={}^\circ$'.format(elevs[i]), fontsize=18)

        cbar = fig.colorbar(pm, ax=ax, pad=0.1, aspect=20, shrink=0.9, extend='both')
        cbar.set_label('dBZ', fontsize=14)
        cbar.ax.tick_params(labelsize=12)
    
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(img_path, bbox_inches='tight')
    plt.close(fig)


def plot_radar_ppi_part(elevs: np.ndarray, refs: np.ndarray, img_path: str, 
                        azimuth_range: List[int, int] = [45, 225], 
                        distance_range: List[int, int] = [0, 80]) -> None:
    thetas = np.arange(azimuth_range[0], azimuth_range[1]) / 180 * np.pi
    rhos = np.arange(distance_range[0], distance_range[1])
    thetas, rhos = np.meshgrid(thetas, rhos)
    refs = refs[:4, azimuth_range[0]: azimuth_range[1], distance_range[0]: distance_range[1]]

    fig = plt.figure(figsize=(18, 12), dpi=600)

    for i in range(4):
        ax = fig.add_subplot(2, 2, i + 1, projection='polar')
        ax.grid(False)
        pm = ax.pcolormesh(thetas, rhos, refs[i].T, cmap=REF_CMAP, norm=REF_NORM)
        ax.set_xlim(azimuth_range[0] / 180 * np.pi, azimuth_range[1] / 180 * np.pi)
        ax.set_rlim(distance_range[0], distance_range[1])
        ax.set_theta_zero_location('N')
        ax.set_theta_direction('clockwise')
        ax.set_theta_offset((azimuth_range[0] + 180) / 180 * np.pi)
        ax.grid(True, linewidth=1)
        ax.tick_params(labelsize=14)
        ax.set_title('$e={}^\circ$'.format(elevs[i]), fontsize=18, pad=-50)

        cbar = fig.colorbar(pm, ax=ax, pad=0.1, aspect=20, shrink=0.5, extend='both')
        cbar.set_label('dBZ', fontsize=14)
        cbar.ax.tick_params(labelsize=12)
    
    plt.subplots_adjust(hspace=-0.2, wspace=0)
    plt.savefig(img_path, bbox_inches='tight')
    plt.close(fig)


def plot_mask(file_path: str, img_path: str) -> None:
    mask = np.loadtxt(file_path, dtype=int)
    thetas = np.arange(reader.AZIMUTH_RANGE) / 180 * np.pi
    rhos = np.arange(reader.MAX_NUM_REF_RANGE_BIN)
    thetas, rhos = np.meshgrid(thetas, rhos)

    fig = plt.figure(figsize=(8, 6), dpi=600)
    ax = fig.add_subplot(1, 1, 1, projection='polar')
    ax.grid(False)
    pm = ax.pcolormesh(thetas, rhos, mask.T, cmap='Greys')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction('clockwise')
    ax.grid(True, linewidth=1)
    ax.tick_params(labelsize=14)
    ax.set_title('Blockage Mask', fontsize=18)

    plt.savefig(img_path, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    example_data_path = r'D:\Data\SBandDataAll\SBandBasicUnzip\20190806\Z_RADR_I_Z9010_20190806183600_O_DOR_SA_CAP.bin'
    mask_path = r'D:\Data\SBandDataAll\BJRadarBlockage.txt'
    elevs, refs = reader.read_radar_bin(example_data_path)
    plot_radar_ppi(elevs, refs, 'ref.png')
    plot_radar_ppi_part(elevs, refs, 'ref_unblocked.png')
    plot_mask(mask_path, 'mask.png')

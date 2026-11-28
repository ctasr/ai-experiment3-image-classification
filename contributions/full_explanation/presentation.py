"""Plot only completed, source-locked actual5000 results; manual review remains."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np
from . import protocol as p
from .analysis import read_full
CLASS_NAMES = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '蛙', '马', '船', '卡车']

def plot_first(output, data_dir, font_family='SimSun', *, presentation_destination):
    output = Path(output)
    summary = p.read(output / 'analysis.json')
    if summary['source_sha256'] != p.sources() or summary['identity_sha256'] != p.sha(output / 'identity.json'):
        raise ValueError('Analysis is not bound to current frozen source/input identity')
    identity, data, receipts = read_full(output)
    if p.data_identity(data_dir) != identity['data']:
        raise ValueError('Figure image directory differs from the frozen original training-batch data')
    font_manager.findfont(font_family, fallback_to_default=False)
    plt.rcParams.update({'font.family': font_family, 'font.size': 10, 'axes.unicode_minus': False})
    destination = Path(presentation_destination)
    destination.mkdir(exist_ok=True)
    files = []

    def save(figure, name):
        path = destination / name
        figure.savefig(path, dpi=180, bbox_inches='tight')
        plt.close(figure)
        files.append(path)
    point = np.asarray(summary['overall']['mean'])
    interval = np.asarray(summary['overall']['conditional_image_ci95'])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    axes[0].errorbar(range(4), point, yerr=np.stack((point - interval[:, 0], interval[:, 1] - point)), fmt='o', capsize=4)
    axes[0].axhline(0, color='gray', linewidth=0.8)
    axes[0].set(xticks=range(4), xticklabels=['42', '17', '73', '三模型逐图均值'], ylabel='高响应窗降幅减随机窗降幅', title='同5000图的类别分层配对区间')
    for seed in p.SEEDS:
        value = np.sort(data[seed]['dq'])
        axes[1].plot(value, np.arange(1, 5001) / 5000, label=str(seed))
    axes[1].axvline(0, color='gray', linewidth=0.8)
    axes[1].set(xlabel='逐图高响应窗降幅减随机窗降幅', ylabel='累计比例', title='全部样本，包括负例与常量图')
    axes[1].legend(title='种子')
    save(fig, '01_full5000_sensitivity.png')
    return files


def main():
    parser=argparse.ArgumentParser(description='CPU rendering of the reviewed first two-view sensitivity figure; frozen source22 and inference/cache stay unchanged')
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--data-dir',type=Path,required=True);parser.add_argument('--destination',type=Path,required=True);parser.add_argument('--font-family',default='SimSun')
    args=parser.parse_args();destination=args.destination.resolve()
    if destination.exists()and any(destination.iterdir()):raise FileExistsError('Use a new presentation folder; do not overwrite original generated figures')
    if destination==args.data_dir.resolve()or destination.is_relative_to(args.data_dir.resolve()):raise ValueError('Presentation output cannot be inside immutable input data')
    if destination==args.output_dir.resolve()or destination==(args.output_dir/'figures').resolve():raise ValueError('Preserve original Source22 figures')
    destination.parent.mkdir(parents=True,exist_ok=True)
    plot_first(args.output_dir,args.data_dir,args.font_family,presentation_destination=destination)
if __name__=='__main__':main()

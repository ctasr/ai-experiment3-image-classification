"""Render two presentation-only crosssplit figures from frozen numeric payload.

No model, data cache, experiment source or statistical resampling is used.
"""
from pathlib import Path
import argparse
import hashlib
import json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(payload_path, expected_sha, output_dir):
    payload_path = Path(payload_path).resolve()
    if sha(payload_path) != expected_sha:
        raise ValueError('Frozen presentation payload SHA differs')
    value = json.loads(payload_path.read_text(encoding='utf-8'))
    if value['kind'] != 'crosssplit_original_artist_payload_v1' or value['seeds'] != [42, 17, 73]:
        raise ValueError('Wrong original fixed presentation protocol')
    if value['modes'] != ['original', 'weighted_single'] or value['epochs'] != list(range(1, 241)):
        raise ValueError('Wrong fixed method/epoch order')
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError('Use a fresh output directory; keep prior figures')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.font_manager import FontProperties
    font = Path(r'C:\Windows\Fonts\simsun.ttc')
    if font.is_file():
        matplotlib.rcParams['font.family'] = FontProperties(fname=str(font)).get_name()
    else:
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = ['Noto Serif CJK SC', 'SimSun', 'DejaVu Sans']
    matplotlib.rcParams.update({'font.size': 10, 'axes.unicode_minus': False})
    colors = {'original': '#4C78A8', 'weighted_single': '#E58A3A'}
    names = {'original': '标准KL', 'weighted_single': '条件加权KL'}
    classes = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '蛙', '马', '船', '卡车']
    if value['class_names_zh'] != classes:
        raise ValueError('Canonical class mapping differs')
    for mode in value['modes']:
        data = value['artists'][mode]
        for field in ('validation_accuracy_pct', 'validation_ce'):
            if np.asarray(data[field]['seed_lines']).shape != (3, 240) or len(data[field]['mean_line']) != 240:
                raise ValueError('Expected unchanged three seed lines and existing mean artist')
        if any(len(data[field]) != 3 for field in ('train_accuracy_pct', 'val_accuracy_pct', 'wall_minutes', 'confusion_counts')):
            raise ValueError('Expected three fixed per-seed artists')
        cm = np.asarray(data['confusion_counts'])
        if cm.shape != (3, 10, 10) or not np.array_equal(cm.sum(axis=2), np.full((3, 10), 500)):
            raise ValueError('Expected six exact row-denominator500 confusion matrices')
    output_dir.mkdir(parents=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), constrained_layout=True)
    for mode in value['modes']:
        data = value['artists'][mode]
        for field, axis, title in (('validation_accuracy_pct', axes[0, 0], '原图验证准确率'),
                                    ('validation_ce', axes[0, 1], '原图验证硬标签CE')):
            for line in data[field]['seed_lines']:
                axis.plot(value['epochs'], line, color=colors[mode], alpha=.18, linewidth=.65)
            axis.plot(value['epochs'], data[field]['mean_line'], color=colors[mode], label=names[mode])
            axis.set_title(title); axis.set_xlabel('轮次'); axis.grid(alpha=.18)
        xs = np.arange(3) + (-.10 if mode == 'original' else .10)
        axes[1, 0].plot(xs, data['train_accuracy_pct'], 'o-', color=colors[mode], label=names[mode] + '训练45k')
        axes[1, 0].plot(xs, data['val_accuracy_pct'], 's--', color=colors[mode], label=names[mode] + '验证5k')
        axes[1, 1].bar(xs, data['wall_minutes'], width=.18, color=colors[mode], label=names[mode])
    for axis in axes[0]:
        axis.legend(fontsize=8)
    axes[1, 0].set_title('LAST同口径原图拟合（准确率%）')
    axes[1, 0].set_xticks(range(3), [str(s) for s in value['seeds']]); axes[1, 0].legend(fontsize=7)
    axes[1, 1].set_title('原始逐轮wall总和（分钟，保留暂停）')
    axes[1, 1].set_xticks(range(3), [str(s) for s in value['seeds']])
    # The only change in this figure: place the legend outside the data axes.
    axes[1, 1].legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.10), ncol=2, frameon=False)
    fig.suptitle('三固定种子：曲线细线与均值、LAST拟合、真实计时')
    first = output_dir / 'curves_fit_cost.png'; fig.savefig(first, dpi=200); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for row, mode in enumerate(value['modes']):
        for column, seed in enumerate(value['seeds']):
            axis = axes[row, column]
            # Original producer divides each class count by5: 500 samples -> percent.
            cm = np.asarray(value['artists'][mode]['confusion_counts'][column]) / 5
            image = axis.imshow(cm, vmin=0, vmax=100, cmap='Blues')
            axis.set_title(names[mode] + ' / ' + str(seed))
            axis.set_xticks(range(10), classes, rotation=45, ha='right', fontsize=9)
            axis.set_yticks(range(10), classes, fontsize=9)
            axis.set_xlabel('预测类'); axis.set_ylabel('真实类')
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.7, label='每真实类别内占比（%）')
    fig.suptitle('每格独立模型的验证5k混淆矩阵，每真实类别500张')
    second = output_dir / 'six_confusion_matrices.png'; fig.savefig(second, dpi=200); plt.close(fig)
    return {path.name: {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}
            for path in (first, second)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', required=True, type=Path)
    parser.add_argument('--payload-sha256', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.payload, args.payload_sha256, args.output_dir), ensure_ascii=False))


if __name__ == '__main__':
    main()

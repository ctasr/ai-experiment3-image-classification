"""Render the frozen external intervals with unambiguous panel titles.

Input is either the original analysis JSON or its public field-selection summary.
No estimates, intervals, bootstrap samples, images or model outputs are computed.
"""
from pathlib import Path
import argparse
import hashlib
import json

ANALYSIS_SHA = 'daa50421606f6626dcadb7c37967266373af71b9573614982dce74c48d150345'
CONTRASTS_SHA = '3d2f09c4e30d448d67533b01699f1bc8618d3a1de2713a2576f805ccf58ab443'
NAMES = ('alignment_B_minus_A', 'weighted_minus_main_LAST', 'confidence_minus_main_LAST',
         'weighted_minus_confidence', 'main_BEST_two_view_minus_raw', 'main_BEST_raw_minus_noKD')
FIGURE = '02_fixed_paired_intervals.png'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(',', ':'), allow_nan=False).encode())


def load_summary(path, expected_sha):
    raw = path.read_bytes()
    require(sha(raw) == expected_sha, 'Actual input JSON SHA differs')
    value = json.loads(raw.decode('utf-8-sig'))
    if value.get('kind') == 'derived_public_field_selection_of_original_generated_summary':
        require(value['original_summary_sha256'] == ANALYSIS_SHA and
                '/fixed_contrasts' in value['selected_JSON_pointers'], 'Public summary is not the fixed original analysis selection')
        data = value['data']
    else:
        require(sha(raw) == ANALYSIS_SHA and value.get('kind') == 'prespecified_external_fixed_analysis_v1',
                'Original fixed analysis JSON required')
        data = value
    contrasts = data['fixed_contrasts']
    require(canonical(contrasts) == CONTRASTS_SHA, 'Frozen point/interval/name values differ')
    require(set(contrasts) == {'primary', 'secondary'} and
            all([row['name'] for row in contrasts[population]] == list(NAMES)
                for population in ('primary', 'secondary')), 'Original six subtraction rows and order required')
    return contrasts


def render(contrasts, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    panels = []
    for pi, population in enumerate(('primary', 'secondary')):
        for mi, metric in enumerate(('accuracy', 'NLL')):
            axis = axes[pi, mi]
            records = contrasts[population]
            factor = 100 if metric == 'accuracy' else 1
            artists = []
            for index, record in enumerate(records):
                entry = record[metric]
                point, lo, hi = [entry[key] * factor for key in ('difference', 'lower95', 'upper95')]
                interval, = axis.plot([lo, hi], [index, index], color='steelblue')
                left, = axis.plot([lo, lo], [index - .08, index + .08], color='steelblue')
                right, = axis.plot([hi, hi], [index - .08, index + .08], color='steelblue')
                marker, = axis.plot(point, index, 'o', color='steelblue')
                # Compare the plotted coordinates to original stored endpoints;
                # accuracy is displayed in pp exactly as in the original plot.
                require(list(interval.get_xdata()) == [lo, hi] and list(marker.get_xdata()) == [point]
                        and list(left.get_xdata()) == [lo, lo] and list(right.get_xdata()) == [hi, hi],
                        'Original point/interval display coordinates changed')
                artists.append({'name': record['name'], 'x': [point, lo, hi], 'y': index})
            axis.set_yticks(range(6), [name.replace('_', ' ') for name in NAMES] if mi == 0 else [])
            axis.invert_yaxis()
            axis.axvline(0, color='gray', linestyle='--')
            axis.grid(axis='x', alpha=.2)
            title = population + ': ' + ('accuracy difference (pp)' if metric == 'accuracy' else 'NLL difference')
            axis.set_title(title)
            require(axis.get_xscale() == 'linear', 'Original linear display scale changed')
            panels.append({'population': population, 'metric': metric, 'title': title,
                           'scale_factor': factor, 'artist_coordinates_sha256': canonical(artists),
                           'six_original_points_and_intervals_exact': True})
    fig.suptitle('Shared-image class-paired bootstrap: conditional descriptive 95% intervals')
    file = output / FIGURE
    fig.savefig(file, dpi=180)
    plt.close(fig)
    return {'figure': {'path': str(file), 'sha256': sha(file.read_bytes()), 'bytes': file.stat().st_size},
            'panels': panels, 'matplotlib_version': matplotlib.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--summary-sha256', required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check-ready', action='store_true')
    args = parser.parse_args()
    contrasts = load_summary(args.summary, args.summary_sha256)
    if args.check_ready:
        print(json.dumps({'state': 'frozen_JSON_ready_no_render', 'source_analysis_sha256': ANALYSIS_SHA,
                          'fixed_contrasts_canonical_sha256': CONTRASTS_SHA, 'new_estimates': 0,
                          'new_images': 0}))
        return 0
    require(args.output_dir is not None, '--output-dir is required for rendering')
    output = args.output_dir.resolve()
    require(not output.exists(), 'Use a new presentation directory; preserve all originals')
    output.mkdir(parents=True)
    result = render(contrasts, output)
    require(sha(args.summary.read_bytes()) == args.summary_sha256, 'Input JSON changed during rendering')
    result.update({'kind': 'external_one_figure_presentation_only_render',
                   'input_JSON': {'path': str(args.summary.resolve()), 'sha256': args.summary_sha256},
                   'source_analysis_sha256': ANALYSIS_SHA, 'fixed_contrasts_canonical_sha256': CONTRASTS_SHA,
                   'renderer_sha256': sha(Path(__file__).read_bytes()),
                   'only_changed_display': 'Panel titles use accuracy difference (pp) / NLL difference, not generic B-A',
                   'row_names_order_factor_line_color_caps_zero_and_interval_scope_retained': True,
                   'new_statistical_estimates_bootstrap_model_forward_raw_images_or_NPZ_read': False,
                   'actual_Root_visual_review_complete': False})
    with (output / 'render_receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'state': 'presentation_only_rendered_Root_visual_review_pending',
                      'figure_sha256': result['figure']['sha256'], 'output_dir': str(output)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

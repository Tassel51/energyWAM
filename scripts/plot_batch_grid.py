"""Plot recorded batch-grid evidence; never infer missing validation points."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT = Path('/cloud/zzc/energyWAM/experiments/batch_grid_micro4_20261105')
COLORS = {8: '#0072B2', 16: '#D55E00', 32: '#009E73', 64: '#CC79A7'}


def render(report, destination):
    destination.mkdir(parents=True, exist_ok=True)
    arms = report['arms']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    views = [('sample_visits', 1, 'Training sample visits'),
             ('active_session_wall_seconds_to_evaluation', 3600,
              'Active session hours (setup / evaluation / saves included)'),
             ('cumulative_unique_logged_step_seconds', 3600,
              'Unique logged training-step hours')]
    for arm in arms:
        batch = arm['batch']
        points = sorted(arm.get('validation', {}).values(),
                        key=lambda p: p['sample_visits'])
        if not points:
            continue
        label = f"B{batch}: VAL through {points[-1]['sample_visits']} visits"
        if arm.get('execution_status')=='failed':
            label += f" (stopped at {arm['completed_sample_visits']})"
        for col, (key, scale, xlabel) in enumerate(views):
            observed = [p for p in points if p.get(key) is not None]
            for row, metric in enumerate(['action_mse', 'future_mse']):
                axes[row, col].plot([p[key] / scale for p in observed],
                                    [p[metric] for p in observed], 'o-',
                                    color=COLORS[batch], label=label,
                                    markersize=4, linewidth=1.6)
                axes[row, col].set_xlabel(xlabel)
    # Baselines are computed from TRAIN only, evaluated on the same quick VAL.
    baseline_arm = next((a for a in arms if a.get('simple_baselines')), None)
    if baseline_arm:
        base = baseline_arm['simple_baselines']['val']
        for ax in axes[0]:
            ax.axhline(base['action']['training_mean_mse'], color='#555555',
                       linestyle='--', linewidth=1, label='TRAIN-mean action')
        for ax in axes[1]:
            ax.axhline(base['future']['current_visual_persistence_mse'],
                       color='#555555', linestyle='--', linewidth=1,
                       label='Current-image persistence')
    for row, label in enumerate(['Quick VAL Action MSE', 'Quick VAL Future MSE']):
        axes[row, 0].set_ylabel(label + ' (lower is better)')
        for ax in axes[row]:
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
    missing = [str(a['batch']) for a in arms if not a.get('validation')]
    placement = report.get('hardware_placement_control') or {}
    placement_note = ('Short B16 placement control available; finalist seed swaps pending.'
                      if placement.get('phase') == 'complete'
                      else 'Hardware groups are not yet controlled.')
    fig.suptitle('Native microbatch 4, K=2, seed42 | ' + report['status'] +
                 (' | Not started: B' + ', B'.join(missing) if missing else '') +
                 '\nMeasured checkpoints only; lines guide the eye. '
                 'One training seed. ' + placement_note, fontsize=11)
    paths = []
    for extension in ['png', 'pdf']:
        path = destination / ('batch_grid_quality.' + extension)
        fig.savefig(path, dpi=160)
        paths.append(str(path))
    # Keep failures visible in the full-range artifact above. A separate
    # explicitly cropped view makes small differences assessable without
    # removing failed trials from either the report or the plotted data.
    stable_values = [p['action_mse'] for arm in arms
                     if arm.get('execution_status') != 'failed'
                     for p in arm.get('validation', {}).values()]
    if stable_values:
        padding = max(.03, .1 * (max(stable_values) - min(stable_values)))
        limits = (max(0, min(stable_values) - padding),
                  max(stable_values) + padding)
        clipped = [(arm['batch'], p['sample_visits'], p['action_mse'])
                   for arm in arms for p in arm.get('validation', {}).values()
                   if not limits[0] <= p['action_mse'] <= limits[1]]
        if clipped:
            for ax in axes[0]:
                ax.set_ylim(*limits)
            detail = '; '.join(f'B{b} at {n} visits: {v:.4f}'
                               for b, n, v in clipped)
            fig.suptitle('Action-axis zoom | ' + report['status'] +
                         '\nOutside Action display range: ' + detail +
                         '\nAll points retained in batch_grid_quality (full range); '
                         'Future axes unchanged. One training seed.', fontsize=10)
            for extension in ['png', 'pdf']:
                path = destination / ('batch_grid_quality_action_zoom.' + extension)
                fig.savefig(path, dpi=160)
                paths.append(str(path))
    plt.close(fig)
    measured = [a for a in arms if a.get('completed_updates', 0) > 0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    panels = [
        ('Steady training samples / second', lambda a:
         1 / a['steady_median_seconds_per_sample']
         if a.get('steady_median_seconds_per_sample') else None),
        ('Sampled process USS peak (GiB)', lambda a:
         a.get('resource_peaks', {}).get('process_uss_gib')),
        ('Maximum assigned-GPU memory peak (GiB)', lambda a:
         max((g['memory_used_mib'] / 1024
              for g in a.get('gpu_peaks', {}).values()
              if g.get('memory_used_mib') is not None), default=None)),
    ]
    for ax, (title, getter) in zip(axes, panels):
        for arm in measured:
            value = getter(arm)
            if value is None:
                continue
            batch = arm['batch']
            ax.bar(f'B{batch}', value, color=COLORS[batch])
            ax.annotate(f'{value:.3g}', (f'B{batch}', value),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.margins(y=.18)
    fig.suptitle('Observed runs only | ' + report['status'] +
                 '\nDifferent GPU groups / incomplete budgets: '
                 'descriptive measurements, not intrinsic batch-speed ranking.',
                 fontsize=10)
    for extension in ['png', 'pdf']:
        path = destination / ('batch_grid_resources.' + extension)
        fig.savefig(path, dpi=160)
        paths.append(str(path))
    plt.close(fig)
    return paths


def render_lr(report, destination):
    """Show measured matched-budget LR points and explicitly mark missing ones."""
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharey='row', constrained_layout=True)
    for column, group in enumerate(report['groups']):
        for row, metric in enumerate(['action_mse', 'future_mse']):
            ax = axes[row, column]
            for candidate in group['candidates']:
                factor = candidate['outer_lr_factor']
                point = candidate.get('validation_at1024')
                if point:
                    ax.scatter(factor, point[metric], color=COLORS[group['batch']], s=35)
                    ax.annotate(f'{point[metric]:.4f}', (factor, point[metric]),
                                xytext=(0, 7), textcoords='offset points',
                                ha='center', fontsize=8)
                else:
                    ax.text(factor, .1, 'not\nmeasured', transform=ax.get_xaxis_transform(),
                            ha='center', fontsize=7, color='#666666')
            ax.set_xscale('log')
            ax.set_xlim(.2, 4.5)
            ax.set_xticks([.3, 1, 3], ['0.3', '1', '3'])
            ax.minorticks_off()
            ax.margins(y=.3)
            ax.grid(alpha=.2)
            if row == 0:
                ax.set_title(f"B{group['batch']} | {group['status']}", fontsize=10)
            else:
                ax.set_xlabel('Outer LR multiplier')
            if column == 0:
                ax.set_ylabel('VAL ' + ('Action' if row == 0 else 'Future') + ' MSE')
    fig.suptitle('Matched 1024 sample visits | ' + report['status'] +
                 '\nMeasured fixed100VAL points only; missing trials are not failures. '
                 'One seed; this is not final LR/batch selection.', fontsize=10)
    paths = []
    for extension in ['png', 'pdf']:
        path = destination / ('batch_grid_matched_lr.' + extension)
        fig.savefig(path, dpi=160)
        paths.append(str(path))
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT)
    parser.add_argument('--lr-probes', action='store_true')
    args = parser.parse_args()
    report_path = args.root / 'main_comparison_report.json'
    result = render(json.loads(report_path.read_text()), args.root / 'figures')
    if args.lr_probes:
        result += render_lr(json.loads((args.root / 'matched_lr_comparison_report.json').read_text()),
                            args.root / 'figures')
    print(json.dumps({'figures': result, 'source': str(report_path)}, indent=2))


if __name__ == '__main__':
    main()

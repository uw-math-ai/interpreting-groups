"""Layout adjustments for checkpoint-based geometry figures.

Only colors, labels, legends and layout change. The plotted coordinates are
fingerprinted before and after each adjustment.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from matplotlib.text import Annotation


def numeric_hash(fig):
    values = []
    for ax in fig.axes:
        for line in ax.lines:
            values.append([np.asarray(line.get_xdata()).tolist(), np.asarray(line.get_ydata()).tolist()])
        for collection in ax.collections:
            if hasattr(collection, '_offsets3d'):
                values.append([np.asarray(x).tolist() for x in collection._offsets3d])
            elif hasattr(collection, 'get_offsets') and len(collection.get_offsets()) > 1:
                values.append(np.asarray(collection.get_offsets()).tolist())
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def polish_geometry(fig, stem, output):
    if getattr(fig, '_publication_styled', False):
        return
    before = numeric_hash(fig)
    titles = {
        'figure_geometry': ['(a) SU(2): $\\exp(tX)$', '(b) SO(3): same path', '(c) SU(3): maximal torus'],
        'figure_geometry_app': ['(d) SU(2)\nantipodal pairs', '(e) SO(3)\nantipodal pairs',
                                '(f) SU(2)\nmetric shape', '(g) SO(3)\nmetric shape'],
        'figure_covers': ['(a) Spin(5)\npath $\\exp(tX)$', '(b) SO(5)\nsame path',
                          '(c) SU(3)\ncenter rotation', '(d) PSU(3)\ncenter collapse'],
        'figure_tori': ['(a) Spin(4)', '(b) SO(4)', '(c) SO(3)$\\times$SU(2)', '(d) PSO(4)'],
    }
    axes = list(fig.axes[:len(titles[stem])])
    for ax, title in zip(axes, titles[stem]):
        ax.set_title(title, loc='left', fontsize=9.5, pad=10)
        ax.tick_params(labelsize=8, pad=2)
        ax.xaxis.label.set_size(8.5)
        ax.yaxis.label.set_size(8.5)
        legend = ax.get_legend()
        if legend:
            legend.remove()  # Retracing is explained in the figure caption.
        for text in ax.texts:
            if not isinstance(text, Annotation):
                continue
            if 'and' in text.get_text():
                text.set_text(r'$t=0,\,2\pi$')
            x, y = text.xy
            dx = 8 if x < np.mean(ax.get_xlim()) else -8
            dy = 8 if y < np.mean(ax.get_ylim()) else -12
            text.set_position((dx, dy))
            text.set_ha('left' if dx > 0 else 'right')
            text.set_va('bottom' if dy > 0 else 'top')
            text.set_fontsize(8.5)
            text.set_bbox(dict(facecolor='white', edgecolor='none', alpha=.92, pad=1.2))
    if stem == 'figure_geometry':
        # Give the shared colorbar a dedicated horizontal strip.
        fig.axes[-1].remove()
        fig.subplots_adjust(left=.065, right=.99, bottom=.31, top=.84, wspace=.36)
        cax = fig.add_axes([.10, .105, .47, .028])
        cb = fig.colorbar(axes[0].collections[0], cax=cax, orientation='horizontal',
                         ticks=np.arange(5)*np.pi)
        cb.ax.set_xticklabels(['$0$', '$\\pi$', '$2\\pi$', '$3\\pi$', '$4\\pi$'])
        cb.ax.tick_params(labelsize=7, pad=1, length=2)
        cb.set_label('rotation angle $t$', fontsize=8, labelpad=1)
    elif stem in ('figure_covers', 'figure_geometry_app'):
        fig.set_size_inches(9.6, 2.65)
        fig.subplots_adjust(left=.065, right=.985, bottom=.21, top=.76, wspace=.43)
    else:
        fig.subplots_adjust(left=.025, right=.985, bottom=.03, top=.83, wspace=.18)
    after = numeric_hash(fig)
    assert before == after, (stem, 'Plot coordinates changed')
    fig._publication_styled = True
    path = Path(output) / 'geometry_numeric_audit.json'
    report = json.loads(path.read_text()) if path.exists() else {}
    report[stem] = {'numeric_coordinates_unchanged': True, 'sha256': after}
    path.write_text(json.dumps(report, indent=2) + '\n')

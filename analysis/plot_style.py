"""
plot_style.py - shared publication-grade styling for all paper figures
=========================================================================

Conventions enforced:
  - No plot titles (use figure captions in LaTeX instead)
  - Large bold axis labels (14pt bold)
  - Large tick labels (12pt)
  - Consistent color scheme across all figures
  - 2.5pt line width, 10-12pt markers with edge stroke
  - Tight layout, white background, subtle grid
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams


# --- Color palette: Wong's color-blind-safe scheme ------------------------
# Wong (2011) Nature Methods 8:441 - recommended by Nature Publishing for
# color-blind accessibility and print friendliness.  Perceptually uniform
# under deuteranopia, protanopia, and tritanopia.  Distinguishable in greyscale.
COL_AL          = '#0072B2'   # blue           - PINN+AL (Our Work)        primary
COL_SEQ_RANDOM  = '#E69F00'   # orange         - Sequential Random         secondary
COL_RAND_RANDOM = '#D55E00'   # vermilion      - Random-Random             tertiary
COL_IC_CORESET  = '#56B4E9'   # sky blue       - IC-coreset variant        light
COL_ANN_AL      = '#999999'   # mid grey       - ANN+AL (deemphasized)
COL_ANN_RR      = '#BDBDBD'   # light grey     - ANN+random (deemphasized)
COL_PHYSICS     = '#009E73'   # bluish green   - physics signal
COL_GRADIENT    = '#CC79A7'   # reddish purple - gradient signal
COL_QBC         = '#F0E442'   # yellow         - committee disagreement
COL_BADGE       = '#0072B2'   # blue           - alias for AL (BADGE is the AL we use)

METHOD_COLORS = {
    'pinn_al':           COL_AL,
    'pinn_ic_coreset':   COL_IC_CORESET,
    'pinn_seqrand':      COL_SEQ_RANDOM,
    'pinn_oneshot':      COL_RAND_RANDOM,
    'ann_al':            COL_ANN_AL,
    'ann_oneshot':       COL_ANN_RR,
    'sequential_al':     COL_AL,
    'sequential_random': COL_SEQ_RANDOM,
    'random_random':     COL_RAND_RANDOM,
}

METHOD_LABELS = {
    'pinn_al':           'PINN+AL (Our Work)',
    'pinn_ic_coreset':   'PINN+AL (IC-coreset)',
    'pinn_seqrand':      'Sequential Random',
    'pinn_oneshot':      'Random-Random',
    'ann_al':            'ANN+AL',
    'ann_oneshot':       'ANN+random',
    'sequential_al':     'PINN+AL (Our Work)',
    'sequential_random': 'Sequential Random',
    'random_random':     'Random-Random',
}

METHOD_MARKERS = {
    'pinn_al':           'o',
    'pinn_ic_coreset':   'P',
    'pinn_seqrand':      's',
    'pinn_oneshot':      'D',
    'ann_al':            '^',
    'ann_oneshot':       'v',
    'sequential_al':     'o',
    'sequential_random': 's',
    'random_random':     'D',
}


def apply_paper_style():
    """Set matplotlib rcParams for publication-grade output.
    Call once at the start of any plotting script.
    """
    rcParams.update({
        # Fonts - bumped sizes for journal readability
        'font.family':       'DejaVu Sans',
        'font.size':         13,
        'axes.titlesize':    14,
        'axes.titleweight':  'bold',
        'axes.labelsize':    16,      # was 14
        'axes.labelweight':  'bold',
        'xtick.labelsize':   13,      # was 12
        'ytick.labelsize':   13,      # was 12
        'legend.fontsize':   12,
        'figure.titlesize':  16,
        'figure.titleweight': 'bold',

        # Axes
        'axes.linewidth':    1.4,
        'axes.edgecolor':    '#333333',
        'axes.grid':         True,
        'axes.axisbelow':    True,
        'axes.spines.top':   False,
        'axes.spines.right': False,

        # Grid (subtle)
        'grid.linestyle':    '-',
        'grid.linewidth':    0.6,
        'grid.color':        '#cccccc',
        'grid.alpha':        0.6,

        # Ticks
        'xtick.major.size':  6,
        'xtick.major.width': 1.2,
        'ytick.major.size':  6,
        'ytick.major.width': 1.2,
        'xtick.minor.size':  3,
        'ytick.minor.size':  3,
        'xtick.direction':   'out',
        'ytick.direction':   'out',

        # Lines / markers
        'lines.linewidth':     2.4,
        'lines.markersize':    9,
        'lines.markeredgewidth': 1.0,

        # Legend
        'legend.frameon':      True,
        'legend.framealpha':   0.92,
        'legend.facecolor':    'white',
        'legend.edgecolor':    '#888888',
        'legend.borderpad':    0.6,
        'legend.handlelength': 1.6,

        # Save
        'savefig.dpi':         200,
        'savefig.bbox':        'tight',
        'figure.dpi':          110,
        'figure.facecolor':    'white',
    })


def get_color(method):    return METHOD_COLORS.get(method, '#555555')
def get_label(method):    return METHOD_LABELS.get(method, method)
def get_marker(method):   return METHOD_MARKERS.get(method, 'o')


def annotate_bar(ax, bars, values, fontsize=11, dy_frac=0.02, fmt='{:.0f}%'):
    """Annotate bar heights with their numeric values."""
    ymin, ymax = ax.get_ylim()
    dy = (ymax - ymin) * dy_frac
    for bar, v in zip(bars, values):
        if v is None or (isinstance(v, float) and not (v == v)):  # nan
            continue
        ax.text(bar.get_x() + bar.get_width()/2, v + dy,
                 fmt.format(v), ha='center', va='bottom',
                 fontsize=fontsize, fontweight='bold')


def style_boxplot(bp, colors):
    """Color and stroke boxplot elements."""
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
        patch.set_edgecolor('#222222')
        patch.set_linewidth(1.3)
    for whisker in bp.get('whiskers', []):
        whisker.set_color('#222222')
        whisker.set_linewidth(1.2)
    for cap in bp.get('caps', []):
        cap.set_color('#222222')
        cap.set_linewidth(1.2)
    for median in bp.get('medians', []):
        median.set_color('black')
        median.set_linewidth(1.8)

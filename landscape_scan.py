# landscape_scan.py
# Fitness landscape diagnostic for Markov regime-switching calibration.
# Computes all 6 parameter pair grids. Renders interactive Plotly 3D
# surfaces with hover coordinates, HMM seed marker, grid minimum marker,
# detailed annotations, and dropdown pair switching.
# Saves one PNG + one HTML per pair, plus one combined interactive HTML.
#
# NOTE: This file has been scrubbed for public release as a code sample
# submitted in support of OAG application RAD_NYC_DAT_6444. Internal
# project paths, credential references, and pipeline wiring have been
# removed. The grid computation, Plotly rendering, and diagnostic logic
# are intact and representative of the production code.

# SELECTOR -- which pair is displayed first when the window opens.
# Switch any time using the dropdown in the browser figure.
PARAM_PAIR = 'sigma2_lambda1'
# Valid values:
#   'sigma1_sigma2'   -- sigma1 vs sigma2,  lambda1 and lambda2 fixed
#   'sigma1_lambda1'  -- sigma1 vs lambda1, sigma2 and lambda2 fixed
#   'sigma1_lambda2'  -- sigma1 vs lambda2, sigma2 and lambda1 fixed
#   'sigma2_lambda1'  -- sigma2 vs lambda1, sigma1 and lambda2 fixed
#   'sigma2_lambda2'  -- sigma2 vs lambda2, sigma1 and lambda1 fixed
#   'lambda1_lambda2' -- lambda1 vs lambda2, sigma1 and sigma2 fixed

# GRID AND AXIS BOUNDS
# _AXIS_LOW  : lower bound as fraction of HMM center value.
# _AXIS_HIGH : upper bound as fraction of HMM center value.
_GRID_POINTS = 30
_AXIS_LOW    = 0.05
_AXIS_HIGH   = 2.70

import time
import datetime
import numpy as np
from pathlib import Path
import plotly.graph_objects as go

_PAIR_ORDER = [
    'sigma1_sigma2',
    'sigma1_lambda1',
    'sigma1_lambda2',
    'sigma2_lambda1',
    'sigma2_lambda2',
    'lambda1_lambda2',
]

_VALID_PAIRS = {
    'sigma1_sigma2'  : ('sigma1',  'sigma2',  'lambda1', 'lambda2'),
    'sigma1_lambda1' : ('sigma1',  'lambda1', 'sigma2',  'lambda2'),
    'sigma1_lambda2' : ('sigma1',  'lambda2', 'sigma2',  'lambda1'),
    'sigma2_lambda1' : ('sigma2',  'lambda1', 'sigma1',  'lambda2'),
    'sigma2_lambda2' : ('sigma2',  'lambda2', 'sigma1',  'lambda1'),
    'lambda1_lambda2': ('lambda1', 'lambda2', 'sigma1',  'sigma2'),
}

# Dark Bloomberg-style background
_BG      = 'rgba(10,10,30,1)'
_GRID_CL = 'rgba(180,180,200,0.25)'


def _make_param_array(val_map):
    """Return numpy array [sigma1, sigma2, lambda1, lambda2] from val_map."""
    return np.array([val_map['sigma1'], val_map['sigma2'],
                     val_map['lambda1'], val_map['lambda2']])


def _compute_grid(pair_name, center, all_expiry_data, fitness_fn):
    """
    Evaluate fitness across _GRID_POINTS x _GRID_POINTS grid for one pair.
    fitness_fn : callable(params_array, all_expiry_data) -> float
    Returns dict of grid arrays, metadata, and grid minimum location.
    """
    param_a, param_b, param_c, param_d = _VALID_PAIRS[pair_name]
    a_center = center[param_a]
    b_center = center[param_b]
    c_val    = center[param_c]
    d_val    = center[param_d]

    a_vals = np.linspace(_AXIS_LOW  * a_center,
                         _AXIS_HIGH * a_center, _GRID_POINTS)
    b_vals = np.linspace(_AXIS_LOW  * b_center,
                         _AXIS_HIGH * b_center, _GRID_POINTS)

    Z = np.full((_GRID_POINTS, _GRID_POINTS), float('nan'))

    t_start = time.time()

    for i, a_val in enumerate(a_vals):
        t_row = time.time()
        for j, b_val in enumerate(b_vals):
            val_map = {param_a: a_val, param_b: b_val,
                       param_c: c_val,  param_d: d_val}
            # Enforce sigma1 < sigma2 (state ordering constraint)
            if val_map['sigma1'] >= val_map['sigma2']:
                Z[i, j] = float('nan')
                continue
            params = _make_param_array(val_map)
            Z[i, j] = fitness_fn(params, all_expiry_data)

        elapsed_row  = time.time() - t_row
        elapsed_grid = time.time() - t_start
        if (i + 1) % max(1, _GRID_POINTS // 2) == 0 or (i + 1) == _GRID_POINTS:
            print('[landscape]   [%s] Row %d/%d  row=%.1fs  total=%.1fs'
                  % (pair_name, i + 1, _GRID_POINTS, elapsed_row, elapsed_grid))

    X, Y = np.meshgrid(a_vals, b_vals, indexing='ij')

    # Grid minimum -- lowest fitness found on the grid
    valid_mask = ~np.isnan(Z)
    if valid_mask.any():
        flat_idx     = int(np.nanargmin(Z))
        min_i, min_j = np.unravel_index(flat_idx, Z.shape)
        min_a        = float(a_vals[min_i])
        min_b        = float(b_vals[min_j])
        min_z        = float(Z[min_i, min_j])
        z_max        = float(np.nanmax(Z))
        z_min        = min_z
    else:
        min_a = min_b = float('nan')
        min_z = z_max = float('nan')
        z_min         = float('nan')

    return {
        'X'       : X,       'Y'       : Y,       'Z'      : Z,
        'a_vals'  : a_vals,  'b_vals'  : b_vals,
        'param_a' : param_a, 'param_b' : param_b,
        'param_c' : param_c, 'param_d' : param_d,
        'a_center': a_center,'b_center': b_center,
        'c_val'   : c_val,   'd_val'   : d_val,
        'min_a'   : min_a,   'min_b'   : min_b,   'min_z'  : min_z,
        'z_max'   : z_max,   'z_min'   : z_min,
    }


def _build_traces(grid, center_z, visible=True):
    """
    Build three Plotly traces for one parameter pair:
      1. Surface     -- fitness landscape with hover coordinates
      2. Seed cone   -- red upward arrow at HMM seed (up = worse fitness)
      3. Grid min    -- lime diamond at the lowest fitness point on the grid
    """
    param_a  = grid['param_a']
    param_b  = grid['param_b']
    param_c  = grid['param_c']
    param_d  = grid['param_d']
    a_center = grid['a_center']
    b_center = grid['b_center']
    c_val    = grid['c_val']
    d_val    = grid['d_val']
    z_max    = grid['z_max'] if not np.isnan(grid['z_max']) else 1.0

    # Cone height: 12% of the fitness range so it is always visible
    cone_h = max(z_max * 0.12, 1e-6)

    # 1. Surface -- viridis colorscale, hover shows x/y/z coordinates
    surface = go.Surface(
        x=grid['X'],
        y=grid['Y'],
        z=grid['Z'],
        colorscale='Viridis',
        colorbar=dict(
            title=dict(text='Fitness', side='right',
                       font=dict(size=12, color='white')),
            tickformat='.3e',
            nticks=10,
            thickness=18,
            len=0.75,
            bgcolor='rgba(20,20,50,0.8)',
            bordercolor='white',
            tickfont=dict(color='white', size=10),
        ),
        contours=dict(
            z=dict(
                show=True,
                usecolormap=True,
                highlightcolor='white',
                project=dict(z=False),
                width=1,
            ),
        ),
        opacity=0.88,
        hovertemplate=(
            '<b>%s</b>: %%{x:.7f}<br>'
            '<b>%s</b>: %%{y:.7f}<br>'
            '<b>Fitness</b>: %%{z:.6e}<br>'
            '<extra>%s vs %s</extra>'
        ) % (param_a, param_b, param_a, param_b),
        visible=visible,
        showlegend=True,
        name='Fitness surface',
        legendgroup='surface_%s_%s' % (param_a, param_b),
    )

    # 2. HMM seed cone -- red, pointing upward (toward higher/worse fitness)
    seed_cone = go.Cone(
        x=[a_center],
        y=[b_center],
        z=[center_z],
        u=[0], v=[0], w=[cone_h],
        colorscale=[[0, 'red'], [1, 'red']],
        showscale=False,
        sizemode='absolute',
        sizeref=cone_h * 0.35,
        anchor='tail',
        hovertemplate=(
            '<b>HMM Seed</b><br>'
            '<b>%s (seed)</b>: %.7f<br>'
            '<b>%s (seed)</b>: %.7f<br>'
            '<b>Fitness at seed</b>: %.6e<br>'
            '<b>Fixed %s</b>: %.7f<br>'
            '<b>Fixed %s</b>: %.7f<br>'
            'Arrow points toward worse fitness (+z)'
            '<extra>HMM Seed</extra>'
        ) % (param_a, a_center, param_b, b_center, center_z,
             param_c, c_val, param_d, d_val),
        visible=visible,
        showlegend=True,
        name='HMM seed (red arrow)',
        legendgroup='seed_%s_%s' % (param_a, param_b),
    )

    # 3. Grid minimum -- lime green diamond
    grid_min = go.Scatter3d(
        x=[grid['min_a']],
        y=[grid['min_b']],
        z=[grid['min_z']],
        mode='markers',
        marker=dict(
            symbol='diamond',
            size=7,
            color='lime',
            line=dict(color='darkgreen', width=2),
        ),
        hovertemplate=(
            '<b>Grid Minimum</b><br>'
            '<b>%s</b>: %.7f<br>'
            '<b>%s</b>: %.7f<br>'
            '<b>Fitness</b>: %.6e<br>'
            'Lowest fitness found on this %dx%d grid'
            '<extra>Grid Min</extra>'
        ) % (param_a, grid['min_a'], param_b, grid['min_b'],
             grid['min_z'], _GRID_POINTS, _GRID_POINTS),
        visible=visible,
        showlegend=True,
        name='Grid min (lime diamond)',
        legendgroup='min_%s_%s' % (param_a, param_b),
    )

    return [surface, seed_cone, grid_min]


def _build_title(grid, center_z, ticker_label):
    """Full title with all seed and fixed parameter values."""
    return (
        '<b>%s Fitness Landscape</b>  |  <b>%s</b> vs <b>%s</b><br>'
        '<sup>'
        'HMM Seed:  %s=<b>%.6f</b>  %s=<b>%.6f</b>  '
        'fitness=<b>%.4e</b>'
        '&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;'
        'Fixed:  %s=<b>%.6f</b>  %s=<b>%.6f</b>'
        '<br>'
        'Grid min fitness: <b>%.4e</b>'
        '&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;'
        'Scan: [%.4f, %.4f] x [%.4f, %.4f]'
        '</sup>'
    ) % (
        ticker_label, grid['param_a'], grid['param_b'],
        grid['param_a'], grid['a_center'],
        grid['param_b'], grid['b_center'],
        center_z,
        grid['param_c'], grid['c_val'],
        grid['param_d'], grid['d_val'],
        grid['min_z'],
        float(grid['a_vals'][0]),  float(grid['a_vals'][-1]),
        float(grid['b_vals'][0]),  float(grid['b_vals'][-1]),
    )


def _build_scene(grid):
    """Build Plotly scene dict with readable axis ticks and labels."""
    def _ax(label, vals):
        lo = float(vals[0])
        hi = float(vals[-1])
        span = hi - lo
        raw_step  = span / 8.0
        magnitude = 10 ** int(np.floor(np.log10(raw_step + 1e-15)))
        step      = round(raw_step / magnitude) * magnitude
        if step == 0:
            step = raw_step
        ticks = list(np.arange(lo, hi + step * 0.5, step))
        return dict(
            title=dict(text='<b>%s</b>' % label, font=dict(size=13, color='white')),
            tickvals=ticks,
            ticktext=['%.5f' % t for t in ticks],
            tickfont=dict(size=9, color='white'),
            gridcolor=_GRID_CL,
            backgroundcolor=_BG,
            showbackground=True,
            zerolinecolor='rgba(255,255,255,0.3)',
        )

    z_vals = grid['Z'][~np.isnan(grid['Z'])]
    z_lo   = float(np.min(z_vals)) if len(z_vals) else 0.0
    z_hi   = float(np.max(z_vals)) if len(z_vals) else 1.0
    z_span = z_hi - z_lo
    z_step = z_span / 8.0
    if z_step > 0:
        mag    = 10 ** int(np.floor(np.log10(z_step + 1e-15)))
        z_step = round(z_step / mag) * mag
    z_ticks = (list(np.arange(z_lo, z_hi + z_step * 0.5, z_step))
               if z_step > 0 else [z_lo, z_hi])

    return dict(
        xaxis=_ax(grid['param_a'], grid['a_vals']),
        yaxis=_ax(grid['param_b'], grid['b_vals']),
        zaxis=dict(
            title=dict(text='<b>Fitness</b>', font=dict(size=13, color='white')),
            tickvals=z_ticks,
            ticktext=['%.3e' % t for t in z_ticks],
            tickfont=dict(size=9, color='white'),
            gridcolor=_GRID_CL,
            backgroundcolor=_BG,
            showbackground=True,
            zerolinecolor='rgba(255,255,255,0.3)',
        ),
        bgcolor=_BG,
        camera=dict(eye=dict(x=1.6, y=1.6, z=1.1)),
        aspectmode='auto',
    )


def _base_layout(title_text, scene_dict):
    """Shared layout settings for all figures."""
    return dict(
        title=dict(
            text=title_text,
            x=0.5, xanchor='center',
            font=dict(size=13, color='white'),
        ),
        scene=scene_dict,
        legend=dict(
            x=0.01, y=0.95,
            bgcolor='rgba(20,20,50,0.85)',
            bordercolor='rgba(180,180,255,0.5)',
            borderwidth=1,
            font=dict(color='white', size=11),
        ),
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(color='white'),
        margin=dict(l=0, r=0, t=130, b=0),
        width=1280,
        height=860,
    )


def _save_pair_figure(pair_name, grid, center_z,
                      ticker_label, timestamp_str, save_dir):
    """Build, save HTML and PNG for a single parameter pair."""
    traces     = _build_traces(grid, center_z, visible=True)
    title_text = _build_title(grid, center_z, ticker_label)
    scene      = _build_scene(grid)

    fig = go.Figure(data=traces)
    fig.update_layout(**_base_layout(title_text, scene))

    stem      = '%s_fitness_%s_%s' % (ticker_label, pair_name, timestamp_str)
    html_path = save_dir / (stem + '.html')
    fig.write_html(str(html_path))
    print('[landscape]   Saved HTML : %s' % html_path.name)

    try:
        png_path = save_dir / (stem + '.png')
        fig.write_image(str(png_path), scale=2)
        print('[landscape]   Saved PNG  : %s' % png_path.name)
    except Exception as e:
        print('[landscape]   PNG skipped (kaleido not available).')
        print('[landscape]   Install: pip install kaleido==0.2.1')
        print('[landscape]   (%s)' % str(e)[:80])


def _build_combined_figure(grids, center_z, ticker_label,
                            timestamp_str, save_dir):
    """
    One Plotly figure containing all 6 pairs.
    Dropdown switches which pair is visible.
    Hover coordinates work on the active surface.
    """
    n_per_pair = 3  # surface + cone + grid_min
    all_traces = []
    active_idx = _PAIR_ORDER.index(PARAM_PAIR)

    for p_idx, pair_name in enumerate(_PAIR_ORDER):
        is_active = (p_idx == active_idx)
        traces    = _build_traces(grids[pair_name], center_z, visible=is_active)
        all_traces.extend(traces)

    # Dropdown buttons -- each button shows one pair, hides others
    buttons = []
    n_pairs = len(_PAIR_ORDER)
    for p_idx, pair_name in enumerate(_PAIR_ORDER):
        grid       = grids[pair_name]
        visibility = [False] * (n_pairs * n_per_pair)
        base       = p_idx * n_per_pair
        visibility[base]     = True   # surface
        visibility[base + 1] = True   # seed cone
        visibility[base + 2] = True   # grid min

        scene_upd = _build_scene(grid)
        title_upd = _build_title(grid, center_z, ticker_label)

        buttons.append(dict(
            label=pair_name,
            method='update',
            args=[
                {'visible': visibility},
                {
                    'title.text'  : title_upd,
                    'scene.xaxis' : scene_upd['xaxis'],
                    'scene.yaxis' : scene_upd['yaxis'],
                    'scene.zaxis' : scene_upd['zaxis'],
                    'scene.camera': scene_upd['camera'],
                },
            ],
        ))

    active_grid = grids[PARAM_PAIR]
    fig = go.Figure(data=all_traces)

    layout = _base_layout(
        _build_title(active_grid, center_z, ticker_label),
        _build_scene(active_grid),
    )
    layout['updatemenus'] = [dict(
        type='dropdown',
        direction='down',
        x=0.01, y=1.13,
        xanchor='left',
        bgcolor='rgba(30,30,70,0.95)',
        bordercolor='rgba(180,180,255,0.6)',
        font=dict(color='white', size=12),
        buttons=buttons,
        active=active_idx,
        showactive=True,
    )]
    layout['width']  = 1400
    layout['height'] = 920
    layout['margin'] = dict(l=0, r=0, t=160, b=0)

    fig.update_layout(**layout)

    html_path = save_dir / ('%s_fitness_all_pairs_%s.html' % (ticker_label, timestamp_str))
    fig.write_html(str(html_path))
    print('[landscape]   Saved combined HTML: %s' % html_path.name)

    return fig


def run_landscape_scan(ticker, all_expiry_data, center, fitness_fn,
                       save_dir=None):
    """
    Computes all 6 parameter pair grids then opens an interactive Plotly
    3D surface in the browser. Saves HTML and PNG for each pair plus one
    combined interactive HTML with dropdown switching.

    ticker          : str   ticker label for figure titles and filenames
    all_expiry_data : list  pre-built expiry data list (market data + draws)
    center          : dict  HMM seed parameters:
                            {'sigma1': float, 'sigma2': float,
                             'lambda1': float, 'lambda2': float}
    fitness_fn      : callable(params_array, all_expiry_data) -> float
    save_dir        : Path or None  output directory (defaults to ./landscape_output)
    """
    if PARAM_PAIR not in _VALID_PAIRS:
        raise ValueError(
            'PARAM_PAIR %r is not valid. '
            'Choose one of: %s' % (PARAM_PAIR, ', '.join(_PAIR_ORDER))
        )

    if save_dir is None:
        save_dir = Path('./landscape_output')
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ticker_label = ticker.lstrip('$')

    # Center-point fitness
    center_z = fitness_fn(_make_param_array(center), all_expiry_data)
    print('[landscape] Center point:')
    print('[landscape]   sigma1=%.6f  sigma2=%.6f  lambda1=%.6f  lambda2=%.6f'
          % (center['sigma1'], center['sigma2'],
             center['lambda1'], center['lambda2']))
    print('[landscape]   Center-point fitness: %.6e' % center_z)

    # Compute all 6 grids
    total = 6 * _GRID_POINTS * _GRID_POINTS
    print('[landscape] Computing all 6 grids. %d x %d x 6 = %d total evaluations ...'
          % (_GRID_POINTS, _GRID_POINTS, total))
    t_all = time.time()
    grids = {}

    for pair_name in _PAIR_ORDER:
        t_pair = time.time()
        print('[landscape]   Grid: %s ...' % pair_name)
        grids[pair_name] = _compute_grid(pair_name, center, all_expiry_data, fitness_fn)
        g = grids[pair_name]
        print('[landscape]   Grid %s done.'
              '  grid_min=%.4e  seed_fitness=%.4e  elapsed=%.1fs'
              % (pair_name, g['min_z'], center_z, time.time() - t_pair))

    print('[landscape] All 6 grids done. total=%.1fs' % (time.time() - t_all))

    # Save figures
    timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d_%I_%M_%S%p')
    print('[landscape] Saving figures to: %s' % save_dir)

    for pair_name in _PAIR_ORDER:
        _save_pair_figure(pair_name, grids[pair_name], center_z,
                          ticker_label, timestamp_str, save_dir)

    combined_fig = _build_combined_figure(
        grids, center_z, ticker_label, timestamp_str, save_dir)

    print('[landscape] Done.')

    try:
        combined_fig.show()
    except Exception:
        pass

"""
Figure 1 of the paper: the Peprika pipeline (A to C), the label evidence, effective units and similarity components of the
corpus (D to F), and the audit and application strip (G). The canvas is 165 mm wide, the text width of the paper, so the
sizes below are the sizes on the page. Writes paper/figures/overview.pdf and overview.png.

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/make_figure_overview.py > logs/make_figure_overview.log 2>&1
"""

from __future__ import annotations

import itertools
import math
import sys
import textwrap
from functools import lru_cache
from pathlib import Path
from contextlib import contextmanager
from matplotlib.transforms import Affine2D
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon, Rectangle, Wedge
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ampx.evaluation import load_frame, task_selection  # noqa: E402
from make_figures_main import ALL_TASKS, CORPUS, SIX_TASKS, TABLES, TASK_LABELS, effective_units  # noqa: E402

# ---------------------------------------------------------------- constants
PT_MM = 25.4 / 72.0
ROW1_DROP = 6.6
W, H = 165.0, 137.0 - ROW1_DROP
YMIN = -10.0
MIN_SIZE = 5.5
SIZE_SMALL, SIZE_LABEL, SIZE_TITLE, SIZE_LETTER = 6.0, 6.5, 8.5, 11.0
LINE_FACTOR = 1.2

GREEN, BLUE, GREY, ORANGE, DATA = "#2a9d5c", "#1f77b4", "#6b6b6b", "#d98e04", "#5f8fc4"
ACTIVE, INACTIVE, UNKNOWN = "#2a9d5c", "#c0392b", "#bdbdbd"
OUTLINE, INK, RULE = "#4a4a4a", "#1a1a1a", "#cfcfcf"
FONT_CANDIDATES = ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]

EVIDENCE = (
    ("Positive-only annotation", "#65A3CE"),
    ("Measurement", "#286AA5"),
    ("Curated label", "#2D925D"),
    ("Measured over annotation", "#AF80B8"),
)
EVIDENCE_COLUMNS = {
    "active": ("active_annotation", "active_measurement", "active_explicit", None),
    "inactive": (None, "inactive_measurement", "inactive_explicit", "inactive_over_annotation"),
}
OFFSETS = {"active": 0.23, "inactive": -0.23}
PEPTIDE_COLOR, COMPONENT_COLOR, EFFECTIVE_COLOR = "#8A8A8A", "#65A3CE", "#286AA5"

COL_X = (2.0, 49.0, 100.0)
COL_WIDTHS = (42.0, 46.0, 63.0)
HEAD_Y = 133.0
RESULT_Y = 89.8
SECTION_RULE_Y = 80.0
DATA_HEADING_Y = 76.0
ROW_BOTTOM, ROW_HEIGHT = 38.0, 33.5
STRIP_RULE_Y, STRIP_HEADING_Y = 22.5, 19.0

AMINO_GROUP = {}
for _letters, _color in (("AVILMFWP", "#f4c06a"), ("KRH", "#7fb0e3"), ("DE", "#e88a8a"), ("STNQGCY", "#9bd29b")):
    for _letter in _letters:
        AMINO_GROUP[_letter] = _color


def tint(color, strength):
    return tuple(1.0 - (1.0 - value) * strength for value in to_rgb(color))


def resolve_font():
    for name in FONT_CANDIDATES:
        try:
            findfont(FontProperties(family=name), fallback_to_default=False)
            return name
        except ValueError:
            continue

    return "DejaVu Sans"


FONT_FAMILY = resolve_font()
plt.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "hatch.linewidth": 0.3,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
    }
)

FIG = plt.figure(figsize=(W / 25.4, (H - YMIN) / 25.4), facecolor="white")
AX = FIG.add_axes([0, 0, 1, 1], zorder=1)
AX.set_xlim(0, W)
AX.set_ylim(YMIN, H)
AX.set_aspect("equal")
AX.axis("off")
Z = itertools.count(1)
LABELS = []

SHIFT = [0.0]


@contextmanager
def row_shift(dy):
    SHIFT[0] = dy

    try:
        yield
    finally:
        SHIFT[0] = 0.0
# ---------------------------------------------------------------- text
@lru_cache(maxsize=None)
def text_width(text, size, weight="normal", style="normal"):
    properties = FontProperties(family=FONT_FAMILY, weight=weight, style=style)
    extent = TextPath((0, 0), text, size=size, prop=properties).get_extents()

    return (extent.width + 0.06 * size) * PT_MM


def line_height(size):
    return size * PT_MM * LINE_FACTOR


def wrap_text(text, size, width, weight="normal", style="normal"):
    for characters in range(len(text), 0, -1):
        lines = textwrap.wrap(text, width=characters, break_long_words=False, break_on_hyphens=False)

        if all(text_width(line, size, weight, style) <= width for line in lines):
            return lines

    raise AssertionError(f"text does not fit the width {width:.1f} mm: {text!r}")


def put_text(x, y, text, size=SIZE_LABEL, ha="center", weight="normal", style="normal", color=INK):
    assert size >= MIN_SIZE
    LABELS.append((x, y + SHIFT[0], text, size, ha, weight, style, color))


def put_lines(x, y_top, text, size, width, ha="center", weight="normal", style="normal", color=INK):
    y = y_top

    for line in wrap_text(text, size, width, weight, style):
        put_text(x, y - line_height(size) / 2, line, size, ha, weight, style, color)
        y -= line_height(size)

    return y


def put_centered(x, y_center, text, size, width, weight="normal", style="normal", color=INK):
    count = len(wrap_text(text, size, width, weight, style))

    return put_lines(x, y_center + count * line_height(size) / 2, text, size, width, weight=weight, style=style, color=color)


def draw_heading(x, y, letter, text):
    put_text(x, y, letter, SIZE_LETTER, ha="left", weight="bold")
    put_text(x + text_width(letter, SIZE_LETTER, "bold") + 1.8, y, text, SIZE_TITLE, ha="left", weight="bold")


def flush_labels():
    for x, y, text, size, ha, weight, style, color in LABELS:
        AX.text(x, y, text, fontsize=size, ha=ha, va="center", fontweight=weight, fontstyle=style, color=color, zorder=next(Z), clip_on=False)


# ---------------------------------------------------------------- drawing tools
def add(artist):
    artist.set_zorder(next(Z))

    if SHIFT[0]:
        artist.set_transform(Affine2D().translate(0.0, SHIFT[0]) + AX.transData)

    if isinstance(artist, Line2D):
        AX.add_line(artist)
    else:
        AX.add_patch(artist)

    return artist


def draw_box(rect, fc, ec=OUTLINE, lw=0.6, rounding=0.8, hatch=None):
    x, y, width, height = rect
    patch = FancyBboxPatch((x, y), width, height, boxstyle=f"round,pad=0,rounding_size={rounding:g}", fc=fc, ec=ec, lw=lw)

    if hatch:
        patch.set_hatch(hatch)

    return add(patch)


def draw_stacked_cards(rect, fc, ec=OUTLINE, lw=0.6):
    x, y, width, height = rect

    for offset in (2, 1, 0):
        draw_box((x + offset, y + offset, width, height), fc, ec, lw)


def draw_database(rect, fc, ec=OUTLINE, lw=0.6):
    x, y, width, height = rect
    cap = min(1.8, 0.4 * height)
    add(Ellipse((x + width / 2, y + cap / 2), width, cap, fc=fc, ec=ec, lw=lw))
    add(Rectangle((x, y + cap / 2), width, height - cap, fc=fc, ec="none"))

    for edge in (x, x + width):
        add(Line2D([edge, edge], [y + cap / 2, y + height - cap / 2], color=ec, lw=lw, solid_capstyle="butt"))

    add(Ellipse((x + width / 2, y + height - cap / 2), width, cap, fc=fc, ec=ec, lw=lw))


def draw_stacked_databases(rect, fc, ec=OUTLINE, lw=0.6):
    x, y, width, height = rect

    for offset in (2, 1, 0):
        draw_database((x + offset, y + offset, width, height), fc, ec, lw)


def draw_line(points, color="black", lw=0.9):
    xs, ys = zip(*points)

    return add(Line2D(xs, ys, color=color, lw=lw, solid_capstyle="projecting", solid_joinstyle="miter"))


def draw_arrow(start, end, lw=0.8, head_length=1.5):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    head = min(head_length, 0.55 * length)
    half = 0.4 * head
    base = (end[0] - ux * head, end[1] - uy * head)
    add(Line2D([start[0], base[0] + ux * 0.3], [start[1], base[1] + uy * 0.3], color="black", lw=lw, solid_capstyle="butt"))
    add(Polygon([end, (base[0] - uy * half, base[1] + ux * half), (base[0] + uy * half, base[1] - ux * half)], closed=True, fc="black", ec="none"))


def draw_rule(x0, x1, y):
    draw_line([(x0, y), (x1, y)], RULE, 0.5)


def draw_chip(rect, fc, ec=OUTLINE, lw=0.5, hatch=None):
    return draw_box(rect, fc, ec, lw, rounding=min(rect[2], rect[3]) * 0.3, hatch=hatch)


def draw_amino_row(x0, y, letters, radius, pitch):
    for index, letter in enumerate(letters):
        x = x0 + index * pitch
        add(Circle((x, y), radius, fc=AMINO_GROUP[letter], ec=OUTLINE, lw=0.4))
        put_text(x, y, letter, SIZE_SMALL, weight="bold")


def draw_state_chips(x, y):
    width, height, gap = 5.2, 4.0, 1.2

    for index, (color, glyph) in enumerate(((ACTIVE, "check"), (INACTIVE, "cross"), (UNKNOWN, "dash"))):
        left = x + index * (width + gap)
        draw_chip((left, y, width, height), color)
        mx, my = left + width / 2, y + height / 2

        if glyph == "check":
            draw_line([(mx - 1.1, my), (mx - 0.3, my - 0.9), (mx + 1.2, my + 1.0)], "white", 0.9)
        elif glyph == "cross":
            draw_line([(mx - 1.0, my - 1.0), (mx + 1.0, my + 1.0)], "white", 0.9)
            draw_line([(mx - 1.0, my + 1.0), (mx + 1.0, my - 1.0)], "white", 0.9)
        else:
            draw_line([(mx - 1.1, my), (mx + 1.1, my)], "#777777", 0.9)


def draw_network(x_positions, yc, counts, pitches):
    layers = [[yc + pitch * (index - (count - 1) / 2) for index in range(count)] for count, pitch in zip(counts, pitches)]

    for column in range(len(layers) - 1):
        for ya in layers[column]:
            for yb in layers[column + 1]:
                draw_line([(x_positions[column], ya), (x_positions[column + 1], yb)], "#a9b4c0", 0.12)

    for column, ys in enumerate(layers):
        color = GREEN if column == len(layers) - 1 else BLUE

        for y in ys:
            add(Circle((x_positions[column], y), 0.62, fc=tint(color, 0.35), ec=color, lw=0.4))


def draw_wheel(cx, cy, radius):
    add(Circle((cx, cy), radius, fc="white", ec="none"))
    add(Wedge((cx, cy), radius, 30, 150, fc=tint(ORANGE, 0.45), ec="none"))
    add(Circle((cx, cy), radius, fc="none", ec=OUTLINE, lw=0.5))
    order = {0: "1", 2: "2", 12: "3", 4: "4"}

    for index in range(14):
        degrees = (90 + index * 360 / 14) % 360
        angle = math.radians(degrees)
        inside = 30 <= degrees <= 150
        add(Circle((cx + radius * math.cos(angle), cy + radius * math.sin(angle)), 0.6, fc=tint(ORANGE, 0.8) if inside else "white", ec=OUTLINE, lw=0.4))

        if index in order:
            put_text(cx + (radius + 2.2) * math.cos(angle), cy + (radius + 2.2) * math.sin(angle), order[index], SIZE_SMALL)


def draw_result(x0, width, text):
    assert text_width(text, SIZE_LABEL, "bold") <= width, text
    put_text(x0, RESULT_Y, text, SIZE_LABEL, ha="left", weight="bold")


def draw_key(x, y, kind, color, label):
    if kind == "square":
        add(Rectangle((x, y - 1.0), 2.0, 2.0, fc=color, ec="none"))
    elif kind == "circle":
        add(Circle((x + 1.0, y), 1.0, fc=color, ec="none"))
    else:
        add(Polygon([(x + 1.0, y + 1.4), (x + 2.4, y), (x + 1.0, y - 1.4), (x - 0.4, y)], closed=True, fc=color, ec="none"))

    put_text(x + 3.4, y, label, SIZE_SMALL, ha="left")

def draw_component(cx, cy, offsets, color, threshold=4.6):
    points = [(cx + dx, cy + dy) for dx, dy in offsets]

    for first, second in itertools.combinations(points, 2):
        if math.dist(first, second) <= threshold:
            draw_line([first, second], OUTLINE, 0.35)

    for x, y in points:
        add(Circle((x, y), 0.8, fc=tint(color, 0.55), ec=color, lw=0.5))


def draw_label_chip(rect, state, lw=0.4):
    if state == "A":
        draw_chip(rect, ACTIVE, lw=lw)
    elif state == "I":
        draw_chip(rect, INACTIVE, lw=lw)
    else:
        draw_chip(rect, "white", ec=UNKNOWN, lw=lw, hatch="////")


def draw_layers(x_positions, yc, counts, pitches, output_states):
    layers = [[yc + pitch * (index - (count - 1) / 2) for index in range(count)] for count, pitch in zip(counts, pitches)]

    for column in range(len(layers) - 1):
        for ya in layers[column]:
            for yb in layers[column + 1]:
                draw_line([(x_positions[column], ya), (x_positions[column + 1], yb)], "#a9b4c0", 0.12)

    for column, ys in enumerate(layers[:-1]):
        for y in ys:
            add(Circle((x_positions[column], y), 0.6, fc=tint(BLUE, 0.35), ec=BLUE, lw=0.4))

    for y, state in zip(layers[-1], output_states):
        draw_label_chip((x_positions[-1] - 1.3, y - 0.375, 2.6, 0.75), state, lw=0.3)
        
# ---------------------------------------------------------------- columns A to C
def draw_column_a():
    x0 = COL_X[0]
    draw_heading(x0, HEAD_Y, "A", "Corpus and folds")

    draw_component(x0 + 9.0, 123.5, [(-3.5, 2.5), (0.0, 3.5), (3.2, 2.2), (-2.2, -0.6), (1.2, -0.8), (4.2, -1.8), (-4.5, -3.0), (0.5, -4.2)], BLUE)
    draw_component(x0 + 23.0, 125.5, [(-2.5, 1.5), (1.0, 2.0), (2.8, -0.8), (-1.0, -1.8)], BLUE)
    draw_component(x0 + 35.0, 122.5, [(-2.0, 1.5), (1.5, 2.0), (0.5, -1.8)], GREEN)

    for dx, dy in ((16.0, 116.5), (27.0, 118.0), (40.0, 128.0), (31.0, 129.5), (39.5, 117.0)):
        add(Circle((x0 + dx, dy), 0.8, fc=tint(BLUE, 0.55), ec=BLUE, lw=0.5))

    put_text(x0 + 21.0, 112.4, r"peptides $\times$ 10 tasks", SIZE_SMALL)

    for row, pattern in enumerate(("AAIUAUUUUU", "UIAUAUUUIU", "IUUUAIUUUU", "AUAUUUUAUU")):
        for column, state in enumerate(pattern):
            draw_label_chip((x0 + 3.3 + column * 3.6, 108.05 - row * 2.5, 3.0, 2.0), state, lw=0.3)

    put_text(x0 + 21.0, 98.75, "ten folds, components stay whole", SIZE_SMALL)

    for index in range(10):
        rect = (x0 + 4.8 + index * 3.3, 94.85, 2.6, 2.6)

        if index == 0:
            draw_box(rect, tint(GREEN, 0.5), GREEN, lw=0.8, rounding=0.3)
        else:
            draw_box(rect, tint(GREY, 0.25), GREY, lw=0.4, rounding=0.3)

    draw_result(x0, COL_WIDTHS[0], "Train-test similarity below 0.7")


def draw_column_b():
    x0 = COL_X[1]
    draw_heading(x0, HEAD_Y, "B", "Predictor")

    draw_amino_row(x0 + 2.0, 128.5, ["K", "W", "L", "K", "R"], 1.4, 3.0)
    put_text(x0 + 17.0, 128.5, "1,861-d features", SIZE_LABEL, ha="left", weight="bold")
    draw_arrow((x0 + 7.0, 127.0), (x0 + 7.0, 125.0))

    left = x0 + 1.0

    for width, label, color in (
        (21.0, "ESM C", tint(BLUE, 0.4)),
        (11.0, "ESM-2", tint(BLUE, 0.2)),
        (2.0, "", tint(ORANGE, 0.6)),
        (11.0, "dipeptide", tint(GREEN, 0.3)),
    ):
        draw_box((left, 121.0, width, 3.8), color, OUTLINE, lw=0.4, rounding=0.3)

        if label:
            put_text(left + width / 2, 122.9, label, MIN_SIZE)

        left += width

    put_text(x0 + 23.0, 119.4, "960 + 480 + 21 + 400 = 1,861", SIZE_SMALL)
    draw_arrow((x0 + 6.0, 121.0), (x0 + 6.0, 116.0))

    xs = (x0 + 6.0, x0 + 16.0, x0 + 26.0, x0 + 36.0)
    draw_layers(xs, 110.5, (6, 5, 4, 10), (1.7, 2.0, 2.2, 0.9), "AIAUUAIUAU")

    for x, label in zip(xs, ("1,861", "512", "256", "10")):
        put_text(x, 103.6, label, SIZE_SMALL)

    draw_chip((x0 + 1.0, 95.0, 10.0, 3.6), tint(BLUE, 0.22), BLUE)
    put_text(x0 + 6.0, 96.8, "MLP", MIN_SIZE)
    put_text(x0 + 12.8, 96.8, "+", SIZE_LABEL)
    draw_chip((x0 + 14.5, 95.0, 15.0, 3.6), tint(BLUE, 0.22), BLUE)
    put_text(x0 + 22.0, 96.8, "ESM C trees", MIN_SIZE)
    draw_arrow((x0 + 30.5, 96.8), (x0 + 33.0, 96.8))
    draw_box((x0 + 33.5, 95.0, 12.0, 3.6), tint(GREEN, 0.3), GREEN)
    put_text(x0 + 39.5, 96.8, "Ensemble", MIN_SIZE, weight="bold")

    draw_result(x0, COL_WIDTHS[1], "AUROC 0.754 vs 0.732 for ESM C")


def draw_column_c():
    x0 = COL_X[2]
    draw_heading(x0, HEAD_Y, "C", "Generator")
    fill = tint(GREEN, 0.15)

    for y, text in ((126.4, "task labels"), (122.0, "properties"), (117.6, "wheel class")):
        draw_box((x0, y, 16.0, 3.6), fill, GREEN)
        put_text(x0 + 8.0, y + 1.8, text, SIZE_SMALL)
        draw_line([(x0 + 16.0, y + 1.8), (x0 + 17.3, y + 1.8)])

    draw_line([(x0 + 17.3, 128.2), (x0 + 17.3, 119.4)])
    draw_arrow((x0 + 17.3, 123.8), (x0 + 19.8, 123.8))
    put_text(x0 + 8.0, 114.6, "26 axes", SIZE_SMALL)

    draw_box((x0 + 20.0, 120.3, 8.0, 7.0), tint(BLUE, 0.22), BLUE)
    put_text(x0 + 24.0, 123.8, "MLP", SIZE_SMALL)
    draw_arrow((x0 + 28.5, 123.8), (x0 + 32.7, 123.8))
    put_text(x0 + 30.6, 125.6, r"$u$", SIZE_SMALL)

    draw_stacked_cards((x0 + 33.0, 113.2, 17.0, 14.3), tint(GREEN, 0.3), GREEN)
    draw_box((x0 + 34.5, 122.05, 14.0, 3.6), "white", GREEN, lw=0.4, rounding=0.4)
    put_text(x0 + 41.5, 123.85, "Attention", MIN_SIZE)
    draw_box((x0 + 34.5, 115.05, 14.0, 3.6), "white", GREEN, lw=0.4, rounding=0.4)
    put_text(x0 + 41.5, 116.85, "Feed-forward", MIN_SIZE)
    draw_arrow((x0 + 41.5, 121.75), (x0 + 41.5, 118.95))
    put_text(x0 + 42.5, 131.0, "12 AdaLN blocks", SIZE_SMALL)
    draw_arrow((x0 + 41.5, 112.9), (x0 + 41.5, 109.9))

    sequence = "KALLKIR"

    for row, (revealed, label) in enumerate((({3}, r"$s=1$"), ({3, 2, 5, 1}, r"$s=5$"), (set(range(7)), r"$s=L$"))):
        y = 106.9 - row * 3.3

        for position, letter in enumerate(sequence):
            rect = (x0 + 21.0 + position * 3.1, y, 2.8, 2.6)

            if position in revealed:
                draw_box(rect, AMINO_GROUP[letter], OUTLINE, lw=0.4, rounding=0.4)
                put_text(rect[0] + 1.4, y + 1.3, letter, SIZE_SMALL, weight="bold")
            else:
                draw_chip(rect, "white", ec=UNKNOWN, lw=0.5, hatch="////")

        put_text(x0 + 43.9, y + 1.3, label, SIZE_SMALL, ha="left")

    draw_wheel(x0 + 8.5, 105.2, 3.4)
    put_text(x0 + 9.0, 98.8, "22 wheel classes", SIZE_SMALL)

    put_text(x0 + 36.0, 95.0, r"$\ell_\gamma=\ell_u+\gamma(\ell_c-\ell_u),\ \gamma=2$", SIZE_SMALL)

    draw_result(x0, COL_WIDTHS[2], "Charge and GRAVY within 0.1 SD; under 0.5% copies")


# ---------------------------------------------------------------- row D to F
def data_axes(left, width, **kwargs):
    total = H - YMIN
    ax = FIG.add_axes([left / W, (ROW_BOTTOM - YMIN) / total, width / W, ROW_HEIGHT / total], zorder=2, **kwargs)
    ax.patch.set_alpha(0.0)
    ax.tick_params(labelsize=SIZE_SMALL, length=2.0, width=0.5, pad=1.5)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    return ax


def draw_bands(ax, positions):
    for index, position in enumerate(positions):
        if index % 2 == 0:
            ax.axhspan(position - 0.5, position + 0.5, color="#F5F5F5", zorder=0, lw=0)


def draw_data_row():
    draw_rule(2.0, 163.0, SECTION_RULE_Y)
    positions = np.arange(len(ALL_TASKS))[::-1].astype(float)
    divider = positions[len(SIX_TASKS) - 1] - 0.5
    table = pd.read_csv(TABLES / "corpus_tasks.csv").set_index("task")
    summary = __import__("json").loads((ROOT / "reference" / "corpus_summary.json").read_text())

    ax_d = data_axes(26.0, 36.0)
    ax_e = data_axes(74.0, 31.0, sharey=ax_d)
    ax_f = data_axes(126.0, 34.0)

    # D: evidence behind the labels
    draw_bands(ax_d, positions)

    for position, task in zip(positions, ALL_TASKS):
        for state in ("active", "inactive"):
            total = float(table.loc[task, state])
            left = 0.0

            for column, (_, color) in zip(EVIDENCE_COLUMNS[state], EVIDENCE):
                if column is None:
                    continue

                share = float(table.loc[task, column]) / total
                ax_d.barh(position + OFFSETS[state], share, left=left, height=0.42, color=color, edgecolor="white", linewidth=0.25)
                left += share

            ax_d.text(
                1.03, position + OFFSETS[state], f"{int(total):,}", va="center", ha="left", fontsize=MIN_SIZE,
                transform=ax_d.get_yaxis_transform(),
            )

    ax_d.set_xlim(0, 1)
    ax_d.set_xticks([0.0, 0.5, 1.0])
    ax_d.set_yticks(positions)
    ax_d.set_yticklabels([TASK_LABELS[task] for task in ALL_TASKS], fontsize=SIZE_SMALL)
    ax_d.set_ylim(-0.6, len(ALL_TASKS) - 0.4)
    ax_d.axhline(divider, color=GREY, lw=0.5, zorder=1)
    ax_d.set_xlabel("Share of labels", fontsize=SIZE_SMALL, labelpad=1.5)

    # E: independent units
    draw_bands(ax_e, positions)

    for position, task in zip(positions, ALL_TASKS):
        pass
        pass

        for state, code in (("active", 1), ("inactive", 0)):
            peptides, components, effective = summary["effective"][task][state]
            row = position + OFFSETS[state]
            ax_e.hlines(row, min(peptides, components, effective), max(peptides, components, effective), color="#E4E4E4", lw=0.7, zorder=1)
            ax_e.scatter(peptides, row, marker="s", s=7, color=PEPTIDE_COLOR, zorder=3, linewidths=0)
            ax_e.scatter(components, row, marker="o", s=7, color=COMPONENT_COLOR, zorder=3, linewidths=0)
            ax_e.scatter(effective, row, marker="D", s=9, color=EFFECTIVE_COLOR, zorder=4, linewidths=0)
            ax_e.text(
                1.03, row, f"{effective:.0f}" if effective >= 100 else f"{effective:.1f}", va="center", ha="left",
                fontsize=MIN_SIZE, color=EFFECTIVE_COLOR, transform=ax_e.get_yaxis_transform(),
            )

    ax_e.set_xscale("log")
    ax_e.set_xlim(1.5, 3e4)
    ax_e.xaxis.set_minor_formatter(NullFormatter())
    ax_e.tick_params(axis="y", labelleft=False, length=0)
    ax_e.axhline(divider, color=GREY, lw=0.5, zorder=1)
    ax_e.set_xlabel("Count", fontsize=SIZE_SMALL, labelpad=1.5)
    ax_e.text(1.03, 1.0, "$N_{\\mathrm{eff}}$", va="bottom", ha="left", fontsize=SIZE_SMALL, color=EFFECTIVE_COLOR, transform=ax_e.transAxes)

    for ax in (ax_d, ax_e):
        ax.grid(axis="x", color="#E4E4E4", lw=0.4, zorder=0)
        ax.set_axisbelow(True)

    # F: components
    sizes = np.array(summary["component_sizes"])
    ax_f.loglog(np.arange(1, len(sizes) + 1), sizes, color=EFFECTIVE_COLOR, lw=0.9)
    ax_f.xaxis.set_minor_formatter(NullFormatter())
    ax_f.yaxis.set_minor_formatter(NullFormatter())
    ax_f.set_xlabel("Component rank", fontsize=SIZE_SMALL, labelpad=1.5)
    ax_f.set_ylabel("Peptides in the component", fontsize=SIZE_SMALL, labelpad=1.5)
    ax_f.text(
        0.97, 0.96, f"{len(sizes):,} components\n{int((sizes == 1).sum()):,} singletons\nlargest {sizes[0]:,} ({sizes[0] / sizes.sum():.1%})",
        transform=ax_f.transAxes, ha="right", va="top", fontsize=MIN_SIZE,
    )

    draw_heading(2.0, DATA_HEADING_Y, "D", "Label evidence")
    draw_heading(72.5, DATA_HEADING_Y, "E", "Effective units")
    draw_heading(117.5, DATA_HEADING_Y, "F", "Similarity components")

    for index, (label, color) in enumerate(EVIDENCE):
        draw_key(3.0 + 31.0 * (index % 2), 29.0 - 3.0 * (index // 2), "square", color, label)

    draw_key(74.5, 29.0, "square", PEPTIDE_COLOR, "Peptides")
    draw_key(90.0, 29.0, "circle", COMPONENT_COLOR, "Components")
    draw_key(74.5, 26.0, "diamond", EFFECTIVE_COLOR, "Effective units; upper marker active, lower inactive")


# ---------------------------------------------------------------- strip G
def draw_chain(x, y, letters, radius=0.62, pitch=1.35):
    for index, letter in enumerate(letters):
        add(Circle((x + index * pitch, y), radius, fc=AMINO_GROUP[letter], ec=OUTLINE, lw=0.25))


def draw_star(cx, cy, outer, color):
    points = []

    for index in range(10):
        radius = outer if index % 2 == 0 else outer * 0.42
        angle = math.radians(90 + index * 36)
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))

    add(Polygon(points, closed=True, fc=color, ec="none"))


def draw_bottle(cx, base):
    add(Polygon([(cx - 1.3, base + 6.3), (cx + 1.3, base + 6.3), (cx + 0.8, base + 8.2), (cx - 0.8, base + 8.2)],
                closed=True, fc=tint(ORANGE, 0.25), ec=ORANGE, lw=0.5))
    draw_box((cx - 2.7, base + 5.4, 5.4, 0.9), tint(BLUE, 0.45), BLUE, lw=0.5, rounding=0.3)
    draw_box((cx - 2.3, base, 4.6, 5.4), "white", BLUE, lw=0.6, rounding=0.8)
    draw_box((cx - 2.3, base, 4.6, 3.2), tint(BLUE, 0.16), "none", lw=0.0, rounding=0.8)

    for step in range(3):
        draw_line([(cx + 0.8, base + 1.2 + 1.3 * step), (cx + 1.9, base + 1.2 + 1.3 * step)], BLUE, 0.35)

def draw_wave(x0, y0, x1, y1, amplitude, cycles, color, lw):
    count = 24
    length = math.hypot(x1 - x0, y1 - y0)
    nx, ny = -(y1 - y0) / length, (x1 - x0) / length
    points = []

    for index in range(count + 1):
        t = index / count
        offset = amplitude * math.sin(2.0 * math.pi * cycles * t)
        points.append((x0 + (x1 - x0) * t + nx * offset, y0 + (y1 - y0) * t + ny * offset))

    return draw_line(points, color, lw)


def draw_tube(points, width, fc, ec, lw=0.6):
    xs, ys = zip(*points)
    add(Line2D(xs, ys, color=ec, lw=width / PT_MM + 2.0 * lw, solid_capstyle="butt", solid_joinstyle="round"))
    add(Line2D(xs, ys, color=fc, lw=width / PT_MM, solid_capstyle="butt", solid_joinstyle="round"))


def draw_stomach(sx, sy, scale):
    wall = "#b85c54"
    outer = tint("#e8a5a0", 0.6)
    inner = tint("#e8a5a0", 0.25)
    body = [
        (3.5, 8.6),
        (3.6, 6.6), (5.2, 5.6), (5.0, 3.9),
        (4.9, 3.2), (3.6, 3.0), (2.9, 3.0),
        (2.9, 1.6),
        (4.0, 0.4), (7.5, 0.1), (10.0, 1.0),
        (12.4, 1.9), (13.9, 4.6), (13.0, 7.2),
        (12.3, 9.8), (9.5, 11.2), (7.2, 10.2),
        (6.2, 9.8), (5.4, 9.4), (4.7, 8.6),
    ]
    codes = [MplPath.MOVETO] + [MplPath.CURVE4] * 6 + [MplPath.LINETO] + [MplPath.CURVE4] * 12

    def place(points, shrink=1.0):
        return [(sx + scale * (8.0 + shrink * (px - 8.0)), sy + scale * (5.8 + shrink * (py - 5.8))) for px, py in points]

    outline = place(body)
    add(PathPatch(MplPath(outline + [outline[0]], codes + [MplPath.CLOSEPOLY]), fc=outer, ec="none"))

    add(Rectangle((sx + scale * 3.5, sy + scale * 8.4), scale * 1.2, scale * 3.0, fc=outer, ec="none"))

    for px in (3.5, 4.7):
        draw_line([(sx + scale * px, sy + scale * 8.6), (sx + scale * px, sy + scale * 11.4)], wall, 0.7)

    add(PathPatch(MplPath(outline, codes), fc="none", ec=wall, lw=0.8, capstyle="butt", joinstyle="round"))

    lumen = place(body, 0.78)
    add(PathPatch(MplPath(lumen + [lumen[0]], codes + [MplPath.CLOSEPOLY]), fc=inner, ec=tint(wall, 0.45), lw=0.3))

    for (xa, ya), (xb, yb) in (((6.8, 2.5), (11.0, 3.2)), ((7.4, 3.6), (11.4, 4.3))):
        draw_wave(sx + scale * xa, sy + scale * ya, sx + scale * xb, sy + scale * yb, 0.2, 2.0, tint(wall, 0.55), 0.35)

    for (px, py), letters in (((6.2, 7.5), "LKF"), ((9.0, 7.9), "RAK"), ((6.6, 4.9), "KLA"), ((9.4, 5.8), "AWR")):
        draw_chain(sx + scale * px, sy + scale * py, letters, 0.4 * scale, 0.9 * scale)

    duodenum = [(sx + scale * px, sy + scale * py) for px, py in ((3.1, 2.3), (1.8, 2.3), (0.8, 3.3), (0.8, 5.2))]
    draw_tube(duodenum, 1.4 * scale, outer, wall)
    draw_line([(sx + scale * 2.9, sy + scale * 1.6), (sx + scale * 2.9, sy + scale * 3.0)], wall, 1.2)


def draw_funnel(cx):
    half, rim_y, rim_ry = 8.0, 11.9, 1.1
    stem_half, stem_top, stem_bottom = 1.1, 6.4, 4.8
    slope = (half - stem_half) / (rim_y - stem_top)

    add(Polygon(
        [(cx - half, rim_y), (cx + half, rim_y), (cx + stem_half, stem_top), (cx + stem_half, stem_bottom),
         (cx - stem_half, stem_bottom), (cx - stem_half, stem_top)],
        closed=True, fc=tint(BLUE, 0.2), ec="none",
    ))

    for sign in (-1, 1):
        draw_line([(cx + sign * half, rim_y), (cx + sign * stem_half, stem_top), (cx + sign * stem_half, stem_bottom)], BLUE, 0.7)

    draw_line([(cx - 6.4, 11.35), (cx - 3.65, 9.15)], "white", 0.7)

    add(Ellipse((cx, rim_y), 2.0 * half, 2.0 * rim_ry, fc=tint(BLUE, 0.4), ec=BLUE, lw=0.7))
    add(Ellipse((cx, rim_y - 0.1), 2.0 * (half - 0.9), 2.0 * (rim_ry - 0.45), fc=tint(BLUE, 0.6), ec="none"))

    for index, letter in enumerate("LKAFRIW"):
        y = rim_y + (0.15 if index % 2 else -0.15)
        add(Circle((cx - 5.4 + index * 1.8, y), 0.55, fc=AMINO_GROUP[letter], ec=OUTLINE, lw=0.25))

    for dx, dy, letter in ((-3.2, 13.6, "K"), (0.4, 13.8, "D"), (3.4, 13.5, "S")):
        add(Circle((cx + dx, dy), 0.55, fc=AMINO_GROUP[letter], ec=OUTLINE, lw=0.25))

    mesh_y = 8.6
    mesh_half = stem_half + (mesh_y - stem_top) * slope
    add(Line2D([cx - mesh_half, cx + mesh_half], [mesh_y, mesh_y], color=BLUE, lw=0.8, linestyle=(0, (2.0, 1.2))))

    for dx, letter in ((-2.4, "A"), (0.0, "F"), (2.4, "R")):
        add(Circle((cx + dx, 9.5), 0.55, fc=AMINO_GROUP[letter], ec=OUTLINE, lw=0.25))

    for y, letter in ((7.4, "L"), (3.9, "K")):
        add(Circle((cx, y), 0.6, fc=AMINO_GROUP[letter], ec=GREEN, lw=0.9))


def draw_model(cx):
    draw_network([cx - 15.2, cx - 11.0, cx - 6.8, cx - 2.6], 9.0, (4, 6, 5, 3), (2.4, 1.8, 2.0, 3.0))
    draw_arrow((cx - 1.4, 9.0), (cx + 2.6, 9.0))

    ox, oy, width, height = cx + 3.6, 4.6, 11.8, 9.2
    curve = [(ox + width * step / 24, oy + height * (1.0 - (1.0 - step / 24) ** 3.5)) for step in range(25)]

    add(Polygon(curve + [(ox + width, oy)], closed=True, fc=tint(BLUE, 0.2), ec="none"))
    add(Line2D([ox, ox + width], [oy, oy + height], color=GREY, lw=0.4, linestyle=(0, (3.0, 2.0))))
    draw_line(curve, BLUE, 1.0)
    draw_line([(ox, oy + height), (ox, oy), (ox + width, oy)], OUTLINE, 0.6)

    px, py = ox + width * 0.22, oy + height * (1.0 - 0.78 ** 3.5)
    add(Line2D([px, px], [oy, py], color=ORANGE, lw=0.7, linestyle=(0, (2.0, 1.5))))
    add(Circle((px, py), 0.75, fc=ORANGE, ec="white", lw=0.4))


def draw_ranking(cx):
    for index, length in enumerate((15.0, 12.0, 9.0, 6.0)):
        green = index == 0
        draw_box((cx - 7.0, 12.0 - index * 2.6, length, 1.8), tint(GREEN if green else BLUE, 0.6 if green else 0.4),
                 GREEN if green else BLUE, lw=0.5 if green else 0.4, rounding=0.4)

    draw_star(cx + 10.6, 12.9, 1.7, ORANGE)


def draw_strip():
    draw_rule(2.0, 163.0, STRIP_RULE_Y)
    draw_heading(2.0, STRIP_HEADING_Y, "G", "Application to human milk peptides")

    centers = (24.0, 63.0, 102.0, 141.0)
    colors = ("#4a4a4a", "#286AA5", "#286AA5", ACTIVE)
    numbers = ("11,612", "6,075", "5%", "32")
    captions = (
        "peptides from the stomachs of preterm infants",
        "novel: similarity below 0.7, 8 to 50 residues",
        "false-positive threshold, every activity scored",
        "antibacterial; four other activities exploratory",
    )

    for cx, color, number, caption in zip(centers, colors, numbers, captions):
        draw_box((cx - 17.5, -8.4, 35.0, 24.0), "white", RULE, lw=0.5, rounding=1.2)
        put_text(cx, 0.2, number, 11.0, weight="bold", color=color)
        put_lines(cx, -2.3, caption, SIZE_SMALL, 33.0)

    cx = centers[0]
    draw_bottle(cx - 13.2, 4.9)
    draw_chain(cx - 7.4, 11.8, "LKF")
    draw_chain(cx - 5.8, 9.0, "KLR")
    draw_chain(cx - 7.0, 6.2, "AKW")
    draw_arrow((cx - 3.0, 13.0), (cx + 4.8, 13.0))
    draw_stomach(cx + 2.4, 3.6, 0.95)

    draw_funnel(centers[1])
    draw_model(centers[2])
    draw_ranking(centers[3])

    for cx_left in centers[:3]:
        draw_line([(cx_left + 18.7, 6.6), (cx_left + 20.3, 3.6), (cx_left + 18.7, 0.6)], ACTIVE, 1.4)



# ---------------------------------------------------------------- main
def main():
    with row_shift(-ROW1_DROP):
        draw_column_a()
        draw_column_b()
        draw_column_c()

        draw_arrow((COL_X[0] + COL_WIDTHS[0] + 0.8, 122.5), (COL_X[1] - 0.8, 122.5))
        draw_arrow((COL_X[1] + COL_WIDTHS[1] + 0.8, 122.5), (COL_X[2] - 0.8, 122.5))

    draw_data_row()
    draw_strip()
    flush_labels()

    output = ROOT / "figures"
    output.mkdir(parents=True, exist_ok=True)
    FIG.savefig(output / "overview.pdf", facecolor="white")
    FIG.savefig(output / "overview.png", dpi=600, facecolor="white")
    print(f"wrote {output / 'overview.pdf'} and {output / 'overview.png'}", flush=True)


if __name__ == "__main__":
    main()
from typing import Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch
import matplotlib.colors as mcolors


def d(x, y):
    return (x**2 + y**2) / (2 * x)

def CR1t(x, y, t):
    return (np.sqrt((x - 1)**2 + y**2) + 2 * (1 - t)) / (np.sqrt((x - t)**2 + y**2) + (1 - t))

def CRd(x, y):
    if (x - 1)**2 + y**2 > 1:
        return float('inf')
    elif x**2 + y**2 < x:
        return (x + x**2 + y**2) / (x * (1 + np.sqrt(x**2 + y**2)))
    else:
        return 1 + (y**2) / (x * (np.sqrt(x) + 1)**2)

def CR0(x, y):
    return (np.sqrt(x**2 + y**2) + 1) / np.sqrt((x - 1)**2 + y**2)

def tCR1(x, y):
    if np.isclose(x,1):
        return 1 - 3*abs(y)/4
    z1 = np.sqrt((-1 + x)**2 + y**2)
    return 1 / (2 * (-1 + x)) * (-1 + x**2 + y**2 + z1 - x * z1 - np.sqrt((z1**2 * (1 + y**2 - z1 + x * (-2 + x + z1)))))

def CR1(x, y):
    if (x - 1)**2 + y**2 <= 1:
        return float('inf')
    elif tCR1(x, y) > 0:
        return CR1t(x, y, tCR1(x, y))
    else:
        return (2 + np.sqrt((-1 + x)**2 + y**2)) / (1 + np.sqrt(x**2 + y**2))

def make_plot(num_points: int,
              plot_range: Tuple[float, float, float, float],
              boxes: List[Tuple[str, Tuple[float, float, float, float]]] = [],
              curves: List[Tuple[np.ndarray, np.ndarray]] = [],
              points: List[Tuple[float, float]] = [],
              plot_title: Optional[str] = None,
              draw_legend: bool = False) -> plt.Figure:
    x = np.linspace(plot_range[0], plot_range[1], num_points)
    y = np.linspace(plot_range[2], plot_range[3], num_points)
    X, Y = np.meshgrid(x, y)

    Z0 = np.vectorize(CR0)(X, Y)
    Z1 = np.vectorize(CR1)(X, Y)
    Zd = np.vectorize(CRd)(X, Y)

    # if inside disk centered at (1, 0) with radius 1
    disk_big = (X - 1)**2 + Y**2 <= 1
    # if inside disk centered at (1/2, 0) with radius 1/2
    disk_small = (X - 1/2)**2 + Y**2 <= 1/4

    plt.figure(figsize=(6, 6))

    Z = np.zeros_like(Z0)
    Z[(Z0 <= Z1) & (Z0 <= Zd)] = 2
    Z[(Z1 <= Z0) & (Z1 <= Zd) & ~disk_big] = 3
    Z[(Zd <= Z0) & (Zd <= Z1) & disk_big] = 4

    colors = [
        (1, 1, 1, 0),
        (0.5529411764705883, 0.8274509803921568, 0.7803921568627451, 1), # turquoise
        (1.0, 0.9294117647058824, 0.43529411764705883, 1), # yellow
        (0.7450980392156863, 0.6823529411764706, 0.8313725490196079, 1), # purple
        (0.5, 0.5, 0.5, 0.5),
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list('custom1', colors, N=len(colors))
    levels = list(range(0, len(colors)+1))

    plt.contourf(X, Y, Z, levels=levels, cmap=cmap)
    # add contour line of color colors[2] but a bit darker
    plt.contour(
        X, Y, Z, levels=[2], linewidths=2,
        colors=[colors[1]],
        zorder=1,
    )

    # # Add regions tCR1 > 0 and outside of disk_big as transparent gray
    ZtCR1 = np.vectorize(tCR1)(X, Y)
    ZtCR1_nonzero = np.zeros_like(Z0)
    ZtCR1_nonzero[(ZtCR1 > 0) & ~disk_big] = 5
    plt.contourf(X, Y, ZtCR1_nonzero, levels=levels, cmap=cmap)
    plt.contour(X, Y, ZtCR1_nonzero, levels=levels, colors=[colors[4]], linewidths=1)

    # add dashed-line circle with center (1/2, 0) and radius 1/2
    plt.contour(X, Y, disk_small, levels=[0], colors='black', linestyles='dashed', linewidths=1)

    # Adding legends and labels
    if draw_legend:
        legend_labels = [
            Patch(facecolor=cmap(1), edgecolor=cmap(1), label=r'$Z_{\mathcal{A}_0}$'),
            Patch(facecolor=cmap(2), edgecolor=cmap(2), label=r'$Z_{\mathcal{A}_1}$'),
            Patch(facecolor=cmap(3), edgecolor=cmap(3), label=r'$Z_{\mathcal{A}_d}$'),
            Patch(facecolor=cmap(4), edgecolor=cmap(4), label=r'$t_1 > 0$'),
            # simple horizontal line legend (not a patch) for curve 1
            plt.Line2D([0], [0], color='black', linewidth=1, label=r'$y = \pm \frac{\sqrt{1 - 4 x + 2 x^2 + 4 x^3 - 3 x^4}}{2 x}$'),
        ]

        plt.legend(handles=legend_labels)

    # Adding other plot features
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.scatter([0, 1], [0, 0], color='black', zorder=5)

    for x, y in curves:
        plt.plot(x, y, color='black', linewidth=1)

    for x, y in points:
        plt.scatter([x], [y], color='black', zorder=5)

    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')

    # set the limits of the plot to the limits of the data
    plt.xlim(plot_range[0], plot_range[1])
    plt.ylim(plot_range[2], plot_range[3])

    # don't display the grid
    plt.grid(False)

    for box_color, box_range in boxes:
        zoom_x_min, zoom_x_max, zoom_y_min, zoom_y_max = box_range
        plt.plot([zoom_x_min, zoom_x_max], [zoom_y_min, zoom_y_min], color=box_color, linewidth=1)
        plt.plot([zoom_x_min, zoom_x_max], [zoom_y_max, zoom_y_max], color=box_color, linewidth=1)
        plt.plot([zoom_x_min, zoom_x_min], [zoom_y_min, zoom_y_max], color=box_color, linewidth=1)
        plt.plot([zoom_x_max, zoom_x_max], [zoom_y_min, zoom_y_max], color=box_color, linewidth=1)

    if plot_title is not None:
        plt.title(plot_title)

    # keep aspect ratio square but allow the plot to be resized
    plt.gca().set_aspect('equal')
    

    # remove whitespace around the image
    plt.tight_layout()

    return plt.gcf()

def main():
    num_points = 1000
    file_type = "pdf"
    full_area = (-1.5, 2.5, -2, 2)
    boxes = [
        ("black", (1/10, 1/2, 1/2+1/10, 1-1/10+1/10)),
        ("red", (0, 1/3, 0, 3/4)),
    ]

    # curve 1: Sqrt[1 - 4 x + 2 x^2 + 4 x^3 - 3 x^4]/(2 x)
    x1 = np.linspace(0.01, 1/3, num_points)
    y1 = np.sqrt(1 - 4 * x1 + 2 * x1**2 + 4 * x1**3 - 3 * x1**4) / (2 * x1)
    x1 = x1[(y1 > full_area[2]) & (y1 < full_area[3])]
    y1 = y1[(y1 > full_area[2]) & (y1 < full_area[3])]

    # curve 2: -Sqrt[1 - 4 x + 2 x^2 + 4 x^3 - 3 x^4]/(2 x)
    x2, y2 = x1, -y1

    fig = make_plot(
        num_points=num_points,
        plot_range=full_area,
        boxes=boxes,
        curves=[(x1, y1), (x2, y2)],
        draw_legend=True,
    )
    fig.savefig(f'regions.{file_type}', dpi=300)
    plt.close(fig)

    x_inflection = 1/3 * (-5 - 44 * (2/(47 + 9 * np.sqrt(93)))**(1/3) + 2 * 2**(2/3) * (47 + 9 * np.sqrt(93))**(1/3))
    y_inflection = np.sqrt(1/4 - (x_inflection - 1/2)**2)

    for i, (_, box_range) in enumerate(boxes):
        fig = make_plot(
            num_points=num_points,
            plot_range=box_range,
            curves=[(x1, y1), (x2, y2)],
            draw_legend=False,
            points=[
                (0.275257, 0.689019),
                (x_inflection, y_inflection),
            ]
        )
        fig.savefig(f'regions_box_{i}.{file_type}', dpi=300)
        plt.close(fig)



if __name__ == '__main__':
    main()

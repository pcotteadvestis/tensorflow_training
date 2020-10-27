from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.layouts import row

from transparentpath import TransparentPath as Path
import pandas as pd

files = {}
kinds = []
for dir_ in Path("models").glob("/*"):
    files[dir_.stem] = list(dir_.glob("/*.csv"))[0]
    kind = files[dir_.stem].stem.split("_")[0]
    if kind not in kinds:
        kinds.append(kind)

accuracies = {}
for key in files:
    kind = files[key].stem.split("_")[0]
    if kind not in accuracies:
        accuracies[kind] = pd.DataFrame(dtype=float, columns=["Accuracy"])
    if kind == "nodes":
        ind = int(files[key].stem.split(kind)[1].split("_")[1])
    else:
        ind = files[key].stem.split(kind)[1].split("_")[1]
    accuracies[kind].loc[ind] = files[key].read(index_col=0)["accuracy"].iloc[-1]


figures = []
for kind in accuracies:
    accuracies[kind].sort_index(inplace=True)

    outfile = output_file((Path("outputs") / kind).with_suffix(".html"))
    source = ColumnDataSource(accuracies[kind])
    p = figure()
    p.circle(x='index', y='Accuracy',
             source=source,
             size=10, color='green')
    p.title.text = kind
    p.xaxis.axis_label = kind
    p.yaxis.axis_label = "Accuracy"
    hover = HoverTool()
    hover.tooltips = [
        (kind, '@index'),
        ('Accuracy', '@Accuracy'),
    ]

    p.add_tools(hover)
    figures.append(p)

show(row(figures))

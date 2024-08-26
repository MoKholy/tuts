import plotly.express as px
import transformer_lens.utils as utils


def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        labels={"x": xaxis, "y": yaxis},
        **kwargs
    ).show(renderer=renderer)


def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x": xaxis, "y": yaxis}, **kwargs).show(
        renderer
    )


def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(
        y=y, x=x, labels={"x": xaxis, "y": yaxis, "color": caxis}, **kwargs
    ).show(renderer)

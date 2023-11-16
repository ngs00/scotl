import pandas
import numpy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mendeleev.vis.utils import create_vis_dataframe
from plotly.graph_objects import Figure
from plotly.graph_objs.layout import Shape
from plotly.graph_objs.layout import Annotation


class PeriodicTable:
    def __init__(self,
                 vals: list,
                 width: int = 1400,
                 height: int = 800,
                 cmap: str = 'RdPu',
                 include_f_block: bool = True):
        self.__vals = [round(v, 3) for v in vals]
        self.__fig = Figure()
        self.__width = width
        self.__height = height
        self.__init_figure(cmap, include_f_block)

    def __init_figure(self,
                      cmap: str,
                      include_f_block: bool):
        elements = create_vis_dataframe(include_f_block=include_f_block, wide_layout=False)
        colored = self.__get_colormap_column(elements, cmap=cmap)
        elements.loc[:, 'attribute_color'] = colored
        elements['display_attribute'] = self.__vals

        tiles = [self.create_tile(row, color='attribute_color') for _, row in elements.iterrows()]
        self.__fig.layout['shapes'] += tuple(tiles)

        self.__fig.layout['annotations'] += tuple(
            elements.apply(self.create_annotation, axis=1, raw=False, args=('symbol',), y_offset=0.15))

        self.__fig.layout['annotations'] += tuple(elements.apply(self.create_annotation, axis=1, raw=False,
                                                                 args=('atomic_number',), y_offset=-0.2))
        # self.__fig.layout['annotations'] += tuple(
        #     elements.apply(self.create_annotation, axis=1, raw=False, args=('name',),
        #                    y_offset=0.2, size=7))

        # self.__fig.layout['annotations'] += tuple(
        #     elements.apply(self.create_annotation, axis=1, raw=False, args=('display_attribute',), y_offset=0.35,
        #                    size=7))

        self.__fig.update_layout(template='plotly_white', width=self.__width, height=self.__height,
                                 xaxis={'range': [0.5, 18.5], 'showgrid': False, 'fixedrange': True, 'side': 'top',
                                        'tickvals': tuple(range(1, 19))},
                                 yaxis={'range': [10.0, 0.5], 'showgrid': False, 'fixedrange': True,
                                        'tickvals': tuple(range(1, 8))},
                                 margin=dict(l=0, r=0, b=0, t=0))

    def __get_colormap_column(self,
                              elements: pandas.DataFrame,
                              cmap: str = 'RdBu_r') -> pandas.Series:
        colormap = plt.get_cmap(cmap)
        cnorm = colors.Normalize(vmin=min(self.__vals), vmax=max(self.__vals))
        scalarmap = cmx.ScalarMappable(norm=cnorm, cmap=colormap)
        rgba = scalarmap.to_rgba(self.__vals)

        for idx in numpy.where(numpy.array(self.__vals) == min(self.__vals)):
            rgba[idx, 0] = 246 / 255
            rgba[idx, 1] = 246 / 255
            rgba[idx, 2] = 246 / 255

        colored = pandas.Series(index=elements.index, data=[colors.rgb2hex(row) for row in rgba])

        return colored

    def show(self):
        self.__fig.show()

    def save(self,
             path_img_file: str):
        self.__fig.write_image(path_img_file)

    @staticmethod
    def create_tile(element: pandas.Series,
                    color: str,
                    opacity: float = 0.8,
                    x_offset: float = 0.45,
                    y_offset: float = 0.45) -> Shape:
        return Shape(type='rect', x0=element['x'] - x_offset, y0=element['y'] - y_offset, x1=element['x'] + x_offset,
                     y1=element['y'] + y_offset, line=dict(color=element[color]), fillcolor=element[color],
                     opacity=opacity)

    @staticmethod
    def create_annotation(row: pandas.Series,
                          attr: str,
                          size: int = 24,
                          x_offset: float = 0.0,
                          y_offset: float = 0.0) -> Annotation:
        return Annotation(x=row['x'] + x_offset, y=row['y'] + y_offset, xref='x', yref='y', text=row[attr],
                          showarrow=False, font=dict(family='Roboto', size=size, color='#333333'),
                          align='center', opacity=0.9)

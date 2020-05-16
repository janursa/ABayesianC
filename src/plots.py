import plotly.graph_objects as go
import plotly.offline
def box_plot(scalled_posteriors,path_to_save):
    colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)',
              'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)']
    traces = []
    ii = 0
    for key,value in scalled_posteriors.items():
        traces.append(go.Box(
            y=value,
            name=key,
            boxpoints='all',
            jitter=0,
            fillcolor=colors[ii],
            marker_size=5,
            whiskerwidth=0.2,
            line_width=2)
                     )
        ii += 1
    layout = go.Layout(yaxis=dict(
    #                             autorange=True,
    #                             showgrid=False,
                                dtick=0.2,
                                zeroline = False,range= [-0.1,1.1]
                                ),
                        margin=dict(
                                l=40,
                                r=30,
                                b=80,
                                t=100
                            ),
                        showlegend=False,
                        paper_bgcolor='rgb(243, 243, 243)',
                        plot_bgcolor='rgb(243, 243, 243)',
                       )
    fig = { "data": traces,"layout":layout }
    plotly.io.write_image(fig = { "data": traces,"layout":layout }, file=path_to_save+'/box_plot.svg',format="svg",scale=None, width=None, height=None)
    
'''
osint_reseacher.py
'''

import dash
from dash import dcc, html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash.long_callback import DiskcacheLongCallbackManager
import diskcache
import plotly.express as px
import plotly.graph_objs as go
from researcher import OSINTResearcher
import os
import networkx as nx
import pyLDAvis
import pyLDAvis.gensim_models
import json
import logging
from waitress import serve
from datetime import datetime, timedelta


USE_CYBERPUNK_THEME = True

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set environment variables
os.environ['GOOGLE_API_KEY'] = ''
os.environ['GOOGLE_CSE_ID'] = ''

# Initialize the long callback manager
cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

# Initialize the Dash app
if not USE_CYBERPUNK_THEME:
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], long_callback_manager=long_callback_manager)
elif USE_CYBERPUNK_THEME:
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, '/osint_assets/custom.css'], long_callback_manager=long_callback_manager)

# Periodically check for mailboxes if enabled
app.callback(Output("search-emails", "id"), Input("interval-component", "n_intervals"))(
    lambda n: dash.no_update
)

# Define the layout
app.layout = dbc.Container([
    html.H1("OSINT Research Dashboard", className="my-4", style={'textAlign': 'center'}),
    dbc.Row([
        dbc.Col([
            dbc.Input(id="search-input", placeholder="Enter search term", type="text"),
            dbc.Button("Search", id="search-button", color="primary", className="mt-2"),
        ], width=6),
        dbc.Col([
            dbc.Input(id="rss-input", placeholder="Enter RSS feed URL", type="text"),
            dbc.Button("Add RSS Feed", id="add-rss-button", color="secondary", className="mt-2"),
            dbc.Button("Show RSS Feeds", id="show-rss-button", color="info", className="mt-2 ml-2"),
            html.Span(id="rss-feed-count", className="ml-2"),
            html.Div(id="rss-feedback", style={'color': 'black'}),
        ], width=6),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([
            dbc.Checkbox(id="search-emails", label="Search Outlook Emails"),
            dbc.Checkbox(id="use-embeddings", label="Use Embeddings for Email Search", disabled=True),
            dbc.Button("Check & Load Mailboxes", id="check-mailboxes-button", color="info", className="mt-2 ml-2", disabled=True),
        ], width=6),
        # dbc.Col([
            # dbc.Input(id="mailbox-input", placeholder="Enter mailbox name", type="text", disabled=True),
            # dbc.Button("Add Mailbox", id="add-mailbox-button", color="primary", className="mt-2", disabled=True),
            # dbc.Button("Check & Load Mailboxes", id="check-mailboxes-button", color="info", className="mt-2 ml-2", disabled=True),
        # ], width=6),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(id="mailbox-select", multi=True, placeholder="Select mailboxes to search", disabled=True, style={'color': 'black'}),
            dbc.Input(id="email-folders", placeholder="Enter folder names (comma-separated)", type="text", disabled=True),
        ], width=12),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([
            dbc.Label("Email Date Range:"),
            html.Span("  "),
            dcc.DatePickerRange(
                id='email-date-range',
                start_date=datetime.now() - timedelta(days=30),  # Default to last 30 days
                end_date=datetime.now(),
                display_format='YYYY-MM-DD'
            ),
        ], width=12),
    ], className="mb-4"),
    dbc.Modal([
        dbc.ModalHeader("RSS Feeds"),
        dbc.ModalBody(id="rss-feed-list"),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-rss-modal", className="ml-auto")
        ),
    ], id="rss-modal", size="lg"),
    dcc.Loading(
        id="loading",
        type="circle",
        children=[
            dbc.Accordion([
                dbc.AccordionItem([
                    html.Div(id="analysis-placeholder", children="Perform a search to generate an analysis of the data, if any."),
                    html.Div(id="analysis-content", style={'display': 'none'}, children=[
                        dbc.Row([
                            dbc.Col([
                                html.H3("Keyword Frequency"),
                                dcc.Graph(id="keyword-frequency-graph"),
                            ], width=6),
                            dbc.Col([
                                html.H3("Source Distribution"),
                                dcc.Graph(id="source-distribution-graph"),
                            ], width=6),
                        ], className="mb-4"),
                        dbc.Row([
                            dbc.Col([
                                html.H3("Entity Network"),
                                dcc.Graph(id="entity-network-graph"),
                            ], width=6),
                            dbc.Col([
                                html.H3("Sentiment Analysis"),
                                dcc.Graph(id="sentiment-gauge"),
                            ], width=6),
                        ], className="mb-4"),
                        dbc.Row([
                            dbc.Col([
                                html.H3("Word Cloud"),
                                html.Img(id="wordcloud-image", style={"width": "80%"}),
                            ], width=10),
                        ], className="mb-4"),
                        dbc.Row([
                            dbc.Col([
                                html.H3("Topic Modeling"),
                                html.Div(id="topic-modeling"),
                            ], width=10),
                        ], className="mb-4")
                    ]),
                ], title="Analysis", item_id="analysis"),
                dbc.AccordionItem([
                    dbc.Row([
                        dbc.Col([
                            html.H3("Search Results"),
                            html.Div(id="search-results"),
                        ], width=10),
                    ]),
                ], title="Search Results", item_id="search_results"),
            ], start_collapsed=False, always_open=True, active_item="search_results"),
        ]
    ),
    dcc.Interval(
        id='interval-component',
        interval=600*1000,  # in milliseconds, updates every 10 minute
        n_intervals=0
    ),
], fluid=True)

researcher = OSINTResearcher()

@app.callback(
    [Output("mailbox-select", "options"),
    #  Output("mailbox-input", "value"),
     Output("mailbox-select", "value"),
     Output("rss-feedback", "style", allow_duplicate=True),
     Output("rss-feedback", "children", allow_duplicate=True)],
    [
        # Input("add-mailbox-button", "n_clicks"),
        Input("check-mailboxes-button", "n_clicks")
    ],
    [
        # State("mailbox-input", "value"),
        State("mailbox-select", "value")
    ],
    prevent_initial_call=True
)
def manage_mailboxes(add_clicks, check_clicks):
    logger.info("manage_mailboxes callback triggered")
    ctx = dash.callback_context
    if not ctx.triggered:
        logger.info("No trigger for manage_mailboxes callback")
        raise PreventUpdate

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    logger.info(f"Button triggered: {button_id}")

    if button_id == "check-mailboxes-button":
        logger.info("Checking mailboxes")
        options = get_mailbox_options()
        logger.info(f"Available mailboxes: {options}")
        return options, dash.no_update, {"color": "green"}, "Mailboxes updated"

    # if button_id == "add-mailbox-button" and mailbox_name:
    #     logger.info(f"Attempting to add mailbox: {mailbox_name}")
    #     success, message = researcher.add_mailbox(mailbox_name)
    #     color = "green" if success else "red"
    #     logger.info(f"Add mailbox result: success={success}, message={message}")
    #     options = get_mailbox_options()
    #     new_selection = current_selection or []
    #     if success:
    #         new_selection.append(mailbox_name)
    #     return options, "", message, {"color": color}, new_selection
    # elif button_id == "check-mailboxes-button":
    #     logger.info("Checking mailboxes")
    #     options = get_mailbox_options()
    #     logger.info(f"Available mailboxes: {options}")
    #     return options, dash.no_update, "Mailboxes updated", {"color": "green"}, current_selection

    logger.info("No action taken in manage_mailboxes callback")
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update


def get_mailbox_options():
    mailboxes = researcher.get_available_mailboxes()
    logger.info(f"Retrieved mailboxes: {mailboxes}")
    return [{"label": mailbox, "value": mailbox} for mailbox in mailboxes]

@app.callback(
    [Output("use-embeddings", "disabled"),
    #  Output("mailbox-input", "disabled"),
    #  Output("add-mailbox-button", "disabled"),
     Output("check-mailboxes-button", "disabled"),
     Output("mailbox-select", "disabled"),
     Output("email-folders", "disabled")],
    [Input("search-emails", "value")]
)
def toggle_email_inputs(search_emails_checked):
    logger.info(f"Email search checkbox state changed: {search_emails_checked}")
    is_disabled = not search_emails_checked

    if is_disabled == False:
        get_mailbox_options()

    return is_disabled, is_disabled, is_disabled, is_disabled

# @app.callback(
#     [Output("search-emails", "disabled"),
#      Output("search-emails", "checked")],
#     [Input("search-emails", "id")]
# )
# def update_email_search_availability(_):
#     is_available = researcher.is_email_search_available()
#     logger.info(f"Email search availability: {is_available}")
#     return not is_available, False

@app.callback(
    output=[Output("search-results", "children"),
     Output("keyword-frequency-graph", "figure"),
     Output("source-distribution-graph", "figure"),
     Output("entity-network-graph", "figure"),
     Output("sentiment-gauge", "figure"),
     Output("topic-modeling", "children"),
     Output("analysis-placeholder", "children"),
     Output("analysis-placeholder", "style"),
     Output("analysis-content", "style"),
     Output("wordcloud-image", "src")],
    inputs=[Input("search-button", "n_clicks")],
    state=[State("search-input", "value"),
           State("search-emails", "value"),
           State("use-embeddings", "value"),
           State("mailbox-select", "value"),
           State("email-folders", "value"),
           State("email-date-range", "start_date"),
           State("email-date-range", "end_date")],
    running=[
        (Output("search-button", "disabled"), True, False),
        (Output("search-input", "disabled"), True, False),
    ],
    prevent_initial_call=True,
    background=True,
)
def update_results(n_clicks, search_term, search_emails, use_embeddings, mailboxes, email_folders, start_date, end_date):
    if n_clicks is None:
        raise PreventUpdate

    if not search_term:
        return dash.no_update

    try:
        web_results = researcher.web_search(search_term)
        rss_results = researcher.search_rss_feeds(search_term)
        
        email_results = []
        if search_emails and researcher.is_email_search_available():
            folder_names = [name.strip() for name in email_folders.split(',')] if email_folders else None
            start_date = datetime.strptime(start_date[:10], "%Y-%m-%d") if start_date else None
            end_date = datetime.strptime(end_date[:10], "%Y-%m-%d") if end_date else None
            email_results = researcher.search_outlook(search_term, mailboxes, folder_names, use_embeddings, start_date, end_date)

        all_results = web_results + rss_results + email_results
        
        keyword_freq, source_dist, entities, sentiment, lda_model, corpus, dictionary, entity_network, wordcloud_img, sentiment_by_source = researcher.analyze_results(all_results)

        if not all_results:
            return (html.P("No search results found."), None, None, None, None, None, "No search results found", {'display': 'block'}, {'display': 'none'}, "")

    except Exception as e:
        logger.error(f"Error in search and analysis: {e}")
        logger.exception("Exception details:")
        error_message = html.Div([
            html.H4("Error in analysis", className="text-danger"),
            html.P("An error occurred while analyzing the search results. Please try again later.")
        ])
        empty_figure = px.scatter(x=[0], y=[0]).update_layout(
            title="Analysis error",
            xaxis_title="",
            yaxis_title=""
        )
        return (error_message, empty_figure, empty_figure, empty_figure, 
                empty_figure, "Analysis error", "An error occurred during analysis", 
                {'display': 'block'}, {'display': 'none'}, "")

    search_results_html = [html.Div([
        html.H4(html.A(result['title'], href=result.get('link', '#'), target="_blank")),
        html.P(result['snippet']),
        html.Small(f"Source: {result['source']} | Published: {result.get('published', 'N/A')}"),
    ], style={'margin-bottom': '20px', 'word-wrap': 'break-word'}) for result in all_results]

    keyword_freq_fig = create_figure_with_cyberpunk_theme(px.bar(keyword_freq, x=keyword_freq.index, y=keyword_freq.values,
                            labels={'x': 'Keyword', 'y': 'Frequency'}))

    source_dist_fig = create_figure_with_cyberpunk_theme(px.pie(names=source_dist.index, values=source_dist.values))

    # Entity Network Graph
    pos = nx.spring_layout(entity_network)
    edge_trace = go.Scatter(
        x=[], y=[], line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    node_trace = go.Scatter(
        x=[], y=[], text=[], mode='markers', hoverinfo='text',
        marker=dict(showscale=True, colorscale='YlGnBu', size=10, colorbar=dict(thickness=15, title='Centrality', xanchor='left', titleside='right')))

    for edge in entity_network.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    for node in entity_network.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['text'] += tuple([f"{node}<br>Degree Centrality: {entity_network.nodes[node]['degree_centrality']:.2f}<br>Betweenness Centrality: {entity_network.nodes[node]['betweenness_centrality']:.2f}<br>Eigenvector Centrality: {entity_network.nodes[node]['eigenvector_centrality']:.2f}"])

    node_trace.marker.color = [entity_network.nodes[node]['degree_centrality'] for node in entity_network.nodes()]

    entity_network_fig = create_figure_with_cyberpunk_theme(go.Figure(data=[edge_trace, node_trace],
                                layout=go.Layout(
                                    title='Entity Network',
                                    showlegend=False,
                                    hovermode='closest',
                                    margin=dict(b=20,l=5,r=5,t=40),
                                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))))

    # Sentiment Gauge
    sentiment_fig = create_figure_with_cyberpunk_theme(go.Figure(go.Indicator(
        mode = "gauge+number",
        value = sentiment,
        title = {'text': "Sentiment"},
        gauge = {'axis': {'range': [-1, 1]},
                'bar': {'color': "#00ff00"},
                'steps' : [
                    {'range': [-1, -0.5], 'color': "#ff0000"},
                    {'range': [-0.5, 0.5], 'color': "#ffff00"},
                    {'range': [0.5, 1], 'color': "#00ff00"}],
                'threshold': {
                    'line': {'color': "#00ffff", 'width': 4},
                    'thickness': 0.75,
                    'value': sentiment}})))

    # Topic Modeling
    topic_model_html = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    topic_model_html = pyLDAvis.prepared_data_to_html(topic_model_html)

    return (
        search_results_html,
        keyword_freq_fig,
        source_dist_fig,
        entity_network_fig,
        sentiment_fig,
        html.Iframe(srcDoc=topic_model_html, style={"height": "800px", "width": "100%"}),
        "",  # Empty string for analysis placeholder
        {'display': 'none'},  # Hide the placeholder when results are shown
        {'display': 'block'},  # Show the analysis content when results are available
        f"data:image/png;base64,{wordcloud_img}" if wordcloud_img else ""
    )

@app.callback(
    [Output("rss-input", "value"),
     Output("rss-feed-list", "children"),
     Output("rss-modal", "is_open"),
     Output("rss-feedback", "children", allow_duplicate=True),
     Output("rss-feedback", "style", allow_duplicate=True),
     Output("rss-feed-count", "children")],
    [Input("add-rss-button", "n_clicks"),
     Input("show-rss-button", "n_clicks"),
     Input("close-rss-modal", "n_clicks"),
     Input({'type': 'delete-feed', 'index': dash.ALL}, 'n_clicks')],
    [State("rss-input", "value"),
     State("rss-modal", "is_open"),
     State({'type': 'delete-feed', 'index': dash.ALL}, 'id')],
    prevent_initial_call=True
)
def manage_rss_feeds(add_clicks, show_clicks, close_clicks, delete_clicks, rss_url, is_open, delete_ids):
    ctx = dash.callback_context
    if not ctx.triggered:
        feed_count = len(researcher.get_rss_feeds())
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, f"  ({feed_count} saved feeds)"

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    feed_count = len(researcher.get_rss_feeds())
    feed_count_display = f"({feed_count} feeds)"
    
    if button_id == "add-rss-button" and rss_url:
        success, message = researcher.add_rss_feed(rss_url)
        if success:
            feed_count += 1
            feed_count_display = f"({feed_count} feeds)"
            return "", create_rss_feed_list(), is_open, "RSS feed added successfully!", {"color": "green"}, feed_count_display
        else:
            return rss_url, dash.no_update, is_open, message, {"color": "red"}, feed_count_display
    elif button_id == "show-rss-button":
        return dash.no_update, create_rss_feed_list(), not is_open, None, None, feed_count_display
    elif button_id == "close-rss-modal":
        return dash.no_update, dash.no_update, False, None, None, feed_count_display
    elif "delete-feed" in button_id:
        deleted_index = json.loads(button_id)['index']
        feeds = researcher.get_rss_feeds()
        if 0 <= deleted_index < len(feeds):
            researcher.remove_rss_feed(feeds[deleted_index])
        feed_count = len(researcher.get_rss_feeds())
        feed_count_display = f"({feed_count} feeds)"
        return dash.no_update, create_rss_feed_list(), is_open, None, None, feed_count_display
    
    return dash.no_update, dash.no_update, is_open, None, None, feed_count_display

def create_rss_feed_list():
    feeds = researcher.get_rss_feeds()
    if not feeds:
        return html.Div("No RSS feeds added yet.")
    
    return html.Ul([
        html.Li([
            html.Span(feed),
            html.Span(" ", className="text-muted"),
            html.Button("Delete", id={'type': 'delete-feed', 'index': i}, className="ml-2 btn btn-danger btn-sm")
        ]) for i, feed in enumerate(feeds)
    ])

# @app.callback(
#     [Output({'type': 'delete-feed', 'index': dash.ALL}, 'disabled')],
#     [Input({'type': 'delete-feed', 'index': dash.ALL}, 'n_clicks')],
#     [State({'type': 'delete-feed', 'index': dash.ALL}, 'id')]
# )
# def delete_rss_feed(n_clicks, ids):
#     ctx = dash.callback_context
#     if not ctx.triggered:
#         raise PreventUpdate
    
#     button_id = ctx.triggered[0]['prop_id'].split('.')[0]
#     deleted_index = json.loads(button_id)['index']
    
#     feeds = researcher.get_rss_feeds()
#     if 0 <= deleted_index < len(feeds):
#         researcher.remove_rss_feed(feeds[deleted_index])
    
#     feed_count = len(researcher.get_rss_feeds())
#     feed_count_display = f"({feed_count} feeds)"
    
#     return [False] * len(ids), feed_count_display

@app.callback(
    [Output({'type': 'delete-feed', 'index': dash.ALL}, 'disabled')],
    [Input({'type': 'delete-feed', 'index': dash.ALL}, 'n_clicks')],
    [State({'type': 'delete-feed', 'index': dash.ALL}, 'id')]
)
def delete_rss_feed(n_clicks, ids):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    deleted_index = json.loads(button_id)['index']
    
    feeds = researcher.get_rss_feeds()
    if 0 <= deleted_index < len(feeds):
        researcher.remove_rss_feed(feeds[deleted_index])
    
    return [False] * len(ids)

def create_figure_with_cyberpunk_theme(fig):
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#00ff00',
        title_font_color='#00ffff',
        legend_title_font_color='#00ffff',
        legend_font_color='#00ff00',
        xaxis=dict(gridcolor='#1a1a1a', zerolinecolor='#1a1a1a'),
        yaxis=dict(gridcolor='#1a1a1a', zerolinecolor='#1a1a1a')
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
    #serve(app.server, host='127.0.0.1', port=8050, threads=6)




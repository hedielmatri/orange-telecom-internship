"""
app.py - Orange Madagascar analytics dashboard.
"""

import sys
#!pip install XXX --target ./my_custom_packages
sys.path.append('./my_custom_packages') 


import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from dash import Dash, Input, Output, State, callback, dcc, html
import dash_bootstrap_components as dbc


# Data loading
PROJECT = os.path.dirname(os.path.abspath(__file__))

df_raw = pl.read_parquet(os.path.join(PROJECT, "df_merged.parquet")).with_columns([
    pl.col("date").cast(pl.Date),
    pl.col("Typologie").fill_null("Unknown")
])

with open(os.path.join(PROJECT, "madagascar_regions.geojson"), "r") as f:
    geojson_data = json.load(f)


# Filter options
ALL_REGIONS = sorted(df_raw["region_cleaned"].drop_nulls().unique().to_list())
ALL_CHANNELS = sorted(df_raw["Group_canal"].drop_nulls().unique().to_list())
ALL_TYPOLOGIES = sorted(df_raw["Typologie"].drop_nulls().unique().to_list())
ALL_TECH = ["2G", "3G", "4G_FDD", "4G_TDD", "4G+_FDD", "4G+_TDD", "5G"]
DATE_MIN = df_raw["date"].min()
DATE_MAX = df_raw["date"].max()


# Plot defaults
TEMPLATE = go.layout.Template(
    layout = go.Layout(
        paper_bgcolor = "white",
        plot_bgcolor = "white",
        font = dict(family = "sans-serif", color = "black", size = 12),
        xaxis = dict(gridcolor = "gray"),
        yaxis = dict(gridcolor = "gray"),
        margin = dict(l = 60, r = 20, t = 40, b = 60),
    )
)


def fmt_number(n, suffix = ""):
    """Format large numbers."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B{suffix}"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M{suffix}"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K{suffix}"
    return f"{n:,.0f}{suffix}"


def fmt_ariary(n):
    return fmt_number(n, " Ar")


# Dash app
# Detect JupyterHub environment variables 
# defaults to "/" for outside of JupyterHub

#proxy_path = "/jupyterhub/user/he4249/proxy/8050/"

proxy_path = "/"

app = Dash(
    __name__, 
    title = "Orange Madagascar Dashboard",
    requests_pathname_prefix = proxy_path,
    external_stylesheets = [dbc.themes.BOOTSTRAP]
)


# Layout
app.layout = dbc.Container([

    # Header
    dbc.Row([
        dbc.Col([
            html.H2("Orange Madagascar Dashboard", style = {"marginTop": "20px"}),
            html.P("Telecom Dashboard for Analysis", style = {"color": "gray"}),
            html.Hr()
        ], width = 12)
    ]),

    # KPI row
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Total Revenue"),
            html.H3(id = "kpi-revenue", style = {"color": "orange"}),
            html.Small(id = "kpi-revenue-sub", style = {"color": "gray"}),
        ])), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Unique Users"),
            html.H3(id = "kpi-users"),
            html.Small(id = "kpi-users-sub", style = {"color": "gray"}),
        ])), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Avg Spend/User"),
            html.H3(id = "kpi-avg"),
            html.Small(id = "kpi-avg-sub", style = {"color": "gray"}),
        ])), width=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Transactions"),
            html.H3(id = "kpi-txns"),
            html.Small(id = "kpi-txns-sub", style = {"color": "gray"}),
        ])), width=3),
    ], style={"marginBottom": "20px"}),

    # Main content
    dbc.Row([

        # Filters sidebar
        dbc.Col([
            dbc.Card(dbc.CardBody([
                html.H5("Filters", style = {"marginBottom": "20px"}),

                html.Div([
                    html.Label("Region"),
                    dcc.Dropdown(id = "filter-region", options = ALL_REGIONS, value = [], multi = True, placeholder = "All regions"),
                ], style = {"marginBottom": "15px"}),

                html.Div([
                    html.Label("Typology"),
                    dcc.Dropdown(id = "filter-typology", options = ALL_TYPOLOGIES, value = [], multi = True, placeholder = "All typologies"),
                ], style = {"marginBottom": "15px"}),

                html.Div([
                    html.Label("Network Technology"),
                    dbc.Checklist(id = "filter-tech", options = [{"label": t, "value": t} for t in ALL_TECH], value = ALL_TECH),
                ], style = {"marginBottom": "15px"}),

                html.Div([
                    html.Label("Sales Channel"),
                    dcc.Dropdown(id = "filter-channel", options = ALL_CHANNELS, value = [], multi = True, placeholder = "All channels"),
                ], style = {"marginBottom": "15px"}),

                html.Div([
                    html.Label("Date Range"),
                    html.Br(),
                    dcc.DatePickerRange(
                        id = "filter-date",
                        start_date = DATE_MIN, end_date = DATE_MAX,
                        min_date_allowed = DATE_MIN, max_date_allowed = DATE_MAX,
                        display_format = "DD MMM YYYY",
                    ),
                ], style = {"marginBottom": "20px"}),

                dbc.Button("Apply Filters", id = "submit-button", n_clicks = 0, color="warning", style={"width": "100%"}),
            ]))
        ], width = 3),

        # Charts
        dbc.Col([
            dcc.Loading(type = "circle", color = "orange", children = [

                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Trend of Daily Revenue"),
                        dbc.CardBody(dcc.Graph(id = "chart-revenue-day"))
                    ]), width = 6),
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Revenue by Region"),
                        dbc.CardBody(dcc.Graph(id = "chart-revenue-region"))
                    ]), width = 6),
                ], style = {"marginBottom": "20px"}),

                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("ARPU by Technology"),
                        dbc.CardBody(dcc.Graph(id = "chart-arpu-tech"))
                    ]), width = 6),
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Revenue by Plan Type"),
                        dbc.CardBody(dcc.Graph(id = "chart-revenue-plan"))
                    ]), width = 6),
                ], style = {"marginBottom": "20px"}),

                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Site Performance, Revenue to Users"),
                        dbc.CardBody(dcc.Graph(id = "chart-scatter-sites"))
                    ]), width = 6),
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Revenue by Channel"),
                        dbc.CardBody(dcc.Graph(id = "chart-revenue-channel"))
                    ]), width = 6),
                ], style = {"marginBottom": "20px"}),

                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Revenue Flow, Region to Channel to Typology"),
                        dbc.CardBody(dcc.Graph(id = "chart-sankey-flow"))
                    ]), width = 6),
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Revenue by Technology per Region"),
                        dbc.CardBody(dcc.Graph(id = "chart-tech-region"))
                    ]), width = 6),
                ]),
            ]),
        ], width = 9),
    ], style = {"marginBottom": "20px"}),

    # Map
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Map of Network Coverage"),
            dbc.CardBody([
                dcc.Loading(type = "circle", color = "orange", children = [
                    dcc.Graph(id = "map-interactive", style = {"height": "650px"}),
                ]),
            ])
        ]), width=12)
    ], style = {"marginBottom": "40px"}),

], fluid = True, style = {"padding": "20px"})


# Callback
@callback(
    Output("kpi-revenue", "children"),
    Output("kpi-revenue-sub", "children"),
    Output("kpi-users", "children"),
    Output("kpi-users-sub", "children"),
    Output("kpi-avg", "children"),
    Output("kpi-avg-sub", "children"),
    Output("kpi-txns", "children"),
    Output("kpi-txns-sub", "children"),
    
    Output("chart-revenue-day", "figure"),
    Output("chart-revenue-region", "figure"),
    Output("chart-arpu-tech", "figure"),
    Output("chart-revenue-plan", "figure"),
    Output("chart-scatter-sites", "figure"),
    Output("chart-revenue-channel", "figure"),
    Output("chart-sankey-flow", "figure"),
    Output("chart-tech-region", "figure"),
    Output("map-interactive", "figure"),
    
    Input("submit-button", "n_clicks"),
    
    State("filter-region", "value"),
    State("filter-typology", "value"),
    State("filter-tech", "value"),
    State("filter-channel", "value"),
    State("filter-date", "start_date"),
    State("filter-date", "end_date"),
)

def update_dashboard(n_clicks, regions, typologies, techs, channels, start_date, end_date):
    """
    Filter data and update all outputs.
    """

    df = df_raw

    if regions: df = df.filter(pl.col("region_cleaned").is_in(regions))
    if typologies: df = df.filter(pl.col("Typologie").is_in(typologies))
    if techs: df = df.filter(pl.col("Max_RAT").is_in(techs))
    if channels: df = df.filter(pl.col("Group_canal").is_in(channels))
    if start_date: df = df.filter(pl.col("date") >= pl.lit(start_date).str.to_date())
    if end_date: df = df.filter(pl.col("date") <= pl.lit(end_date).str.to_date())

    # KPIs
    if len(df) > 0:
        kpi_data = df.select([
            pl.col("CA").sum().alias("total_revenue"),
            pl.col("msisdn").n_unique().alias("unique_users"),
            pl.col("region_cleaned").n_unique().alias("n_regions"),
            pl.col("Max_RAT").n_unique().alias("n_techs"),
            pl.len().alias("n_txns")
        ]).row(0)
        
        total_revenue, unique_users, n_regions, n_techs, n_txns = kpi_data
        avg_spend = total_revenue / unique_users if unique_users > 0 else 0
    else:
        total_revenue = unique_users = avg_spend = n_regions = n_techs = n_txns = 0

    return (
        fmt_ariary(total_revenue), f"Across {n_regions} regions",
        fmt_number(unique_users), "Active subscribers",
        fmt_ariary(avg_spend), "Per user this month",
        fmt_number(len(df)), f"Across {n_techs} network types",
        
        build_revenue_day(df),
        build_region_revenue(df),
        build_arpu_tech(df),
        build_revenue_plan(df),
        build_scatter_sites(df),
        build_channel_revenue(df),
        build_sankey_flow(df),
        build_tech_region(df),
        build_map(df),
    )


# Chart builders

def build_revenue_day(df):
    """
    Daily revenue trend.
    """
    fig = go.Figure()
    if len(df) > 0:
        daily = df.group_by("date").agg(pl.col("CA").sum()).sort("date").to_pandas()
        fig.add_trace(go.Scatter(x = daily["date"], y = daily["CA"], mode = "lines + markers", line = dict(color = "orange")))
    fig.update_layout(template = TEMPLATE, height = 350, yaxis_title = "Revenue (Ar)")
    return fig


def build_region_revenue(df):
    """
    Revenue by region.
    """
    fig = go.Figure()
    if len(df) > 0:
        agg = df.group_by("region_cleaned").agg(pl.col("CA").sum()).sort("CA", descending=False).to_pandas()
        fig.add_trace(go.Bar(y = agg["region_cleaned"], x = agg["CA"], orientation = "h", marker_color = "orange"))
    fig.update_layout(template = TEMPLATE, height = 350, xaxis_title = "Revenue (Ar)", yaxis = dict(automargin = True))
    return fig


def build_arpu_tech(df):
    """
    ARPU by technology.
    """
    all_techs = ["2G", "3G", "4G_FDD", "4G_TDD", "4G+_FDD", "4G+_TDD", "5G"]
    fig = go.Figure()
    if len(df) > 0:
        agg = df.group_by("Max_RAT").agg([
            pl.col("CA").sum().alias("total_ca"),
            pl.col("msisdn").n_unique().alias("users")
        ]).to_pandas()
        agg["ARPU"] = (agg["total_ca"] / agg["users"]).fillna(0)
        agg = agg.set_index("Max_RAT").reindex(all_techs, fill_value = 0).reset_index()
        fig.add_trace(go.Bar(x = agg["Max_RAT"], y = agg["ARPU"], marker_color = "orange"))
    fig.update_layout(template = TEMPLATE, height = 350, yaxis_title = "ARPU (Ar/User)", xaxis = dict(categoryorder = "array", categoryarray = all_techs))
    return fig


def build_revenue_plan(df):
    """
    Revenue by plan type.
    """
    fig = go.Figure()
    if len(df) > 0:
        agg = df.group_by("Gamme_groupe").agg(pl.col("CA").sum()).to_pandas()
        fig = px.pie(agg, values = "CA", names = "Gamme_groupe", hole = 0.5)
        fig.update_layout(legend = dict(orientation = "h", y = -0.15, x = 0.5, xanchor = "center"))
    fig.update_layout(template = TEMPLATE, height = 350, margin = dict(t = 20, b = 20, l = 20, r = 20))
    return fig


def build_scatter_sites(df):
    """
    Site revenue vs users scatter.
    """
    fig = go.Figure()
    fig.update_layout(template = TEMPLATE, height = 350, xaxis_title = "Users (log)", yaxis_title = "Revenue (log)")

    if len(df) == 0:
        return fig
    
    agg = df.group_by("sig_nom_site").agg([
        pl.col("CA").sum().alias("total_ca"),
        pl.col("msisdn").n_unique().alias("users"),
        pl.col("region_cleaned").first().alias("region")
    ]).filter((pl.col("total_ca") > 0) & (pl.col("users") > 0)).to_pandas()
    
    if len(agg) == 0:
        return fig

    fig = px.scatter(agg, x = "users", y = "total_ca", color = "region", hover_name = "sig_nom_site", log_x = True, log_y = True)
    fig.update_layout(template = TEMPLATE, height = 350, xaxis_title = "Users (log)", yaxis_title = "Revenue (log)", legend = dict(title = "Region", font = dict(size = 9)))
    return fig


def build_channel_revenue(df):
    """
    Revenue by channel.
    """
    fig = go.Figure()
    if len(df) > 0:
        agg = df.group_by("Group_canal").agg(pl.col("CA").sum()).sort("CA", descending=False).to_pandas()
        fig.add_trace(go.Bar(y = agg["Group_canal"], x = agg["CA"], orientation = "h", marker_color = "blue"))
    fig.update_layout(template = TEMPLATE, height = 350, xaxis_title = "Revenue (Ar)", yaxis = dict(automargin = True))
    return fig


def build_sankey_flow(df):
    """
    Revenue flow from region to channel to typology.
    """
    fig = go.Figure()
    fig.update_layout(template = TEMPLATE, height = 350)

    if len(df) == 0:
        return fig

    flow = df.group_by(["region_cleaned", "Group_canal", "Typologie"]).agg(pl.col("CA").sum()).filter(pl.col("CA") > 0).to_pandas()
    flow = flow[flow["CA"] > 0]
    if len(flow) == 0:
        return fig

    regions = flow["region_cleaned"].unique().tolist()
    channels = flow["Group_canal"].unique().tolist()
    typos = flow["Typologie"].unique().tolist()
    nodes = regions + channels + typos
    node_idx = {name: i for i, name in enumerate(nodes)}

    source, target, value = [], [], []

    for _, row in flow.groupby(["region_cleaned", "Group_canal"]).agg({"CA": "sum"}).reset_index().iterrows():
        source.append(node_idx[row["region_cleaned"]])
        target.append(node_idx[row["Group_canal"]])
        value.append(row["CA"])

    for _, row in flow.groupby(["Group_canal", "Typologie"]).agg({"CA": "sum"}).reset_index().iterrows():
        source.append(node_idx[row["Group_canal"]])
        target.append(node_idx[row["Typologie"]])
        value.append(row["CA"])

    fig = go.Figure(data = [go.Sankey(
        node = dict(pad = 15, thickness = 20, label = nodes, color = "orange"),
        link = dict(source = source, target = target, value = value, color = "rgba(255, 165, 0, 0.2)")
    )])
    fig.update_layout(template = TEMPLATE, height = 350)
    return fig


def build_tech_region(df):
    """
    Revenue by technology per region.
    """
    fig = go.Figure()
    fig.update_layout(template = TEMPLATE, height = 350, yaxis_title = "Revenue (Ar)", barmode = "group", xaxis = dict(automargin = True, tickangle = -45), legend = dict(title = "Technology"))

    if len(df) == 0:
        return fig

    agg = df.group_by(["region_cleaned", "Max_RAT"]).agg(pl.col("CA").sum()).to_pandas()
    for tech in ALL_TECH:
        tech_data = agg[agg["Max_RAT"] == tech]
        fig.add_trace(go.Bar(x = tech_data["region_cleaned"], y = tech_data["CA"], name = tech))

    fig.update_layout(template = TEMPLATE, height = 350, yaxis_title = "Revenue (Ar)", barmode = "group", xaxis = dict(automargin = True, tickangle = -45), legend = dict(title = "Technology"))
    return fig


def build_map(df):
    """
    Map with choropleth and tower scatter markers.
    """
    fig = go.Figure()

    # Choropleth
    if len(df) > 0:
        region_agg = df.group_by("region_cleaned").agg(pl.col("CA").sum().alias("value")).to_pandas()
        fig.add_trace(go.Choropleth(
            geojson = geojson_data,
            locations = region_agg["region_cleaned"],
            z = region_agg["value"],
            featureidkey = "properties.region",
            colorscale = [[0, "white"], [1, "orange"]],
            marker_line_color = "gray",
            colorbar = dict(title = "Revenue (Ar)"),
            name = "Revenue",
        ))

    # Scatter per technology
    
    group_cols = ["sig_nom_site", "x", "y", "Max_RAT"]
    
    if len(df) > 0:
        site_agg = df.drop_nulls(["x", "y"]).group_by(group_cols).agg(pl.col("CA").sum().alias("value")).to_pandas()
        max_val = site_agg["value"].max() if len(site_agg) > 0 else 1
    else:
        site_agg = pd.DataFrame(columns = group_cols + ["value"])
        max_val = 1

    for tech in ALL_TECH:
        tech_data = site_agg[site_agg["Max_RAT"] == tech]

        if len(tech_data) == 0:
            fig.add_trace(go.Scattergeo(lon = [47.0], lat = [-18.0], name = tech, marker = dict(size = 0, opacity = 0), showlegend = True, hoverinfo = "skip"))
            continue

        names = tech_data["sig_nom_site"].fillna("Unknown").astype(str)
        
        hover = names + " | " + tech + " | " + tech_data["value"].apply(lambda v: f"{v:,.0f} Ar")
        sizes = np.clip(tech_data["value"] / max_val * 18 + 3, 3, 22)

        fig.add_trace(go.Scattergeo(lon = tech_data["x"], lat = tech_data["y"], text = hover.tolist(), hoverinfo = "text", marker = dict(size = sizes, opacity = 0.7), name = tech))

    fig.update_geos(fitbounds = "locations", visible = False, bgcolor = "white", projection_type = "mercator")
    fig.update_layout(template = TEMPLATE, height = 650, margin = dict(l = 0, r = 0, t = 10, b = 0), legend = dict(title = "Technology", x = 0.01, y = 0.99), geo = dict(bgcolor = "white", landcolor = "white"))
    return fig


# Run
if __name__ == "__main__":
    print(f"Data: {len(df_raw):,} rows, {df_raw['msisdn'].n_unique():,} users")
    print(f"http://0.0.0.0:8050/jupyterhub/user/he4249/proxy/8050/")
    print("ssh -N -L 8050:localhost:8050 he4249@lambcomp02.ccbb.utexas.edu")
    app.run(host = "0.0.0.0", port = 8050, debug = True, use_reloader = False)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
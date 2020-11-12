# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import plotly
import datetime
import math

# import local functions
import covid19_analysis.dataFun as dataFun
#import covid19_analysis.dataPlot as dataPlot


from covid19_analysis import __version__

__author__ = "J SAYRITUPAC"
__copyright__ = "J SAYRITUPAC"
__license__ = "mit"


# Report daily cases evolution for last three months
def last_daily_cases(df_data, ctry_list, num_days=3*31, rolling_win=True, df_type='cases'):
    '''Display countries last days daily cases trend
        df_data:    <dataframe> contain all countries daily data
        ctry_list:  <list> string list with countries to display
        num_days:   <int> set the number of days to display rolling back from the last day
        rolling_win:<boolean> set weakly rolling window with center on the day
        df_type:    <string> define the type of data displayed, optiones are 'cases', 'recover' & 'fatalities'
    '''

    # define graph object
    fig = plotly.graph_objs.Figure()

    # Loop per country, display daily evolution for last three months
    for c in ctry_list:
        ts_c = dataFun.get_timeseries_from_JHU(df_data, c, verbose=False)
        
        # calculate daily cases (set to 0 if no cases)
        data_temp = np.array(ts_c[1:], dtype=int)  - np.array(ts_c[:-1], dtype=int)
        ts_c_daily = pd.Series(data_temp, index=ts_c.index[1:]).clip(0)
        
        if rolling_win:
            # moving average, 7 days centered in day
            ts_c_daily = ts_c_daily.rolling(7, min_periods=1, center=True).mean()
        

        # Plot graph for a define time interval
        mask = (ts_c_daily.index >= (ts_c_daily.index[-1] - pd.Timedelta(num_days, unit='days')))
        
        fig.add_trace(
            plotly.graph_objs.Scatter(
                mode = 'lines',
                name = c,
                x = ts_c_daily.index[mask],
                y = ts_c_daily[mask],
                line=dict(width = 1.5),
            )
        )

    fig.update_layout(
        plot_bgcolor='white', 
        xaxis_title = 'Dates [Days]',
        yaxis_title = 'Daily ' + df_type,
        title = 'Lasts month '+ df_type + ' evolution' + datetime.datetime.today().strftime(', %B %d, %Y'),
        title_x = .5
    )

    fig.update_yaxes(showgrid=True, gridwidth=.3, gridcolor='gainsboro')
    fig.show()


# Report growth rates over time
def growth_rates(data_ts, label = 'Cases ', trend_line = False, y_range = [1, 1.07], Percentage=True):
    '''Display growth rates over time for cases/cures/fatalities for one dataset array'''
    # fill nan values with previous values
    data_tmp = data_ts.fillna(method='bfill')
    # calculate growth rates
    data_tmp = np.array(data_tmp, dtype=int)
    if Percentage:  # display results as a growing percentage
        growth_ratio = 100 * (dataFun.safe_div(data_tmp[1:], data_tmp[:-1]) -1)
    else:
        growth_ratio = dataFun.safe_div(data_tmp[1:], data_tmp[:-1])
    time_vector = data_ts.index

    # display growth ratio over time
    fig = plotly.graph_objs.Figure()
    fig.add_trace(
        plotly.graph_objs.Scatter(
            mode = 'lines+markers',
            x = time_vector[1:],
            y = growth_ratio,
            marker = dict(color = 'Black', line = dict(color = 'DarkGrey', width=1.5)),
            name = label,
        ))

    if trend_line:
        growth_ratio_ma = dataFun.mov_avg(growth_ratio, 7)
        fig.add_trace(
        plotly.graph_objs.Scatter(
            mode = 'lines',
            x = time_vector[1:],
            y = growth_ratio_ma,
            marker = dict(color = 'DarkRed', line = dict(color = 'DarkRed', width=1.5)),
            name = 'data trend',
        ))
        
    fig.update_layout(
        plot_bgcolor='white', 
        xaxis_title = 'Time [Days]',
        yaxis_title = label + ' growth ratio' + (' [%]' if Percentage else '' ),
        title = 'Overall ' + label + 'growth rati trends' + datetime.datetime.today().strftime(', %B %d, %Y'),
        title_x = .5
        )
    fig.update_yaxes(showgrid=True, gridwidth=.3, gridcolor='gainsboro')

    if Percentage:
        fig.update_yaxes(range=[0, np.max(growth_ratio)*1.05])
    else:
        fig.update_yaxes(range=y_range)

    fig.show()
    return fig


# Plot countries growing ratio and doubling time chars
def growing_ratio_countries(df_data, ctry_list, pop_th=100, num_days=37, df_source='JHU', day_filter = np.nan, clear_pop = False ):
    '''Display countries cases over time compare to standards doubling-time ratios
        df_data:    <dataframe> contain all countries daily data
        ctry_list:  <list> string list with countries to display
        pop_th:     <int> population threshold, allows to set chart starting point
        num_days:   <int> set the number of days to display
        df_source:  <str> set the dataframe data source, options are: 'JHU' (default), 'SPF', 'raw_data'
        day_filter: <str> define a date string as a time filter, no filter as default
        clear_pop:  <bool> substract population from first day, useful if counting from a different day from first outbreak
        
    Graph inspired on the work or Lisa Charlotte ROST, designer & blogger at Datawrapper (March 2020)
    https://lisacharlotterost.de/
    Original graph from Lisa https://www.datawrapper.de/_/w6x6z/ 
    '''

    # build growing rates template
    fig_gr = doublingtime_chart(pop_th, num_days)

    # Extract timeseries & add trace to figure
    if df_source == 'JHU':
        max_cases = 1
        for country_name in ctry_list:    
            ts_country = dataFun.get_timeseries_from_JHU(df_data, country_name)

            # post first-outbreak filters
            if not pd.isna(day_filter):    # a time filter is included
                ts_country = ts_country[ts_country.index >= day_filter]

                if clear_pop:   # substract first date population
                    ts_country -= ts_country[0]


            if max_cases < np.max(ts_country):
                max_cases = np.max(ts_country)

            fig_gr.add_trace(
            plotly.graph_objs.Scatter(
                mode = 'lines',
                #mode = 'lines+markers',
                x = np.array(range(0, ts_country[ts_country > pop_th*.5].size)),
                y = ts_country[ts_country > pop_th*.5],
                name = country_name
            ))

        # set graph extra details
        fig_gr.update_layout(
            title = 'Doubling rates per country' + datetime.datetime.today().strftime(', %B %d, %Y'),
            title_x = .5
        )
        # correct y axis
        fig_gr.update_yaxes(range=[math.log10(pop_th), np.maximum(math.log10(max_cases)+.2, math.log10(pop_th)+3.5)])    

    # Display simple data overtime
    elif df_source == 'raw_data':
        data_flt = df_data > pop_th

        fig_gr.add_trace(
            plotly.graph_objs.Scatter(
                mode = 'lines',
                #mode = 'lines+markers',
                x = np.array(range(0, len(data_flt))),
                y = df_data[data_flt],
                name = ctry_list,
                line = dict(color = 'Black')
            ))

    elif df_source == 'SPF':
        ts_cases = pd.Series(data=df_data.cas_confirmes.fillna(0).values, index=df_data.date)
        ts_ftlts = pd.Series(data=df_data.deces.fillna(0).values, index=df_data.date)
        
        # post first-outbreak filters
        if not pd.isna(day_filter):    # a time filter is included
                ts_cases[ts_cases.index >= day_filter]
                ts_ftlts[ts_ftlts.index >= day_filter]

                if clear_pop:   # substract first date population
                    ts_cases = ts_cases - ts_country[0]
                    ts_ftlts = ts_ftlts - ts_country[0]  


        t_idx = ts_cases > pop_th
        # trace french cases
        fig_gr.add_trace(
            plotly.graph_objs.Scatter(
                #mode = 'lines',
                mode = 'lines+markers',
                x = np.array(range(0, ts_cases[t_idx].size)),
                y = ts_cases[t_idx],
                name = 'Cases',
                line=dict(color='CornflowerBlue'),
                #marker=dict(color='CornflowerBlue')
            ))
        # trace french fatalities
        fig_gr.add_trace(
            plotly.graph_objs.Scatter(
                #mode = 'lines',
                mode = 'lines+markers',
                x = np.array(range(0, ts_ftlts[t_idx].size)),
                y = ts_ftlts[t_idx],
                name = 'Fatalities',
                line=dict(color='Black'),
            ))
        # set graph extra details
        fig_gr.update_layout(
            title = 'Doubling rates in France' + datetime.datetime.today().strftime(', %B %d, %Y'),
            title_x = .5
        )

    fig_gr.show()


# Plot countries growing ratio and doubling time chars
def growing_ratio_country(df_data, pop_th=100, num_days=90, df_source=None, date_filter = None, clear_pop = False ):
    '''Display countries cases over time compare to standards doubling-time ratios
        df_data:    <dataframe> contain all countries daily data
        pop_th:     <int> population threshold, allows to set chart starting point
        num_days:   <int> set the number of days to display
        df_source:  <str> set the dataframe data source, options are: 'JHU' (default), 'SPF', 'raw_data'
        date_filter:<str> define a date string as a time filter, default: no filter (None)
        clear_pop:  <bool> substract population from first day, useful if counting from a different day from first outbreak
        
    Graph inspired on the work or Lisa Charlotte ROST, designer & blogger at Datawrapper (March 2020)
    https://lisacharlotterost.de/
    Original graph from Lisa https://www.datawrapper.de/_/w6x6z/ 
    '''

    # build growing rates template
    fig_gr = doublingtime_chart(pop_th, num_days)


    # Extract data to plot as a function of the datasource
    if df_source is 'datagouv':
        ts_cases = pd.Series(data=df_data.total_cas_confirmes.fillna(0).values, index=df_data.date)
        ts_ftlts = pd.Series(data=df_data.total_deces_hopital.fillna(0).values, index=df_data.date)

    elif df_source is None:
        raise NotImplementedError
    
    # post first-outbreak filters: reshape date to start from a given date
    if not (date_filter is None):
        ts_cases = ts_cases[ts_cases.index >= date_filter]
        ts_ftlts = ts_ftlts[ts_ftlts.index >= date_filter]

        # substract first date population, evaluate the evolution counting population growth from a given date
        if clear_pop:   # substract first date population
            ts_cases -= ts_cases[0]
            ts_ftlts -= ts_ftlts[0]  

    # plot the doubling rate over time
    t_idx = ts_cases > pop_th
    # trace cases
    fig_gr.add_trace(
        plotly.graph_objs.Scatter(
            mode = 'lines+markers', #'lines'
            x = np.arange(0, len(t_idx)),
            y = ts_cases[t_idx],
            name = 'Cases',
            line=dict(color='CornflowerBlue'),
            #marker=dict(color='CornflowerBlue')
        ))
    # trace fatalities
    fig_gr.add_trace(
        plotly.graph_objs.Scatter(
            mode = 'lines+markers', # 'lines',
            x = np.arange(0, len(t_idx)),
            y = ts_ftlts[t_idx],
            name = 'Fatalities',
            line=dict(color='Black'),
        ))
    # set graph extra details
    fig_gr.update_layout(
        title = 'Doubling rates in a Country' + datetime.datetime.today().strftime(', %B %d, %Y'),
        title_x = .5
    )
    # correct y axis
    fig_gr.update_yaxes(range=[math.log10(pop_th), np.maximum(math.log10(np.max(ts_cases))+.2, math.log10(pop_th)+3.5)])
    
    fig_gr.show()


# Explore the growing rate over time (call chart growing rate countries)
def doublingtime_chart(pop_th=100, num_days=37):
    '''Build a doubling time chart template
        pop_th:     <int> population threshold, identify min days per contry and set the chart starting point
        num_days:   <int> set the number of days to display
        
    Graph inspired on Lisa Charlotte ROST work https://www.datawrapper.de/_/w6x6z/ 
    '''
    
    # define some working variables
    ndays = np.array(range(0, num_days))
    symbols_list = ['circle', 'square', 'diamond', 'triangle_up', 'cross', 'x', 'asterisk', 'hash']
    gr_array = [1, 2, 3, 5, 7, 30]
    gr_labels = ['daily', 'two days', 'three days', 'five days', 'weekly', 'monthly']
    
    # build a chart template (growing ratios references)
    fig = plotly.graph_objs.Figure()
    for gr_idx, gr_value in enumerate(gr_array):
        # calculate references growing rates
        ncase = dataFun.doubling_time_fun(pop_th, num_days, gr_value)

        # add a growing rate
        fig.add_trace(
            plotly.graph_objs.Scatter(
                #mode = 'lines+markers',
                mode = 'lines',
                name = gr_labels[gr_idx],
                x = ndays,
                y = ncase,
                marker_symbol = 100+2*gr_idx, #select 'open' symbols
                line=dict(color='DarkGray', width = 1.5, dash = 'dashdot'),
                #visible = 'legendonly'
                showlegend=False,
                hoverinfo='skip'
        ))
    
    # anotation style
    annotation_style=dict(size=10, color='DimGray')
    # Text for daily grow
    fig.add_annotation(x = 9, y = math.log10(dataFun.doubling_time_equation(pop_th, 9, 1)),
                        text = 'Doubles every day', font = annotation_style, arrowcolor='DimGray')
    # Text for 2 days grow
    fig.add_annotation(x = 19, y = math.log10(dataFun.doubling_time_equation(pop_th, 19, 2)),
                        text = 'Doubles every 2nd day', font = annotation_style, arrowcolor='DimGray')
    # Text for 3 days grow
    fig.add_annotation(x = 29, y = math.log10(dataFun.doubling_time_equation(pop_th, 29, 3)),
                        text = 'Doubles every 3rd day', font = annotation_style, arrowcolor='DimGray')
    # Text for 5 days grow
    fig.add_annotation(x = 31, y = math.log10(dataFun.doubling_time_equation(pop_th, 31, 5)),
                        text = 'Doubles every 5th day', font = annotation_style, arrowcolor='DimGray')
    # Text for weekly grow
    fig.add_annotation(x = 33, y = math.log10(dataFun.doubling_time_equation(pop_th, 33, 7)), 
                        text = 'Doubles every week', font = annotation_style, arrowcolor='DimGray')
    # Text for monthly grow
    fig.add_annotation(x = 35, y = math.log10(dataFun.doubling_time_equation(pop_th, 35, 30)),
                        text = 'Doubles every month', font = annotation_style, arrowcolor='DimGray')

    
    # set chart style and names
    fig.update_yaxes(range=[math.log10(pop_th), math.log10(pop_th)+3.5])
    fig.update_xaxes(range=[0, num_days])
    fig.update_layout(
        yaxis_type="log",
        plot_bgcolor='white', 
        xaxis_title = 'Days',
        yaxis_title = 'Number of cases <br> <sub>Log axe</sub>',
    )
    fig.update_yaxes(showgrid=True, gridwidth=.3, gridcolor='gainsboro')
    #fig.show()
    return fig


# Countries comparison
def disp_countries_comp(df_data, ctry_list, mask=0, plot_type='line'):
    '''Routine to plot countries cases over time so a visual comparison is possible
        df_data:    <dataframe> information from JHU for each case per country over time
        ctry_list:  <list> string list with countries to compare
        mask:       <boolean> vector with period to display, all period by default (0)
        plot_type:  TO BE DONE LATER

    '''
    fig = plotly.graph_objs.Figure()

    ctry_max = 1
    for country in ctry_list:
        # get country timeseries
        ctry_ts = dataFun.get_timeseries_from_JHU(df_data, country, verbose=False)

        if ctry_max < np.max(ctry_ts):
            ctry_max = np.max(ctry_ts)

        # check for time filter
        if mask is 0:
            mask = ctry_ts.index >= ctry_ts.index[0]
        elif type(mask) == str:
            mask = ctry_ts.index >= mask


        if plot_type == 'Bar':
            fig.add_trace(
                plotly.graph_objs.Bar(
                    x = ctry_ts.index[mask],
                    y = ctry_ts[mask], 
                    name = country
                ))

        elif plot_type == 'line':
            fig.add_trace(
                plotly.graph_objs.Scatter(
                    mode = 'lines', #'lines+markers',
                    x = ctry_ts.index[mask],
                    y = ctry_ts[mask], 
                    name = country
                ))
        
    # set background and axis chart style
    fig.update_layout(
        xaxis_title = 'Time [Days]',
        yaxis_title = 'Cases',
        title = 'COVID-19 cases per country' + datetime.datetime.today().strftime(', %B %d, %Y'),
        title_x = 0.5,
        plot_bgcolor='white', 
        yaxis_type="log"
    )

    # display horizontal grid lines
    fig.update_yaxes(showgrid=True, gridwidth=.3, gridcolor='gainsboro')
    fig.update_yaxes(range=[np.log10(np.min(ctry_ts[mask])+1), np.log10(ctry_max)+.5])

    fig.show()


# Generate recoveries and fatalities rates for JHU dataframe source
def disp_country_rates_jhu(ts_case, ts_recov, ts_death, loc_name, mask=0):
    '''Routine to display the evolution of recovery and fatalies rates compare to all cases reported by JHU datasource
        ts_case:    <timeserie> information over time for each case
        ts_recov:   <timeserie> information over time for each recovery
        ts_death:   <timeserie> information over time for each fatality
        loc_name:   <string> name of the location under study
        mask:       <boolean> vector with period to display, all period by default (0)

        '''
    # Check for time filter
    if mask is 0:
        mask = ts_case.index >= ts_case.index[0]

    # Calculate rates faces to total cases diagnosed
    rate_recov = dataFun.safe_div(ts_recov.values, ts_case.values) *100
    rate_death = dataFun.safe_div(ts_death.values, ts_case.values) *100   

    # display rates
    fig = plotly.subplots.make_subplots(rows=2, cols=1)

    fig.add_bar(
        row=1, col=1,
        x = ts_case.index[mask],
        y = rate_recov[mask],
        name = 'Recoveries',
        marker = dict(color = 'darkseagreen', line=dict(color='forestgreen', width=1.5)),
    )
    fig.update_xaxes(title_text="Time [Days]", row=1, col=1)
    fig.update_yaxes(title_text="Percentage [%]", row=1, col=1, showgrid=True, gridwidth=.3, gridcolor='gainsboro')

    fig.add_bar(
        row=2, col=1,
        x = ts_case.index[mask],
        y = rate_death[mask],
        name = 'Fatalities',
        marker = dict(color = 'DimGray', line=dict(color='Black', width=1.5)),
    )
    fig.update_xaxes(title_text="Time [Days]", row=2, col=1)
    fig.update_yaxes(title_text="Percentage [%]", row=2, col=1, showgrid=True, gridwidth=.3, gridcolor='gainsboro')

    fig.update_layout(
        title_text = 'Recovery & Fatalities rates for ' + loc_name + datetime.datetime.today().strftime(', %B %d, %Y'),
        title_x = .5,
        plot_bgcolor='white')
    
    fig.show()



# Generate cumulative graph over time for JHU dataframe source
def disp_cum_jhu(ts_case, ts_recov, ts_death, loc_name, mask=0):
    '''Routine to display the normal/log tendency of the cumulated cases for JHU datasource only
        ts_case:    <timeserie> information over time for each case
        ts_recov:   <timeserie> information over time for each recovery
        ts_death:   <timeserie> information over time for each fatality
        loc_name:   <string> name of the location under study
        mask:       <boolean> vector with period to display, default=0 all period

        '''
    # Check for time filter
    if mask is 0:
        mask = ts_case.index >= ts_case.index[0]

    # Build plot for basic data display
    fig = plotly.graph_objs.Figure()
    # diagnosed cases
    fig.add_trace(
        plotly.graph_objs.Scatter(
            mode='lines+markers',
            x=ts_case.index[mask], 
            y=ts_case[mask],  
            name = 'All cases',
            marker=dict(color='CornflowerBlue')
    ))
    # recover cases
    fig.add_trace(
        plotly.graph_objs.Scatter(
            mode='lines+markers',
            x=ts_recov.index[mask], 
            y=ts_recov[mask],
            name = 'Recover',
            marker=dict(color='forestgreen')
    ))
    # death cases
    fig.add_trace(
        plotly.graph_objs.Scatter(
            mode='lines+markers',
            x=ts_death.index[mask], 
            y=ts_death[mask],  
            name = 'Fatalities',
            marker=dict(color='black')
    ))

    if ts_case.max() > 100:
        fig.update_layout(yaxis_title = 'Cases [Log]', yaxis_type="log")
    else:
        fig.update_layout(yaxis_title = 'Cases')

    fig.update_layout(
        xaxis_title = 'Time [Days]',
        plot_bgcolor='white',  
        title = 'Current situation in ' + loc_name + datetime.datetime.today().strftime(', %B %d, %Y'),
        title_x = .5
    )

    fig.update_yaxes(showgrid=True, gridwidth=.3, gridcolor='gainsboro')
    
    fig.show()


# Generate a graph in original axis with current active cases
def disp_daily_cases(df_data, loc_name, df_source='JHU', mask=None, trend=False):
    '''Display daily cases evolution for confirmed & fatalities for two different data sources. 
        df_data:    <dataframe> daily information per case
        loc_name:   <string> name of the location under study
        df_source:  <string> select the type of dataframe source
        trend: display a trend line for each plot (default: False)
        
        '''
    if df_source == 'SPF':
        # time vector
        date_time = pd.DataFrame(index=df_data.date).index

        # Daily cases       
        cases_d = np.insert(np.diff(df_data.cas_confirmes).clip(0), 0, 0)

        # daily fatalities
        fatal_d = np.insert(np.diff(df_data.deces).clip(0) ,0, 0)

        # daily recov
        recov_d = 0
    
    elif df_source is 'datagouv':
        # time vector
        date_time = pd.DataFrame(index=df_data.date).index
        # Daily cases       
        cases_d = np.insert(np.diff(df_data.total_cas_confirmes).clip(0), 0, 0)
        # daily fatalities
        fatal_d = np.insert(np.diff(df_data.total_deces_hopital).clip(0) ,0, 0)
        recov_d = 0 # daily recov


    elif df_source == 'JHU':
        # time vector
        date_time = df_data.index

        # Daily cases
        cases_d = np.insert(np.diff(df_data.cases).clip(0), 0, 0)

        # Daily fatalities
        fatal_d = np.insert(np.diff(df_data.death).clip(0) ,0, 0)

        # Daily recovery
        recov_d = np.insert(np.diff(df_data.recov).clip(0) ,0, 0)

    else:
        print('Error: Not valid value for df_source')
        return

    # Check for time filter
    if mask is None:
        mask = date_time >= date_time[0]

    # Build plot for daily variation
    fig = plotly.graph_objs.Figure()

    # daily cases
    fig.add_trace(
        plotly.graph_objs.Bar(
            x = date_time[mask],
            y = cases_d[mask],
            marker = dict(color = 'CornflowerBlue', line = dict(color = 'DarkBlue', width=1.5)),
            name = 'Cases'
    ))
    if trend:
        cases_trend = dataFun.mov_avg(cases_d[mask], 7)
        fig.add_trace(
        plotly.graph_objs.Scatter(
            x = date_time[mask],
            y = cases_trend,
            line = dict(color = 'CornflowerBlue', width=1.5),
            name = 'Trend cases, 7 days mean'
        ))

    # daily fatalities
    fig.add_trace(
        plotly.graph_objs.Bar(
            x = date_time[mask],
            y = fatal_d[mask],
            marker = dict(color = 'DimGray', line = dict(color = 'Black', width=1.5)),
            name = 'Fatalities'
    ))

    if df_source is 'JHU': # exclude SPF
        # daily recoveries
        fig.add_trace(
            plotly.graph_objs.Bar(
                x = date_time[mask],
                y = recov_d[mask],
                marker = dict(color = 'DarkSeaGreen', line = dict(color = 'ForestGreen', width=1.5)),
                name = 'Recoveries'
        ))

    fig.update_layout(
        plot_bgcolor='white', 
        #barmode = 'stack',
        xaxis_title = 'Time [Days]',
        yaxis_title = 'Cases',
        title = 'Daily progression in ' + loc_name + datetime.datetime.today().strftime(', %B %d, %Y'),
        title_x = .5
    )
    fig.update_yaxes(showgrid=True, gridwidth=.3, gridcolor='gainsboro')

    fig.show()


# Generate a graph in original axis with current active cases
def disp_current_cases(df_data, loc_name, pop_factor=1, source=None):
    '''Display current cases from cumulative and fatalities 
        df_data:    <dataframe> daily information per case
        loc_name:   <string> name of the location under study
        pop_factor: <integer> mutiplicative factor for yaxis chart
        
        '''
    # Calculate current cases from confimed & fatalities
    if source is 'datagouv':
        fat_c = np.array(df_data.total_deces_hopital, dtype=int)
        fat_c[fat_c<0] = 0
        liv_c = np.array(df_data.total_cas_confirmes, dtype=int) - fat_c

    else:        
        fat_c = np.array(df_data.deces, dtype=int)
        fat_c[fat_c<0] = 0
        liv_c = np.array(df_data.cas_confirmes, dtype=int) - fat_c
    
    # Build plot for basic data display
    fig = plotly.graph_objs.Figure()
    # Confirmed cases
    fig.add_trace(
        plotly.graph_objs.Bar(
            x=df_data.date, 
            y=np.maximum(0, liv_c) / pop_factor,  
            name = 'On going cases',
            marker=dict(color='CornflowerBlue')
    ))
    # Fatalities
    fig.add_trace(
        plotly.graph_objs.Bar(
            x=df_data.date, 
            y=np.maximum(0, fat_c) / pop_factor, 
            name = 'Fatalities',
            marker=dict(color='Black')
    ))

    fig.update_yaxes(showgrid=True, gridwidth=.3, gridcolor='gainsboro')
    #title= 'Cases, 1 by ' + str(pop_factor) + ' peoples'

    if pop_factor == 1: 
        fig.update_yaxes(title = 'Number of confirmed cases')
    else:
        fig.update_yaxes(title = 'Number of confirmed cases <br>  <sub>factor of 1 by ' + str(format(pop_factor, ",").replace(",", ".")) +' peoples</sub>')
    
    fig.update_layout(
        barmode = 'stack',
        xaxis_title = 'Time [Days]',
        plot_bgcolor='white',  
        title = 'Current active cases in ' + loc_name + datetime.datetime.today().strftime(', %B %d, %Y'),
        title_x = .5
    )
    fig.show()


# Generate a cumulative chart for SPF datasets
def disp_cumulative(df_data, loc_name, pop_factor=1, source=None):
    '''Routine to display the normal/log tendency of the cumulated cases
        df_data:        <dataframe> information over time for each case
        loc_name:     <string> name of the location under study
        pop_factor:     <int> multiplicative factor for number of cases
                        default value 1, for other values is display in the 
                        vertical axis the multiplicative magnitude
        
        '''
    if source is 'datagouv':
        cases_cum = df_data.total_cas_confirmes
        fatalities = df_data.total_deces_hopital
    else:
        cases_cum = df_data.cas_confirmes
        fatalities = df_data.deces

    fig = plotly.graph_objs.Figure()
    # add scatter chart for confirmed cases
    fig.add_trace(
        plotly.graph_objs.Scatter(
            mode = 'lines+markers',
            #mode = 'markers',
            x=df_data.date, 
            y=cases_cum,  
            name = 'Confirmed cases',
            marker=dict(color='CornflowerBlue'),
    ))
    # add scatter chart for fatalities
    fig.add_trace(
        plotly.graph_objs.Scatter(
            mode='lines+markers',
            #mode='markers',
            x=df_data.date, 
            y=fatalities, 
            name = 'Fatalities',
            marker=dict(color='black')
    ))

    fig.update_layout(yaxis_title = 'Cases [Log]', yaxis_type="log")

    fig.update_layout(
        xaxis_title = 'Time [Days]',
        plot_bgcolor='white',  
        title = 'Current status in ' + loc_name + ' , ' + datetime.datetime.today().strftime('%B %d, %Y'),
        title_x = .5
    )
    fig.update_yaxes(showgrid=True, gridwidth=.3, gridcolor='gainsboro')

    fig.show()
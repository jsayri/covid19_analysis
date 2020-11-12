# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import plotly
import datetime
import math

# import local functions
import covid19_analysis.dataFun as dataFun


from covid19_analysis import __version__

__author__ = "J SAYRITUPAC"
__copyright__ = "J SAYRITUPAC"
__license__ = "mit"


# Function library to treat and plot results that comes from
# the french dataset. Variables and graph are adapted to those
# dataframes.

# Report daily evolution at hospital for one department
def disp_dep_hosp(df_donnes, nom_dep):
    '''
    Display daily evolution at deparment hospital
    '''
    fig = plotly.graph_objs.Figure()
    # Ajout trace des cas d'hospitalisation
    fig.add_trace(
        plotly.graph_objs.Scatter(
            mode = 'lines+markers',
            x=df_donnes.jour, 
            y=df_donnes.hosp,  
            name = 'Hospitalisation',
            marker=dict(color='CornflowerBlue'),
    ))
    # Ajout trace des décès
    fig.add_trace(
        plotly.graph_objs.Scatter(
            mode='lines+markers',
            x=df_donnes.jour, 
            y=df_donnes.dc, 
            name = 'Décès ',
            marker=dict(color='black')
    ))
    # Ajout trace des cas en reanimation
    fig.add_trace(
        plotly.graph_objs.Scatter(
            mode='lines+markers',
            x=df_donnes.jour, 
            y=df_donnes.rea, 
            name = 'Réanimation  ',
    ))
    # Ajout trace des retours a domicile
    fig.add_trace(
        plotly.graph_objs.Scatter(
            mode='lines+markers',
            x=df_donnes.jour, 
            y=df_donnes.rad, 
            name = 'Retour à domicile',
    ))

    fig.update_layout(yaxis_title = 'Nombre de personnes ')

    fig.update_layout(
        xaxis_title = 'Temps [Jours]',
        plot_bgcolor='white',  
        title = 'État actuel, ' + nom_dep + ' , ' + datetime.datetime.today().strftime('%B %d, %Y'),
        title_x = .5
    )
    fig.update_yaxes(showgrid=True, gridwidth=.3, gridcolor='gainsboro')

    fig.show()

    # display current cases in hospital divided by age
def disp_regions_comp(df_data, y_log=False):
    ''' Display cases per region
        df_data:    <dataframe> daily hospitalizations for regions and age categorie    
    '''
    dict_regions_code = {'84' : 'Auvergne-Rhône-Alpes', '27' : 'Bourgogne-Franche-Comté', '53' : 'Bretagne',
                     '24' : 'Centre-Val de Loire', '94' : 'Corse', '44' : 'Grand Est', 
                     '32' : 'Hauts-de-France', '11' : 'Île-de-France', '28' : 'Normandie', 
                     '75' : 'Nouvelle-Aquitaine', '76' : 'Occitanie', '52' : 'Pays de la Loire', 
                     '93' : "Provence-Alpes-Côte d'Azur", '01' : 'Guadeloupe', '02' : 'Martinique', 
                     '03' : 'Guyane', '04' : 'La Réunion', '06' : 'Mayotte'}

    fig = plotly.graph_objs.Figure()

    for reg_code in df_data.reg.unique():
        # select one region df for all ages
        df_reg = df_data[(df_data['reg'] == reg_code) & (df_data['cl_age90'] == 0)]

        # define label
        label_name = dict_regions_code.get('{:02}'.format(reg_code))

        fig.add_trace(
            plotly.graph_objs.Scatter(
                mode = 'lines',
                name = label_name,
                x = df_reg.jour,
                y = df_reg.hosp,
                line=dict(width = 1.5),
            )
        )
    
    if y_log:
        fig.update_layout(yaxis_title = 'Cases [Log]', yaxis_type="log")

    fig.update_layout(
        plot_bgcolor='white', 
        xaxis_title = 'Dates [Days]',
        yaxis_title = 'Cases',
        title = 'Hospitalization par region, evolution' + datetime.datetime.today().strftime(', %B %d, %Y'),
        title_x = .5
    )
    
    fig.show()

# Generate a cumulative chart
def disp_cumulative(df_data, loc_name, pop_factor=1, source='datagouv'):
    '''Routine to display the normal/log tendency of the cumulated cases
        df_data:        <dataframe> information over time for each case
        loc_name:       <string> name of the location under study
        pop_factor:     <int> multiplicative factor for number of cases
                        default value 1, for other values is display in the 
                        vertical axis the multiplicative magnitude
        
        '''
    if source is 'datagouv':
        dates = df_data.jour
        cases_cum = df_data.hosp
        fatalities = df_data.dc
    else:
        cases_cum = df_data.cas_confirmes
        fatalities = df_data.deces

    fig = plotly.graph_objs.Figure()
    # add scatter chart for confirmed cases
    fig.add_trace(
        plotly.graph_objs.Scatter(
            mode = 'lines+markers',
            #mode = 'markers',
            x=dates, 
            y=cases_cum,  
            name = 'Confirmed cases',
            marker=dict(color='CornflowerBlue'),
    ))
    # add scatter chart for fatalities
    fig.add_trace(
        plotly.graph_objs.Scatter(
            mode='lines+markers',
            #mode='markers',
            x=dates, 
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


# Generate a graph in original axis with current active cases
def disp_daily_cases(df_data, loc_name, df_source='datagouv', trend=False):
    '''Display daily cases evolution for confirmed & fatalities. 
        df_data:    <dataframe> daily information per case
        loc_name:   <string> name of the location under study
        df_source:  <string> select the type of dataframe source
        trend: display a trend line for each plot (default: False)
        
        '''
    if df_source is 'datagouv':
        # time vector
        date_time = pd.DataFrame(index=df_data.jour).index
        # Daily cases       
        cases_d = np.insert(np.diff(df_data.hosp).clip(0), 0, 0)
        # daily fatalities
        fatal_d = np.insert(np.diff(df_data.dc).clip(0) ,0, 0)
        recov_d = 0 # daily recov

    else:
        print('Error: Not valid value for df_source')
        return

    # Build plot for daily variation
    fig = plotly.graph_objs.Figure()

    # daily cases
    fig.add_trace(
        plotly.graph_objs.Bar(
            x = date_time,
            y = cases_d,
            marker = dict(color = 'CornflowerBlue', line = dict(color = 'DarkBlue', width=1.5)),
            name = 'Cases'
    ))
    if trend:
        cases_trend = dataFun.mov_avg(cases_d, 7)
        fig.add_trace(
        plotly.graph_objs.Scatter(
            x = date_time,
            y = cases_trend,
            line = dict(color = 'CornflowerBlue', width=1.5),
            name = 'Trend cases, 7 days mean'
        ))

    # daily fatalities
    fig.add_trace(
        plotly.graph_objs.Bar(
            x = date_time,
            y = fatal_d,
            marker = dict(color = 'DimGray', line = dict(color = 'Black', width=1.5)),
            name = 'Fatalities'
    ))
    if trend:
        fatal_trend = dataFun.mov_avg(fatal_d, 7)
        fig.add_trace(
        plotly.graph_objs.Scatter(
            x = date_time,
            y = fatal_trend,
            line = dict(color = 'DimGray', width=1.5),
            name = 'Trend fatalities, 7 days mean'
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
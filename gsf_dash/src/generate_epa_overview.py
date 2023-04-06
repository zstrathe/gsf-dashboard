## TODO implement a way of plotting this EPA overview as an additional page to the PDF output implemented in main()
    def plot_epa(self, ponds_data: dict, save_output=True, plot_title='Pond EPA Overview') -> str: # returns output file name as a string 
        def subplot_ax_format(subplot_ax, fill_color):
                subplot_ax.set_facecolor(fill_color)
                subplot_ax.spines['top'].set_visible(False)
                subplot_ax.spines['right'].set_visible(False)
                subplot_ax.spines['bottom'].set_visible(False)
                subplot_ax.spines['left'].set_visible(False)
                subplot_ax.get_xaxis().set_ticks([])
                subplot_ax.get_yaxis().set_ticks([])
                fig.add_subplot(subplot_ax)

        # function to plot each pond to ensure that data for each plot is kept within a local scope
        def plot_each_pond(ponds_data, fig, pond_plot, pond_name):
            inner_plot = gridspec.GridSpecFromSubplotSpec(5,3,subplot_spec=pond_plot, wspace=0, hspace=0)
            
            # get pond active status (True or False) from ponds_active_status dict
            pond_active = self.ponds_active_status[pond_name]
            
            # get data for individual pond only, as a list
            # each list entry should consist of a tuple containing the date and the epa value, sorted in descending order by date
            single_pond_data = list(ponds_data[pond_name]['epa_data'].items())
            
             # check and update the title_date global variable if this pond has a more recent date
            if len(single_pond_data) != 0:
                nonlocal title_date
                if single_pond_data[0][0] > title_date:
                    title_date = single_pond_data[0][0]
            
            # title subplot
            title_ax = plt.Subplot(fig, inner_plot[0,:])
            title_str = r'$\bf{' + pond_name + '}$' 
            if len(ponds_data[pond_name]['source']) > 0:
                title_str += '\nsource: ' + ponds_data[pond_name]['source']
            title_ax.text(0.5, 0.1, title_str , ha = 'center', va='bottom', fontsize='large')
            subplot_ax_format(title_ax, 'white')

            # epa data subplots
            epa_ax = plt.Subplot(fig, inner_plot[1:3,:])
            
            if pond_active:
                # set fill color based on latest EPA reading
                if len(single_pond_data) > 0:
                    # set lastest_epa_data as the first value in single_pond_data (which should be reverse sorted)
                    latest_epa_data = single_pond_data[0][1] 
                    fill_color_dict = {(0,0.0199999999):'red', 
                                       (0.02,0.0349999999): 'yellow',
                                       (0.035,99999999999): 'mediumspringgreen'}
                    for idx, (key, val) in enumerate(fill_color_dict.items()):
                        if latest_epa_data >= key[0] and latest_epa_data < key[1]:
                            fill_color = val 
                else:
                    # set fill color as light grey if there is no EPA data for the pond
                    fill_color = 'lightgrey'

                if len(single_pond_data) > 0:
                    # set the center-point of each data point on the subplot, depending on number of data points available
                    if len(single_pond_data) == 1:
                        text_x = [0.5]
                    elif len(single_pond_data) == 2:
                        text_x = [0.3, 0.7]
                    else:
                        text_x = [0.195, 0.5, 0.805]

                    for idx, item in enumerate(single_pond_data):
                        text_date_formatted =  item[0].strftime("%-m/%-d/%y")
                        epa_ax.text(text_x[idx], 0.7, text_date_formatted, ha='center', va='center')
                        text_epa_data_formatted =  f'{item[1]*100: .2f}%'
                        epa_ax.text(text_x[idx], 0.3, text_epa_data_formatted, ha='center', va='center', fontsize='large', weight='bold')
                else:
                    epa_ax.text(0.5, 0.5, 'Active but no data', ha='center', va='center')
            else: # if pond is inactive
                fill_color = 'whitesmoke'
                epa_ax.text(0.5, 0.5, 'Inactive', ha='center', va='center')
            
            subplot_ax_format(epa_ax, fill_color)
            
            # epa change subplot
            epa_pct_ax = plt.Subplot(fig, inner_plot[3:,:])
            
            if pond_active:
            # Get the % change in EPA values if there is >1 entry (use the first and last indexes of the single_pond_data list)
                if len(single_pond_data) != 0:
                    epa_pct_fill = 'xkcd:light grey'    

                    def calc_epa_pct_chg(data_curr: list, data_prev: list):
                        '''
                        inputs:
                            data_curr: [datetime value, epa float value]
                            data_prev: [datetime value, epa float value]

                        output:
                            pct_chg: calculated absolute change in percentage rounded to 2 decimal places and formatted with % character (str)
                            delta_days: number of days between readings (int)
                            pct_format_color: color for displaying percentage (str)
                        '''
                        delta_days = (data_curr[0] - data_prev[0]).days    

                        pct_chg = (data_curr[1] - data_prev[1]) * 100
                        if pct_chg > 0:
                            pct_format_color = 'xkcd:emerald green'
                            pct_chg = f'+{pct_chg:.2f}%'
                        elif pct_chg < 0: 
                            pct_format_color = 'xkcd:fire engine red'
                            pct_chg = f'{pct_chg:.2f}%'
                        else:
                            pct_format_color = 'black'
                            pct_chg = f'{pct_chg:.2f}%'
                        return [pct_chg, delta_days, pct_format_color]
                        # else:
                        #     return ['n/a', delta_days, 'black']

                    if len(single_pond_data) == 2:
                        epa_pct_chg, delta_days, epa_pct_color = calc_epa_pct_chg(single_pond_data[0], single_pond_data[1])
                        text_formatted1 =  f'Change ({delta_days} day{"s" if delta_days > 1 else ""}):' 
                        epa_pct_ax.text(0.5, 0.7, text_formatted1, ha='center', va='center', fontsize='large')
                        epa_pct_ax.text(0.5, 0.3, epa_pct_chg, ha='center', va='center', fontsize='large', color=epa_pct_color, weight='bold')
                    elif len(single_pond_data) == 3:
                        epa_pct_chg1, delta_days1, epa_pct_color1 = calc_epa_pct_chg(single_pond_data[0], single_pond_data[1])
                        epa_pct_chg2, delta_days2, epa_pct_color2 = calc_epa_pct_chg(single_pond_data[0], single_pond_data[2])
                        text_formatted1 =  f'Change ({delta_days1} day{"s" if delta_days1 > 1 else ""}, {delta_days2} days):'
                        epa_pct_ax.text(0.5, 0.7, text_formatted1, ha='center', va='center', fontsize='large')
                        epa_pct_ax.text(0.3, 0.3, epa_pct_chg1, ha='center', va='center', fontsize='large', color=epa_pct_color1, weight='bold')
                        epa_pct_ax.text(0.7, 0.3, epa_pct_chg2, ha='center', va='center', fontsize='large', color=epa_pct_color2, weight='bold')
                    else: # if there is only one data point so no percentage change
                        epa_pct_ax.text(0.5, 0.7, 'Change:', ha='center', va='center', fontsize='large')
                        epa_pct_ax.text(0.5, 0.3, 'n/a', ha='center', va='center', fontsize='large')

                else: # when there is no data for this pond
                    epa_pct_fill = 'lightgrey'

            else: # if pond is inactive
                epa_pct_fill = 'whitesmoke'
                
            subplot_ax_format(epa_pct_ax, epa_pct_fill) 

        ##############################
        ### START OF PLOTTING CODE ###
        ##############################

        n_rows_small = 12
        n_rows_large = 10
        n_cols_small = 4
        n_cols_large = 2

        # generate labels for ponds in a list (with each row being a sublist)
        title_labels = []
        for row in range(1,n_rows_small+1):
            title_labels.append([])
            for col in range(1,n_cols_small+1):
                if row < 10:
                    title_labels[row-1].append(f'0{row}0{col}')  
                else:
                    title_labels[row-1].append(f'{row}0{col}')  
        for idx_row, row in enumerate(title_labels):
            if idx_row < n_rows_large:
                for col in [6,8]:
                    if idx_row+1 < 10:
                        row.append(f'0{idx_row+1}0{col}')
                    else:
                        row.append(f'{idx_row+1}0{col}')
            else: # append blanks for the two large columns with only 10 rows 
                [row.append(f'BLANK {idx_row+1}-{c}') for c in [6,8]]

        # Add 2 blank rows at end to make room for extra data aggregations to be listed        
        #[title_labels.append([f'BLANK {r}-{c}' for c in range(1,7)]) for r in range(13,15)]

        # flatten title_labels for indexing from plot gridspec plot generation
        flat_title_labels = [label for item in title_labels for label in item]
        
        # initialize title_date as a global variable to track the most recent date to print in the title
        title_date = datetime.min
        
        # Initialize main plot
        plt_width = 8.5
        plt_height = 11
        scale_factor = 2.5
        fig = plt.figure(figsize=(plt_width*scale_factor, plt_height*scale_factor))
        
        outer_plots = gridspec.GridSpec(len(title_labels), len(title_labels[0]), wspace=0.05, hspace=0.2)

        for idx_plot, pond_plot in enumerate(outer_plots):
            pond_name = flat_title_labels[idx_plot]
            if 'BLANK' not in pond_name:

                # plot each pond with a function to ensure that data for each is isolated within a local scope
                plot_each_pond(ponds_data, fig, pond_plot, pond_name)

            else: # for plotting subplots labeled 'BLANK' in 'title_labels' list:  (the four lower right subplots) and the BLANK rows
                ax = plt.Subplot(fig, pond_plot)
                ax.axis('off')
                if 'BLANK 11-6' in pond_name: # plot the color key in the first blank subplot
                    legend_text = (
                                    r'$\bf{Color\/\/key\/\/(latest\/\/EPA\/\/reading)}$:' "\n"
                                    "0% - 1.99%:      Red\n"
                                    "2% - 3.49%:      Yellow\n"
                                    "3.5% and up:    Green\n"
                                    "No EPA data:     Grey"
                                   )
                    t = ax.text(0.1,0.8,legend_text,ha='left', va='top', fontsize = 'x-large',
                           bbox=dict(facecolor='xkcd:bluegrey', alpha=0.5), multialignment='left')
                if 'BLANK 12-6' in pond_name: # plot the legend in the first blank subplot    
                    pass
                elif 'BLANK 13-1' in pond_name:
                    pass
                elif 'BLANK 13-3' in pond_name:
                    pass            
                elif 'BLANK 14-1' in pond_name:
                    pass
                elif 'BLANK 14-3' in pond_name:
                    pass

                fig.add_subplot(ax) # add the subplot for the 'BLANK' entries
        
        title_date_formatted = title_date.strftime('%-m/%-d/%Y')
        fig.suptitle(f'{plot_title}\nas of {title_date_formatted}', fontweight='bold', fontsize=16, y=0.905)
        
        out_filename = f'./output_files/{plot_title} {title_date.strftime("%Y-%m-%d")}.pdf'
        if save_output == True:
            plt.savefig(out_filename, bbox_inches='tight')     
        fig.show() 
        return out_filename
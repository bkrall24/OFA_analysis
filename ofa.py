import pandas as pd
import numpy as np
import streamlit as st
import io
import re


def parse_dict(txt):
    """ Parses list of strings with format string 1: string 2 as key,value pairs in a dict

    Args:
        txt (list of str)

    Returns:
        dict, all possible key, value pairs from txt 
    """

    data = [x for x in txt if ':' in x] # pull out lines containing a colon
    dict = {x.split(':',1)[0].strip(): x.split(':',1)[1].strip() for x in data} # splits each line of data at the colon, removes whitespace

    # Check for situations where the subject ID is formmatted as C##M## and extract second number as int
    if "Subject ID" in dict.keys():
        sid = dict['Subject ID']
        pattern = re.compile(r'^C(\d+)M(\d+)$')
        match = pattern.match(sid)
        if match:
            dict['Subject ID']= int(match.group(2))
        
    return dict

def match_and_rename(result, keys):
    """ Summary files have abbreviations in the table and full names in the totals. This matches the abbreviations to names
    
    Args:
        result (pandas DataFrame): column names are measurements to be renamed
        keys (dict keys, list of str): measurement names from totals
    
    """
    for c in result.columns:
        characters_to_match = set(char for char in c if char != '.' and char != 's')
        new_col = [x for x in keys if characters_to_match.issubset(set(char for char in x))]
        
        if len(new_col) > 0:
            result.rename(columns={c: new_col[0]}, inplace=True)
    
    # return result

def parse_time(time_str):
    """ Change strings to TimeDelta format

    Args:
        time_str (str): format mm:ss
    
    Returns:
        timedelta 
    """

    minutes, seconds = time_str.split(':')
    minutes = int(minutes)
    seconds = float(seconds)
    return pd.Timedelta(minutes=minutes, seconds=seconds)

def cast_numeric(data, keys):

    """ Reformat dict or pandas dataframe to represent data as numeric
    Args:
        data (pandas DataFrame or dict): data to be made numeric
        keys (dict keys, pd columns, or list of str): list of data keys/columns to be cast as numeric
    """
    for c in keys:
        if 'Time' in c:
            if type(data[c]) is pd.Series:
                data[c] = data[c].apply(parse_time)
            else:
                data[c] = parse_time(data[c])

            if c == 'Session Time':
                if type(data[c]) is pd.Series:
                    data[c] = data[c].dt.total_seconds() / 60
                else:
                    data[c] = data[c].total_seconds() / 60
            else:
                if type(data[c]) is pd.Series:
                    data[c] = data[c].dt.total_seconds()
                else:
                    data[c] = data[c].total_seconds()
        else:
            data[c] = pd.to_numeric(data[c], errors= 'coerce')

def parse_summary_file(example_file):
    # with open(example_file, 'r') as file:
    #     lines = file.readlines()
    lines = [line.decode('utf-8') for line in example_file.readlines()]  # reads each line of uploaded file

    sum_start = [ind for (ind,x) in enumerate(lines) if 'Printed' in x]

    summaries = []
    for ind in range(len(sum_start)):
        if ind < len(sum_start)-1:
            single_sum = lines[sum_start[ind]-2: sum_start[ind+1]-2]
        else:
            single_sum = lines[sum_start[ind]-2:len(lines)]

        summaries.append(single_sum)
    
    return summaries

def parse_txt_table(head, c):
    h2 = [[x for x in q.strip().split(' ') if x != '' ] for q in head]
    header = []
    for a,b in zip(h2[0], h2[1]):
        header.append(a+" "+ b)   
        
    c2 = [[x for x in q.strip().split(' ') if x != '' ] for q in c]
    df = pd.DataFrame(c2, columns = header)
 
    return df


def parse_single_summary(summ):

    end_meta = [ind for ind,x in enumerate(summ) if 'Detail Reporting Mode' in x]
    meta_dict = parse_dict(summ[0:end_meta[0]+1])
    dat = summ[end_meta[0]+1:]
    data_less = [True for x in dat if "No Block information found" in x]
    
    
    if len(data_less) > 0:
        return None, meta_dict, None


    else:
        tbl1_start = [ind for ind,x in enumerate(dat) if 'Dist.   Time' in x]
        tbl2_start = [ind for ind,x in enumerate(dat) if 'Jump    Jump' in x]
        totals_start = [ind for ind,x in enumerate(dat) if 'Distance Traveled' in x]

        tbl1 = parse_txt_table(dat[tbl1_start[0]: tbl1_start[0]+2], dat[tbl1_start[0]+3:tbl2_start[0]-1])
        tbl2 = parse_txt_table(dat[tbl2_start[0]: tbl2_start[0]+2], dat[tbl2_start[0]+3:totals_start[0]-3])
        totals = parse_dict(dat[totals_start[0]:-1])

        result = pd.merge(tbl1, tbl2, on = 'Session Time', how = 'inner')

        match_and_rename(result, totals.keys()) # rename columns 
        result = result[['Session Time'] + [col for col in result.columns if col != 'Session Time']] # pull Session Time as first column

        cast_numeric(result, result.columns) # cast data as numeric
        cast_numeric(totals, totals.keys())
        

        return result, meta_dict, totals


def compile_summary_files(files):
    """ Compiles multiple summary files together
    
    Args:
        files (list): List of fileUploader objects of .summary files
    
    Returns:
        dat (dict): tables, totals and meta data combined
    """
    metas = []
    totals = []
    results = []

    for f in files:
        try:
            
            summaries = parse_summary_file(f)
            if len(summaries) > 1:
                # st.write("Multiple datasets found in "+ f.name)
                data_choice = st.multiselect("Multiple datasets found, which do you want to analyze in "+f.name+" ?", options = np.arange(1,len(summaries)+1))
            else:
                data_choice = [1]

            for ind,x in enumerate(summaries):
                if ind+1 in data_choice:
                    r,m,t = parse_single_summary(x)
                    if r is not None:
                        metas.append(m)
                        totals.append(t)
                        results.append(r)
                    if r is None:
                        st.warning("Dataset "+str(ind+1)+ " in "+f.name +" was aborted, no data loaded")

        except:
            st.warning(f.name+" could not be loaded correctly")

    if len(metas)> 0:
        all_meta = pd.DataFrame(metas)
        durations = np.unique(all_meta['Session Time (min)'].values.astype(int))

        if len(durations)> 1:
            st.warning("Summary files with different session durations loaded. This might cause errors in final data output")
        
        empty_group = all_meta['Group ID'] == ''
        if empty_group.any():
            empty_experiments = all_meta['Experiment ID'].loc[empty_group]
            for x in empty_experiments:
                st.warning(x+" has no Group ID")
            
            st.error("Must have group IDs for each experiment - fix data and try again")


        empty_group = all_meta['Subject ID'] == ''
        if empty_group.any():
            empty_experiments = all_meta['Experiment ID'].loc[empty_group]
            for x in empty_experiments:
                st.warning(x+" has no Subject ID")
            
            st.error("Must have subject IDs for each experiment - fix data and try again")
        # add something to check to see if group ID and subject ID are all entered

        all_totals = pd.DataFrame(totals) 
        all_totals.index= pd.MultiIndex.from_arrays([all_meta['Group ID'], all_meta['Subject ID']])
        # all_totals.index= pd.MultiIndex.from_arrays([all_meta['Group ID'].astype(int), all_meta['Subject ID']])

        tbls = {}
        for col in all_totals.columns:
            dat = compile_measurement(col, results, metas)
            tbls[col] = dat.sort_index(level='Group', axis=1)

        all_totals = table_to_dict(all_totals)
        dat = {'tables': tbls, 'totals': all_totals, 'meta': all_meta}
    
        return dat
    else:
        st.error("No data loaded")
        return None

# def parse_summary(example_file):
#     """ Parses .summary files to extract data
    
#     Args:
#         example_file (UploadedFile): file uploaded into streamlit
    
#     Returns:
#         result (pandas DataFrame): contains binned data from summary
#         meta_dict (dict): contains meta data
#         totals (dict): contains totals
#     """

#     # with open(example_file, 'r') as file:
#     #     lines = file.readlines()
#     # lines = example_file.readlines() 

   
#     lines = [line.decode('utf-8') for line in example_file.readlines()]  # reads each line of uploaded file
#     meta_data = lines[:36] # pulls first 37 lines as metadata
#     meta_dict = parse_dict(meta_data) # parses metadata as a dictionary
 
#     tablular_data = lines[37:] #pulls remaining lines as data
    
#     # stops = [x for x,y in enumerate(tablular_data) if len(y) == 2] # identifies new lines as 2 char lines (not ideal)
#     stops = [ind for ind,line in enumerate(tablular_data) if line.strip() == '']

#     totals = tablular_data[stops[-2]:stops[-1]] # assumes 'totals' data will be between that second to last two new lines (file ends with two)
#     totals = parse_dict(totals)
#     breaks = [x for x,y in enumerate(tablular_data) if '=' in y] #identifies breaks in remaining table as separated by === lines
#     breaks.append(len(tablular_data)) 

#     # Iterate through new lines --> equal sign containing lines to identify chunks of data
#     frames =[]
#     for a,b in zip(breaks, stops): 
#         c = tablular_data[a+1:b] # data past the equal sign line to new line
#         head = tablular_data[a-2:a] # two lines prior to equal sign line 

#         # Parse the lines as strings, save in DataFrame
#         if len(c) > 0:
#             h2 = [[x for x in q.strip().split(' ') if x != '' ] for q in head]
#             header = []
#             for a,b in zip(h2[0], h2[1]):
#                 header.append(a+" "+ b)   
            
#             c2 = [[x for x in q.strip().split(' ') if x != '' ] for q in c]
#             df = pd.DataFrame(c2, columns = header)
#             frames.append(df)

#     # Merge all tables using the "Session Time" column
#     result = frames[0]
#     for df in frames[1:]:
#         result = pd.merge(result, df, on = 'Session Time', how = 'inner')

    
#     match_and_rename(result, totals.keys()) # rename columns 

#     result[['Session Time'] + [col for col in result.columns if col != 'Session Time']] # pull Session Time as first column
    
#     cast_numeric(result, result.columns) # cast data as numeric
#     cast_numeric(totals, totals.keys())
#     # cast_numeric(meta, )

#     return result, meta_dict, totals

def compile_measurement(measurement, results, metas):
    """ Compiles tables of binned data by subject and group id from meta data

    Args:
        measurement (str): Column of results data
        results (list of pandas DataFrame): list of all dataframes of binned data to be combined
        metas (list of dicts) : list of dataframes of metadata to pull group and subject number from
    
    Returns:
        dataframe of binned data foe measurement passed
    """

    columns = pd.MultiIndex(levels=[[], []], codes=[[], []], names=[ 'Group', 'Subject'])
    df = pd.DataFrame(columns=columns)

    for experiment, meta in zip(results, metas):
        # experiment_name = meta["Experiment ID"]
        # group_number = int(meta["Group ID"])
        group_number = (meta["Group ID"])
        subject_id = meta["Subject ID"]
        column_index = (group_number, subject_id)
        df[column_index] = experiment[measurement]
        
    df.index = np.array(results[0]['Session Time'])

    return df

def table_to_dict(table):
    """Transforms a table to a dictionary"""
    table_dict = {}
    for col in table.columns:
        table_dict[col] = table[col].transpose()
    
    return table_dict

# def compile_summary(files):
#     """ Compiles multiple summary files together
    
#     Args:
#         files (list): List of fileUploader objects of .summary files
    
#     Returns:
#         dat (dict): tables, totals and meta data combined
#     """
#     metas = []
#     totals = []
#     results = []

#     for f in files:
#         try:
#             r,m,t = parse_summary(f)
#             metas.append(m)
#             totals.append(t)
#             results.append(r)
#         except:
#             st.warning(f.name+" could not be loaded correctly")

#     all_meta = pd.DataFrame(metas)

#     durations = np.unique(all_meta['Session Time (min)'].values.astype(int))
#     if len(durations)> 1:
#         st.warning("Summary files with different session durations loaded. This might cause errors in final data output")
#     # add something here to indicate if there are multiple time courses 
#     all_totals = pd.DataFrame(totals) 
#     all_totals.index= pd.MultiIndex.from_arrays([all_meta['Group ID'].astype(int), all_meta['Subject ID']])

#     tbls = {}
#     for col in all_totals.columns:
#         dat = compile_measurement(col, results, metas)
#         tbls[col] = dat.sort_index(level='Group', axis=1)

#     all_totals = table_to_dict(all_totals)
#     dat = {'tables': tbls, 'totals': all_totals, 'meta': all_meta}
    
#     return dat

def combine_sessions(a, b):

    """ Combining two sesions
    
    Args:
        a (dict) & b (dict): dictionary of results, totals, meta
    
    Returns:
        r (dict) : results, totals, meta
    """
    c  = {}
    for k in a['tables']:
        b['tables'][k].index = b['tables'][k].index + max(a['tables'][k].index)
        c[k] = pd.concat([a['tables'][k],b['tables'][k]], axis = 0)

    new = {}
    for k in a['totals']:
        new[k] = pd.concat([a['totals'][k], b['totals'][k]], axis = 1).transpose()
        new[k].index = ['Session 1', 'Session 2']

    am =  a['meta'].sort_values(by='Subject').reset_index(drop=True)
    bm = b['meta'].sort_values(by= 'Subject').reset_index(drop=True)

    identical_columns = [col for col in am.columns if am[col].equals(bm[col])]
    merged_df = am.merge(bm, on = identical_columns, how='outer', suffixes=('_1', '_2'))
    
    r = {'tables': c, 'totals': new, 'meta':merged_df}
    return r


def compile_excel(d):
    """ Compile data from excel
    
    Args:
        d (fileUploader): excel file uploaded to streamlit
    
    Returns:
        dat (dict): results, totals, dict from excel sheet
    """

    # All possible measurements that could be in file
    pm = [ 'Ambulatory Distance', 'Ambulatory Time', 'Ambulatory Counts', 'Vertical Counts',
                'Vertical Time',  'Stereotypic Time', 'Stereotypic Counts', 'Resting Time', 'Jump Counts', 'Jump Time',
                'Ambulatory Episodes Average Speed',  'Ambulatory Episodes', 'Ambulatory Episodes Total Speed']

    # Splits columns into meta/data based on if contains an element in pm
    meta_columns = [x for x in d.columns if x.split(' Bin ')[0] not in pm]
    data_columns  = [x for x in d.columns if x.split(' Bin ')[0] in pm]
    
    # Separate dataframe
    data = d.loc[:, data_columns]
    meta = d.loc[:, meta_columns]
    
    # Get measurements based on actual data
    column_split = np.array([x.split(' Bin ') for x in data.columns])
    measurements = np.unique(column_split[:,0])

    # Determine number of bins based splitting columns names on ' Bin '
    if column_split.shape[1] > 1:
        bin_ids = np.unique(column_split[:,1]).astype(int)
    else:
        bin_ids = [1]

    # Determine bin length based on 
    bin_length = np.unique(meta['Session Duration'] / max(bin_ids))

    if len(bin_length) > 1:
        st.error('Not all subjects have the same session duration')
        # bin_length = bin_length[0]
    else:
        bin_length = bin_length[0]/60

    bins = np.linspace(bin_length, bin_length * max(bin_ids), max(bin_ids))

    tbls = {}
    totals = pd.DataFrame()
    totals.index = pd.MultiIndex.from_arrays([meta['Group'].astype(str), meta['Subject']])

    for m in measurements:
        m_col = column_split[:,0] == m
        dm = data.loc[:, m_col].transpose()

        dm.index = bins
        dm.columns = pd.MultiIndex.from_arrays([meta['Group'].astype(str), meta['Subject']])
        dm = dm.sort_index(axis = 1)

        tbls[m] = dm
        totals[m] = dm.sum().transpose()


    totals.sort_index()
    totals = table_to_dict(totals)
    dat = {'tables': tbls, 'totals': totals, 'meta': meta}

    return dat 



def match_old_to_new(s):
    """ Takes old data and matches to new conventions"""

    match_key = {'Distance Traveled': 'Ambulatory Distance', 'Time Ambulatory': 'Ambulatory Time', 
                'Ambulatory Count': 'Ambulatory Counts', 'Vertical Count': 'Vertical Counts',
                'Time Vertical': 'Vertical Time', 'Time Stereotypic': 'Stereotypic Time', 'Stereotypic Count' : 'Stereotypic Counts',
                'Time Resting': 'Resting Time', 'Jump Count': 'Jump Counts', 'Time Jumping': 'Jump Time',
                'Average Velocity': 'Ambulatory Episodes Average Speed', 'Ambulatory Episodes': 'Ambulatory Episodes'}

    renamed_tables = {match_key.get(key, key): value for key, value in s['tables'].items()}
    renamed_totals = {match_key.get(key, key): value for key, value in s['totals'].items()}

    meta_key = {'Group ID': 'Group', 'Subject ID': 'Subject', 'Experiment ID': 'Protocol', 'Session No': 'Session Number'}

    # renamed_meta = {meta_key.get(key, key): value for key, value in s['meta'].items()}
    renamed_meta = s['meta'].rename(meta_key, axis =1 )

    renamed_meta['Start Date'] = pd.to_datetime(renamed_meta['Start Date'], format='%m/%d/%Y')
    renamed_meta['Time'] = pd.to_datetime(renamed_meta['Start Time'], format='%H:%M:%S').dt.time
    renamed_meta['Start Time'] = pd.to_datetime(renamed_meta['Start Date'].astype(str) + ' ' + renamed_meta['Time'].astype(str))
    renamed_meta['Start Time'] = renamed_meta['Start Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    renamed_meta = renamed_meta.drop(['Start Date', 'Time'], axis=1)

    renamed_meta['Session Duration'] = renamed_meta['Session Time (min)'].astype(int) * 60
    renamed_meta = renamed_meta.drop(['Session Time (min)'], axis=1)

    n = {'tables': renamed_tables, 'totals': renamed_totals, 'meta': renamed_meta}
    return n


def combine_data_dicts(s, c):
    """ Combines two dictionaries of data
    
    Args:
        s,c (dicts): totals, results, meta 
    
    Returns:
        combined_dict (dict)
        subject_1 (list): strings of subject IDs from s
        subejct_2 (list): strings of subject IDs from c
    """

    combined_tables = {}
    combined_totals = {}
    combined_keys = set(s['tables'].keys()).intersection(c['tables'].keys())
    
    for k in combined_keys:
        combined_tables[k] = pd.concat([s['tables'][k], c['tables'][k]], axis = 1).sort_index(axis =1)

        if len(s['totals'][k].shape) == 1 and len(c['totals'][k].shape) == 1:
            combined_totals[k] = pd.concat([s['totals'][k], c['totals'][k]]).sort_index()
        else:
            combined_totals[k]= pd.concat([s['totals'][k], c['totals'][k]], axis = 1).sort_index(axis =1)

    combined_meta = pd.concat([s['meta'], c['meta']]).reset_index(drop = True)
    combined_dict = {'tables': combined_tables, 'totals': combined_totals, 'meta': combined_meta}

    subject_1 = s['meta']['Subject'].values
    subject_2 = c['meta']['Subject'].values

    return combined_dict, subject_1, subject_2


def reformat_totals(df):
    """ Takes a dataframe of totals with subjects & groups as columns and sessions as rows and reformats as groups as columns

    Args:
        df (pandas DataFrame): totals
    
    Returns:
        df2 (pandas DataFrame)
    
    """

    # first check if input is a Series (i.e one row only)
    if len(df.shape) == 1:
        df.index = df.index.droplevel(1)
        groups = np.unique(df.index)
        test = {}
        for g in groups:
            test[g] = np.array(df.loc[g]) if isinstance(df.loc[g], pd.Series) else np.array([df.loc[g]])

        max_length = max(len(k) if isinstance(k, np.ndarray) else 1 for k in test.values())

        # Pad shorter arrays with NaN values
        for g in groups:
            test[g] = np.concatenate([test[g], [np.nan] * (max_length - len(test[g]))])

        df2 = pd.DataFrame(test)

    # If input has multiple rows, reformat each session seperately and concatenate horizontally
    else:
        df = df.droplevel(1, axis = 1)
        groups = df.columns.unique()
        df2 = pd.DataFrame()

        for session in df.index:
            test = {}
            for g in groups:
                test[g] = np.array(df.loc[session, g]) if isinstance(df.loc[session, g], pd.Series) else np.array([df.loc[session, g]])
            
            max_length = max(len(k) if isinstance(k, np.ndarray) else 1 for k in test.values())
            for g in groups:
                test[g] = np.concatenate([test[g], [np.nan] * (max_length - len(test[g]))])

            mm =  pd.MultiIndex.from_arrays([[session] * len(groups), groups], names = ['Session', 'Group'])
            df3 = pd.DataFrame(test)
            df3.columns = mm
            df2 = pd.concat([df2, df3], axis = 1)


    df2.sort_index(axis = 1, inplace = True)
    
    return df2



# Formatting functions

def center_excel_sheet(writer, sheet_name, shape):
    """ Center all cells in excel output"""
    workbook  = writer.book
    worksheet = writer.sheets[sheet_name]

    # Get the dimensions of the DataFrame
    num_rows, num_cols = shape

    # Create a cell format with center alignment
    center_alignment = workbook.add_format({'align': 'center', 'valign': 'vcenter'})

    # Apply the cell format to the entire sheet
    worksheet.set_column(0, num_cols - 1, cell_format=center_alignment)

def center_add_lines(writer, sheet_name, df):
    """ Center data and add lines in table"""
    workbook  = writer.book
    worksheet = writer.sheets[sheet_name]
    border_format = workbook.add_format({'right': 1, 'border_color': 'black', 'align': 'center', 'valign': 'vcenter'})

    num_rows, num_cols = df.shape
    center_alignment = workbook.add_format({'align': 'center', 'valign': 'vcenter'})

    worksheet.set_column(0, num_cols - 1, cell_format=center_alignment)
    if isinstance(df.columns, pd.MultiIndex):
        # ch = np.diff([int(x.split(" ")[1]) for x in df.columns.get_level_values(0)])
        # bold_lines = np.argwhere(ch).flatten().tolist()

        unique_labels, numeric_values = np.unique(df.columns.get_level_values(0), return_inverse=True)
        bold_lines = np.where(np.diff(numeric_values) != 0)[0] + 1
        for l in bold_lines:
            worksheet.set_column(l, l, cell_format = border_format)


def center_with_lines_and_color(writer, sheet_name, df, s1, s2):
    """ Center data, add lines, color subject IDs"""
    workbook  = writer.book
    worksheet = writer.sheets[sheet_name]

    num_rows, num_cols = df.shape
    center_alignment = workbook.add_format({'align': 'center', 'valign': 'vcenter'})
    worksheet.set_column(0, num_cols - 1, cell_format= center_alignment)

    border_format = workbook.add_format({'left': 1, 'border_color': 'black', 'align': 'center', 'valign': 'vcenter'})
    blue_format = workbook.add_format({'bg_color': '#E6F7FF', 'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1})
    orange_format = workbook.add_format({'bg_color': '#FFEFD5', 'bold': True, 'align': 'center', 'valign': 'vcenter', 'border':1})

    # ch = np.diff(df.columns.get_level_values(0))
    # ch = np.insert(ch, 0, 0)
    # ch = np.append(ch, 1000)
    # bold_lines = np.argwhere(ch).flatten().tolist()

    unique_labels, numeric_values = np.unique(df.columns.get_level_values(0), return_inverse=True)
    bold_lines = np.where(np.diff(numeric_values) != 0)[0] + 1

    for l in bold_lines:
        worksheet.set_column(l+1, l+1, cell_format = border_format)

    subject_col_index = df.columns.get_level_values('Subject')

    # Iterate through the 'Subject' level and apply formatting based on conditions
    for ind,sub in enumerate(subject_col_index):

        if sub in s1:
            worksheet.write(1, ind+1, sub, blue_format)
            
        if sub in s2:
            worksheet.write(1, ind+1,  sub, orange_format)

def identify_duplicate_subjects(s1):
    unique_sub = np.unique(s1)
    
    if len(unique_sub) < len(s1):
        counts = np.array( [s1.count(x) for x in unique_sub])
        dupes = unique_sub[np.greater(counts, 1)]
        num_dupes = counts[np.greater(counts, 1)]

        for dupe,n in zip(dupes, num_dupes):
            st.warning("Subject ID "+str(dupe)+" loaded "+str(n)+" times")
        
        st.warning("Check output, multiple subjects with same ID")

def confirm_matching_data(dat, s1, s2):

    mismatched = {}
    all_sub = dat['meta']['Subject'].values
    for k in dat['tables']:
        subs = dat['tables'][k].columns.get_level_values('Subject').values
        missing = [sub for sub in subs if sub not in all_sub]
        if len(missing) > 0:
            mismatched[k] = missing

        subject_ids_by_group = {group: len(dat['tables'][k].columns.get_level_values('Subject')[dat['tables'][k].columns.get_level_values('Group') == group].tolist()) 
                                for group in dat['tables'][k].columns.get_level_values('Group').unique()}
    
    if (len(mismatched)) > 0:
        for k in mismatched:
            st.write(str(k)+" Intervals is missing "+ str(", ".join(mismatched[k]))+ " subjects")

    st.write("Loaded "+str(len(all_sub))+" animals total")
    st.write("Old system: "+str(len(s1))+ " animals")

    if len(s1) > 0:
        identify_duplicate_subjects(s1.tolist())
    
    st.write("New system: "+str(len(s2))+ " animals") 
    if len(s2) > 0:
        identify_duplicate_subjects(s2.tolist())
    for k in subject_ids_by_group:
        st.write('Group '+str(k)+': '+str(subject_ids_by_group[k])+ " animals")
    
    st.write("Measurements included: "+', '.join(dat['tables'].keys()))

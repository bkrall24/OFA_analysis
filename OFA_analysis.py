import pandas as pd
import numpy as np
import streamlit as st
import io
import re


def parse_dict(txt):
    data = [x for x in txt if ':' in x]
    dict = {x.split(':',1)[0].strip(): x.split(':',1)[1].strip() for x in data}

    if "Subject ID" in dict.keys():
        sid = dict['Subject ID']
        pattern = re.compile(r'^C(\d+)M(\d+)$')
        match = pattern.match(sid)
        if match:
            dict['Subject ID']= int(match.group(2))
        

    return dict

def match_and_rename(result, keys):
    for c in result.columns:
        characters_to_match = set(char for char in c if char != '.' and char != 's')
        
        new_col = [x for x in keys if characters_to_match.issubset(set(char for char in x))]
        
        if len(new_col) > 0:
            result.rename(columns={c: new_col[0]}, inplace=True)
    
    return result

def parse_time(time_str):
    minutes, seconds = time_str.split(':')
    minutes = int(minutes)
    seconds = float(seconds)
    return pd.Timedelta(minutes=minutes, seconds=seconds)

def cast_numeric(data, keys):

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


def parse_summary(example_file):
    # with open(example_file, 'r') as file:
    #     lines = file.readlines()
    # lines = example_file.readlines()       
    lines = [line.decode('utf-8') for line in example_file.readlines()]
    meta_data = lines[:36]
    meta_dict = parse_dict(meta_data)

    tablular_data = lines[37:]
    
    stops = [x for x,y in enumerate(tablular_data) if len(y) == 2]
    totals = tablular_data[stops[-2]:stops[-1]]
    totals = parse_dict(totals)

    breaks = [x for x,y in enumerate(tablular_data) if '=' in y]
    breaks.append(len(tablular_data))

    frames =[]
    for a,b in zip(breaks, stops):
        c = tablular_data[a+1:b]
        head = tablular_data[a-2:a]
        if len(c) > 0:
            h2 = [[x for x in q.strip().split(' ') if x != '' ] for q in head]
            header = []
            for a,b in zip(h2[0], h2[1]):
                header.append(a+" "+ b)   
            
            c2 = [[x for x in q.strip().split(' ') if x != '' ] for q in c]
            df = pd.DataFrame(c2, columns = header)
            frames.append(df)

    result = frames[0]
    for df in frames[1:]:
        result = pd.merge(result, df, on = 'Session Time', how = 'inner')

    result = match_and_rename(result, totals.keys())
    result = result[['Session Time'] + [col for col in result.columns if col != 'Session Time']]
    
    cast_numeric(result, result.columns)
    cast_numeric(totals, totals.keys())

    
    return result, meta_dict, totals

def compile_measurement(measurement, results, metas):

    columns = pd.MultiIndex(levels=[[], []], codes=[[], []], names=[ 'Group', 'Subject'])
    df = pd.DataFrame(columns=columns)

    for experiment, meta in zip(results, metas):
        # experiment_name = meta["Experiment ID"]
        group_number = int(meta["Group ID"])
        subject_id = meta["Subject ID"]
        column_index = (group_number, subject_id)
        df[column_index] = experiment[measurement]
        
    df.index = np.array(results[0]['Session Time'])

    return df

def table_to_dict(table):

    table_dict = {}
    for col in table.columns:
        table_dict[col] = table[col].transpose()
    
    return table_dict

def compile_summary(files):
    metas = []
    totals = []
    results = []

    for f in files:
        r,m,t = parse_summary(f)
        metas.append(m)
        totals.append(t)
        results.append(r)

    all_meta = pd.DataFrame(metas)

    all_totals = pd.DataFrame(totals) 
    all_totals.index= pd.MultiIndex.from_arrays([all_meta['Group ID'].astype(int), all_meta['Subject ID']])

    tbls = {}
    for col in all_totals.columns:
        dat = compile_measurement(col, results, metas)
        tbls[col] = dat.sort_index(level='Group', axis=1)

    all_totals = table_to_dict(all_totals)
    dat = {'tables': tbls, 'totals': all_totals, 'meta': all_meta}
    
    return dat

def combine_sessions(a, b):
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
    pm = [ 'Ambulatory Distance', 'Ambulatory Time', 'Ambulatory Counts', 'Vertical Counts',
                'Vertical Time',  'Stereotypic Time', 'Stereotypic Counts', 'Resting Time', 'Jump Counts', 'Jump Time',
                'Ambulatory Episodes Average Speed',  'Ambulatory Episodes', 'Ambulatory Episodes Total Speed']

    meta_columns = [x for x in d.columns if x.split(' Bin ')[0] not in pm]
    data_columns  = [x for x in d.columns if x.split(' Bin ')[0] in pm]
    

    data = d.loc[:, data_columns]
    meta = d.loc[:, meta_columns]

    column_split = np.array([x.split(' Bin ') for x in data.columns])
    measurements = np.unique(column_split[:,0])

    if column_split.shape[1] > 1:
        bin_ids = np.unique(column_split[:,1]).astype(int)
    else:
        bin_ids = [1]

    bin_length = np.unique(meta['Session Duration'] / max(bin_ids))

    if len(bin_length) > 1:
        st.text('Multiple time courses contained in data')
    else:
        bin_length = bin_length[0]/60

    bins = np.linspace(bin_length, bin_length * max(bin_ids), max(bin_ids))

    tbls = {}
    totals = pd.DataFrame()
    totals.index = pd.MultiIndex.from_arrays([meta['Group'], meta['Subject']])

    for m in measurements:
        m_col = column_split[:,0] == m
        dm = data.loc[:, m_col].transpose()

        dm.index = bins
        dm.columns = pd.MultiIndex.from_arrays([meta['Group'], meta['Subject']])
        dm = dm.sort_index(axis = 1)

        tbls[m] = dm
        totals[m] = dm.sum().transpose()


    totals.sort_index()

    totals = table_to_dict(totals)
    dat = {'tables': tbls, 'totals': totals, 'meta': meta}

    return dat



def match_old_to_new(s):

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

    combined_tables = {}
    combined_totals = {}
    combined_keys = set(s['tables'].keys()).intersection(c['tables'].keys())
    

    for k in combined_keys:
        combined_tables[k] = pd.concat([s['tables'][k], c['tables'][k]], axis = 1).sort_index(axis =1)

        if len(s['totals'][k].shape) == 1 and len(c['totals'][k].shape) == 1:
            combined_totals[k] = pd.concat([s['totals'][k], c['totals'][k]]).sort_index()
        else:
            combined_totals[k]= pd.concat([s['totals'][k], c['totals'][k]], axis = 1).sort_index(axis =1)
        # st.text(combined_tables[k])


    combined_meta = pd.concat([s['meta'], c['meta']]).reset_index(drop = True)
    combined_dict = {'tables': combined_tables, 'totals': combined_totals, 'meta': combined_meta}

    subject_1 = s['meta']['Subject'].values
    subject_2 = c['meta']['Subject'].values

    return combined_dict, subject_1, subject_2


def reformat_totals(df):
    

    if len(df.shape) == 1:
        df.index = df.index.droplevel(1)
        groups = np.unique(df.index)
        test = {}
        for g in groups:
            test[g] = np.array(df.loc[g]) if isinstance(df.loc[g], pd.Series) else np.array([df.loc[g]])
            
        # if max([len(k.shape) for  k in test.values()]) == 0:
        #     df2 = pd.DataFrame(test, index = [0])
        # else:
        #     df2 = pd.DataFrame(test)

        max_length = max(len(k) if isinstance(k, np.ndarray) else 1 for k in test.values())

        # Pad shorter arrays with NaN values
        for g in groups:
            # test[g] = np.pad(test[g], (0, max_length - len(test[g])), constant_values=-1)
            test[g] = np.concatenate([test[g], [np.nan] * (max_length - len(test[g]))])

        df2 = pd.DataFrame(test)

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
                # test[g] = np.pad(test[g], (0, max_length - len(test[g])), constant_values=-1)
                test[g] = np.concatenate([test[g], [np.nan] * (max_length - len(test[g]))])

            mm =  pd.MultiIndex.from_arrays([[session] * len(groups), groups], names = ['Session', 'Group'])
            df3 = pd.DataFrame(test)
            df3.columns = mm
            df2 = pd.concat([df2, df3], axis = 1)


    df2.sort_index(axis = 1, inplace = True)
    
    return df2



def center_excel_sheet(writer, sheet_name, shape):
    workbook  = writer.book
    worksheet = writer.sheets[sheet_name]

    # Get the dimensions of the DataFrame
    num_rows, num_cols = shape

    # Create a cell format with center alignment
    center_alignment = workbook.add_format({'align': 'center', 'valign': 'vcenter'})

    # Apply the cell format to the entire sheet
    worksheet.set_column(0, num_cols - 1, cell_format=center_alignment)

def center_add_lines(writer, sheet_name, df):
    workbook  = writer.book
    worksheet = writer.sheets[sheet_name]
    border_format = workbook.add_format({'right': 1, 'border_color': 'black', 'align': 'center', 'valign': 'vcenter'})

    num_rows, num_cols = df.shape
    center_alignment = workbook.add_format({'align': 'center', 'valign': 'vcenter'})

    # Apply the cell format to the entire sheet
    worksheet.set_column(0, num_cols - 1, cell_format=center_alignment)
    if isinstance(ref.columns, pd.MultiIndex):
        ch = np.diff([int(x.split(" ")[1]) for x in df.columns.get_level_values(0)])

        bold_lines = np.argwhere(ch).flatten().tolist()
        for l in bold_lines:
            worksheet.set_column(l+1, l+1, cell_format = border_format)
            #     empty_col = pd.MultiIndex.from_tuples([(' ', ' ')], names=('Session', 'Group'))
            #     ref = pd.concat([ref.loc[:, ('Session 1', slice(None))], pd.DataFrame(columns=empty_col), ref.loc[:, ('Session 2', slice(None))]], axis=1)

def center_with_lines_and_color(writer, sheet_name, df, s1, s2):
    workbook  = writer.book
    worksheet = writer.sheets[sheet_name]

    num_rows, num_cols = df.shape
    center_alignment = workbook.add_format({'align': 'center', 'valign': 'vcenter'})
    worksheet.set_column(0, num_cols - 1, cell_format= center_alignment)

    border_format = workbook.add_format({'left': 1, 'border_color': 'black', 'align': 'center', 'valign': 'vcenter'})
    blue_format = workbook.add_format({'bg_color': '#E6F7FF', 'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1})
    orange_format = workbook.add_format({'bg_color': '#FFEFD5', 'bold': True, 'align': 'center', 'valign': 'vcenter', 'border':1})

    ch = np.diff(df.columns.get_level_values(0))
    ch = np.insert(ch, 0, 0)
    ch = np.append(ch, 1000)
    bold_lines = np.argwhere(ch).flatten().tolist()

    for l in bold_lines:
        worksheet.set_column(l+1, l+1, cell_format = border_format)

    subject_col_index = df.columns.get_level_values('Subject')

    # Iterate through the 'Subject' level and apply formatting based on conditions
    for ind,sub in enumerate(subject_col_index):

        if sub in s1:
            worksheet.write(1, ind+1, sub, blue_format)
            
        if sub in s2:
            worksheet.write(1, ind+1,  sub, orange_format)



def confirm_matching_data(dat, s1, s2):

    mismatched = {}
    missing_totals = {}
    all_sub = dat['meta']['Subject'].values
    for k in dat['tables']:
        subs = dat['tables'][k].columns.get_level_values('Subject').values
        missing = [sub for sub in subs if sub not in all_sub]
        if len(missing) > 0:
            mismatched[k] = missing

        subject_ids_by_group = {group: len(dat['tables'][k].columns.get_level_values('Subject')[dat['tables'][k].columns.get_level_values('Group') == group].tolist()) 
                                for group in dat['tables'][k].columns.get_level_values('Group').unique()}
        
        
        # missing_vals = len(all_sub) - dat['totals'][k].shape[1]
        # if missing_vals != 0:
        #     missing_totals[k]= missing_vals

    # if (len(missing_totals)) > 0:
    #     for k in missing_totals:
    #         st.write(str(k)+" Totals is missing "+ str(missing_totals[k])+ " values")
    
    if (len(mismatched)) > 0:
        for k in mismatched:
            st.write(str(k)+" Intervals is missing "+ str(", ".join(mismatched[k]))+ " subjects")

    st.write(str(len(all_sub))+" animals total")
    st.write(str(len(s1))+ " animals from old system")  
    st.write(str(len(s2))+ " animals from new system") 
    for k in subject_ids_by_group:
        st.write('Group '+str(k)+': '+str(subject_ids_by_group[k])+ " animals")
    
    st.write("Measurements included: "+', '.join(dat['tables'].keys()))

    

# def add_lines_sheet(writer, sheet_name, df):
#     workbook  = writer.book
#     worksheet = writer.sheets[sheet_name]
#     border_format = workbook.add_format({'left': 1, 'border_color': 'black'})

#     ch = np.diff(df.columns.get_level_values(0))
#     ch = np.insert(ch, 0, 0)
#     ch = np.append(ch, 1000)
#     bold_lines = np.argwhere(ch) + 1

#     for l in bold_lines:
#         worksheet.set_column(l[0], l[0], cell_format = border_format)


# def color_code_by_room(writer, sheet_name, df, s1, s2):

#     workbook  = writer.book
#     worksheet = writer.sheets[sheet_name]

#     # Get the column index for the 'Subject' level
#     subject_col_index = df.columns.get_level_values('Subject')
#     # st.text(s2)
#     # Define formats for different colors
#     blue_format = workbook.add_format({'bg_color': '#99CCFF', 'bold': True})
#     orange_format = workbook.add_format({'bg_color': '#FF9900', 'bold': True})

#     # Iterate through the 'Subject' level and apply formatting based on conditions
#     for ind,sub in enumerate(subject_col_index):
#         # cell_value = df.iloc[1, col]
#         # st.text(cell_value)
     
#         if sub in s1:
#             worksheet.write(1, ind+1, sub, blue_format)
            
#         if sub in s2:
#             worksheet.write(1, ind+1,  sub, orange_format)
    
       

       
                   

st.title('Open Field Activity Data Processing')
sys = st.multiselect(label = 'Choose the OFA System', options = ['New (Rm 8)', 'Old (Rm 7)'])

if 'Old (Rm 7)' in sys:
    mult_old = st.toggle(label = 'Two Sessions in Old (Rm 7) files?')

    if mult_old:
        session1_files = st.file_uploader(label = "Select Session 1 Files", accept_multiple_files= True)
        if len(session1_files) > 0:
            d1 = compile_summary(session1_files)
            d1 = match_old_to_new(d1)
        else:
            st.text('Choose session 1 summary files, or deselect two sessions')
            d1 = None
            
        session2_files = st.file_uploader(label = "Select Session 2 Files", accept_multiple_files= True)
        if len(session2_files) > 0:
            d2 = compile_summary(session2_files)
            d2 = match_old_to_new(d2)
        else:
            st.text('Choose session 2 summary files, or deselect two sessions')
            d2 = None

        if d1 is not None and d2 is not None:
            old_dat = combine_sessions(d1, d2)
        else:
            old_dat = None


    else:
        session1_files = st.file_uploader(label = "Select Summary Files for Old (Rm 7) Analysis", accept_multiple_files= True)
        if len(session1_files) > 0:
            old_dat = compile_summary(session1_files)
            old_dat = match_old_to_new(old_dat)
        else:
            st.text('Choose files, or deselect "Old (Rm 7)"')
            old_dat = None
else:
    old_dat = None

if 'New (Rm 8)' in sys:
    mult_new = st.toggle(label = 'Two Sessions in New (Rm 8) file?')
    excel_file = st.file_uploader(label = "Select Excel File for New (Rm 8) analysis", accept_multiple_files= False)

    if excel_file is not None:
        excel_dat = pd.read_excel(excel_file)
        
        if mult_new:
            sessions = excel_dat['Protocol'].unique()

            d1 = excel_dat.loc[excel_dat['Protocol'] == sessions[0], :]
            d1 = d1.dropna(axis =1, how = 'all')
        

            d2 = excel_dat.loc[excel_dat['Protocol'] == sessions[1], :]
            d2 = d2.dropna(axis = 1, how = 'all')

            a = compile_excel(d1)
            b = compile_excel(d2)
            new_dat = combine_sessions(a, b)
        else:
            excel_dat = excel_dat.dropna(axis = 1, how = 'all')
            new_dat = compile_excel(excel_dat)
            
    
    else:
        st.text('Choose files, or deselect "New (Rm 8)"')
        new_dat = None
else:
    new_dat = None



if new_dat is not None and old_dat is not None:
    dat,s1, s2  = combine_data_dicts(old_dat, new_dat)

elif new_dat is not None:
    dat = new_dat
    s2 = dat['meta']['Subject'].values
    s1 = []
elif old_dat is not None:
    dat = old_dat
    s1 = dat['meta']['Subject'].values
    s2 = []
else:
    dat = None



if dat is not None:
    confirm_matching_data(dat, s1, s2)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer: 


        for k in dat['totals'].keys():
            
            ref = reformat_totals(dat['totals'][k])
            
            if len(k) > 21:
                k2 = k[:21]
            else:
                k2 = k

            ref.index = ref.index+1

            # if isinstance(ref.columns, pd.MultiIndex):
            #     empty_col = pd.MultiIndex.from_tuples([(' ', ' ')], names=('Session', 'Group'))
            #     ref = pd.concat([ref.loc[:, ('Session 1', slice(None))], pd.DataFrame(columns=empty_col), ref.loc[:, ('Session 2', slice(None))]], axis=1)
            ref.to_excel(writer, sheet_name = k2 + " Total")
            center_add_lines(writer, k2 + " Total", ref)

            # center_excel_sheet(writer, k2 + " Total", ref.shape)
            dat['tables'][k].to_excel(writer, sheet_name = k2 + " Intervals")
            center_with_lines_and_color(writer, k2 + " Intervals", dat['tables'][k], s1, s2)
            

        dat['meta'] = dat['meta'][['Subject'] + [col for col in dat['meta'].columns if col != 'Subject']]      
        dat['meta'].to_excel(writer, sheet_name = 'Metadata')
        center_excel_sheet(writer, 'Metadata', dat['meta'].shape)


       

    st.download_button(
        label="Download Excel workbook",
        data= buffer,
        file_name="workbook.xlsx",
        mime="application/vnd.ms-excel"
    )



from ofa import *
import pandas as pd
import streamlit as st
import io

st.title('Open Field Activity Data Processing')
sys = st.multiselect(label = 'Choose the OFA System', options = ['New (Rm 8)', 'Old (Rm 7)'])

if 'Old (Rm 7)' in sys:
    mult_old = st.toggle(label = 'Two Sessions in Old (Rm 7) files?')

    if mult_old:
        session1_files = st.file_uploader(label = "Select Session 1 Files", accept_multiple_files= True)
        if len(session1_files) > 0:
            d1 = compile_summary_files(session1_files)
            if d1 is not None:
                d1 = match_old_to_new(d1)
        else:
            st.text('Choose session 1 summary files, or deselect two sessions')
            d1 = None
            
        session2_files = st.file_uploader(label = "Select Session 2 Files", accept_multiple_files= True)
        if len(session2_files) > 0:
            d2 = compile_summary_files(session2_files)
            if d2 is not None:
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
            old_dat = compile_summary_files(session1_files)
            if old_dat is not None:
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


import os
import re
from datetime import datetime as dt
from dateutil.parser import parse
from thefuzz import fuzz
import math
import numpy as np


UPLOAD_FOLDER = '/flask_app/files/xlsx/'
import pandas as pd


def format_doc(doc_type, doc_name, extracted_data, pathfile, dataFrameForSorting):
    if doc_type == 'vgm':
        return format_vgm(doc_name, extracted_data)
    elif doc_type == 'bolinstruction':
        return format_bolinstruction(doc_name, extracted_data)
    elif doc_type == 'manifesto':
        return format_manifesto(doc_name, extracted_data, dataFrameForSorting)
    elif doc_type == 'passiveinvoice':
        return format_passiveinvoice(doc_name, extracted_data, dataFrameForSorting)
    else:
        return


def prune_text(text):
    chars = "\\`*_\{\}[]\(\)\|/<>#-\'\"+!$,\."
    for c in chars:
        if c in text:
            text = text.replace(c, "")
    return text


def cleanup_text(text):
    result = re.sub(r'[^a-zA-Z0-9-]+', '', text) # added hyphen (ex: ARKU 98702-0)
    return result


def get_container_type(text):
    return text


def get_quantity(text):
    if text == '':
        text = 1
    return text


def get_pack_quantity(text):
    return text


def equals_or_nan(list1, list2):
    if len(list1) != len(list2):
        return False
    
    for i in range(len(list1)):
        if list1[i] != list2[i] and not (is_nan(list1[i]) or is_nan(list2[i])):
            return False
    
    return True


def is_nan(value):
    return (isinstance(value, float) and math.isnan(value)) or value == "nan"


def normalize_weight(weight):
    if "\\" in weight:
        weight = weight.replace("\\", "")
    if "kgs." in weight.lower():
        weight = weight.lower().replace("kgs.", "")
    if "." in weight and "," in weight:
        if weight.find(".") < weight.find(","):
            # . is for thousands and , for decimals
            weight = weight.replace(".", "")
            weight = weight.replace(",", ".")
        else:
            # , is for thousands and . for decimals
            weight = weight.replace(",", "")
    elif "," in weight[-3:]:
        # , is the decimal sign
        weight = weight.replace(",", ".")
    elif "," in weight[:3]:
        # , is for thousands
        weight = weight.replace(",", "")
    elif "." in weight[-3:]:
        # . is the decimal sign
        pass
    elif "." in weight[:3]:
        # . is for thousands
        weight = weight.replace(".", "")
    return weight



# Remove the charachter from Tare column. for example: T:1200 -> 1200  or  Tare:1200 -> 1200
def remove_prefix_and_convert(value):
    value = re.sub(r'[^0-9.]', '', str(value))
    if value.startswith('.'):
        # for example if value is: KGS.3900 , value after line above will be: .3900
        value = value[1:]
    return value


def clean_weight(string):
    words_to_replace = ['KGS.', 'KGS', 'KG.', 'KG']
    pattern = r'\b(?:{})\b'.format('|'.join(words_to_replace))
    cleaned_string = re.sub(pattern, '', str(string), flags=re.IGNORECASE)
    cleaned_string = cleaned_string.replace(",", "") # remove the comma 22,820. 000 --> 22820. 000
    cleaned_string = cleaned_string.replace(".", ",") # remove the comma 22820, 000 --> 22820. 000
    cleaned_string = cleaned_string.replace(" ", "") # remove the space 22820,000 --> 22820.000

    # Check if the first character is a comma or dot and remove it
    if cleaned_string and (cleaned_string[0] == ',' or cleaned_string[0] == '.'):
        cleaned_string = cleaned_string[1:]
    return cleaned_string

# Clean container ID
def clean_containerID(value):
    value = str(value).replace(" ", "")
    value = value.replace("_", "")
    return value

# Remove prefixes like SN, SN:, SEAL, SEALS, N.
def clean_seal_number(value):
    if pd.isnull(value) or value.lower() == 'nan':
        return np.nan

    cleaned_value = re.sub(r'^N\.?\s*', '', str(value), flags=re.IGNORECASE)
    characters_to_remove = ['SN:', 'SN', 'SEALS:', 'SEAL:', 'Seal:', 'SEALS', 'SEAL', 'Seal:']
    for char in characters_to_remove:
        cleaned_value = cleaned_value.replace(char, '')
    if cleaned_value.startswith(':'): # removing the ':'
        cleaned_value = cleaned_value[1:]
    cleaned_value = cleaned_value.strip()
    return cleaned_value
    

def sort_filtered_df(filtered_df):
    print(f'length of filter: {len(filtered_df)}')
    # Group filtered_df by 'page' in ascending order
    grouped_df = filtered_df.groupby('page', sort=True)

    # Initialize a list to store the sorted rows
    sorted_rows = []

    # Iterate over the groups (Page by Page) and sort them based on their coordinates (box)
    for _, group_df in grouped_df:
        # Create a zipped list of (_preds, _bbox, _words) for each group
        zipped_list = list(zip(group_df['label'], group_df['box'], group_df['word']))

        # Sort the zipped list based on the custom key function
        sorted_list = sorted(zipped_list, key=lambda x: (x[1][1], x[1][0]))

        # Extract the sorted rows from the sorted list
        sorted_group = pd.DataFrame(sorted_list, columns=['label', 'box', 'word'])

        # Append the sorted rows to the list
        sorted_rows.extend(sorted_group.values.tolist())

    # Create a new DataFrame from the sorted rows
    sorted_df = pd.DataFrame(sorted_rows, columns=['label', 'box', 'word'])

    print(f'length of sorted_df: {len(sorted_df)}')
    return sorted_df

def container_id_pattern(row):
    # This function will check entities like SGCU8120 (third and fourth letters are 'U' and
    # first and second are just letters (Aa-Zz) followed by a number or letter
    patternSGCU = r'^[A-Za-z]{3}U\w+'
    if re.match(patternSGCU, row['word']):
        return 'container_id'
    else:
        return row['label']

def manually_correct_predictions(df):
    # Check SGCU12000 to be container_id
    df['label'] = df.apply(container_id_pattern, axis=1)
    return df

def rearrange_service_df(detail_df):
    last_seen_prediction = ''
    # Iterate over the groups (Page by Page) and re-arrange them based on their coordinates (box)
    for name, group in detail_df.groupby('page', sort=True):
        for index, row in group.iterrows():
            if row['label'] == last_seen_prediction:
                # here we have two consecutive equal label, ex: service_key service_key
                # which we're going to control their coordinates to prevent ocr unaligned boxes
                # check the length of dataframe and next prediction
                if index + 1 < len(group) and group.iloc[index + 1]['label'] != last_seen_prediction:
                    distance = abs(row['box'][1] - group.iloc[index + 1]['box'][1])
                    if distance < 8:
                        print(f'row {index + 1} can be changed with row: {index}')
                        group.loc[index, :], group.loc[index + 1, :] = group.loc[index + 1, :].copy(), group.loc[index,
                                                                                                       :].copy()
            else:
                last_seen_prediction = row['label']

    return detail_df

def format_passiveinvoice(doc_name, extracted_data, dataFrameForSorting):

    doc_name_contents = re.split("_", doc_name, 2)
    if len(doc_name_contents) == 3:
        attach_filename = doc_name_contents[2]
    else:
        attach_filename = doc_name
    reference_number = attach_filename.replace(".PDF", "").replace(".pdf", "")

    xls_filepath = os.path.join(UPLOAD_FOLDER, reference_number + ".xlsx")
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    writer = pd.ExcelWriter(xls_filepath, engine='xlsxwriter')

    ##### new way to create table developed by Ata 13 July 2023 #####

    # Function below will check some labels manually to be correct
    dataFrameForSorting = manually_correct_predictions(dataFrameForSorting)

    # Define a threshold for x-coordinate similarity (this threshold will remove predictions that they're
    # not in same area because in this project we're struggling with tables)
    x_threshold = 100

    # Group the DataFrame by 'label'
    grouped = dataFrameForSorting.groupby('label')

    # print(f'labels are: {dataFrameForSorting["label"]}')


    # Create a new DataFrame from the filtered rows
    filtered_df = dataFrameForSorting

    # sort dataframe by pages
    #filtered_df = sort_filtered_df(filtered_df)
    df = dataFrameForSorting
    # define two dataframes, header & service part
    services_df = df[(df['label'] == 'service_key') | (df['label'] == 'service_value')]
    services_df = services_df.reset_index(drop=True) # reset index and sort services_df
    services_df = rearrange_service_df(services_df)
    # Reset the index if needed
    services_df.reset_index(drop=True, inplace=True)
    header_df = df[(df['label'] != 'service_key') & (df['label'] != 'service_value')]











    # then we have to group the dataframe by page, then proceed it page by page into two different
    # dataframes, service_df and header_df
    output = pd.DataFrame(
        columns=['container_id', 'seal_number', 'container_quantity', 'container_type', 'package_quantity',
                 'package_type', 'tare',
                 'weight', 'MERCE'])

    preds = list(filtered_df['label'])
    words = list(filtered_df['word'])
    boxes = list(filtered_df['box'])
    # print(f'preds: {preds}')
    # print(f'words: {words}')


    flag_label = preds[0]
    flag_last_row_number = 0  # store last row of our flag (we use flag to know when we should go to next line)
    for i in range(len(preds)):
        if output.empty:
            output.at[0, preds[i]] = words[i]
            continue
        if preds[i] == flag_label:
            flag_last_row_number = output.shape[0]
            output.at[flag_last_row_number, preds[i]] = words[i]
            continue
        else:
            # it's not the flag, so we have to check
            if (pd.isna(output[preds[i]].iloc[flag_last_row_number])):
                output.at[flag_last_row_number, preds[i]] = words[i]
            else:
                # we need to put it in last valid index
                output.at[output[preds[i]].last_valid_index() + 1, preds[i]] = words[i]

    # in this part we have to rename the columns as client requested
    output.rename(columns={"container_id": "SIGLA"}, inplace=True)
    output.rename(columns={"seal_number": "SIGILLI"}, inplace=True)
    output.rename(columns={"container_quantity": "QUANTITA CONTAINER"}, inplace=True)
    output.rename(columns={"container_type": "TIPOLOGIA CONTAINER"}, inplace=True)
    output.rename(columns={"package_quantity": "COLLI"}, inplace=True)
    output.rename(columns={"package_type": "IMBALLO"}, inplace=True)
    output.rename(columns={"tare": "TARA"}, inplace=True)
    output.rename(columns={"weight": "PESO LORDO"}, inplace=True)

    ################################################################################
    ############################  CLEANING THE TABLE  ##############################
    ################################################################################
    # Iterate over rows and split container type into container type and container quantity
    for i, row in output.iterrows():
        container_type = str(row['TIPOLOGIA CONTAINER']).strip()
        container_quantity = row['QUANTITA CONTAINER']
        # print(f'container type here is: {container_type}')
        match = re.match(r'(\d+)\s*[xX]\s*(.*)', container_type)
        if match:
            # print('matched!')
            if pd.isnull(container_quantity) or str(container_quantity).lower() in ['nan', 'NaN']:
                output.at[i, 'QUANTITA CONTAINER'] = match.group(1)
                output.at[i, 'TIPOLOGIA CONTAINER'] = match.group(2)

    output['TARA'] = output['TARA'].apply(remove_prefix_and_convert)
    output['SIGILLI'] = output['SIGILLI'].apply(clean_seal_number)
    output['PESO LORDO'] = output['PESO LORDO'].apply(clean_weight)
    output['SIGLA'] = output['SIGLA'].apply(clean_containerID)
    # remove QUANTITA CONTAINER
    output.drop('QUANTITA CONTAINER', axis=1, inplace=True)

    # Fill 'IMBALLO' column
    output['IMBALLO'] = output['COLLI'].astype(str).str.extract(r'(\D+)$')[0].str.strip()

    # Extract the quantity or number from 'COLLI' and store it in 'COLLI' column
    output['COLLI'] = output['COLLI'].astype(str).str.extract(r'^(\d+)')
    output['COLLI'] = output['COLLI'].astype(float).fillna(np.nan).astype(pd.Int64Dtype())

    ##### Changes by Vale on 2023-07-27 #####
    output['TAR A'] = output['TARA']
    columns_order = ['MERCE', 'TIPOLOGIA CONTAINER', 'SIGLA', 'COLLI', 'PESO LORDO', 'PESO NETTO', 'VOLUME', 'SIGILLI',
                     'IMBALLO', 'TAR A']
    for c in columns_order:
        if c not in output.columns:
            output[c] = ""
    output_order = output[columns_order]
    output_order.to_excel(writer, sheet_name="Sheet1", index=False)  # send df to writer
    ############ End of changes #############
    worksheet = writer.sheets["Sheet1"]  # pull worksheet object
    for idx, col in enumerate(output_order):  # loop through all columns
        series = output_order[col]
        max_len = max((
            series.astype(str).map(len).max(),  # len of largest item
            len(str(series.name))  # len of column name/header
        )) + 1  # adding a little extra space
        worksheet.set_column(idx, idx, max_len)  # set column width
    writer.close()

    print('excel file has been created!')
    # print(df)

    return xls_filepath, reference_number + ".xlsx"
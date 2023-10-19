import os
import io
import json
import re
import fitz
import numpy as np
import pandas as pd
from itertools import groupby
from PIL import Image
#from paddleocr import PaddleOCR
import torch
import pytesseract
from skimage import io as skio
from scipy.ndimage import interpolation as inter
import cv2
from transformers import AutoProcessor
from transformers import AutoModelForTokenClassification
from transformers import LayoutLMv3ImageProcessor
import logging
from utils import AWSutils
from dotenv import load_dotenv
import random
#Google Vision OCR
from google.cloud import vision_v1p3beta1 as vision
from google.cloud import vision_v1p3beta1 as vision
from google.cloud import vision
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/ml_worker/utils/platformwenda-vgm-test-apikey.json"

logger = logging.getLogger('ad_logger')

UPLOAD_FOLDER = '/flask_app/files/'
# ---- THIS DEPENDS ON THE MODEL ----
PROCESSOR_PATH = 'microsoft/layoutlmv3-base'
MODEL_PATH = 'atatavana/passive_invoices_v2.2'
lang = 'eng'
# -----------------------------------

# define id2label: list of entities the model was trained for
# ---- THIS DEPENDS ON THE MODEL ----
id2label= {0: 'client_code', 1: 'client_id', 2: 'client_reference', 3: 'client_vat', 4: 'delivery_place', 5: 'detail_desc', 6: 'detail_packnr', 7: 'detail_packtype', 8: 'detail_qty', 9: 'detail_weight', 10: 'detail_weight_um', 11: 'doc_date', 12: 'doc_nr', 13: 'doc_type', 14: 'issuer_addr', 15: 'issuer_cap', 16: 'issuer_city', 17: 'issuer_contact', 18: 'issuer_contact_email', 19: 'issuer_contact_phone', 20: 'issuer_fax', 21: 'issuer_name', 22: 'issuer_prov', 23: 'issuer_state', 24: 'issuer_tel', 25: 'issuer_vat', 26: 'operation_code', 27: 'order_date', 28: 'order_nr', 29: 'others', 30: 'pickup_date', 31: 'pickup_place', 32: 'receiver_addr', 33: 'receiver_cap', 34: 'receiver_city', 35: 'receiver_fax', 36: 'receiver_name', 37: 'receiver_prov', 38: 'receiver_state', 39: 'receiver_tel', 40: 'receiver_vat', 41: 'recipient_name', 42: 'ref_nr', 43: 'sender_name', 44: 'service_date', 45: 'service_date-end', 46: 'service_key', 47: 'service_order', 48: 'service_value', 49: 'shipment_nr', 50: 'time', 51: 'tot_value'}

header_keys = ['client_code', 'client_id', 'client_reference', 'client_vat', 'delivery_place', 'detail_desc', 'detail_packnr', 'detail_packtype', 'detail_qty', 'detail_weight', 'detail_weight_um', 'doc_date', 'doc_nr', 'doc_type', 'issuer_addr', 'issuer_cap', 'issuer_city', 'issuer_contact', 'issuer_contact_email', 'issuer_contact_phone', 'issuer_fax', 'issuer_name', 'issuer_prov', 'issuer_state', 'issuer_tel', 'issuer_vat', 'operation_code', 'order_date', 'order_nr', 'others', 'pickup_date', 'pickup_place', 'receiver_addr', 'receiver_cap', 'receiver_city', 'receiver_fax', 'receiver_name', 'receiver_prov', 'receiver_state', 'receiver_tel', 'receiver_vat', 'recipient_name', 'ref_nr', 'sender_name', 'service_date', 'service_date-end', 'service_order', 'shipment_nr', 'time', 'tot_value']
details_keys = ['service_key', 'service_value']
# LOAD TOKENS FROM HUGGINGFACE
load_dotenv()
auth_token = os.environ.get("TOKEN") or True
# -----------------------------------

processor = AutoProcessor.from_pretrained(MODEL_PATH, apply_ocr=False, use_auth_token=auth_token)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH, use_auth_token=auth_token)

#ppocr = PaddleOCR(lang='en', use_gpu=False)

logger.info('Model loaded')

def elab(filePath, ocr):
    logger.info('elab start')
    # response = mockupElab(filePath)
    elab_data, dataFrameForSorting = process_PDF(filePath, ocr)
    response = dict()
    count_pag = 0
    for page in elab_data:
        if len(page) != 0:
            response[count_pag] = structuredResponse(elab_data[page], count_pag)
        else:
            pass
        count_pag += 1
    logger.info('elab end')
    return unify_response(response), response, dataFrameForSorting


def mockupElab(filePath):
    response = dict()
    response[0] = structuredResponse("test", 0)
    return response


def compute_detection_index(expected_keys, found_keys, found_details, expected_details_keys):
    # TODO: UPDATE
    # for the key-value pairs, we check which labels we found wrt the set of expected labels
    details_total_score = 0  # for the details, we check if the values are not null
    if len(found_details) != 0:
        for row in found_details:
            row_count = 0
            for element in row:
                if isinstance(element, float) and element != np.nan:
                    row_count += 1
                elif isinstance(element, list) and len(element) != 0:
                    row_count += 1
                elif isinstance(element, str) and element != "":
                    row_count += 1
            row_score = row_count/len(expected_details_keys)
            details_total_score += row_score
        details_score = details_total_score/len(found_details)
    else:
        details_score = 0
    return ((len(found_keys)/len(expected_keys)) + details_score)/2


def prune_text(text):
    chars = "\\`*_{}[]()>#+-.!$"
    for c in chars:
        if c in text:
            text = text.replace(c, "\\" + c)
    return text


def structuredResponse(content, n_pag):
    # ---- THIS DEPENDS ON THE MODEL ----
    header_keys = ['client_code', 'client_id', 'client_reference', 'client_vat', 'delivery_place', 'detail_desc',
                   'detail_packnr', 'detail_packtype', 'detail_qty', 'detail_weight', 'detail_weight_um', 'doc_date',
                   'doc_nr', 'doc_type', 'issuer_addr', 'issuer_cap', 'issuer_city', 'issuer_contact',
                   'issuer_contact_email', 'issuer_contact_phone', 'issuer_fax', 'issuer_name', 'issuer_prov',
                   'issuer_state', 'issuer_tel', 'issuer_vat', 'operation_code', 'order_date', 'order_nr', 'others',
                   'pickup_date', 'pickup_place', 'receiver_addr', 'receiver_cap', 'receiver_city', 'receiver_fax',
                   'receiver_name', 'receiver_prov', 'receiver_state', 'receiver_tel', 'receiver_vat', 'recipient_name',
                   'ref_nr', 'sender_name', 'service_date', 'service_date-end', 'service_order', 'shipment_nr', 'time',
                   'tot_value']
    det_keys = ['service_key', 'service_value']
    # -----------------------------------
    
    result = dict()
    data = []

    if str(content) == "test":
        main_keys = header_keys
        details_keys = det_keys
        main_values = []
        details_values = [['CRXU 721825/0',
                       'SEAL 3781204 31237276',
                       '1',
                       '20\' OPEN TOP',
                       'KGS 2000',
                       '2 PACKAGES',
                       'KGS 14000']]
    else:
        values = []
        det_values = []
        details = dict()
        count = 0
        if len(content) != 0:
            content_main = content[0]
            content_details = content[1]
            if len(content_main) != 0:
                main_keys = content_main["labels"]
                main_values = content_main["values"]
            else:
                main_keys = []
                main_values = []
            if len(content_details) != 0:
                details_keys = content_details.columns.values.tolist()
                details_values = content_details.values.tolist()
            else:
                details_keys = det_keys
                details_values = []
        else:
            main_keys = []
            main_values = []
            details_keys = det_keys
            details_values = []
    
    # detection_index = compute_detection_index(keys, main_keys, details_values, det_keys)
    # TODO: UPDATE
    detection_index = 0.8
    result['detection_index'] = "{:.2f}".format(detection_index)

    header = dict()
    header['key'] = 'Header'
    header['type'] = 'Inputs'
    h_values = []
    for k,v in zip(main_keys, main_values):
        val = dict()
        val['key'] = k
        val['value'] = str(v).strip()
        val['state'] = 'INCOMPLETE'
        # val['coordinates'] = []
        h_values.append(val)
    header['value'] = h_values
    # header['coordinates'] = [[0,0], [0,0], [0,0], [0,0], [0,0]]
    header['page'] = n_pag + 1
    if len(h_values) != 0:
        data.append(header)

    # detail part
    details = dict()
    details['key'] = 'Details'
    details['type'] = 'Table'
    d_values = dict()
    d_values['header'] = details_keys
    d_data = []
    for row in details_values:
        stripped = [str(s).strip() for s in row]
        clean = [prune_text(str(s)) for s in stripped]
        d_data.append(clean)
    d_values['data'] = d_data
    details['value'] = d_values
    # details['coordinates'] = [[0,0], [0,0], [0,0], [0,0], [0,0]]
    details['page'] = n_pag + 1
    if len(d_data) != 0:
        data.append(details)
    
    result['data_to_review'] = data
    
    # with open("files/test.json", "w") as outfile:
    #     json.dump(result, outfile)
    return result


def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]


def intersect(w, z):
    # this method will detect if there is any intersect between two boxes or not
    x1 = max(w[0], z[0]) #190  | 881  |  10
    y1 = max(w[1], z[1]) #90   | 49   | 273
    x2 = min(w[2], z[2]) #406  | 406  | 1310
    y2 = min(w[3], z[3]) #149  | 703  | 149
    if (x1 > x2 or y1 > y2):
        return 0
    else:
        # because sometimes in annotating, it is possible to overlap rows or columns by mistake 
        # for very small pixels, we check a threshold to delete them
        area = (x2-x1) * (y2-y1)
        if (area > 0):  #500 is minumum accepted area
            return [int(x1), int(y1), int(x2), int(y2)]
        else:
            return 0

# calculates the verticle distance between boxes
def dist_height(y1,y2):
    return abs(y1- y2)


def mergeBoxes(df):
    xmin, ymin, xmax, ymax = [], [], [], []
    for i in range(df.shape[0]):
        box = df['bbox_column'].iloc[i]
        xmin.append(box[0])
        ymin.append(box[1])
        xmax.append(box[2])
        ymax.append(box[3])
    return [min(xmin), min(ymin), max(xmax), max(ymax)]


def mergeCloseBoxes(pr, bb, wr, threshold):
    # This code will merge boxes that they have same label and they are close
    # to each other. the distance between two boxes has to be below the threshold x 2 (left and right)
    idx = 0
    final_bbox =[]
    final_preds =[]
    final_words=[]
    for box, pred, word in zip(bb, pr, wr):
        if (pred=='others'):
            continue
        else:
            flag = False
            for b, p, w in zip(bb, pr, wr):
                if (p == 'others'):
                    continue
                elif (box==b): # we shouldn't check each item with itself
                    continue
                else:
                    XMIN, YMIN, XMAX, YMAX = box
                    xmin, ymin, xmax, ymax = b
                    if (p == 'package_quantity' or p == 'weight'):
                        intsc = intersect([XMIN, YMIN, XMAX+threshold, YMAX+(15)], [xmin-threshold, ymin-(15), xmax, ymax])
                    else:
                        intsc = intersect([XMIN, YMIN, XMAX+threshold, YMAX], [xmin-threshold, ymin, xmax, ymax])
                    if (intsc != 0 and pred==p):
                        flag = True
                        # we have to check if there is any intersection between box and all the other boxes in final_bbox list
                        # because if we have added it before, just we have to update (expand) it
                        merged_box = [
                            min(XMIN, xmin),
                            min(YMIN, ymin),
                            max(XMAX, xmax),
                            max(YMAX, ymax)
                        ]
                        merged_words = word + ' ' + w
                        # add to final_bbox
                        wasAvailable = False
                        for id, fbox in enumerate(final_bbox):
                            if (intersect(box, fbox) != 0 and pred==final_preds[id]):
                                wasAvailable = True
                                if (isInside(merged_box, fbox)):
                                    # box is EXACTLY inside another processed box, so we added it before and we should leave it now
                                    # example: N  1  20  BOX --> intersect(N&20)=True
                                    break
                                # box has intersect with another added box, so we have to update it
                                merged_box = [
                                    min(fbox[0], min(XMIN, xmin)),
                                    min(fbox[1], min(YMIN, ymin)), 
                                    max(fbox[2], max(XMAX, xmax)), 
                                    max(fbox[3], max(YMAX, ymax))
                                ]
                                final_bbox[id] = merged_box
                                final_words[id] = final_words[id] + ' ' + w
                                break
                        
                        if (not wasAvailable):
                            # there was no intersect, bbox is not added before
                            final_bbox.append(merged_box)
                            final_preds.append(pred)
                            final_words.append(merged_words)
            # ---- for loop finishes here ----
            if (flag == False):
                # there was no intersect between bbox and the other bboxes
                # we will check for last time if box is inside the others, because if the word is last word (like Juan + Mulian + Alexander) (Alexander)
                # it is added before but it has not intersection with others, so we will check to prevent
                for id, fbox in enumerate(final_bbox):
                    if (intersect(box, fbox) != 0 and pred==final_preds[id]):
                        flag = True

                if (not flag):
                    final_bbox.append(box)
                    final_preds.append(pred)
                    final_words.append(word)

    return final_bbox, final_preds, final_words


def isInside(w, z):
    # return True if w is inside z, if z is inside w return false
    if(w[0] >= z[0] and w[1] >= z[1] and w[2] <= z[2] and w[3] <= z[3]):
        return True
    return False


def removeSimilarItems(final_bbox, final_preds, final_words):
    _bb =[] 
    _pp=[] 
    _ww=[]
    for i in range(len(final_bbox)):
        _bb.append(final_bbox[i])
        _pp.append(final_preds[i])
        _ww.append(final_words[i])
        for j in range(len(final_bbox)):
            if (final_bbox[i] == final_bbox[j]):
                continue
            elif (isInside(final_bbox[i], final_bbox[j]) and final_preds[i]==final_preds[j] ):
                # box i is inside box j, so we have to remove it
                _bb = _bb[:-1]
                _pp = _pp[:-1]
                _ww = _ww[:-1]
                continue
    return _bb, _pp, _ww


def rearrange_service_df(detail_df):
    # this function will rearrange order of service_key and service_value to fix some of the issues
    # caused by OCR (for some cases OCR is not able to detect the exact coordinates of a bbox)
    last_seen_prediction = ''
    for index, row in detail_df.iterrows():
        if row['predictions'] == last_seen_prediction:
            print(f'error in row: {index}')
            # check the length of dataframe and next prediction
            if index + 1 < len(detail_df) and detail_df.iloc[index + 1]['predictions'] != last_seen_prediction:
                # calculate their distances
                distance = abs(row['bbox'][1] - detail_df.iloc[index + 1]['bbox'][1])
                if distance < 8:
                    print(f'row {index + 1} can be changed with row: {index}')
                    detail_df.loc[index, :], detail_df.loc[index + 1, :] = detail_df.loc[index + 1,
                                                                           :].copy(), detail_df.loc[index, :].copy()

        else:
            last_seen_prediction = row['predictions']

    return detail_df

def getDetailDataframeFromServicesDataframe(services_df):
    # This function will receive the "service_key"s and "service_value"s and match them
    # by expanding the bounding boxes and calculating the intersection between two boxes
    # and deliver a new dataframe named detail_df
    detail_df = pd.DataFrame(columns=['service_key', 'service_value'])
    matched_indexes = []
    data_to_concat = []
    for index, row in services_df.iterrows():
        if index not in matched_indexes:
            flag = False  # Reset the flag for each row
            service_key = row['words']
            # Get the bounding box coordinates of the current row
            x1, y1, x2, y2 = row['bbox']
            # Iterate through the other rows in detail_df
            for other_index, other_row in services_df.iterrows():
                # Skip the same row
                if index == other_index:
                    continue

                expanded_box_from_right = [x1, y1, other_row['bbox'][2], y2]
                expanded_box_from_left = [other_row['bbox'][0], y1, x2, y2]

                if intersect(expanded_box_from_right, other_row['bbox']) != 0 or intersect(
                        expanded_box_from_left, other_row['bbox']) != 0:
                    if (row["predictions"] == "service_key" and other_row["predictions"] == "service_value") or (
                            row["predictions"] == "service_value" and other_row["predictions"] == "service_key"):
                        print(f'{row["words"]} ---> {other_row["words"]}')
                        flag = True
                        matched_indexes.append(index)
                        matched_indexes.append(other_index)
                        data_to_concat.append(
                            {row["predictions"]: row["words"], other_row["predictions"]: other_row["words"]})
                        break
                    else:
                        print('same labels have intersection!')
            if flag:
                continue
            else:
                print(f'{row["words"]}')
                matched_indexes.append(index)
                data_to_concat.append({row["predictions"]: row["words"]})

    detail_df = pd.concat([detail_df, pd.DataFrame(data_to_concat)], ignore_index=True)
    return detail_df

def createDataframes(predictions, words, boxes):
    data = {'predictions': predictions, 'words': words, 'bbox': boxes}
    df = pd.DataFrame(data)

    services_df = df[(df['predictions'] == 'service_key') | (df['predictions'] == 'service_value')]
    # reset index and sort services_df
    services_df = services_df.reset_index(drop=True)
    services_df = rearrange_service_df(services_df)
    detail_df = getDetailDataframeFromServicesDataframe(services_df)

    header_df = df[(df['predictions'] != 'service_key') & (df['predictions'] != 'service_value')]
    header_df = header_df.drop(columns=['bbox'])
    header_df = header_df.rename(columns={'predictions': 'labels', 'words': 'values'})

    return header_df, detail_df


def process_results(preds, words, bboxes, prev_df, page_number, dataFrameForSorting):
    logger.info('process_results start')
    # first we need to merge boxes that have same label and close to each  other
    # last argument is threshold value to set the maximum distance between words
    final_bbox, final_preds, final_words = mergeCloseBoxes(preds, bboxes, words, 70)
    _bbox, _preds, _words = removeSimilarItems(final_bbox, final_preds, final_words)
    # convert float list to int
    _bbox = [[int(x) for x in item ] for item in _bbox]
    # sorting from top left, to bottom right 
    if (len(_preds) != 0):
        zipped_list = list(zip(_preds, _bbox, _words))
        sorted_list = sorted(zipped_list, key=lambda x: (x[1][1], x[1][0]))
        _preds, _bbox, _words = zip(*sorted_list)
    
    print('#preds: ' + str(len(_preds))+ ', #words: ' + str(len(_words)))

    #dataFrameForSorting = createDataFrameForSorting(_preds, _words, _bbox, page_number, dataFrameForSorting)
    df_main, df_details = createDataframes(_preds, _words, _bbox)
    #prev_df = df_details.copy().reset_index(drop = True)

    logger.info('process_form end')
    #return [df_main, df_details], prev_df, dataFrameForSorting
    return [df_main, df_details]

def createDataFrameForSorting(predictions, words, bboxes, page_number, previous_dataframe):
    if len(predictions) > 0:
        # Create a new DataFrame with the values from the lists
        new_data = pd.DataFrame({'page': page_number, 'box': bboxes, 'word': words, 'label': predictions})

        # Concatenate the new DataFrame with the existing DataFrame
        updated_dataframe = pd.concat([previous_dataframe, new_data], ignore_index=True)
        return updated_dataframe
    else:
        return previous_dataframe

def process_PDF(filePath, ocr):
    logger.info('process_PDF start')
    # we unpack the PDF into multiple images
    doc = fitz.open(filePath)

    result = {}
    prev_df = pd.DataFrame(columns = details_keys)
    dataFrameForSorting = pd.DataFrame(columns=['page', 'box', 'word', 'label'])
    for i in range(0, doc.page_count):
        page = doc.load_page(i)     # number of page
        zoom = 2                    # zoom factor
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix = mat, dpi = 300)
        if filePath[-4:] == ".pdf":
            imgOutput = filePath.replace(".pdf", "_{}.png".format(i))
        elif filePath[-4:] == ".PDF":
            imgOutput = filePath.replace(".PDF", "_{}.png".format(i))
        pix.save(imgOutput)
        processedImgOutput = imgOutput.replace(".png", "_processed.png")
        rotation = checkRotation(imgOutput)
        if rotation == -1:
            # img is blank: delete it?
            os.remove(imgOutput)
            result[(imgOutput.replace(UPLOAD_FOLDER, ""))] = []
            continue
        elif rotation != 0:
            im = Image.open(imgOutput)
            rotated = im.rotate(-(int(rotation)), expand=True)
            angle, skewed_image = correct_skew(rotated)
            out = remove_borders(skewed_image)
            cv2.imwrite(imgOutput, skewed_image)
            cv2.imwrite(processedImgOutput, out)
            # out.save(imgOutput)
        else:
            im = Image.open(imgOutput)
            angle, skewed_image = correct_skew(im)
            out = remove_borders(skewed_image)
            cv2.imwrite(imgOutput, skewed_image)
            cv2.imwrite(processedImgOutput, out)
        # each image goes through the model
        #pageResult, prev_df, dataFrameForSorting = process_page(imgOutput, processedImgOutput, ocr, prev_df, i, dataFrameForSorting)
        pageResult = process_page(imgOutput, processedImgOutput, ocr, prev_df, i, dataFrameForSorting)
        # result is saved in a dict-like shape to be returned
        result[(imgOutput.replace(UPLOAD_FOLDER, ""))] = pageResult

    # clean up function to delete local files - both the original pdf 
    # (that was previously uploaded to S3) and the newly created images
    if filePath[-4:] == ".pdf":
        pattern = (filePath.replace(".pdf","")) + "_*"
    elif filePath[-4:] == ".PDF":
        pattern = (filePath.replace(".PDF","")) + "_*"
    # cleanup(pattern)

    logger.info('process_PDF end')
    return result, dataFrameForSorting


def process_image(filePath, ocr):
    #page_result, _, _ = process_page(filePath, filePath, ocr, pd.DataFrame(), 0, pd.DataFrame())
    page_result = process_page(filePath, filePath, ocr, pd.DataFrame(), 0, pd.DataFrame())
    result = {(filePath.replace(UPLOAD_FOLDER, "")): page_result}
    
    # clean up function to delete local files - both the original pdf 
    # (that was previously uploaded to S3) and the newly created images
    pattern = filePath
    cleanup(pattern)

    return result


def process_page(filePath, processedImgOutput, ocr, prev_df, page_number, dataFrameForSorting):
    logger.info('process_page start')
    # load image (at this stage all pdf pages have been transformed to images)
    image = Image.open(filePath).convert("RGB")
    processedImage = Image.open(processedImgOutput).convert("RGB")
    bboxes, prds, words, image = infer(image, processedImage, ocr)
    predictions = []
    for id in prds:
        predictions.append(id2label.get(id))

    # output of function below would be: dfs = [df_main, df_detail]
    # dataFrameForSorting could be used for getting Excel files as an output
    dfs = process_results(predictions, words, bboxes, prev_df, page_number, dataFrameForSorting)

    logger.info('process_page end')
    return dfs #, prev_df, dataFrameForSorting


def create_bounding_box_paddle(bbox_data, width_scale, height_scale):
    xs = []
    ys = []
    for x, y in bbox_data:
        xs.append(x)
        ys.append(y)

    left = int(max(0, min(xs) * width_scale))
    top = int(max(0, min(ys) * height_scale))
    right = int(min(1000, max(xs) * width_scale))
    bottom = int(min(1000, max(ys) * height_scale))

    return [left, top, right, bottom]


def aws_processor(image, width, height):
    # extract text from image
    response = AWSutils.detect_document_text(image)

    # process response to get text and bounding boxes
    words = []
    bboxes = []
    for item in response['Blocks']:
        if item['BlockType'] == 'WORD':
            words.append(item['Text'])
            bbox = item['Geometry']['BoundingBox']
            # rescale bbox coordinates to be within 0-1000 range
            x1 = int(bbox['Left'] * 1000)
            y1 = int(bbox['Top'] * 1000)
            x2 = int((bbox['Left'] + bbox['Width']) * 1000)
            y2 = int((bbox['Top'] + bbox['Height']) * 1000)
            bboxes.append((x1, y1, x2, y2))

    return words, bboxes


def create_bounding_box_vision_ocr(vertices, width_scale, height_scale):
    # Get the x, y coordinates
    x1 = int(vertices[0].x * width_scale)
    y1 = int(vertices[0].y * height_scale)

    x2 = int(vertices[2].x * width_scale)
    y2 = int(vertices[2].y * height_scale)

    # Validate x1 < x2
    if x1 > x2:
        x1, x2 = x2, x1

    # Validate y1 < y2
    if y1 > y2:
        y1, y2 = y2, y1

    # Return valid bounding box
    return [x1, y1, x2, y2]


def google_vision_processor(image, width, height):
    #inference_image = [image.convert("RGB")]
    client = vision.ImageAnnotatorClient()
    with io.BytesIO() as output:
        image.save(output, format='JPEG')
        content = output.getvalue()
    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    # Get the bounding box vertices and remove the first item
    bboxes = [text.bounding_poly.vertices[1:] for text in texts]
    # Create the list of words and boxes
    words = [text.description for text in texts]
    boxes = [create_bounding_box_vision_ocr(bbox, 1000 / width, 1000 / height) for bbox in bboxes]
    return words, boxes


def paddle_processor(image,width,height):
    width, height = image.size
    width_scale = 1000 / width
    height_scale = 1000 / height
    
    # Perform OCR on the image
    #results = ppocr.ocr(np.array(image))
    results = []
    
    # Extract the words and bounding boxes from the OCR results
    words = []
    boxes = []
    for line in results:
        for bbox in line:
            words.append(bbox[1][0])
            boxes.append(create_bounding_box_paddle(bbox[0], width_scale, height_scale))
    return words, boxes


# function infer might change with new findings on model
def infer(image, processedImage, ocr):
    width, height = image.size
    lang = 'eng'
    custom_config = r'--oem 3 --psm 6'
    
    if ocr == 'aws':
        words, boxes = aws_processor(processedImage, width, height)
        logger.info('infer - aws_processor OK')
    elif ocr == 'paddle':
        words, boxes = paddle_processor(processedImage, width, height)
        logger.info('infer - paddle_processor OK')
    elif ocr == 'google':
        words, boxes = google_vision_processor(processedImage, width, height)
        logger.info('infer - google_vision_processor OK')
    else:
        feature_extractor = LayoutLMv3ImageProcessor(apply_ocr = True,
                                                    config = custom_config,
                                                    lang = lang)
        # feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr = True, lang = lang, config = custom_config)
        logger.info('infer - feature extractor OK')
        encoding_feature_extractor = feature_extractor(processedImage, return_tensors="pt", truncation = True)
        logger.info('infer - encoding feature extractor OK')
        words, boxes = encoding_feature_extractor.words, encoding_feature_extractor.boxes

    # encode
    encoding = processor(image, words, boxes = boxes, truncation = True, return_offsets_mapping = True, return_tensors="pt", 
                         padding = "max_length", stride = 128, max_length = 512, return_overflowing_tokens = True)
    logger.info('infer - encoding OK')
    offset_mapping = encoding.pop('offset_mapping')
    overflow_to_sample_mapping = encoding.pop('overflow_to_sample_mapping')

    # change the shape of pixel values
    encoding['pixel_values'] = torch.stack(encoding['pixel_values'])

    logger.info('infer - change shape of pixel values OK')

    # forward pass
    with torch.no_grad():
        outputs = model(**encoding)

    # get predictions
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoding.bbox.squeeze().tolist()

    if (len(token_boxes) == 512):
        predictions = [predictions]
        token_boxes = [token_boxes]


    box_token_dict = {}
    for i in range(0, len(token_boxes)):
        # skip first 128 tokens from second list to last one (128 is the stride value) except the first window
        initial_j = 0 if i == 0 else 129
        for j in range(initial_j, len(token_boxes[i])):
            unnormal_box = unnormalize_box(token_boxes[i][j], width, height)
            if (np.asarray(token_boxes[i][j]).shape != (4,)):
                continue
            elif (token_boxes[i][j] == [0, 0, 0, 0] or token_boxes[i][j] == 0):
                continue
            else:
                bbox = tuple(unnormal_box)  # Convert the list to a tuple
                token = processor.tokenizer.decode(encoding["input_ids"][i][j])
                if bbox not in box_token_dict:
                    box_token_dict[bbox] = [token]
                else:
                    box_token_dict[bbox].append(token)

    box_token_dict = {bbox: [''.join(words)] for bbox, words in box_token_dict.items()}
    boxes = list(box_token_dict.keys())
    words = list(box_token_dict.values())
    preds = []

    box_prediction_dict = {}
    for i in range(0, len(token_boxes)):
        for j in range(0, len(token_boxes[i])):
            if (np.asarray(token_boxes[i][j]).shape != (4,)):
                continue
            elif (token_boxes[i][j] == [0, 0, 0, 0] or token_boxes[i][j] == 0):
                continue
            else:
                bbox = tuple(token_boxes[i][j])  # Convert the list to a tuple
                prediction = predictions[i][j]
                if bbox not in box_prediction_dict:
                    box_prediction_dict[bbox] = [prediction]
                else:
                    box_prediction_dict[bbox].append(prediction)

    for i, (bbox, predictions) in enumerate(box_prediction_dict.items()):
        count_dict = {}  # Dictionary to store the count of each prediction label
        for prediction in predictions:
            if prediction in count_dict:
                count_dict[prediction] += 1
            else:
                count_dict[prediction] = 1

        max_count = max(count_dict.values())  # Find the maximum count of a prediction label
        max_predictions = [key for key, value in count_dict.items() if
                           value == max_count]  # Find the predictions with the maximum count
        # If there is only one prediction with the maximum count so we can add it to the preds, but
        # if they're more than one, so we don't know which one should be label for the box
        #print(f'count_dict: {count_dict}, word: {words[i]}')
        if len(max_predictions) == 1:
            preds.append(max_predictions[0])
        else:
            others_id = next((id_ for id_, label in id2label.items() if label == 'others'), None)
            #print(f'others_id:{others_id}')
            print(f'count dict for word: {words[i]} (prediction:votes) is: {count_dict}')
            if others_id in count_dict:
                count_dict.pop(others_id)
                new_max_count = max(count_dict.values())
                new_max_predictions = [key for key, value in count_dict.items() if
                                       value == new_max_count]  # Find the predictions with the maximum count
                if len(new_max_predictions) == 1:
                    preds.append(new_max_predictions[0])
                    continue
            # the idea is to look at next and previous items in the dictionary, if they have a label like the labels
            # that we have for this box, so we can select it for label
            # ex: ANKORIG SRL - ANKORIG = [0, 6] SRL = [6] so--> ANKORIG could be considered as 6
            max_next_prev_item = []
            # Check the previous item (if it exists)
            if i - 1 >= 0:
                prev_predictions = box_prediction_dict[
                    list(box_prediction_dict.keys())[i - 1]]  # Get the predictions of the previous item
                prev_count_dict = {key: prev_predictions.count(key) for key in
                                   set(prev_predictions)}  # Count the occurrences of each prediction
                max_prev_count = max(prev_count_dict.values())  # Find the maximum count of a prediction label
                max_prev_predictions = [key for key, value in prev_count_dict.items() if
                                        value == max_prev_count]  # Find the predictions with the maximum count
                max_next_prev_item.extend(max_prev_predictions)  # Add the predictions to the max_next_prev_item list

            # Check the next item (if it exists)
            if i + 1 < len(box_prediction_dict):
                next_predictions = box_prediction_dict[
                    list(box_prediction_dict.keys())[i + 1]]  # Get the predictions of the next item
                next_count_dict = {key: next_predictions.count(key) for key in
                                   set(next_predictions)}  # Count the occurrences of each prediction
                max_next_count = max(next_count_dict.values())  # Find the maximum count of a prediction label
                max_next_predictions = [key for key, value in next_count_dict.items() if
                                        value == max_next_count]  # Find the predictions with the maximum count
                max_next_prev_item.extend(max_next_predictions)  # Add the predictions to the max_next_prev_item list
            print(f'max_next_prev_item is: {max_next_prev_item}')
            if any(prediction in max_next_prev_item for prediction in max_predictions):  # If there are common predictions between max_next_prev_item and max_predictions
                common_predictions = list(
                    set(max_predictions).intersection(max_next_prev_item))  # Find the common predictions
                # Add a randomly chosen common prediction to the preds, random would work if we have again more than one choice
                preds.append(random.choice(common_predictions))
            else:
                preds.append(random.choice(
                    max_predictions))  # Add a randomly chosen prediction with the maximum count to the preds

    flattened_words = [word[0].strip() for word in words]  # because words are list of lists
    # for word, prediction in zip(flattened_words, preds):
    #     print(f'word: {word} ---> {prediction}')
    return boxes, preds, flattened_words, image
    

def cleanup(pattern):
    for f in os.listdir(UPLOAD_FOLDER):
        if re.search(pattern, os.path.join(UPLOAD_FOLDER, f)):
            os.remove(os.path.join(UPLOAD_FOLDER, f))


def checkRotation(filePath):
    im = skio.imread(filePath)
    try:
        newdata = pytesseract.image_to_osd(im, nice=1)
        rotation = re.search('(?<=Rotate: )\d+', newdata).group(0)
    except:
        # Exception might happen with blank pages (tesseract not detecting anything)
        # so to mark it we set rotation = -1
        rotation = -1
    return rotation


# correct the skewness of images
def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    # Convert the PIL Image object to a numpy array
    image = np.asarray(image.convert('L'), dtype=np.uint8)

    # Apply thresholding
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)
    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
            borderMode=cv2.BORDER_REPLICATE)
    return best_angle, corrected


def remove_borders(img):
    result = img.copy()
    
    try:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) # convert to grayscale
    except:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        gray = result[:, :, 0]
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (255,255,255), 5)

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
    remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (255,255,255), 5)
    return result


def unify_response(response):
    unified_response = dict()
    detection_index = 0
    data_to_review = []
    count_important = 0
    for pag_nr in response:
        pag_data = response[pag_nr]
        if len(pag_data["data_to_review"]) != 0:
            count_important += 1
            detection_index += float(pag_data["detection_index"])
            data_to_review.append(pag_data["data_to_review"])
    unified_response["detection_index"] = detection_index/count_important
    unified_response["data_to_review"] = data_to_review
    return unified_response

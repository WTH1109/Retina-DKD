import os
import xlrd
import xlwt
import xlutils.copy
import re


def read_xlsx_5(path_r):
    id_search = {}
    id_data_non_invasive = []
    id_data_invasive = []
    id_lesion_type = []

    data = xlrd.open_workbook(path_r)
    table = data.sheets()[0]
    for i_r in range(1, table.nrows):
        id_usr = table.cell(i_r, 1).value
        lesion_type_str = table.cell_value(i_r, 2)
        press_r = table.cell_value(i_r, 3)
        course_r = table.cell_value(i_r, 4)
        mashing_r = table.cell_value(i_r, 5)
        protein_r = table.cell_value(i_r, 6)
        eGFR_r = table.cell_value(i_r, 7)
        if lesion_type_str == 'DN':
            lesion_type = 0
        elif lesion_type_str == 'NDRD':
            lesion_type = 2
        else:
            lesion_type = 1
        if not isinstance(id_usr, str):
            id_usr = int(id_usr)
            id_usr = str(id_usr)
        id_search[id_usr] = i_r - 1
        if mashing_r == '':
            mashing_r = 0.0
            protein_r = 0.0
        if course_r == '':
            course_r = 0.0
        press_r = press_r / 202.0
        course_r = course_r / 564.0
        mashing_r = mashing_r / 13.8
        protein_r = protein_r / 207.0
        eGFR_r = eGFR_r / 132.0
        non_invasive_r = [press_r, course_r]
        invasive_r = [mashing_r, protein_r, eGFR_r]
        id_data_non_invasive.append(non_invasive_r)
        id_data_invasive.append(invasive_r)
        id_lesion_type.append(lesion_type)
    return id_search, id_data_non_invasive, id_data_invasive, id_lesion_type


def get_id(img_name_r):
    split_str = re.split('-|_', img_name_r)
    if len(split_str) > 3:
        if len(split_str[0]) < 4:
            id_name = split_str[1]
        else:
            id_name = split_str[0]
    else:
        if len(split_str[0]) < 4:
            id_name = split_str[-2]
        else:
            id_name = split_str[0]
    if id_name[0] == 'N' or id_name[0] == 'n':
        id_name = id_name[1:]
    id_name = id_name.capitalize()
    return id_name


def test_data_write_xlsx(xlsx_name, model_name, dataset, dataset_num, data_all, model_ori_name, sta_num):
    if not os.path.exists('./excel_data'):
        os.mkdir('./excel_data')
    if not os.path.exists('excel_data/' + xlsx_name):
        excel_write = xlwt.Workbook(encoding='utf-8', style_compression=0)
    else:
        excel_rd = xlrd.open_workbook('excel_data/' + xlsx_name, formatting_info=True)
        excel_write = xlutils.copy.copy(excel_rd)
    try:
        sheet1 = excel_write.get_sheet(model_name)
    except:
        sheet1 = excel_write.add_sheet(model_name, cell_overwrite_ok='True')

    if dataset_num == 1:
        sheet1.write_merge(0, 1, 0, 1, model_ori_name + ' epoc:' + str(sta_num))
        sheet1.write_merge(0, 0, 2, 3, 'Acc')
        sheet1.write_merge(0, 0, 4, 5, 'Sen')
        sheet1.write_merge(0, 0, 6, 7, 'Spec')
        sheet1.write_merge(0, 0, 8, 9, 'F1_S')
        sheet1.write_merge(0, 0, 10, 11, 'AUC')
        for i_r in range(5):
            sheet1.write(1, 2 * i_r + 2, 'ave')
            sheet1.write(1, 2 * i_r + 3, 'var')
        sheet1.write(dataset_num + 1, 0, dataset)
        sheet1.write(dataset_num + 1, 1, model_name)
        for i_r in range(10):
            sheet1.write(dataset_num + 1, i_r + 2, data_all[i_r])
        excel_write.save('excel_data/' + xlsx_name)
    else:
        sheet1.write(dataset_num + 1, 0, dataset)
        sheet1.write(dataset_num + 1, 1, model_name)
        for i_r in range(10):
            sheet1.write(dataset_num + 1, i_r + 2, data_all[i_r])
        excel_write.save('excel_data/' + xlsx_name)


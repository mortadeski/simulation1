# Writing to an excel
# sheet using Python
import xlwt
from xlwt import Workbook


def write_res_to_excel(names, res):

    # Workbook is created
    wb = Workbook()

    # add_sheet is used to create sheet.
    sheet1 = wb.add_sheet('Sheet 1')


    # structer of write - col, row, what to write

    #cols names
    for i in range(0, len(names)):
        col = i + 1
        sheet1.write(col, 0, names[i])

    #records values
    for i in range(0, len(res)):
        distribution_res = res[i]
        col = i + 1
        for j in range(0, len(distribution_res)):
            num = distribution_res[j]
            row = j + 1
            sheet1.write(col, row, num)



    # name of the file
    wb.save('xlwt example.xls')

# import PyPDF2
import tabula

import pandas as pd

# pdf1File = open('12321.pdf', 'rb')
# pdf1Reader = PyPDF2.PdfFileReader(pdf1File)
#
# pageObj = pdf1Reader.getPage(0)
# print(pageObj.extractText())

# df = tabula.read_pdf('12321.pdf', pages=0)[0]
#
# print(df)
#
# tabula.convert_into("12321.pdf", "12321.csv", output_format="csv", pages='all')

df = pd.read_csv('12321.csv')

print(df)
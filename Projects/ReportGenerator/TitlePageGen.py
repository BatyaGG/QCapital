from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Image, Table, TableStyle, Paragraph
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
import reportlab


from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from MergePDFs import merge_pdfs

# pdfmetrics.registerFont(TTFont('Circle_Bold', 'DejaVuSans.ttf'))
# pdfmetrics.registerFont(TTFont('FreeSans', 'FreeSans.ttf'))
# pdfmetrics.registerFont(TTFont('VeraBd', 'VeraBd.ttf'))
# pdfmetrics.registerFont(TTFont('VeraIt', 'VeraIt.ttf'))
# pdfmetrics.registerFont(TTFont('VeraBI', 'VeraBI.ttf'))

# import sys
# sys.path[0] = '../..'
# import config
from QuantumCapital import constants
from QuantumCapital.DBManager import DBManager
from QuantumCapital.PlotManager import *
# print(str(config.PROJ_DIR) + '/Projects/ReportGenerator')

pdfmetrics.registerFont(TTFont('Circe_Bold', 'Circe-Bold.ttf'))
pdfmetrics.registerFont(TTFont('Circe_Reg', 'Circe-Light.ttf'))

# A4	210 mm x 297 mm
A4_H = 297
A4_W = 210


class TitlePageGen:
    def __init__(self, out_name, project_dir):
        reportlab.rl_config.TTFSearchPath.append(str(project_dir) + '/Projects/ReportGenerator')
        self.canvas = Canvas(f"{out_name}")
        # self.canvas.setFont('Circe_Bold', 25)
        # self.canvas.setFont('Circe_Reg', 25)

    def main_drawer(self, name, surname, sex, txt):
        assert sex in ('m', 'f')
        head_img_h = 70
        head_image = Image('head_img.jpg', A4_W * mm, head_img_h * mm)
        head_image.drawOn(self.canvas, 0, (A4_H - head_img_h) * mm)

        ahmet_img_h = 767 / 7
        ahmet_img_w = 486 / 7
        ahmet_image = Image('ahmet.png', ahmet_img_w * mm, ahmet_img_h * mm)
        ahmet_image.drawOn(self.canvas, 120 * mm, 0)

        sign_img_h = 767 / 11
        sign_img_w = 486 / 11
        sign_image = Image('ahmet_sign.png', sign_img_w * mm, sign_img_h * mm)

        style_bold = ParagraphStyle(
            name='Normal',
            fontName='Circe_Bold',
            fontSize=20,
            alignment=TA_CENTER,
            borderPadding=1,
            spaceShrinkage=0.05,
        )

        style_reg = ParagraphStyle(
            alignment=TA_LEFT,
            name='Normal',
            fontName='Circe_Reg',
            fontSize=18,
            leading=16,

        )

        style_sign = ParagraphStyle(
            name='Normal',
            fontName='Circe_Bold',
            fontSize=16,
            leading=16
        )

        hello_txt = Paragraph(f"Дорог{'ой' if sex == 'm' else 'ая'} {name} {surname}", style_bold)

        txt = Paragraph(txt, style_reg)

        sign = Paragraph(f"""С уважением,<br/><br/><br/><br/><br/><br/><br/>
                                    Ахмет Бяшимов,<br/>
                                    CEO, Co-Founder Quantum Capital""", style_sign)

        data = [[hello_txt], [txt], [sign_image], [sign]]

        table_w = A4_W * 0.8
        table_h = A4_H * 0.48
        table = Table(data, colWidths=[table_w * mm, table_w * mm, table_w * mm / 2, table_w * mm / 2],
                      rowHeights=[0.1*table_h * mm, 0.65 * table_h * mm, 0 * table_h * mm, 0.16 * table_h * mm])
        table.wrapOn(self.canvas, 0, 0)
        table.setStyle(TableStyle([
            # ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
            # ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
            # ('BACKGROUND', (0, 0), (-1, 2), colors.lightgrey),
            # ('FONTNAME', (0,0), (0,-1), 'Circe_Bold'),
            ('FONTSIZE', (0, 1), (-1, 1), 50),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        table.drawOn(self.canvas, mm * (A4_W - table_w) / 2, (ahmet_img_h - 30) * mm)

    def save(self):
        self.canvas.save()


if __name__ == '__main__':
    gen = TitlePageGen('ahmet.pdf', '/home/batyrkhan/Coding_Projects/QCapital')
#     txt = """Все позиции портфеля Quantum Capital,
#  отчитывающиеся на ежеквартальной основе, превзошли
# консенсус-прогнозы и подтвердили наше фундаментальное
# видение.
# Также на прошлой неделе были опубликованы отчеты по
# первоначальным заявкам на пособие по безработице и ВВП США.
# Показатели оказались в прогнозируемом диапазоне, что указывает
# на дальнейшее восстановление экономики США:
# − заявки на пособие по безработице: 406K (ожидалось 425K);
# − рост ВВП США в 1 квартале: 6.4% годовых (ожидалось 6.5%).
# При этом рост потребительских расходов был компенсирован
# увеличением торгового дефицита в структуре ВВП."""
    title_page_text_path = 'Reports/title_text.txt'

    with open(title_page_text_path, 'r') as file:
        title_page_text = file.read().replace('\n', ' ')
    print(title_page_text)
    gen.main_drawer('', "Клиент", 'm', title_page_text)
    gen.save()
    report_path = 'Reports/Годовой2.pdf'
    merge_pdfs('ahmet.pdf', report_path, 'годовой.pdf')

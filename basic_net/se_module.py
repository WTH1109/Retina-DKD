from torch import nn
import os
import xlrd
import xlwt
import xlutils.copy


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.Attention_CNN_set = []
        self.Attention_Trans_set = []
        self.write = []

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # attention = y.reshape((-1))
        # attention_cnn = attention[0:512].cpu().abs().sum().detach().numpy()
        # attention_trans = attention[512:1024].cpu().abs().sum().detach().numpy()
        # self.Attention_CNN_set.append(attention_cnn)
        # self.Attention_Trans_set.append(attention_trans)
        # sheet1, excel_write = attention_xlsx_initial()
        # sheet1.write(len(self.Attention_CNN_set), 1, str(attention_cnn))
        # sheet1.write(len(self.Attention_Trans_set), 2, str(attention_trans))
        # if len(self.Attention_CNN_set) == 1:
        #     sheet1.write(0, 1, 'cnn')
        # else:
        #     sheet1.write(0, 2, 'trans')
        # excel_write.save('/mnt/disk10T/wengtaohan/Code/DKD/test_file/excel_data/' + 'attention.xls')
        return x * y.expand_as(x)


def attention_xlsx_initial():
    xlsx_name = 'attention.xls'

    model_name = 'sheet1'
    if not os.path.exists('/mnt/disk10T/wengtaohan/Code/DKD/test_file/excel_data'):
        os.mkdir('/mnt/disk10T/wengtaohan/Code/DKD/test_file/excel_data')
    if not os.path.exists('/mnt/disk10T/wengtaohan/Code/DKD/test_file/excel_data/' + xlsx_name):
        excel_write = xlwt.Workbook(encoding='utf-8', style_compression=0)
    else:
        excel_rd = xlrd.open_workbook('/mnt/disk10T/wengtaohan/Code/DKD/test_file/excel_data/' + xlsx_name,
                                      formatting_info=True)
        excel_write = xlutils.copy.copy(excel_rd)
    try:
        sheet1 = excel_write.get_sheet(model_name)
    except:
        sheet1 = excel_write.add_sheet(model_name, cell_overwrite_ok='True')
    return sheet1, excel_write

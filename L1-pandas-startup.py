import pandas as pd

data = pd.read_excel('./ex01.xlsx', sheet_name = None)
xls = pd.ExcelFile('./ex01.xlsx')
xls.sheet_names
print("工作表：", xls.sheet_names, '\n')

df= data.get('範例')
cols = df.columns
print("欄位：")
print(cols, '\n')

head = df.head(2)
print("前2筆資料：")
print(head, '\n')

print("範例資料表相關資訊：")
info = df.info()
print(info, '\n')

print("a x b：")
shape = df.shape
print(shape, '\n')

row, col = df.shape
print("row:", row)
print("column:", col)
print

# https://note.nkmk.me/en/python-pandas-len-shape-size/
# 玩玩看
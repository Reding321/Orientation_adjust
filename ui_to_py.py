import os
import os.path

# UI文件所在的路径
dir = r'C:\Users\HP\Desktop\pythonProject\Orientation_adjust_tool_Remake'


# 列出目录下的所有UI文件
def listUiFile():
	list = []
	files = os.listdir(dir)
	for filename in files:
		if os.path.splitext(filename)[1] == '.ui':
			list.append(filename)
	return list


# 把扩展名为.ui的文件改成扩展名为.py的文件
def transPyFile(filename):
	return os.path.splitext(filename)[0] + '.py'


# 调用系统命令把UI文件转换成Python文件
def runMain():
	list = listUiFile()
	for uifile in list:
		pyfile = transPyFile(uifile)
		cmd = 'pyuic5 -o {pyfile} {uifile}'.format(pyfile=pyfile,uifile=uifile)
		os.system(cmd)


if __name__ == "__main__":
	runMain()









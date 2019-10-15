#-*-coding: utf-8-*-
#author: lxz-HXY

'''
代码测试根据照片ID去将文件夹中的图片取出来并保存到新的目录
'''
import os
import shutil


def move_pic(old_database, new_database):    
    ID = [1, 2, 3, 4, 5, 6]
    file_names = os.listdir(old_database)
    for file in file_names:
        img_name = os.listdir(old_database + file)
        
        for id in ID:
            if str(id) == file:
                if not os.path.exists(new_database  + str(id)):
                    os.mkdir(new_database  + str(id))
                shutil.copy(old_database + file + '/' + str(img_name[0]), new_database + '/' + str(id))
                
    

if __name__ == '__main__':
    move_pic(old_database='/home/lxz/project/faceid/debug_code/old_face/', new_database='/home/lxz/project/faceid/debug_code/new_face/')
     
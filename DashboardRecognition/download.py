# download.py
# automatically download dashboard images from Internet
import os


def GetBaiduImages(images_path, prefix):
       # rename all png files to their original postfix
       files = os.listdir(images_path)
       print(len(files))
       counter = 0
       for _f in files:
              if _f.startswith(prefix):
                     _f_new = '%06d.jpg' % counter
                     counter += 1
                     os.renames(os.path.join(images_path, _f), os.path.join(images_path, _f_new))
              else:
                     os.remove(os.path.join(images_path, _f))
                     print(_f)


if __name__ == '__main__':
       GetBaiduImages('data/Baidu_YaLiBiao/', 'u=')

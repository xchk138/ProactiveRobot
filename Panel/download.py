# download.py
# automatically download dashboard images from Internet
import os


def GetBingImages(images_path, prefix):
       # rename all png files to their original postfix
       files = os.listdir(images_path)
       print(len(files))
       for _f in files:
              if _f.startswith(prefix):
                     _f_new = _f + '.jpg'
                     os.renames(os.path.join(images_path, _f), os.path.join(images_path, _f_new))
              else:
                     os.remove(os.path.join(images_path, _f))
                     print(_f)


if __name__ == '__main__':
       #GetBingImages('data/Baidu_BiLeiQi/', 'u=')
       GetBingImages('data/Bing_BiLeiQi/', 'OIP-C')

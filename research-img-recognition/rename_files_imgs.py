from pathlib import Path
t = "research-img-recognition"
folder = "input_img_cropped_modified"
d = Path.cwd() / t / t / folder

import sys
for dir in d.iterdir():
    s = dir.name.split('.')
    if len(s) != 2:
        raise AssertionError()
    if "_" == s[0][-1]:
        print('found _:', dir.name)
        new = s[0][:-1] + '.' + s[1]
        print('new:', new)
        print('s:', s)
        dir.rename(d / new)
        print('new name:', s)
    else:
        print('nothing wrong')
    

    
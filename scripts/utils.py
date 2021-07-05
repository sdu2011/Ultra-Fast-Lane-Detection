import os
from shutil import copyfile

if __name__ == "__main__":
    """
    python utils.py
    """
    train_dir = '/home/suchang/data/lane/autocore_0622'
    test_dir = '/home/suchang/data/lane/autocore_0622_test'
    for e in os.listdir(train_dir):
        # print(e)
        if e[-3:] == 'jpg':
            #img_name = e[:-4] + 'jpg'
            idx = e[5:-4]
            print(idx)

            # if int(idx) > 400 and int(idx) < 700:
            if int(idx) < 700:
                print(e)
                src = '{}/{}'.format(train_dir,e)
                dst = '{}/{}'.format(test_dir,e)
                copyfile(src, dst)

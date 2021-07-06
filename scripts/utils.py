import os
from shutil import copyfile

if __name__ == "__main__":
    """
    python utils.py
    """
    train_dir = '/home/suchang/data/lane/autocore_0622'
    test_dir = '/home/suchang/data/lane/autocore_0622_test'
    debug_dir = '/home/suchang/data/lane/autocore_0622_debug'
    for e in os.listdir(train_dir):
        # print(e)
        if e[-3:] == 'jpg':
            #img_name = e[:-4] + 'jpg'
            idx = e[5:-4]
            # print(idx)

            # if int(idx) > 400 and int(idx) < 700:
            if int(idx) < 4354:
                # print(e)
                src = '{}/{}'.format(train_dir,e)
                dst = '{}/{}'.format(test_dir,e)
                copyfile(src, dst)
                print('copy {} to {}'.format(src,dst))

        # 拷贝有json文件的到debug目录
        if e[-4:] == 'json':
            img_name = e[:-4] + 'jpg'
            # print(''.format(img_name))

            src = '{}/{}'.format(train_dir,img_name)
            dst = '{}/{}'.format(debug_dir,img_name)
            print('copy {} to {}'.format(src,dst))
            copyfile(src, dst)

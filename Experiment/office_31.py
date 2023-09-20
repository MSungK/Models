import os
import os.path as osp

os.chdir('/workspace/Minsung/dataset/office_31')

class_dict = dict()

webcam_root = 'webcam/images'
dslr_root = 'dslr/images'
amazon_root = 'amazon/images'

index = 0

domains = [webcam_root, dslr_root, amazon_root]

for domain in domains:

    f = open(f'{osp.dirname(domain)}.txt', 'w')

    for dir in os.listdir(domain):
        
        if dir not in class_dict.keys():
            class_dict[dir] = index
            index += 1

        dir_root = osp.join(domain, dir)
        
        for file in os.listdir(dir_root):
            f.write(f'{osp.join(dir_root, file)} {class_dict[dir]}\n')

    f.close()
    


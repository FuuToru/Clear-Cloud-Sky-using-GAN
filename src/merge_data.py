import os
import shutil

org_path = "data/archive/EuroSAT"
classes = os.listdir(org_path)
path_target = "data/targets"

file_idx = 1
for kind in classes:
    path = os.path.join(org_path,kind)
    if os.path.isfile(path):
        continue
    
    for i, f in enumerate(os.listdir(path)):
        #copy file
        shutil.copyfile(os.path.join(path,f), os.path.join(path_target,f))
        #rename file
        os.rename(os.path.join(path_target,f), os.path.join(path_target, "IMG_{}.jpg".format(file_idx)))
        file_idx += 1


        

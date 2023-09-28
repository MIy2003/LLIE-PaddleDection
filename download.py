import os
import zipfile
os.chdir('/home/aistudio/data/data241073')
extracting = zipfile.ZipFile('ppyoloe_crn_l_300e_coco.zip')
extracting.extractall('/home/aistudio/work/PPYOLOE/')

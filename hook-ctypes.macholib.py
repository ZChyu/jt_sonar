from PyInstaller.utils.hooks import copy_metadata
datas = copy_metadata('cffi') + copy_metadata('greenlet')
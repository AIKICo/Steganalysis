import os;
from subprocess import Popen, PIPE

root, dirs, files = next(os.walk('D:\\Databases\\PDA\\StegHide'))
secret_filename = 'SecretFile.txt'
for file in files:
    if file.lower().endswith('.wav'):
        p = Popen('steghide embed -cf D:\\Databases\\PDA\\StegHide\\' + file +
                  ' --dontcompress -ef  D:\\Databases\\PDA\\SecretFile\\' + secret_filename
                  + ' -p kabinet95', shell=True,
                  stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()
        print(file)
        if err != '':
            print(err)
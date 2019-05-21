import subprocess
import inspect
import os
import re
import ast

class GlobalFeatures():

    def __init__(self,img):
        """Constructor"""
        self.jar = os.path.join(os.path.dirname(inspect.getfile(GlobalFeatures)),'lirecolordescriptors.jar')
        self.img = img
        self.out = {}

    def extract(self):
        args = ["java", "-jar", self.jar, self.img]
        print(args)
        process = subprocess.Popen(args, stdout=subprocess.PIPE)
        data = process.communicate()
        st=0
        key=""
        for line in data[0].decode("utf-8").splitlines():
            if (st==1):
                self.out[key]=ast.literal_eval(line)
                st=0
            m = re.search('(\w+):', line)
            if m:
                key=m.group(1)
                st=1
        return self.out

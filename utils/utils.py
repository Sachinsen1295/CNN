import base64

def decodeimage(imgstring, filename):
    imgdata = base64.b64decode(imgstring)
    with open(filename, 'wb') as f:
        f.write(imgdata)
        f.close()

def encodeimage(croppedimagepath):
    with open(croppedimagepath,'r') as f:
        return base64.b64encode(f.read())
import ExifTags
import Image

img = Image.open('TestImages/01.jpg')
print(img._getexif().items())
exif=dict((ExifTags.TAGS[k], v) for k, v in img._getexif().items() if k in ExifTags.TAGS)
if not exif['Orientation']:
    img=img.rotate(90, expand=True)
img.thumbnail((1000,1000), Image.ANTIALIAS)
img.save(output_fname, "JPEG")
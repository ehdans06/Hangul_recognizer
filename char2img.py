from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np
import pickle as pk

def char2img(encoding, image_size, fontname, save_img=False, save_dataset=False):
    num_cho, num_jung, num_jong = 19, 21, 28
    if save_dataset:
        data = {
            'images': [],
            'labels': []
        }

    chars = []
    if encoding == 'euc-kr':
        chars = euc_kr_gen()
    elif encoding == 'unicode':
        chars = unicode_gen()

    for rot_times in range(0, len(fontname)):
        for char in chars:
            img = Image.new('L', (image_size, image_size))
            draw = ImageDraw.Draw(img)
#            draw.text((0, 0), char, 255, font=ImageFont.truetype('fonts/'+random.choice(fontname)+'.ttf', random.randrange((int)(image_size/1.5),image_size)))
            draw.text((0, 0), char, 255, font=ImageFont.truetype('fonts/'+fontname[rot_times]+'.ttf', random.randrange((int)(image_size/1.5),image_size)))
            if save_img:
                img = img.rotate(random.randrange(-30,30))
                img.save(char + '.bmp')

            if save_dataset:
                img_data = np.asarray(img, dtype=np.uint8) / 256  # img to ndarray
                cho, jung, jong = jamo_decomposition(char)
                label_onehot = np.zeros(num_cho + num_jung + num_jong)
                label_onehot[cho] = 1
                label_onehot[num_cho + jung] = 1
                label_onehot[num_cho + num_jung + jong] = 1
                data['images'].append(img_data)
                data['labels'].append(label_onehot)

    if save_dataset:
        data['images'] = np.array(data['images']).reshape((-1, image_size, image_size, 1))
        data['labels'] = np.array(data['labels'])
        file_path = 'dataset/' + encoding + '_' + fontname[0] + '.pkl'
        with open(file_path, 'wb') as f:
            pk.dump(data, f, pk.HIGHEST_PROTOCOL)

# EUC-KR
def euc_kr_gen():
    for i in range(0x30+128, 0x49+128):
        for j in range(0x21+128, 0x7F+128):
            yield bytes([i, j]).decode('euc-kr')

# Unicode
def unicode_gen():
    for i in range(0xAC00, 0xD7A4):
        yield chr(i)

def unicode_one(image_size, fontname):
    img = Image.new('RGB', (image_size, image_size))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), '뷁', (255,255,255), font=ImageFont.truetype('fonts/'+fontname[0]+'.ttf', image_size))
    img.save('unicode-one.png')

def jamo_decomposition(char):
    # Jamo decomposition
    # num_cho, num_jung, num_jong = 19, 21, 28
    # 한글유니코드값 = 0xAC00 + ( (초성순서 * 21) + 중성순서 ) * 28 + 종성순서
    hangul_code = ord(char) - int('0xAC00',16)
    cho_jung, jong = hangul_code // 28, hangul_code % 28
    cho, jung = cho_jung // 21, cho_jung % 21
    return cho, jung, jong

def ndarray_to_img(img_data):
    # ndarray to img
    #img_data.reshape((image_size, image_size))
    img_data = img_data * 256
    img = Image.fromarray(np.uint8(img_data))

def main():
    image_size = 28
    fontname_train = ['NanumBarunGothic', 'gungsoe', 'NanumGothic', 'NanumPen', 'NanumMyeongjo', 'NanumBrush', 'BMDOHYEON_ttf', 'BMHANNA_11yrs_ttf', 'BMJUA_ttf', 'Busan', 'JejuGothic', 'JejuHallasan', 'JejuMyeongjo', 'SungDong Gothic B', 'SungDong Gothic EB', 'SungDong Myungjo R','SeoulNamsanB', 'SeoulNamsanL']
#    fontname_test = []
    char2img(encoding='unicode', image_size=image_size, fontname=fontname_train, save_dataset=True)
#    char2img(encoding='unicode', image_size=image_size, fontname=fontname_test, save_dataset=True)

if __name__ == '__main__':
    main()

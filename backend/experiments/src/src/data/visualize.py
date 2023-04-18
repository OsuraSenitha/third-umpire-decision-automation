import cv2 as cv

def drawRects(img, txt_cntnt):
    lines = list(map(lambda line: list(map(float, line.split())), txt_cntnt.strip().split("\n")))
    drawn_img = img.copy()
    img_h, img_w, _ = img.shape
    for line in lines:
        l, x, y, w, h = line
        l, x, y, w, h = int(l), x*img_w, y*img_h, w*img_w, h*img_h
        drawn_img = cv.rectangle(drawn_img, (int(x-w/2), int(y-h/2)), (int(x+w/2),int(y+h/2)), (255,0,0), int(min(img_h, img_w)/100))

    return drawn_img
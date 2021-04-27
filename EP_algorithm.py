import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import cv2
import pydicom
import os
import numpy as np
from scipy.misc import derivative
from scipy import integrate
"""
get data matrix by simpleitk
"""
# read single image
# pydicom
# img_head1 = pydicom.read_file(r"D:\yexuehua\MyProject\DCEUS\data\RawDicom\J2LCTE9I")
# img_array1 = img_head1.pixel_array

# simpleitk
img_head1 = sitk.ReadImage(r"D:\yexuehua\MyProject\DCEUS\data\RawDicom\J2LCTE9I")
img_array1 = sitk.GetArrayFromImage(img_head1)
print(img_head1)
drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
pi = 3.14159
ROI_coor = []



#show image
# plt.subplot(1,2,1)
# plt.imshow(img_array1[0])
# plt.subplot(1,2,2)
# plt.imshow(img_array1[1])
# plt.show()


"""
Part I convert dicom to echo power data
formulation:
EP(x, t) = Vmax**2 * 10^(( h^(-1)(I(x,t)) -1 ) * DR)
DR : 60~80 dB
h(x) : [0,1] -> [0,255]
I(x,t) : x is the position of pixel, t is the instant time
Vmax : maximum positive amplitude of signed 16-bit integer(2^15 - 1)
"""
# convert to echo-power data
def convert2EchoPower(x, DR, Vmax=32767):
    """
    :param x: input image
    :param DR: (dynamic range = 10log10(PowerMax/PowerMin))
    :param Vmax: maximum positive amplitude of signed 16-bit integer(2^15 - 1)
    :return:
    """
    # Gray = R*0.299 + G*0.587 + B*0.114 rgb convert to gray
    gray = np.squeeze(x[:, :, :, 0]*0.299 + x[:, :, :, 1]*0.587 + x[:, :, :, 2]*0.114)
    return (Vmax**2) * np.power(10,((gray/255 -1) * DR)/10)


def average_power(ROI, EP_map):
    """
    :param ROI: binary label matrix, 1 is ROI
    :param EP_map: converted Echo power
    :return: average_power
    """
    # set bouding box to label
    ROI_full = np.zeros([EP_map.shape[1], EP_map.shape[2]])
    ROI_full[ROI[0]:ROI[2], ROI[1]:ROI[3]] = 1
    sum_img =  np.sum(ROI_full * EP_map,axis=(1,2))
    return sum_img/np.sum(ROI_full)

def fetch_EP(EP_map, x, y):
    return EP_map[:, x, y]


def bolus_model(t, O, A, s, m):
    return A * (1 / (s * (t) * (2 * pi) ** 0.5)) * (np.exp(-np.square((np.log(t) - m) / (2 * s)))) + O


def arrival_time(ROI_EP):
    for i in range(len(ROI_EP)):
        if ROI_EP[i] != ROI_EP[i+1]:
            return i+1


def bolus_fit(ROI_EP):
    print("strat to fit ........")
    t1 = np.arange(1, len(ROI_EP) + 1)
    A = np.sum(ROI_EP)
    popt,pcov = curve_fit(bolus_model, t1, ROI_EP ,p0=[0,A,1.1,4])
    return popt


def quantitation_param(ROI_EP, model_param):
    print("start quantitation......")
    O = model_param[0]
    A = model_param[1]
    s = model_param[2]
    m = model_param[3]

    def f(t):
        return A * (1 / (s * (t) * (2 * pi) ** 0.5)) * (np.exp(-np.square((np.log(t) - m) / (2 * s)))) + O

    # store slope
    slope = []
    x = np.arange(1,len(ROI_EP)+1)
    for t in range(1,len(ROI_EP)+1):
        slope.append(derivative(f, t, dx=1e-6))
    xi = slope.index(max(slope))
    xo = slope.index(min(slope))
    PE = np.max(f(x))
    TTP = np.argmax(f(x))
    WiR = max(slope)
    WoR = min(slope)
    TI = xi - f(xi) / slope[xi]
    TO = xo - f(xo) / slope[xo]
    WiAUC, _ = integrate.quad(f, TI, TTP)
    WoAUC, _ = integrate.quad(f, TTP, TO)
    WioAUC = WiAUC + WoAUC
    RT = TTP - TI
    FT = TO - TTP
    WiPI = WiAUC / RT


    print("PE:", PE)
    print("TTP:", TTP)
    print("WiR:", WiR)
    print("WoR:", WoR)
    print("TI:", TI)
    print("TO:", TO)
    print("WiAUC", WiAUC)
    print("WoAUC", WoAUC)
    print("WioAUC", WioAUC)
    print("RT", RT)
    print("FT", FT)
    print("WiPI", WiPI)

    plt.plot(x, ROI_EP, "r-")
    plt.plot(x,bolus_model(x, *model_param),'g--')
    plt.show()
    plt.savefig("average_power_curve.png")



# EP = convert2EchoPower(img_array1, 60)
#
# point_curve = fetch_EP(EP, 245, 294)
# x = np.arange(0,len(point_curve))
# plt.plot(x, point_curve)
# plt.show()


def visualize(name, data, EP):
    def draw_ROI(event, x, y, flag, param):
        global ix, iy, drawing, mode
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                img_copy = data[pos].copy()
                if mode == True:
                    cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 0, 255), 2)
                    cv2.imshow(name, img_copy)
                else:
                    cv2.circle(img_copy, (x, y), 1, (0, 0, 255), 2)
                    cv2.imshow(name, img_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            ROI_coor.append([ix, iy, x, y])
            if mode == True:
                cv2.rectangle(data[pos], (ix, iy), (x, y), (0, 0, 255), 2)
            else:
                cv2.circle(data[pos], (x, y), 1, (0, 0, 255), 2)


    def draw_all(pos):
        for ix, iy, x, y in ROI_coor:
            cv2.rectangle(data[pos], (ix, iy), (x, y), (0, 0, 255), 2)

    cv2.namedWindow(name, 0)
    # cv2.resizeWindow(name, 800, 600)
    loop_flag = 0
    pos = 0
    cv2.createTrackbar('time', name, 0, len(data), draw_all)
    cv2.setMouseCallback(name, draw_ROI)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        elif cv2.waitKey(1) & 0xFF == ord("a"):
            print("--you pressed a key--")
            avg_EP = average_power(ROI_coor[1], EP)
            popt = bolus_fit(avg_EP)
            quantitation_param(avg_EP, popt)
        pos = cv2.getTrackbarPos('time', name)
        img_new = data[pos]
        cv2.imshow(name, img_new)

# region [206, 157, 341, 268] for test

EP = convert2EchoPower(img_array1, 60)
# x = np.arange(0,len(img_array1))
# #gray = np.squeeze(img_array1[:, :, :, 0]*0.299 + img_array1[:, :, :, 1]*0.587 + img_array1[:, :, :, 2]*0.114)
# popt = bolus_fit(average_power([206, 157, 341, 268], EP))
# print(popt)
# quantitation_param(average_power([206, 157, 341, 268], EP), popt)
visualize("us_img", img_array1, EP)

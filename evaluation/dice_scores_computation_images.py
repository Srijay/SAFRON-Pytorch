# This script computes average dice index scores between two
# sets of class outputs or images in 2-d with each pixel
# representing the class
import numpy
import glob
import random
import os
from PIL import Image
import numpy as np
import scipy.io
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

image_size = 256

color_dict_conic = {
        1: [0, 0, 0],  # neutrophil  : black
        2: [0, 255, 0],  # epithelial : green
        3: [255, 255, 0],  # lymphocyte : Yellow
        4: [255, 0, 0],  # plasma : red
        5: [0, 0, 255],  # eosinophil : Blue
        6: [255, 0, 255],  # connectivetissue : fuchsia
        0: [255, 255, 255]  # Background : white
    }


def get_images(folder,file_names):
    images = []
    for fname in file_names:
        img = Image.open(os.path.join(folder, fname))
        img = numpy.asarray(img)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        images.append(img)
    return images


def compute_dice_score(x,y):
    k=1
    denom = (np.sum(x) + np.sum(y))
    if(denom==0):
        return 1
    return np.sum(x[y==k])*2.0 / denom


def evaluate_dice_score(images1,images2,class_id='all'):

    dice_score = 0
    l = len(images1)

    for i in range(0,len(images1)):

        # matplotlib.image.imsave("rgb.png", images1[i])

        if(class_id=='all'):
            img1 = np.ones((image_size, image_size))
            img2 = np.ones((image_size, image_size))
            for j in range(image_size):
                for k in range(image_size):
                    if ((images1[i][j][k] == [255, 255, 255]).all()):
                        img1[j][k] = 0
                    if ((images2[i][j][k] == [255, 255, 255]).all()):
                        img2[j][k] = 0
        else:
            img1 = np.zeros((image_size, image_size))
            img2 = np.zeros((image_size, image_size))
            for j in range(image_size):
                for k in range(image_size):
                    if ((images1[i][j][k] == color_dict_conic[class_id]).all()):
                        img1[j][k] = 1
                    if ((images2[i][j][k] == color_dict_conic[class_id]).all()):
                        img2[j][k] = 1

        dice_score += compute_dice_score(img1,img2)

    dice_score=dice_score/l*1.0
    return dice_score


def colored_images_from_classes(x,image_size):
    k = np.zeros((image_size, image_size, 3))
    l = image_size
    for i in range(0, l):
        for j in range(0, l):
            k[i][j] = color_dict_conic[x[i][j]]
    return k/255.0


folder_1 = "F:/Datasets/conic/CoNIC_Challenge/challenge/valid/masks"
folder_2 = "F:/Datasets/conic/CoNIC_Challenge/challenge/valid/hovernet_results/color_segmentation"

image_names = [file for file in os.listdir(folder_2) if file.endswith('.png')]

#image_names = ['100.png', '101.png', '102.png', '103.png', '1037.png', '1038.png', '1039.png', '104.png', '1040.png', '1042.png', '1043.png', '1044.png', '1045.png', '1049.png', '105.png', '1050.png', '1055.png', '106.png', '1066.png', '1067.png', '1068.png', '1069.png', '107.png', '1070.png', '108.png', '109.png', '110.png', '111.png', '112.png', '113.png', '114.png', '1145.png', '1146.png', '1147.png', '1148.png', '115.png', '1150.png', '1151.png', '1152.png', '1153.png', '1154.png', '1155.png', '1156.png', '1157.png', '1158.png', '1159.png', '116.png', '1160.png', '1161.png', '1162.png', '1163.png', '1164.png', '1165.png', '1166.png', '1167.png', '1168.png', '1169.png', '117.png', '1170.png', '1171.png', '1172.png', '1173.png', '1174.png', '1175.png', '1176.png', '1177.png', '1178.png', '1179.png', '118.png', '119.png', '12.png', '120.png', '121.png', '122.png', '123.png', '124.png', '125.png', '126.png', '127.png', '128.png', '129.png', '13.png', '130.png', '131.png', '132.png', '133.png', '134.png', '135.png', '14.png', '1468.png', '1469.png', '1470.png', '1471.png', '1472.png', '1473.png', '1474.png', '1475.png', '1476.png', '1477.png', '1478.png', '1479.png', '1480.png', '1481.png', '1482.png', '1483.png', '1484.png', '1485.png', '1486.png', '1487.png', '1488.png', '1489.png', '1490.png', '1491.png', '1492.png', '1493.png', '1494.png', '1495.png', '1496.png', '1497.png', '1498.png', '1499.png', '15.png', '1500.png', '1501.png', '1502.png', '1503.png', '1648.png', '1649.png', '1650.png', '1651.png', '1652.png', '1653.png', '1654.png', '1655.png', '1656.png', '1657.png', '1658.png', '1659.png', '1660.png', '1661.png', '1662.png', '1663.png', '1664.png', '1665.png', '1666.png', '1667.png', '1668.png', '1669.png', '1670.png', '1671.png', '1672.png', '1673.png', '1674.png', '1675.png', '1676.png', '1677.png', '1678.png', '1679.png', '1680.png', '1681.png', '1682.png', '1683.png', '1756.png', '1757.png', '1758.png', '1759.png', '1760.png', '1761.png', '1762.png', '1763.png', '1764.png', '1765.png', '1766.png', '1767.png', '1768.png', '1769.png', '1770.png', '1771.png', '1773.png', '1774.png', '1775.png', '1776.png', '1777.png', '1778.png', '1779.png', '1780.png', '1781.png', '1782.png', '1783.png', '1784.png', '1785.png', '1786.png', '1787.png', '1788.png', '1789.png', '1790.png', '1791.png', '1829.png', '1830.png', '1831.png', '1832.png', '1834.png', '1835.png', '1836.png', '1837.png', '1840.png', '1844.png', '1845.png', '1858.png', '1859.png', '1860.png', '1972.png', '1974.png', '1975.png', '1976.png', '1977.png', '1978.png', '1979.png', '1980.png', '1981.png', '1982.png', '1983.png', '1984.png', '1985.png', '1986.png', '1987.png', '1988.png', '1989.png', '1990.png', '1991.png', '1992.png', '1993.png', '1994.png', '1995.png', '1996.png', '1997.png', '1998.png', '1999.png', '2000.png', '2001.png', '2002.png', '2003.png', '2004.png', '2005.png', '2006.png', '2007.png', '2574.png', '2575.png', '2576.png', '2577.png', '2578.png', '2579.png', '2580.png', '2581.png', '2582.png', '2583.png', '2584.png', '2585.png', '2586.png', '2587.png', '2588.png', '2589.png', '2590.png', '2591.png', '2592.png', '2593.png', '2594.png', '2595.png', '2596.png', '2597.png', '2598.png', '2599.png', '2600.png', '2601.png', '2602.png', '2603.png', '2734.png', '2735.png', '2736.png', '2737.png', '2738.png', '2739.png', '2740.png', '2741.png', '2742.png', '2743.png', '2744.png', '2745.png', '2746.png', '2747.png', '2748.png', '2749.png', '2750.png', '2751.png', '2752.png', '2753.png', '2754.png', '2755.png', '2756.png', '2757.png', '2758.png', '2759.png', '2760.png', '2761.png', '2762.png', '2763.png', '2764.png', '2765.png', '2766.png', '2767.png', '2768.png', '2769.png', '2770.png', '2771.png', '2772.png', '2773.png', '2774.png', '2775.png', '2776.png', '2777.png', '2778.png', '2779.png', '2780.png', '2781.png', '2782.png', '2783.png', '2784.png', '2785.png', '2786.png', '2787.png', '2788.png', '2789.png', '2790.png', '2791.png', '2792.png', '2793.png', '2794.png', '2795.png', '2796.png', '2797.png', '2798.png', '2799.png', '2800.png', '2801.png', '2802.png', '2803.png', '2804.png', '2805.png', '2806.png', '2807.png', '2808.png', '2809.png', '2810.png', '2811.png', '2812.png', '2813.png', '2814.png', '2815.png', '2816.png', '2817.png', '2818.png', '2819.png', '2820.png', '2821.png', '2822.png', '2823.png', '3059.png', '3060.png', '3061.png', '3062.png', '3063.png', '3064.png', '3065.png', '3066.png', '3067.png', '3068.png', '3069.png', '3070.png', '3071.png', '3072.png', '3073.png', '3074.png', '3075.png', '3076.png', '3077.png', '3078.png', '3079.png', '3080.png', '3081.png', '3082.png', '3083.png', '3085.png', '3086.png', '3087.png', '3088.png', '3225.png', '3226.png', '3227.png', '3228.png', '3229.png', '3230.png', '3231.png', '3232.png', '3233.png', '3234.png', '3235.png', '3236.png', '3237.png', '3238.png', '3239.png', '3240.png', '3241.png', '3242.png', '3243.png', '3244.png', '3245.png', '3246.png', '3247.png', '3248.png', '3249.png', '3250.png', '3251.png', '3252.png', '3253.png', '3254.png', '3255.png', '3256.png', '3257.png', '3258.png', '3259.png', '3285.png', '3286.png', '3287.png', '3288.png', '3289.png', '3290.png', '3291.png', '3292.png', '3293.png', '3294.png', '3295.png', '3296.png', '3297.png', '3298.png', '3299.png', '3300.png', '3301.png', '3302.png', '3303.png', '3304.png', '3305.png', '3306.png', '3307.png', '3308.png', '3309.png', '3439.png', '3440.png', '3441.png', '3442.png', '3443.png', '3444.png', '3445.png', '3446.png', '3447.png', '3448.png', '3449.png', '3450.png', '3451.png', '3452.png', '3453.png', '3454.png', '3455.png', '3456.png', '3457.png', '3458.png', '3459.png', '3460.png', '3461.png', '3462.png', '3463.png', '3464.png', '3465.png', '3466.png', '3467.png', '3468.png', '3469.png', '3470.png', '3471.png', '3472.png', '3473.png', '3474.png', '3475.png', '3476.png', '3477.png', '3478.png', '3479.png', '3480.png', '3481.png', '3482.png', '3483.png', '3484.png', '3485.png', '3486.png', '3487.png', '3488.png', '3489.png', '3490.png', '3491.png', '3492.png', '3493.png', '3494.png', '3495.png', '3496.png', '3497.png', '3498.png', '3499.png', '3500.png', '3501.png', '3502.png', '3503.png', '3504.png', '3505.png', '3506.png', '3507.png', '3508.png', '3509.png', '3510.png', '3556.png', '3557.png', '3558.png', '3559.png', '3560.png', '3561.png', '3562.png', '3563.png', '3564.png', '3565.png', '3566.png', '3567.png', '3568.png', '3569.png', '3570.png', '3571.png', '3572.png', '3573.png', '3574.png', '3575.png', '3576.png', '3577.png', '3578.png', '3579.png', '3580.png', '3581.png', '3582.png', '3583.png', '3584.png', '3585.png', '3586.png', '3587.png', '3588.png', '3589.png', '3590.png', '3591.png', '3592.png', '3593.png', '3594.png', '3595.png', '3596.png', '3597.png', '3598.png', '3599.png', '3600.png', '3601.png', '3602.png', '3603.png', '3604.png', '3605.png', '3606.png', '3607.png', '3608.png', '3609.png', '3610.png', '3611.png', '3612.png', '3613.png', '3614.png', '3615.png', '3616.png', '3617.png', '3618.png', '3619.png', '3798.png', '3799.png', '3800.png', '3801.png', '3802.png', '3803.png', '3804.png', '3805.png', '3806.png', '3807.png', '3808.png', '3809.png', '3810.png', '3811.png', '3812.png', '3813.png', '3814.png', '3815.png', '3816.png', '3817.png', '3948.png', '3949.png', '3950.png', '3951.png', '3952.png', '3953.png', '3954.png', '3955.png', '3956.png', '3957.png', '3958.png', '3959.png', '3960.png', '3961.png', '3962.png', '3963.png', '4251.png', '4252.png', '4253.png', '4254.png', '4255.png', '4256.png', '4257.png', '4258.png', '4259.png', '4260.png', '4261.png', '4262.png', '4263.png', '4264.png', '4265.png', '4266.png', '4267.png', '4268.png', '4281.png', '4282.png', '4283.png', '4284.png', '4285.png', '4286.png', '4287.png', '4288.png', '4289.png', '4290.png', '4291.png', '4292.png', '4305.png', '4306.png', '4307.png', '4308.png', '4309.png', '4310.png', '4311.png', '4312.png', '4313.png', '4314.png', '4315.png', '4316.png', '4341.png', '4342.png', '4343.png', '4344.png', '4345.png', '4346.png', '4347.png', '4348.png', '4349.png', '4350.png', '4351.png', '4352.png', '4395.png', '4396.png', '4397.png', '4398.png', '4399.png', '4400.png', '4401.png', '4402.png', '4403.png', '4404.png', '4405.png', '4406.png', '4419.png', '4420.png', '4421.png', '4422.png', '4423.png', '4424.png', '4425.png', '4426.png', '4427.png', '4428.png', '4429.png', '4430.png', '4539.png', '4540.png', '4541.png', '4542.png', '4543.png', '4544.png', '460.png', '461.png', '462.png', '463.png', '464.png', '465.png', '466.png', '467.png', '468.png', '469.png', '470.png', '471.png', '472.png', '473.png', '474.png', '475.png', '476.png', '477.png', '478.png', '4785.png', '4786.png', '4787.png', '4788.png', '4789.png', '479.png', '4790.png', '4791.png', '4792.png', '4793.png', '4794.png', '4795.png', '4796.png', '480.png', '481.png', '482.png', '4821.png', '4822.png', '4823.png', '4824.png', '4825.png', '4826.png', '4827.png', '4828.png', '4829.png', '483.png', '4830.png', '4831.png', '4832.png', '484.png', '4845.png', '4846.png', '4847.png', '4848.png', '4849.png', '485.png', '4850.png', '4851.png', '4852.png', '4853.png', '4854.png', '4855.png', '4856.png', '4857.png', '4858.png', '4859.png', '486.png', '4860.png', '4861.png', '4862.png', '4863.png', '4864.png', '4865.png', '4866.png', '4867.png', '4868.png', '487.png', '488.png', '4889.png', '489.png', '4890.png', '4891.png', '4892.png', '490.png', '4901.png', '4902.png', '4903.png', '4904.png', '4909.png', '491.png', '4910.png', '4911.png', '4912.png', '492.png', '4929.png', '493.png', '4930.png', '4931.png', '4932.png', '494.png', '4945.png', '4946.png', '4947.png', '4948.png', '495.png', '4953.png', '4954.png', '4955.png', '4956.png', '496.png', '497.png', '498.png', '499.png', '500.png', '501.png', '502.png', '503.png', '504.png', '505.png', '506.png', '507.png', '508.png', '509.png', '510.png', '511.png', '512.png', '513.png', '514.png', '515.png', '516.png', '517.png', '518.png', '519.png', '52.png', '520.png', '521.png', '522.png', '523.png', '524.png', '525.png', '526.png', '527.png', '528.png', '529.png', '53.png', '530.png', '531.png', '532.png', '533.png', '534.png', '535.png', '536.png', '537.png', '538.png', '539.png', '54.png', '540.png', '541.png', '542.png', '543.png', '544.png', '545.png', '546.png', '547.png', '548.png', '549.png', '55.png', '550.png', '551.png', '552.png', '553.png', '554.png', '555.png', '556.png', '557.png', '558.png', '559.png', '56.png', '560.png', '561.png', '562.png', '563.png', '564.png', '565.png', '566.png', '567.png', '57.png', '58.png', '59.png', '604.png', '605.png', '606.png', '607.png', '608.png', '609.png', '610.png', '611.png', '612.png', '613.png', '614.png', '615.png', '616.png', '617.png', '618.png', '619.png', '620.png', '621.png', '622.png', '623.png', '624.png', '625.png', '626.png', '627.png', '628.png', '629.png', '630.png', '631.png', '632.png', '633.png', '634.png', '635.png', '636.png', '637.png', '638.png', '639.png', '640.png', '641.png', '642.png', '643.png', '644.png', '645.png', '646.png', '647.png', '648.png', '649.png', '650.png', '651.png', '652.png', '653.png', '654.png', '655.png', '656.png', '657.png', '658.png', '659.png', '660.png', '661.png', '662.png', '663.png', '664.png', '665.png', '666.png', '667.png', '668.png', '669.png', '670.png', '671.png', '672.png', '673.png', '674.png', '675.png']

length = len(image_names)

print("Number of files to be processed: ",length)

images_real = get_images(folder_1,image_names)
images_gen = get_images(folder_2,image_names)

class_dict = {
        1: 'neutrophil',  # neutrophil  : black
        2: 'epithelial',  # epithelial : green
        3: 'lymphocyte',  # lymphocyte : Yellow
        4: 'plasma',  # plasma : red
        5: 'eosinophil',  # eosinophil : Blue
        6: 'connectivetissue',  # connectivetissue : fuchsia
        0: 'background'  # Background : white
    }

# class_dict = {
#         1: 'Neoplastic',  # Neoplastic  : black
#         2: 'Inflammatory',  # epithelial : green
#         3: 'Soft',  # lymphocyte : Yellow
#         4: 'Dead',  # plasma : red
#         5: 'Epithelial',  # eosinophil : Blue
#         0: 'background'  # Background : white
#     }

classes = [1,2,3,4,5,6]

for c in classes:
    dice_score = evaluate_dice_score(images_real,images_gen,c)
    print("Dice score of class ",class_dict[c]," is ",dice_score)

print("Overall Dice score is ",evaluate_dice_score(images_real,images_gen,'all'))
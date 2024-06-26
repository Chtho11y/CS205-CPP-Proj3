import matplotlib.pyplot as plt
import numpy as np


class Data(object):

    def __init__(self, _siz, _tims, label, deg=3):
        siz = np.array(_siz)
        tims = np.array(_tims)
        n = len(tims)
        self.label = label
        self.tims = tims
        self.N = n
        self.siz = siz
        self.logSiz = np.log2(self.siz)
        self.logT = np.log2(tims)
        fit0 = np.polyfit(self.logSiz[:6], self.logT[:6], 1)
        fit1 = np.polyfit(self.logSiz[3:], self.logT[3:], 1)
        fit3 = np.polyfit(self.siz[:], self.tims[:], deg)
        self.logTfit = np.poly1d(fit1)
        self.logTfit0 = np.poly1d(fit0)
        self.Tfit = np.poly1d(fit3)
        self.gflops = 2.0e-6 * siz * siz * siz / tims


def draw(data_list):
    plt.xlabel("N")
    plt.ylabel("T / ms")
    for data in data_list:
        xd = np.linspace(data.siz[0], data.siz[-1], 100)
        plt.plot(xd, data.Tfit(xd), label=data.label)
        plt.scatter(data.siz, data.tims, marker='x', s=12)
    plt.legend()
    plt.show()

    plt.xlabel("log N")
    plt.ylabel("log T")
    for data in data_list:
        xd = np.linspace(data.logSiz[0], data.logSiz[-1], 100)
        lx = data.logTfit(xd)
        ly = data.logTfit0(xd)
        final = []
        for (x, y) in zip(lx, ly):
            final.append(max(x, y))
        # final = lx
        plt.plot(xd, final, label=data.label)
        plt.scatter(data.logSiz, data.logT, marker='x', s=12)
    plt.legend()
    plt.show()

    plt.xlabel("N")
    plt.ylabel("GFLOPS")
    for data in data_list:
        plt.plot(data.siz, data.gflops, label=data.label)
        plt.scatter(data.siz, data.gflops, marker='x', s=12)
    plt.legend()
    plt.show()

def draw_rate(siz, data, label):
    plt.xlabel("N")
    plt.ylabel("Rate")
    plt.plot(siz, data, label=label)
    plt.scatter(siz, data, marker='x', s=12)
    plt.legend()
    plt.show()


Nplain = [16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408]
Tplain = [0.00260, 0.01627, 0.09697, 1.34747, 9.75737, 31.90780, 72.86347, 158.87603, 282.71483, 442.50140, 2505.47237, 906.09597, 1622.49127, 1729.65843]
plain_data = Data(Nplain, Tplain, 'plain')

Ntrans = [16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048, 2176, 2304, 2432]
Ttrans = [0.00367, 0.00947, 0.06647, 0.66623, 5.77860, 21.88940, 54.26210, 106.03807, 183.28903, 293.63667, 441.94393, 635.82193, 881.87603, 1170.20410, 1529.38183, 1949.34240, 2471.26410, 3054.69970, 3858.83907, 4648.21807, 5634.11387, 6953.20807]
trans_data = Data(Ntrans, Ttrans, 'trans')

Ntrans_simd = [16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944, 3072, 3200, 3328, 3456, 3584]
Ttrans_simd = [0.00753, 0.04660, 0.07863, 0.47167, 2.73307, 8.71690, 19.60207, 35.67397, 59.70667, 91.38723, 135.80820, 191.03820, 255.68853, 336.75497, 439.64763, 552.05867, 686.84350, 878.97043, 1164.80820, 1386.70243, 1712.36177, 2104.40427, 2582.00787, 3047.72487, 3890.23490, 4199.32943, 4954.99750, 5797.04197, 6517.05720, 7438.35770, 8546.06710]
trans_simd = Data(Ntrans_simd, Ttrans_simd, 'trans&simd')

Nreorder = [16, 32, 64, 128, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096]
Treorder = [0.00217, 0.00483, 0.01533, 0.16987, 1.00460, 5.99660, 28.92250, 70.17497, 132.67233, 220.95777, 359.00010, 646.27660, 1146.79447, 1972.01580, 3337.63087, 4671.69813, 5972.50930, 7541.42007, 9646.45933, 11853.38893]
reorder_data = Data(Nreorder, Treorder, 'reorder')

Nreorder_omp = [16, 32, 64, 128, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096]
Treorder_omp = [0.04770, 0.04250, 0.05130, 0.16500, 0.62937, 1.57273, 5.03660, 10.95610, 19.73830, 33.84093, 59.67493, 88.71870, 152.70287, 260.89193, 422.77587, 630.62683, 805.98990, 1062.96297, 1364.50087, 1898.18273]
reorder_omp = Data(Nreorder_omp, Treorder_omp, 'reorder_omp')

Nblock = [16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144]
Tblock = [0.06017, 0.14203, 0.12537, 0.37350, 0.50160, 2.09150, 11.56183, 37.15310, 84.44807, 158.23487, 208.69867, 292.60360, 474.79647, 670.74113, 941.03647, 1345.31753, 1733.54187]
block_data = Data(Nblock, Tblock, 'block')

Nalign = [16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144]
Talign = [0.23320, 0.09993, 0.10480, 0.37987, 0.49210, 1.59110, 8.51443, 25.33033, 61.46040, 118.60253, 211.55683, 255.69380, 390.58470, 572.04867, 801.07603, 1027.59950, 1304.44250]
align_data = Data(Nalign, Talign, 'align')

Nalign_ref = [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144]
Talign_ref = [1.32417, 7.76893, 26.69063, 60.49683, 108.70600, 189.13900, 247.94927, 393.56327, 556.48833, 736.32840, 1037.27350, 1383.57253]
Talign_blas = [1.19437, 5.30383, 17.34890, 38.13243, 60.34600, 104.34113, 173.84167, 246.36697, 375.93730, 468.50700, 680.56970, 947.85047]
align_data_ref = Data(Nalign_ref, Talign_ref, 'aligned')
align_data_blas = Data(Nalign_ref, Talign_blas, 'blas')

Nunroll_ref = [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192]
Tunroll_ref = [1.19963, 5.87727, 17.67207, 42.33807, 73.59463, 136.32740, 175.11183, 273.84837, 378.06137, 523.71667, 741.94097, 965.29933, 1203.64447, 1658.52480, 2065.27627, 2705.93510]
Tunroll_blas = [1.19437, 5.30383, 17.34890, 38.13243, 60.34600, 104.34113, 173.99930, 245.45530, 359.63367, 439.93007, 614.98697, 816.99880, 1072.40223, 1231.76233, 1519.33103, 1950.16010]
unroll_ref = Data(Nunroll_ref, Tunroll_ref, 'unroll')
unroll_blas = Data(Nunroll_ref, Tunroll_blas, 'blas')

Ninst = [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192]
Tinst = [1.20563, 5.83460, 16.49800, 38.84187, 68.30670, 120.75803, 153.63563, 249.62903, 341.46667, 463.99620, 683.64870, 865.98900, 1138.18767, 1497.64727, 1966.43930, 2569.43837]
inst_ref = Data(Ninst, Tinst, 'inst')

Ncuda_plain = [16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192, 9216, 10240]
Tcuda_plain = [0.21427, 0.20508, 0.22632, 0.33171, 3.32104, 2.38298, 7.76807, 17.16398, 34.18132, 57.93172, 111.10425, 161.71069, 206.79435, 588.64868, 815.86768, 954.90226, 1246.77327, 1561.44303, 2219.50073]
cuda_plain = Data(Ncuda_plain, Tcuda_plain, 'cuda_plain')

Ncuda_plain_f = [16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192]
Tcuda_plain_f = [0.16935, 0.20109, 0.23869, 0.40438, 0.83333, 4.96362, 29.95671, 89.47664, 153.78198, 299.09961, 514.53394, 818.13175, 1200.72095, 1708.45524, 2330.12907, 3112.56071, 4066.88363, 5293.52645, 6857.06331, 8600.14665, 10127.72396]
cuda_plain_f = Data(Ncuda_plain_f, Tcuda_plain_f, 'cuda_plain_origin')

Ncuda_shared = [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192, 9216, 10240, 11264, 12288]
Tcuda_shared = [2.12085, 7.11304, 13.61938, 27.74536, 44.48201, 80.08561, 116.56925, 143.56958, 254.76286, 408.81152, 635.75033, 913.37394, 1255.12801, 2048.31730, 2271.82601, 2711.82707]
cuda_shared = Data(Ncuda_shared, Tcuda_shared, 'cuda_shared')

Ncuda_reg = [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192, 9216, 10240, 11264, 12288, 14336, 16384]
Tcuda_reg = [1.87826, 5.56071, 11.19702, 17.68400, 31.25203, 52.20874, 73.27002, 99.00033, 156.67603, 228.27604, 342.40365, 459.13005, 594.99821, 754.98438, 943.02433, 1159.65894, 1655.10978, 2346.94067]
cuda_reg = Data(Ncuda_reg, Tcuda_reg, 'cuda_reg')

NSblas = [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192, 9216, 10240, 12288, 14336, 16384, 20480, 24576, 32768]
TSopenblas = [5.98836, 25.53760, 24.72510, 54.54004, 79.94043, 67.96794, 132.67871, 153.15291, 252.66602, 327.13062, 531.18376, 766.86410, 827.71761, 1530.97225, 2195.76961, 3135.44580, 5472.29305, 10955.84949, 13879.69, 36105.95996]
S_blas = Data(NSblas, TSopenblas, 'openblas')

NSblas_ref = [8192, 9216, 10240, 12288, 14336, 16384]
TSopenblas_ref = [766.86410, 827.71761, 1530.97225, 2195.76961, 3135.44580, 5472.29305]
S_blas_ref = Data(NSblas_ref, TSopenblas_ref, 'openblas')

Nstrassen = [8192, 9216, 10240, 11264, 12288, 14336, 16384]
Tstrassen = [989.63444, 1629.37598, 1950.76994, 2399.63062, 2880.35832, 4943.71663, 6784.64697]
strassen = Data(Nstrassen, Tstrassen, 'strassen')

Nstrassen_buf8k = [8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384, 20480, 24576, 32768]
Tstrassen_buf8k = [595.65503, 1425.48275, 1431.01367, 2013.60872, 2547.23307, 2750.98991, 3426.84424, 3811.40633, 4673.22168, 8670.22949, 13833, ]
strassen_buf8k = Data(Nstrassen_buf8k, Tstrassen_buf8k, 'strassen_buffer')

NSinst_false = [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192, 9216, 10240, 12288, 14336, 16384]
TSinst_false = [9.85734, 17.13908, 21.28573, 37.54468, 58.09131, 83.50562, 117.80176, 156.51497, 250.25293, 551.61271, 716.42000, 877.65072, 1307.87069, 1674.97363, 2657.95565, 5320.24634, 7733.79549]
S_inst_false = Data(NSinst_false, TSinst_false, 'inst')

NSinst = [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192, 9216, 10240, 12288, 14336, 16384, 20480, 24576, 32768]
TSinst = [8.34831, 10.37769, 13.10929, 23.38306, 28.53296, 40.32935, 81.35758, 118.63395, 223.89250, 274.97331, 470.78662, 655.12329, 905.62321, 1157.63989, 1819.61702, 3357.70972, 4573.57894, 8423.69995, 14275.77515, 42136.92188]
S_inst = Data(NSinst, TSinst, 'inst_improved')

Ncuda_multi = [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240, 11264, 12288, 16384, 20480, 24576, 28672, 32768]
Tcuda_multi = [24.64795, 45.85400, 84.09399, 104.29688, 152.09009, 253.85205, 288.22900, 385.45410, 508.07910, 687.02295, 890.41089, 1889.19397, 2203.04993, 3829.43750, 6311.90662, 6793.87000, 8677.59802]
cuda_multi = Data(Ncuda_multi, Tcuda_multi, 'cuda_multi')

Ncuda_prefetch = [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384, 20480, 24576, 28672, 32768, 49152, 65536]
Tcuda_prefetch = [23.29541, 49.20532, 74.26644, 104.38265, 151.56698, 234.30005, 299.16227, 394.71265, 509.62598, 625.71305, 772.54289, 1188.66577, 1117.38973, 1294.51839, 1474.82861, 1785.04956, 2782.23657, 4758.01058, 6175.21826, 8414.91968, 24217.45328, 51367.65]
cuda_prefetch = Data(Ncuda_prefetch, Tcuda_prefetch, 'cuda_prefetch')
cuda_prefetch_cut = Data(Ncuda_prefetch[:-4], Tcuda_prefetch[:-4], 'cuda_prefetch')

draw([cuda_prefetch, S_blas, strassen_buf8k, S_inst])

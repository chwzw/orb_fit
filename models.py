import numpy as np
from functions import M2E, Rz, Rx, plx_correction_relative, light_aberration_correction, e_alpha, e_delta

def no_plx_model(params, t, csst_pos, csst_v):
    
    ra_0, dec_0, pmra, pmdec = params
    t0 = np.min(t)  # 参考时刻，就取观测时间的最小值
    ra_icrs = ra_0 + pmra * (t-t0) / np.cos(np.radians(dec_0)) / 3600e3  # deg
    dec_icrs = dec_0 + pmdec * (t-t0) / 3600e3  # deg
    
    u = np.column_stack([np.cos(np.radians(ra_icrs)) * np.cos(np.radians(dec_icrs)), 
                         np.sin(np.radians(ra_icrs)) * np.cos(np.radians(dec_icrs)),
                         np.sin(np.radians(dec_icrs))]) # 修正视差前的方向矢量
    
    # u_csstcrs = light_aberration_correction(u, csst_v)  # 光行差修正
    u_csstcrs = u # 不考虑光行差的影响(检查光行差是否有问题)

    #############################################################################
    # 重要！这个仿真数据似乎并没有模拟光行差，因为光行差修正的量级大约在0.001deg, 而数据赤经赤纬的波动仅在0.00001deg量级。如果上面光行差修正启用，拟合效果极差。
    #############################################################################
    
    ra_csstcrs = np.degrees(np.arctan2(u_csstcrs[:,1], u_csstcrs[:,0]))
    dec_csstcrs = np.degrees(np.arcsin(u_csstcrs[:,2]))

    return ra_csstcrs, dec_csstcrs


def single_star_model(params, t, csst_pos, csst_v, u):
    '''单星模型（用于初步拟合质心位置，作为后续双星轨道拟合的初值）。不解算视向速度。
    参数:
        ra_0: 质心在t0时刻的相对赤经（deg）
        dec_0: 质心在t0时刻的相对赤纬（deg）
        pmra: 赤经的视差运动（mas/yr）
        pmdec: 赤纬的视差运动（mas/yr）
        plx: 视差（mas）
        t: 时间（Jyr）
    '''
    ra_t0, dec_t0, pmra, pmdec, plx = params
    t0 = np.min(t)  # 参考时刻，就取观测时间的最小值
    ra_t = ra_t0 + pmra * (t-t0)    # mas
    dec_t = dec_t0 + pmdec * (t-t0)   # mas
    
    # 修正到CSST系下（视差修正）
    #  u = np.column_stack([np.cos(np.radians(ra_icrs)) * np.cos(np.radians(dec_icrs)), 
    #                     np.sin(np.radians(ra_icrs)) * np.cos(np.radians(dec_icrs)),
    #                    np.sin(np.radians(dec_icrs))]) # 修正视差前的方向矢量
    
    plx_d_ra, plx_d_dec = plx_correction_relative(u, plx, csst_pos)  # 视差修正
    # u_csstcrs = light_aberration_correction(u_csstcrs_raw, csst_v)  # 光行差修正
    # u_csstcrs = u_csstcrs_raw # 不考虑光行差的影响(详见no_plx_model, 因光行差似乎无需考虑)
    
    # ra_csstcrs = np.degrees(np.arctan2(u_csstcrs[:,1], u_csstcrs[:,0]))
    # dec_csstcrs = np.degrees(np.arcsin(u_csstcrs[:,2]))

    return ra_t + plx_d_ra, dec_t + plx_d_dec



def orbit_model_campbell(params, t, csst_pos, csst_v, u):
    '''
    params: (ra_com_0, dec_com_0, pmra, pmdec, plx, period, ecc, semi_maj, incl, arg_peri, omega, m_0)
    
    模型函数定义（用参数以及时间t，CSST坐标以及速度，来表示出观测量ra,dec)
    使用campbell参数，即经典轨道参数：
        - semi_maj (mas): 轨道半长轴
        - ecc : 轨道偏心率
        - incl (rad): 轨道倾角
        - arg_peri (rad): 近点幅角
        - omega (deg): 升交点经度
        - M_0 (rad): 参考时刻平近点角
    注意:
    1. 题目要求求解的m_0是参考时刻的偏近点角，而非平近点角。（一般M表示平近点角，E表示偏近点角）。
    2. 注意仿真数据的内容的单位：
        - bx，by, bz(CSST位置) 单位为AU， 
        - vx,vy,vz(CSST速度)单位为km/s
        - ra_err, dec_err 单位为mas 
        - t 单位为 Jyear
    3. 模型函数应支持向量化操作。所有标量数组处理为(N,)形状；所有矢量（坐标、速度）处理为（N,3)形状, 每一行为一个坐标/速度矢量'''
    
    # 解包参数
    ra_com_0, dec_com_0, pmra, pmdec, plx, period, ecc, semi_maj, incl, arg_peri, omega, M_0 = params
    
    t0 = np.min(t)  # 参考时刻，假设没有给出就取观测时间的最小值
        
    # 计算时刻t时质心在天球上的位置
    ra_com_t, dec_com_t = single_star_model((ra_com_0, dec_com_0, pmra, pmdec, plx), t, csst_pos, csst_v, u)

    #### 计算可见伴星在椭圆中的“二维”坐标（COM为圆心，近点为x轴，右手系, 尺度使用天球上的角距离（mas)）
    # M_0 = m_0 - ecc * np.sin(m_0)  # 开普勒方程（M=E-e*sinE)，计算参考时刻平近点角
    M = M_0 + 2 * np.pi / period * (t - t0)  # 时刻t的平近点角
    E = M2E(M, ecc)  # 时刻t的偏近点角 （求解开普勒方程）)
    x_ellipse = semi_maj * (np.cos(E) - ecc) 
    y_ellipse = semi_maj * np.sqrt(1 - ecc**2) * np.sin(E)
    
    # 利用轨道指向参数（incl, omega, arg_peri), 旋转椭圆坐标到icrs框架下(轨道运动导致观测到的位置不同于COM)
    pos_ellipse = np.column_stack([x_ellipse, y_ellipse, np.zeros_like(x_ellipse)])  # 椭圆坐标系下的位置向量
    Rotation_matrix = Rz(-omega) @ Rx(-incl) @ Rz(-arg_peri)
    offset_ellipse = (Rotation_matrix @ pos_ellipse.T).T  
    
    d_ra_orbit =  offset_ellipse @ e_alpha(u)  # mas 
    d_dec_orbit = offset_ellipse @ e_delta(u)  # mas 

    ra_t = ra_com_t + d_ra_orbit 
    dec_t = dec_com_t + d_dec_orbit 

    return ra_t, dec_t


def orbit_model_ti(params, t, csst_pos, csst_v, u):
    '''
    params: (ra_com_0, dec_com_0, pmra, pmdec, plx, period, ecc, M0, A, B, F, G)
    
    模型函数定义（用参数以及时间t，CSST坐标以及速度，来表示出观测量ra,dec)
    使用Thiele-Innes参数，即轨道参数的另一种表达方式
    注意:
    1. 题目要求求解的m_0是参考时刻的偏近点角，而非平近点角。（一般M表示平近点角，E表示偏近点角）。
    2. 注意仿真数据的内容的单位：
        - bx，by, bz(CSST位置) 单位为AU， 
        - vx,vy,vz(CSST速度)单位为km/s
        - ra_err, dec_err 单位为mas 
        - t 单位为 Jyear
    3. 模型函数应支持向量化操作。所有标量数组处理为(N,)形状；所有矢量（坐标、速度）处理为（N,3)形状, 每一行为一个坐标/速度矢量'''
    
    # 解包参数
    ra_com_0, dec_com_0, pmra, pmdec, plx, period, ecc, M0, A, B, F, G = params

    t0 = np.min(t)  # 参考时刻，假设没有给出就取观测时间的最小值
        
    # 计算时刻t时质心在天球上的位置
    ra_com_t, dec_com_t = single_star_model((ra_com_0, dec_com_0, pmra, pmdec, plx), t, csst_pos, csst_v, u)
    M = M0 + 2 * np.pi / period * (t - t0)  # 时刻t的平近点角 (period 单位为yr)
    E = M2E(M, ecc)  # 时刻t的偏近点角 （求解开普勒方程）)
    x = np.cos(E) - ecc
    y = np.sqrt(1 - ecc**2) * np.sin(E)
    d_ra_orbit = A * x + F * y  # mas
    d_dec_orbit = B * x + G * y  # mas
    
    ra_t = ra_com_t + d_ra_orbit
    dec_t = dec_com_t + d_dec_orbit
    
    return ra_t, dec_t



def cal_residual(params, t, ra_obs, dec_obs, ra_err, dec_err, csst_pos, csst_v, u, model):
    ra_pred, dec_pred = model(params, t, csst_pos, csst_v, u)
    ra_res = (ra_obs - ra_pred) / ra_err
    dec_res = (dec_obs - dec_pred) / dec_err
    return np.concatenate([ra_res, dec_res])





import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.interpolate import interp1d


def e_alpha(u):
    '''计算赤经方向单位矢量'''
    e_z = np.array([0, 0, 1])
    return np.cross(e_z, u) / np.linalg.norm(np.cross(e_z, u))


def e_delta(u):
    '''计算赤纬方向单位矢量'''
    e_ra = e_alpha(u)
    e_dec_vec = np.cross(u, e_ra)
    return e_dec_vec / np.linalg.norm(e_dec_vec)



def M2E(M, e, max_iter=100, tol=1e-15):
    """
    将平近点角M转换为偏近点角E（支持高偏心率轨道）
    
    参数:
        M (np.array): 平近点角数组（弧度）
        e (float): 轨道偏心率（0 ≤ e < 1）
        max_iter (int): 最大迭代次数
        tol (float): 收敛容差
    
    返回:
        np.array: 偏近点角E数组（弧度）
    """
    # 1. 归一化M到[0, 2π)区间
    M_norm = np.mod(M, 2*np.pi)
    
    # 2. 改进初始猜测值（特别处理高偏心率情况）
    if e < 0.8:
        # 中低偏心率使用标准初始值
        E = np.copy(M_norm)
    else:
        # 高偏心率使用改进初始值
        # 当M接近0或2π时使用立方根近似
        near_periapsis = (M_norm < 0.1) | (M_norm > 2*np.pi - 0.1)
        E = np.where(near_periapsis, np.cbrt(6 * M_norm), M_norm)
        
        # 添加偏移避免导数分母接近零
        E += 0.001 * e * np.sin(E)
    
    # 3. 带安全保护的牛顿迭代
    for i in range(max_iter):
        sinE = np.sin(E)
        cosE = np.cos(E)
        
        # 计算函数值和导数值
        f = E - e * sinE - M_norm
        df = 1 - e * cosE
        
        # 安全保护：避免除以接近零的值
        df_safe = np.where(np.abs(df) < 1e-8, np.sign(df)*1e-8, df)
        
        # 计算步长
        delta = f / df_safe
        
        # 步长限制：防止过大跳跃
        max_delta = 0.5 * np.pi
        delta_clipped = np.clip(delta, -max_delta, max_delta)
        
        # 更新E值
        E -= delta_clipped
        
        # 检查收敛性（考虑高偏心率情况）
        residual = np.abs(f)
        if np.max(residual) < tol:
            return E
            
        # 高偏心率特殊处理：当迭代停滞时添加扰动
        if i > 10 and np.max(np.abs(delta_clipped)) < 1e-8:
            E += 0.01 * np.sin(i) * np.random.rand(*E.shape)
    
    print(f'Not converged')
    print(f'Remaining f: {M_norm - E + e * np.sin(E)}')
    return E


import numpy as np
import time

def M2E(M, e, max_iter=100, tol=1e-15, debug=False):
    """
    高效平近点角M到偏近点角E转换（仅迭代未收敛元素）
    
    参数:
        M (np.array): 平近点角数组（弧度）
        e (float): 轨道偏心率（0 ≤ e < 1）
        max_iter (int): 最大迭代次数
        tol (float): 收敛容差
        debug (bool): 是否输出调试信息
    
    返回:
        np.array: 偏近点角E数组（弧度）
    """
    # 1. 归一化M到[0, 2π)区间
    M_norm = np.mod(M, 2 * np.pi)
    n = M_norm.size
    
    # 2. 初始化结果数组和工作状态
    E_result = np.empty_like(M_norm)
    
    # 创建活动掩码：跟踪哪些元素需要迭代
    active_mask = np.ones(n, dtype=bool)
    
    # 3. 改进初始猜测值（特别处理高偏心率情况）
    E_initial = np.empty_like(M_norm)
    
    if e < 0.8:
        E_initial[active_mask] = M_norm[active_mask]  # 中低偏心率使用标准初始值
    else:
        # 高偏心率改进初始值
        near_periapsis = (M_norm < 0.1) | (M_norm > 2*np.pi - 0.1)
        E_initial = np.where(near_periapsis, np.cbrt(6 * M_norm), M_norm)
        E_initial[active_mask] += 0.001 * e * np.sin(E_initial[active_mask])
    
    # 4. 工作数组初始化（用于迭代）
    E_working = E_initial.copy()
    perturbation_counter = np.zeros(n, dtype=int)  # 跟踪扰动应用次数
    last_delta = np.zeros(n)  # 跟踪上一步的delta值
    
    # 5. 迭代计数器初始化
    iter_count = 0
    start_time = time.perf_counter()
    
    # 6. 带安全保护的牛顿迭代（仅对活动元素）
    while iter_count < max_iter and np.any(active_mask):
        iter_count += 1
        active_count = np.sum(active_mask)
        if debug and iter_count % 10 == 0:
            print(f"Iter {iter_count}: {active_count}/{n} active elements")
        
        # 提取活动元素的子集
        M_active = M_norm[active_mask]
        E_active = E_working[active_mask]
        
        # 计算函数值和导数值（仅活动元素）
        sinE = np.sin(E_active)
        cosE = np.cos(E_active)
        f = E_active - e * sinE - M_active
        df = 1 - e * cosE
        
        # 安全保护：避免除以接近零的值
        df_safe = np.where(np.abs(df) < 1e-8, np.sign(df)*1e-8, df)
        
        # 计算步长
        delta = f / df_safe
        
        # 步长限制：防止过大跳跃
        max_delta = 0.5 * np.pi
        delta_clipped = np.clip(delta, -max_delta, max_delta)
        
        # 更新活动元素的E值
        E_active -= delta_clipped
        E_working[active_mask] = E_active
        
        # 更新最后一步的delta值
        last_delta[active_mask] = delta_clipped
        
        # 计算活动元素的残差
        residual = np.abs(f)
        
        # 找出当前迭代中收敛的活动元素
        converged = residual < tol
        converged_ids = np.where(active_mask)[0][converged]
        
        # 标记已收敛元素
        if np.any(converged):
            E_result[converged_ids] = E_working[converged_ids]
            active_mask[converged_ids] = False
        
        # 高偏心率特殊处理：当迭代停滞时添加扰动
        if e >= 0.8 and np.any(active_mask):
            # 仅对活动元素检查
            active_indices = np.where(active_mask)[0]
            stalled = np.abs(last_delta[active_indices]) < 1e-8
            stalled_ids = active_indices[stalled]
            
            if np.any(stalled) and iter_count > 10:
                # 限制扰动应用次数（防止无限循环）
                can_perturb = perturbation_counter[stalled_ids] < 3
                perturb_ids = stalled_ids[can_perturb]
                
                if np.any(perturb_ids):
                    if debug:
                        print(f"Applying perturbation to {perturb_ids.size} elements")
                    
                    # 应用扰动（不同元素不同扰动）
                    perturbations = 0.01 * np.sin(iter_count) * (np.random.random(perturb_ids.size) - 0.5)
                    E_working[perturb_ids] += perturbations
                    
                    # 更新扰动计数器
                    perturbation_counter[perturb_ids] += 1
    
    # 7. 迭代结束后处理
    elapsed = time.perf_counter() - start_time
    
    # 处理剩余未收敛元素
    if np.any(active_mask):
        unconverged = np.sum(active_mask)
        if debug:
            print(f"Warning: {unconverged} elements not converged after {iter_count} iterations")
            # 计算剩余未收敛元素的残差
            active_indices = np.where(active_mask)[0]
            f_final = E_working[active_mask] - e * np.sin(E_working[active_mask]) - M_norm[active_mask]
            max_residual = np.max(np.abs(f_final))
            print(f"Max residual: {max_residual:.2e}, Average residual: {np.mean(np.abs(f_final)):.2e}")
        
        # 保存当前最佳值
        E_result[active_mask] = E_working[active_mask]
    
    if debug:
        # 计算最终残差（仅用于调试）
        f_final = E_result - e * np.sin(E_result) - M_norm
        avg_residual = np.mean(np.abs(f_final))
        max_residual = np.max(np.abs(f_final))
        print(f"Completed in {elapsed*1000:.2f}ms | Iterations: {iter_count}")
        print(f"Max residual: {max_residual:.2e} | Avg residual: {avg_residual:.2e}")
    
    return E_result

# 测试函数
if __name__ == '__main__':
    # 创建包含多种情况的测试数据
    test_M = np.array([
        # 低偏心率正常值
        0.1, 1.0, 2.0, 3.0,
        # 高偏心率正常值
        0.01, 0.1, 1.0, 
        # 困难的高偏心率值
        0.001, np.pi, 2*np.pi - 0.001,
        # 多个重复值（测试向量化性能）
        *([12.35793846724546] * 1000)
    ])
    
    # 测试不同偏心率
    for ecc in [0.1, 0.5, 0.99]:
        print(f"\nTesting with e = {ecc:.4f}")
        
        # 测试普通版本
        print("Testing optimized version:")
        E_opt = M2E(test_M, ecc, debug=True)
        residual = test_M - E_opt + ecc * np.sin(E_opt)
        print(f"Max residual: {np.max(np.abs(residual)):.2e}")



def Rx(theta):
    '''坐标系绕x轴旋转theta度'''
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), np.sin(theta)],
                     [0, -np.sin(theta), np.cos(theta)]])

def Rz(theta):
    '''绕z轴的旋转矩阵（旋转坐标系的矩阵，注意区分旋转矢量的矩阵）'''
    return np.array([[np.cos(theta), np.sin(theta), 0],
                     [-np.sin(theta),  np.cos(theta), 0],
                     [0,              0,             1]])

# def parallax_correction_old(u_raw, plx, csst_pos):
    '''视差修正(输入应为方向单位矢量)
        plx (mas)
        csst_pos (AU)
        1 pc = 206264.806 AU
        '''
    # 确保视差不为零或负值
    # plx = max(plx, 1e-6)
    
    pos = (u_raw * 206265000 / plx - csst_pos) # AU
    u = pos / np.linalg.norm(pos, axis=1)[:, None]
    return u


# def parallax_correction_new(u_raw, plx, csst_pos, return_delta_u=False):
    '''视差修正(输入应为方向单位矢量)
        plx (mas)
        csst_pos (AU)
        1 pc = 206264.806 AU
        '''
    # 确保视差不为零或负值
    # plx = max(plx, 1e-6)
    
    offset_vector = - csst_pos * plx / 206265000 # 无量纲，准备投影到垂直于u_raw的方向

    delta_u = offset_vector - np.einsum('ij,ij->i', offset_vector, u_raw)[:, np.newaxis] * u_raw
    u = u_raw + delta_u  # 修正视差 (修正完实际上不再是单位矢量)
    
    # test
    # print(np.linalg.norm(delta_u, axis=1) * 360 / 2 / np.pi) # deg
    if return_delta_u:
        return delta_u
    else:
        return u


def plx_correction_relative(u, plx, csst_pos):
    '''u 仅是一个(3,)的numpy array
    返回值： d_ra, d_dec (mas)'''
    offset_vector = - csst_pos * plx / 206265000 # rad，准备投影到垂直于u的方向
    delta_u = offset_vector - np.einsum('ij,j->i', offset_vector, u)[:,np.newaxis] * u
    # 投影到e_ra e_dec方向
    d_ra = np.rad2deg(delta_u @ e_alpha(u)) * 3.6e6 # mas
    d_dec = np.rad2deg(delta_u @ e_delta(u)) * 3.6e6 # mas
    return d_ra, d_dec


def light_aberration_correction(u, csst_v):
    '''光行差修正(输入应为方向单位矢量)
        csst_v (km/s)'''
    beta = csst_v / 299792.458 # km/s
    delta_u = (beta - np.einsum('ij,ij->i', beta, u)[:, np.newaxis] * u)
    u_prime = u + delta_u  # 修正光行差
    
    # test 
    print(f' light aberration correction: {np.linalg.norm(delta_u, axis=1) * 360 / 2 / np.pi}') # deg
    return u_prime

#

def cal_cov_and_print(result):
    print("收敛信息:", result.message)
    print("迭代次数:", result.nfev)
    print("最终残差平方和:", np.sum(result.fun**2))

    # 计算参数的标准误差（需要协方差矩阵）
    if result.status > 0:  # 成功收敛
        jac = result.jac # 计算雅可比矩阵
        residuals_var = np.sum(result.fun**2) / (len(result.fun) - len(result.x)) # 计算残差方差（残差平方和除以自由度）
        # 计算参数协方差矩阵
        try:
            pcov = np.linalg.inv(jac.T.dot(jac)) * residuals_var
            perr = np.sqrt(np.diag(pcov))
        except np.linalg.LinAlgError:
            print("无法计算协方差矩阵，可能是雅可比矩阵奇异")
            
    else:
        print("拟合可能未收敛，无法计算标准误差")
    
    try: 
        return pcov, perr
    except:
        return None, None




def _compute_model_predictions(model, params, t, csst_pos, csst_v, u, num_points):
    """
    内部函数：计算模型的预测值
    """
    # 计算模型预测值
    ra_pred, dec_pred = model(params, t, csst_pos, csst_v, u)
    
    if num_points is not None:
        # 生成一系列t，以便预测结果平滑
        t_min, t_max = np.min(t), np.max(t)
        t_smooth = np.linspace(t_min, t_max, num_points)

        # 对 CSST 位置和速度进行插值(三维插值)
        csst_pos_interp = np.zeros((len(t_smooth), 3))
        csst_v_interp = np.zeros((len(t_smooth), 3))
        
        for i in range(3):  # 对 x, y, z 分量分别插值
            pos_interp_func = interp1d(t, csst_pos[:, i], kind='cubic', fill_value='extrapolate')
            v_interp_func = interp1d(t, csst_v[:, i], kind='cubic', fill_value='extrapolate')
            
            csst_pos_interp[:, i] = pos_interp_func(t_smooth)
            csst_v_interp[:, i] = v_interp_func(t_smooth)
    
        ra_pred_smooth, dec_pred_smooth = model(params, t_smooth, csst_pos_interp, csst_v_interp, u)
        return ra_pred, dec_pred, ra_pred_smooth, dec_pred_smooth
    
    return ra_pred, dec_pred, ra_pred, dec_pred

def _setup_plot(figsize=(5, 5), dpi=250):
    """
    内部函数：设置绘图的基础参数
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    
    # 设置坐标轴标签
    ax.set_xlabel('Relative RA (mas)')
    ax.set_ylabel('Relative Dec (mas)')
    
    '''# 创建自定义刻度格式化函数（度转毫角秒）
    def deg_to_mas(x, pos):
        return f"{x * 3600 * 1000:.0f}"
    
    # 应用自定义格式化器
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(deg_to_mas))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(deg_to_mas))'''
    
    ax.set_aspect('equal', adjustable=None)
    
    ax.grid(True, alpha=0.3)
    return fig, ax


def _finalize_plot(fig, ax, title):
    """内部函数：完成绘图"""
    ax.legend()
    ax.set_title(title)
    
    # 强制设置等比例
    ax.set_aspect('equal', adjustable='box')
    
    # 获取当前坐标轴范围
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # 计算当前范围的比例
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    
    # 计算需要调整的比例
    if x_range > y_range:
        # 需要扩展y轴范围
        center_y = (ylim[0] + ylim[1]) / 2
        new_y_range = x_range
        ax.set_ylim(center_y - new_y_range/2, center_y + new_y_range/2)
    else:
        # 需要扩展x轴范围
        center_x = (xlim[0] + xlim[1]) / 2
        new_x_range = y_range
        ax.set_xlim(center_x - new_x_range/2, center_x + new_x_range/2)
    
    plt.tight_layout()
    plt.show()
    
    
def plot_model(model, params, t, csst_pos, csst_v, u, num_points=200, point_size=1, plot_pred=True, ax=None):
    """
    绘制模型预测曲线
    
    参数:
        model: 模型函数
        params: 模型参数
        t: 观测时间
        csst_pos: CSST位置 (AU)
        csst_v: CSST速度 (AU/day)
        u: 方向矢量
        num_points: 平滑曲线的点数（默认200）
        point_size: 点的大小（默认1）
        plot_pred: 是否在观测时间点绘制预测值（默认True）
    """
    # 计算模型预测
    ra_pred, dec_pred, ra_pred_smooth, dec_pred_smooth = _compute_model_predictions(
        model, params, t, csst_pos, csst_v, u, num_points
    )
    
    # 设置绘图
    if ax is None:
        fig, ax = _setup_plot()
    else:
        fig = ax.figure
        
    # 绘制平滑预测曲线
    ax.plot(ra_pred_smooth, dec_pred_smooth, label='Predicted', color='blue')
    
    # 绘制预测值（观测时间点）
    if plot_pred:
        ax.scatter(ra_pred, dec_pred, label='Predicted (obs times)', color='black', s=10*point_size)

    _finalize_plot(fig, ax, 'Model Predictions')
    plt.show()


def plot_model_and_obs(model, params, t, d_ra_obs, d_dec_obs, ra_err, dec_err, 
                         csst_pos, csst_v, u, num_points=200, point_size=1):
    """
    绘制模型预测与观测值的对比图，包括误差棒
    
    参数:
        model: 模型函数
        params: 模型参数
        t: 观测时间
        d_ra_obs: 观测的赤经偏移 (deg)
        d_dec_obs: 观测的赤纬偏移 (deg)
        ra_err: 赤经观测误差 (deg)
        dec_err: 赤纬观测误差 (deg)
        csst_pos: CSST位置 (AU)
        csst_v: CSST速度 (AU/day)
        u: 方向矢量
        num_points: 平滑曲线的点数（默认200）
    """
    # 设置绘图
    fig, ax = _setup_plot()
    
    # 绘制观测值带误差棒
    ax.errorbar(d_ra_obs, d_dec_obs, 
                xerr=ra_err, yerr=dec_err,
                fmt='.', color='red', markersize=point_size,
                ecolor='red', elinewidth=1, capsize=2,
                label='Observed with error bars')
    
    # 计算并绘制模型预测
    ra_pred, dec_pred, ra_pred_smooth, dec_pred_smooth = _compute_model_predictions(
        model, params, t, csst_pos, csst_v, u, num_points
    )
    
    # 绘制平滑预测曲线
    ax.plot(ra_pred_smooth, dec_pred_smooth, label='Predicted', color='blue')
    
    # 绘制预测值（观测时间点）
    ax.scatter(ra_pred, dec_pred, label='Predicted (obs times)', color='black', s=10*point_size)
    
    _finalize_plot(fig, ax, 'Model vs Observations')
    plt.show()
    
    
    
import seaborn as sns
def correlation_analysis(cov, param_names = [
        'ra_com', 'dec_com', 'pmra_com',
        'pmdec_com', 'plx',
        'period', 'ecc', 'semi_maj',
        'incl', 'arg_peri', 'omega', 'm_0'
    ]):
    if cov is not None:
        corr_matrix = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
        
        # 绘制相关矩阵热图
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, 
                    xticklabels=param_names, 
                    yticklabels=param_names,
                    cmap='RdBu_r', 
                    center=0,
                    annot=True, 
                    fmt='.2f',
                    square=True)
        plt.title('correlation matrix')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()


def plot_norm_residual(residuals, ax=None):
    """
    绘制残差图

    参数:
        residuals: 残差数组
        ax: 目标坐标轴（可选）
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else: 
        fig = ax.figure
    
    n = residuals.shape[0]
    delta_ra_norm = residuals[:n//2]
    delta_dec_norm = residuals[n//2:]

    ax.scatter(delta_ra_norm, delta_dec_norm, s=10, color='black')
    ax.set_xlabel('Residual RA (Normalized)')
    ax.set_ylabel('Residual Dec (Normalized)')
    ax.set_title('Residuals')
    
    
    
def plot_residual(model, params, t, d_ra_obs, d_dec_obs, ra_err, dec_err, 
                        csst_pos, csst_v, u, ax=None):
    # how residual is calculated
    ra_pred, dec_pred = model(params, t, csst_pos, csst_v, u)
    res_ra = (d_ra_obs - ra_pred) 
    res_dec = (d_dec_obs - dec_pred)
    residuals = np.concatenate([res_ra, res_dec])
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    res_ra = residuals[:len(residuals)//2]
    res_dec = residuals[len(residuals)//2:]
    
    ax.scatter(res_ra, res_dec, s=10, color='black')
    ax.set_xlabel('Residual RA (mas)')
    ax.set_ylabel('Residual Dec (mas)')
    ax.set_title('Residuals')
    plt.show()
    
import os
import pickle
import lib.PLOT
import numpy as np
from lib.add import *
import scipy as sp
import matplotlib.pyplot as plt


if __name__ == '__main__':
    ver_name = 'ver1'
    b1_list = []
    b2_list = []
    b3_list = []
    P_self = []
    P_pv = []
    P_wind = []
    P_GP = []
    P_REC = []
    npv_list = []
    gp_list = []
    rec_list = []
    cu_list = []
    ppa_pv_list = []
    ppa_wind_list = []
    pv_lcoe_list = []
    wind_lcoe_list = []
    kepco_list = []
    kepco_yearly_list = []
    BAU_npv_list = []

    pkl_list = os.listdir('../result_pickle/' + ver_name + '/')
    pkl_num = len(pkl_list)

    Cap_load = 1000                                # kW
    N_yr = 10
    r = 4.5 / 100                                                         # 사회적할인율

    demand_2018 = pd.read_csv('../data/' + ver_name + '/' + 'AMR고객별부하현황_업종별_2018.csv', engine='python', encoding='euc-kr')
    demand_2019 = pd.read_csv('../data/' + ver_name + '/' + 'AMR고객별부하현황_업종별_2019.csv', engine='python', encoding='euc-kr')
    demand_name = ['광업', '제조업', '전기, 가스, 증기 및 수도사업', '하수 · 폐기물 처리, 원료재생 및 환경복원업', '건설업']
    demand_2018_data = make_demand_pattern_by_category(demand_pattern=demand_2018, name_list=demand_name)
    demand_2019_data = make_demand_pattern_by_category(demand_pattern=demand_2019, name_list=demand_name)
    demand = make_average_pattern(demand_2018_data, demand_2019_data)
    A_Demand = (demand / demand.max().max()).values
    P_load = np.round(A_Demand * Cap_load, 2)       # 공휴일이나 토요일, 일요일은?
    P_load = extend_2dto3d(data=P_load, N_yr=N_yr, R_array=np.array([1]*N_yr))

    P_load = np.round(P_load, 0)        # 11.22 소수점 없애는걸로 수정 (시간 빨라질 수 있는지 확인용)

    p = lib.PLOT.PlotOpt()
    # 연도별 전략 선택
    for file in pkl_list:
        with open('../result_pickle/' + ver_name + '/' + file, 'rb') as handle:
            result = pickle.load(handle)
        b1_list.append(result['b_1'])
        b2_list.append(result['b_2'])
        b3_list.append(result['b_3'])
        npv_list.append(result['npv'])

        p_self_sum = []
        p_pv_sum = []
        p_wind_sum = []
        for i in range(10):
            p_self_sum.append(result['P_self'][i].sum())
            p_pv_sum.append(result['P_pv'][i].sum())
            p_wind_sum.append(result['P_wind'][i].sum())

        P_self.append(np.array(p_self_sum))
        P_pv.append(np.array(p_pv_sum))
        P_wind.append(np.array(p_wind_sum))
        P_GP.append(result['P_GP'])
        P_REC.append(result['P_REC'])

        gp_list.append(result['mu_gp'])
        rec_list.append(result['mu_rec'])
        cu_list.append(result['mu_cu'])
        ppa_pv_list.append(result['mu_pv_ppa'])
        ppa_wind_list.append(result['mu_wind_ppa'])

        pv_lcoe_list.append(result['r_pv_lcoe'])
        wind_lcoe_list.append(result['r_wind_lcoe'])
        # 347번부터 r_kepco 소수점 달라짐 (350~1000개 사이중에서 무작위로 100개 선택하여 추세 보여줘야할듯)
        r_kepco = np.round((result['mu_C_kepco'][9][0][0] / result['mu_C_kepco'][0][0][0] - 1), 2)
        kepco_list.append(r_kepco * 100)

        # 산업용 107원을 init으로 봄
        kepco_yearly_list.append(make_price_trend(trend_type='linear', init_value=107, rate=r_kepco, N_yr=10, round_loc=1))

        BAU_npv_list.append(make_BAU_scenario_npv(P_load=P_load, mu_C_kepco=result['mu_C_kepco'],
                                                  mu_D_kepco=result['mu_D_kepco'], r=r, Cap_load=Cap_load,
                                                  N_yr=N_yr))

    # # opt1
    # p.set_frame()
    # p.set_axis(xlim_min=21, xlim_max=30, yticks=[-1, 0, 1, 2], ytickslabel=['', 'strategy2', 'strategy1', ''], x_label='Year', y_label='Strategy Choice')
    # x_data = np.arange(21, 31, 1)
    #
    # b1_per = make_b_per(pkl_num=pkl_num, b_list=b1_list)
    # b2_per = make_b_per(pkl_num=pkl_num, b_list=b2_list)
    #
    # p.opt1_text(x_loc=24, y_loc=0.5, text='MonteCarlo count = ' + str(pkl_num))
    # for i in range(10):
    #     p.opt1_text(x_loc=x_data[i], y_loc=1, text=np.round(b1_per[i], 2).astype('str') + '%')
    #     p.opt1_text(x_loc=x_data[i], y_loc=0, text=np.round(b2_per[i], 2).astype('str') + '%')
    # p.set_title(title='Strategy choice')
    # #
    #
    # # opt2
    # p.set_frame()
    # p.set_axis(xlim_min=21, xlim_max=30, yticks=np.arange(0, 120, 20), ytickslabel=np.arange(0, 120, 20).astype('str'), x_label='Year', y_label='%')
    #
    # p_per = make_P_per(pkl_num=pkl_num, P_self=P_self, P_GP=P_GP, P_REC=P_REC, P_pv=P_pv, P_wind=P_wind) * 100
    # p.opt2_bar(x_data=np.arange(21, 31, 1), y_data=p_per)
    # p.set_title(title='')
    #
    # # opt3
    # p.set_frame()
    # p.set_axis(xlim_min=21, xlim_max=30, yticks=np.arange(0, 120, 20), ytickslabel=np.arange(0, 120, 20).astype('str'), x_label='Year', y_label='%')
    #
    # b1_per = make_b_per(pkl_num=pkl_num, b_list=b1_list)
    # b2_per = make_b_per(pkl_num=pkl_num, b_list=b2_list)
    # b_per = np.concatenate((b1_per.reshape(10, 1), b2_per.reshape(10, 1)), axis=1)
    # p.opt3_bar(x_data=np.arange(21, 31, 1), y_data=b_per)
    # p.set_title(title='')
    
    # opt4
    npv_array = np.array(npv_list) / 1135.58       # 환율적용
    npv_min = np.min(npv_array)
    npv_max = np.max(npv_array)

    BAU_npv_array = np.array(BAU_npv_list) / 1135.58
    BAU_min = np.min(BAU_npv_list)
    BAU_max = np.max(BAU_npv_list)

    # # NPV의 평균, 분산, 표준편차
    # npv_mu = np.mean(npv_array)
    # npv_var = np.var(npv_array, ddof=1)
    # npv_std = np.std(npv_array, ddof=1)
    #
    # npv_median = np.median(npv_array)
    #
    # L_ = npv_mu - (1.96 * (npv_std / np.sqrt(npv_array.shape[0])))
    # U_ = npv_mu + (1.96 * (npv_std / np.sqrt(npv_array.shape[0])))
    #
    control = 20
    x_axis_min = math.floor(npv_min / 10**(int(math.log10(npv_min)))) * 10**(int(math.log10(npv_min)))
    x_axis_max = math.ceil(npv_max / 10**(int(math.log10(npv_max)))) * 10**(int(math.log10(npv_max)))
    interval = (x_axis_max - x_axis_min) / control

    x_axis, y_axis = make_npv_axis(npv_array=npv_array, control=20, interval=interval, init=x_axis_min)
    bau_x, bau_y = make_npv_axis(npv_array=BAU_npv_array, control=20, interval=interval, init=x_axis_min)

    p.set_frame()
    p.set_axis2(xlim_min=x_axis_min, xlim_max=x_axis_max, xticks=np.arange(x_axis_min, x_axis_max + interval, interval),
               xtickslabel=np.arange(x_axis_min, x_axis_max + interval, interval).astype('str'),
               yticks=np.arange(0, max(y_axis) + 10, 10),
               ytickslabel=np.arange(0, max(y_axis) + 10, 10).astype('str'),
               x_label='$', y_label='Frequency', ylim_min=0, ylim_max=max(y_axis) + 10)
    p.opt4_bar(x_axis=x_axis, interval=interval, y_axis=y_axis, color='b', alpha=0.3, label='Optimal_Scenario_NPV')
    # plt.close()

    # p.set_frame()
    # p.set_axis2(xlim_min=x_axis_min, xlim_max=x_axis_max, xticks=np.arange(x_axis_min, x_axis_max + interval, interval),
    #            xtickslabel=np.arange(x_axis_min, x_axis_max + interval, interval).astype('str'),
    #            yticks=np.arange(0, max(y_axis) + 10, 10),
    #            ytickslabel=np.arange(0, max(y_axis) + 10, 10).astype('str'),
    #            x_label='$', y_label='Frequency', ylim_min=0, ylim_max=max(y_axis) + 10)
    # p.opt4_bar(x_axis=bau_x, interval=interval, y_axis=bau_y, color='r', alpha=0.5)

    # L_, U_ = st.norm.interval(0.95, npv_avg, scale=st.sem(npv_array, ddof=0))
    BAU_avg = np.sum(np.array(BAU_npv_list) / 1135.58) / len(BAU_npv_list)
    p.axes.axvline(BAU_avg, color='b', label='BAU_scenario_average_NPV')
    # p.axes.axvline(U_, color='r')
    p.axes.legend(loc='upper left', ncol=1, fontsize=14, frameon=True, shadow=True)

    # # opt5
    # p.set_frame()
    # p.set_axis3(xlim_min=21, xlim_max=30, x_label='Year', y_label='won/kWh')
    # p.opt5_line(x_axis=np.arange(21, 31, 1), y_axis=gp_list, color='brown', label='Green Premium', alpha=0.7)
    # p.opt5_line(x_axis=np.arange(21, 31, 1), y_axis=rec_list, color='r', label='REC', alpha=0.7)
    # p.opt5_line(x_axis=np.arange(21, 31, 1), y_axis=cu_list, color='grey', label='ETS', alpha=0.7)
    # p.opt5_line(x_axis=np.arange(21, 31, 1), y_axis=ppa_pv_list, color='r', label='PPA(PV)', alpha=0.7)
    # p.opt5_line(x_axis=np.arange(21, 31, 1), y_axis=ppa_wind_list, color='b', label='PPA(Onshore Wind)', alpha=0.7)
    # p.opt5_line(x_axis=np.arange(21, 31, 1), y_axis=kepco_yearly_list, color='k', label='Utility tariff', alpha=0.7)
    # p.fig.legend(loc='upper center', ncol=2, fontsize=15, frameon=True, shadow=True)
    # p.axes.grid(True, axis='y', alpha=0.5, linestyle='--')


    ## opt6 [부하패턴 그리기]
    # Cap_load = 1000
    #
    # demand_2018 = pd.read_csv('../data/' + ver_name + '/' + 'AMR고객별부하현황_업종별_2018.csv', engine='python', encoding='euc-kr')
    # demand_2019 = pd.read_csv('../data/' + ver_name + '/' + 'AMR고객별부하현황_업종별_2019.csv', engine='python', encoding='euc-kr')
    # demand_name = ['광업', '제조업', '전기, 가스, 증기 및 수도사업', '하수 · 폐기물 처리, 원료재생 및 환경복원업', '건설업']
    # demand_2018_data = make_demand_pattern_by_category(demand_pattern=demand_2018, name_list=demand_name)
    # demand_2019_data = make_demand_pattern_by_category(demand_pattern=demand_2019, name_list=demand_name)
    # demand = make_average_pattern(demand_2018_data, demand_2019_data)
    # A_Demand = (demand / demand.max().max()).values
    # P_load = np.round(A_Demand * Cap_load, 2)       # 공휴일이나 토요일, 일요일은?
    # P_load = np.round(P_load, 0)        # 11.22 소수점 없애는걸로 수정 (시간 빨라질 수 있는지 확인용)
    # flat = P_load.flatten()
    # flat.sort()
    # any_min = flat[7]
    # P_load[np.where(P_load < 600)] = any_min
    # p.set_frame()
    # p.set_axis(xlim_min=1, xlim_max=8760, yticks=np.arange(0, np.max(P_load), 100), ytickslabel=np.arange(0, np.max(P_load), 100).astype('str'), x_label='Hour', y_label='kW')
    # x_data = np.arange(1, 8761, 1)
    # p.axes.plot(x_data, P_load.reshape(-1), color='k', label='Demand_pattern', alpha=0.7)
    # p.fig.legend(loc='upper center', ncol=2, fontsize=15, frameon=True, shadow=True)

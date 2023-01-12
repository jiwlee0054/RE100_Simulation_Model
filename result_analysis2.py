import os
import pickle
import numpy as np
import random
import math
import copy

import lib.PLOT
import lib.parameters as para

if __name__ == '__main__':
    # general_flag = False
    # progressive_flag = False
    uncertainty_flag = False
    fems_flag = False

    options = para.ProbOptions()
    IFN = para.ReadInputData(options)
    input_parameters_pulp = para.ParameterPulpFrom(options, IFN, None)

    loc = options.loc3
    country_list = ['KR', 'US']

    # uncertainty_range = [200]
    # uncertainty_item = ['PVlcoe', 'Onshorelcoe', 'tariff']

    result_dict = dict()

    country = 'KR'
    folder_list = [f"{country}_constant", f"{country}_linear"]

    for folder in folder_list:
        files = os.listdir(f"{loc}/{folder}")
        result_dict[folder] = dict()
        if len(files) != options.general_scen_num:
            print('결과 갯수 부족')
            quit()
        else:
            for f in files:
                p_self_sum_by_year = []
                p_pv_sum_by_year = []
                p_onshore_sum_by_year = []
                p_eac_sum_by_year = []
                p_tariff_sum_by_year = []

                scen_num = f.split('.')[0]
                print(f'file reading_ scenario_number : {country}, {scen_num}')
                result_dict[folder][scen_num] = dict()

                with open(f"{loc}/{folder}/{f}", 'rb') as of:
                    result = pickle.load(of)

                for y in range(options.year1 - options.year0 + 1):
                    p_self_sum_by_year.append(result['p_self_ydh'][y, :, :].sum())
                    p_pv_sum_by_year.append(result['p_pv_ydh'][y, :, :].sum())
                    p_onshore_sum_by_year.append(result['p_onshore_ydh'][y, :, :].sum())
                    p_eac_sum_by_year.append(result['p_eac_y'][y].sum())
                    p_tariff_sum_by_year.append(result['p_tariff_ydh'][y, :, :].sum())

                result_dict[folder][scen_num]['p_self_sum_by_year'] = p_self_sum_by_year
                result_dict[folder][scen_num]['p_pv_sum_by_year'] = p_pv_sum_by_year
                result_dict[folder][scen_num]['p_onshore_sum_by_year'] = p_onshore_sum_by_year
                result_dict[folder][scen_num]['p_eac_sum_by_year'] = p_eac_sum_by_year
                result_dict[folder][scen_num]['p_tariff_sum_by_year'] = p_tariff_sum_by_year
                result_dict[folder][scen_num]['npv'] = result['npv']
                result_dict[folder][scen_num]['u_y'] = result['u_y']
                result_dict[folder][scen_num]['BAU'] = sum([IFN.P_load.sum() * result['tariff'][y] /
                                                             (1 + input_parameters_pulp.discount_rate) **
                                                             (y - options.year0) for y in range(options.year0,
                                                                                                options.year1 + 1)])

    if uncertainty_flag:
        country = 'KR'
        item = 'tariff'
        result_dict.clear()
        file1 = random.sample(os.listdir(f"{loc}/KR_constant"), 300)

        alpha = 0.7
        color1 = 'b'
        color2 = 'r'
        control1 = 150
        control2 = 10
        size1 = 100
        size2 = 200

        folder = f"{country}_{item}_{size2}"
        files2 = os.listdir(f"{loc}/{folder}")

        x_axis_max = 6800000
        x_axis_min = 5500000
        y_max = 20
        label1 = 'Optimal_Scenario_NPV'

        result_dict[f"{country},{item},{size1}"] = dict()
        result_dict[f"{country},{item},{size2}"] = dict()
        if len(files2) != options.uncertainty_scen_num:
            print('결과 갯수 부족')
            quit()
        else:
            for f in file1:
                p_self_sum_by_year = []
                p_pv_sum_by_year = []
                p_onshore_sum_by_year = []
                p_eac_sum_by_year = []
                p_tariff_sum_by_year = []

                scen_num = f.split('.')[0]
                print(f'file reading_ scenario_number : f"{country},{item},{size1}", {scen_num}')
                result_dict[f"{country},{item},{size1}"][scen_num] = dict()

                with open(f"{loc}/KR_constant/{f}", 'rb') as of:
                    result = pickle.load(of)

                for y in range(options.year1 - options.year0 + 1):
                    p_self_sum_by_year.append(result['p_self_ydh'][y, :, :].sum())
                    p_pv_sum_by_year.append(result['p_pv_ydh'][y, :, :].sum())
                    p_onshore_sum_by_year.append(result['p_onshore_ydh'][y, :, :].sum())
                    p_eac_sum_by_year.append(result['p_eac_y'][y].sum())
                    p_tariff_sum_by_year.append(result['p_tariff_ydh'][y, :, :].sum())

                result_dict[f"{country},{item},{size1}"][scen_num]['p_self_sum_by_year'] = p_self_sum_by_year
                result_dict[f"{country},{item},{size1}"][scen_num]['p_pv_sum_by_year'] = p_pv_sum_by_year
                result_dict[f"{country},{item},{size1}"][scen_num]['p_onshore_sum_by_year'] = p_onshore_sum_by_year
                result_dict[f"{country},{item},{size1}"][scen_num]['p_eac_sum_by_year'] = p_eac_sum_by_year
                result_dict[f"{country},{item},{size1}"][scen_num]['p_tariff_sum_by_year'] = p_tariff_sum_by_year
                result_dict[f"{country},{item},{size1}"][scen_num]['npv'] = result['npv']
                result_dict[f"{country},{item},{size1}"][scen_num]['u_y'] = result['u_y']
                result_dict[f"{country},{item},{size1}"][scen_num]['BAU'] = sum(
                    [IFN.P_load.sum() * result['tariff'][y] /
                     (1 + input_parameters_pulp.discount_rate) **
                     (y - options.year0) for y in
                     range(options.year0,
                           options.year1 + 1)])
            fig_name = f'Frequency of constant NPV based on Monte Carlo, {country}, {item}, {size1}_new'
            p = lib.PLOT.PlotOpt()
            npv_list = []
            for k in result_dict[f"{country},{item},{size1}"].keys():
                npv_list.append(result_dict[f"{country},{item},{size1}"][k]['npv'])

            npv_array = np.array(npv_list)
            interval = (x_axis_max - x_axis_min) / control1

            npv_x_axis = []
            npv_y_axis = []
            init = copy.deepcopy(x_axis_min)
            for i in range(control1):
                npv_y_axis.append(np.where((npv_array >= init) * (npv_array < init + interval))[0].shape[0])
                npv_x_axis.append(init)
                init += interval
            npv_x_axis.append(init)

            choice1 = 0
            choice2 = 200
            y_new = npv_y_axis[choice1:choice2]
            x_new = npv_x_axis[choice1:choice2 + 1]

            array_x = np.array(x_new).astype('int')
            array_x_str = []
            for i in range(len(x_new)):
                if i % control2 == 0:
                    v = np.round(array_x[i] / 10 ** 6, 2)
                    array_x_str.append(f"${v}M")
                else:
                    array_x_str.append('')

            array_y = np.arange(0, y_max + 5, 5).astype('str')

            p.set_frame()
            p.set_axis2(xlim_min=min(x_new), xlim_max=max(x_new),
                        xticks=x_new,
                        xtickslabel=array_x_str,
                        yticks=np.arange(0, y_max + 5, 5),
                        ytickslabel=array_y,
                        x_label='', y_label='', ylim_min=0, ylim_max=max(y_new) + 5)

            x_new.pop(-1)
            p.axes.bar(np.array(x_new) + (interval / 2), y_new, width=interval / 4, color=color1, alpha=alpha,
                       label=label1)
            p.axes.legend(loc='upper left', ncol=1, fontsize=20, frameon=True, shadow=True)
            p.save_fig(options.loc4, country, fig_name)

            for f in files2:
                p_self_sum_by_year = []
                p_pv_sum_by_year = []
                p_onshore_sum_by_year = []
                p_eac_sum_by_year = []
                p_tariff_sum_by_year = []

                scen_num = f.split('.')[0]
                print(f'file reading_ scenario_number : f"{country},{item},{size2}", {scen_num}')
                result_dict[f"{country},{item},{size2}"][scen_num] = dict()

                with open(f"{loc}/{folder}/{f}", 'rb') as of:
                    result = pickle.load(of)

                for y in range(options.year1 - options.year0 + 1):
                    p_self_sum_by_year.append(result['p_self_ydh'][y, :, :].sum())
                    p_pv_sum_by_year.append(result['p_pv_ydh'][y, :, :].sum())
                    p_onshore_sum_by_year.append(result['p_onshore_ydh'][y, :, :].sum())
                    p_eac_sum_by_year.append(result['p_eac_y'][y].sum())
                    p_tariff_sum_by_year.append(result['p_tariff_ydh'][y, :, :].sum())

                result_dict[f"{country},{item},{size2}"][scen_num]['p_self_sum_by_year'] = p_self_sum_by_year
                result_dict[f"{country},{item},{size2}"][scen_num]['p_pv_sum_by_year'] = p_pv_sum_by_year
                result_dict[f"{country},{item},{size2}"][scen_num]['p_onshore_sum_by_year'] = p_onshore_sum_by_year
                result_dict[f"{country},{item},{size2}"][scen_num]['p_eac_sum_by_year'] = p_eac_sum_by_year
                result_dict[f"{country},{item},{size2}"][scen_num]['p_tariff_sum_by_year'] = p_tariff_sum_by_year
                result_dict[f"{country},{item},{size2}"][scen_num]['npv'] = result['npv']
                result_dict[f"{country},{item},{size2}"][scen_num]['u_y'] = result['u_y']
                result_dict[f"{country},{item},{size2}"][scen_num]['BAU'] = sum(
                    [IFN.P_load.sum() * result['tariff'][y] /
                     (1 + input_parameters_pulp.discount_rate) **
                     (y - options.year0) for y in
                     range(options.year0,
                           options.year1 + 1)])

            # Uncertainty analysis based on size of uncertainty range
            fig_name = f'Frequency of constant NPV based on Monte Carlo, {country}, {item}, {size2}_new'
            p = lib.PLOT.PlotOpt()
            npv_list = []
            for k in result_dict[f"{country},{item},{size2}"].keys():
                npv_list.append(result_dict[f"{country},{item},{size2}"][k]['npv'])

            npv_array = np.array(npv_list)
            interval = (x_axis_max - x_axis_min) / control1

            npv_x_axis = []
            npv_y_axis = []
            init = copy.deepcopy(x_axis_min)
            for i in range(control1):
                npv_y_axis.append(np.where((npv_array >= init) * (npv_array < init + interval))[0].shape[0])
                npv_x_axis.append(init)
                init += interval
            npv_x_axis.append(init)

            choice1 = 0
            choice2 = 150
            y_new = npv_y_axis[choice1:choice2]
            x_new = npv_x_axis[choice1:choice2 + 1]

            array_x = np.array(x_new).astype('int')
            array_x_str = []
            for i in range(len(x_new)):
                if i % control2 == 0:
                    v = np.round(array_x[i] / 10 ** 6, 2)
                    array_x_str.append(f"${v}M")
                else:
                    array_x_str.append('')
            array_y = np.arange(0, y_max + 5, 5).astype('str')

            p.set_frame()
            p.set_axis2(xlim_min=min(x_new), xlim_max=max(x_new),
                        xticks=x_new,
                        xtickslabel=array_x_str,
                        yticks=np.arange(0, y_max + 5, 5),
                        ytickslabel=array_y,
                        x_label='', y_label='', ylim_min=0, ylim_max=max(y_new) + 5)

            x_new.pop(-1)
            p.axes.bar(np.array(x_new) + (interval / 2), y_new, width=interval / 4, color=color1, alpha=alpha,
                       label=label1)
            p.axes.legend(loc='upper left', ncol=1, fontsize=20, frameon=True, shadow=True)
            p.save_fig(options.loc4, country, fig_name)



    p = lib.PLOT.PlotOpt()
    # NPV of optimal sourcing  strategy for immediate RE100
    country = 'KR'
    alpha = 0.7
    color1 = 'b'
    color2 = 'r'
    label1 = 'Optimal_Scenario_NPV'
    label2 = 'BAU_scenario_average_NPV'
    control1 = 150
    control2 = 10

    x_axis_max = 7000000
    x_axis_min = 5000000
    y_max = 60
    for folder in folder_list:
        npv_list = []
        BAU_npv_list = []
        if 'constant' in folder:
            fig_name = 'Frequency of constant NPV based on Monte Carlo'
        else:
            fig_name = 'Frequency of progressive NPV based on Monte Carlo'

        result_ = result_dict[folder]

        for k in result_.keys():
            npv_list.append(result_[k]['npv'])
            BAU_npv_list.append(result_[k]['BAU'])

        npv_array = np.array(npv_list)
        BAU_avg = np.sum(BAU_npv_list) / len(BAU_npv_list)
        interval = (x_axis_max - x_axis_min) / control1

        npv_x_axis = []
        npv_y_axis = []
        init = copy.deepcopy(x_axis_min)
        for i in range(control1):
            npv_y_axis.append(np.where((npv_array >= init) * (npv_array < init + interval))[0].shape[0])
            npv_x_axis.append(init)
            init += interval
        npv_x_axis.append(init)

    # y축 데이터 -> 0인 데이터는 제외
        y_new = npv_y_axis[0:control1]
        x_new = npv_x_axis[0:control1 + 1]

        array_x = np.array(x_new).astype('int')
        array_x_str = []
        for i in range(len(x_new)):
            if i % control2 == 0:
                v = np.round(array_x[i] / 10 ** 6, 2)
                array_x_str.append(f"${v}M")
            else:
                array_x_str.append('')
        array_y = np.arange(0, y_max + 5, 5).astype('str')
        for i in range(array_y.shape[0]):
            if i % 5 == 0:
                pass
            else:
                array_y[i] = ''
        p.set_frame()
        p.set_axis2(xlim_min=min(x_new), xlim_max=max(x_new),
                    xticks=x_new,
                    xtickslabel=array_x_str,
                    yticks=np.arange(0, y_max + 5, 5),
                    ytickslabel=array_y,
                    x_label='', y_label='', ylim_min=0, ylim_max=max(y_new) + 5)

        x_new.pop(-1)
        p.axes.bar(np.array(x_new)+(interval/2), y_new, width=interval/4, color=color1, alpha=alpha, label=label1)
        p.axes.axvline(BAU_avg, color=color2, label=label2)
        p.axes.legend(loc='upper left', ncol=1, fontsize=20, frameon=True, shadow=True)
        p.save_fig(options.loc4, country, fig_name)

    # # Decision analysis of FEMS installation
    # if fems_flag and general_flag:
    #     for country in country_list:
    #         fig_name = f"Decision analysis of FEMS installation in {country}"
    #         p.opt6_bar(country, options, 'dodgerblue', 'The time of FEMS decision', fig_name, result_dict)
import os
import pickle
import numpy as np
import random

import lib.PLOT
import lib.parameters as para


if __name__ == '__main__':
    general_flag = False
    progressive_flag = False
    uncertainty_flag = True
    fems_flag = False
    
    options = para.ProbOptions()
    IFN = para.ReadInputData(options)
    input_parameters_pulp = para.ParameterPulpFrom(options, IFN, None)

    loc = options.loc3
    country_list = ['KR', 'US']

    # uncertainty_range = [200]
    # uncertainty_item = ['PVlcoe', 'Onshorelcoe', 'tariff']

    result_dict = dict()

    for country in country_list:
        if general_flag:
            folder = f"{country}_constant"
        elif progressive_flag:
            folder = f"{country}_linear"
        else:
            break

        files = os.listdir(f"{loc}/{folder}")
        result_dict[country] = dict()
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
                result_dict[country][scen_num] = dict()

                with open(f"{loc}/{folder}/{f}", 'rb') as of:
                    result = pickle.load(of)

                for y in range(options.year1 - options.year0 + 1):
                    p_self_sum_by_year.append(result['p_self_ydh'][y, :, :].sum())
                    p_pv_sum_by_year.append(result['p_pv_ydh'][y, :, :].sum())
                    p_onshore_sum_by_year.append(result['p_onshore_ydh'][y, :, :].sum())
                    p_eac_sum_by_year.append(result['p_eac_y'][y].sum())
                    p_tariff_sum_by_year.append(result['p_tariff_ydh'][y, :, :].sum())

                result_dict[country][scen_num]['p_self_sum_by_year'] = p_self_sum_by_year
                result_dict[country][scen_num]['p_pv_sum_by_year'] = p_pv_sum_by_year
                result_dict[country][scen_num]['p_onshore_sum_by_year'] = p_onshore_sum_by_year
                result_dict[country][scen_num]['p_eac_sum_by_year'] = p_eac_sum_by_year
                result_dict[country][scen_num]['p_tariff_sum_by_year'] = p_tariff_sum_by_year
                result_dict[country][scen_num]['npv'] = result['npv']
                result_dict[country][scen_num]['u_y'] = result['u_y']
                result_dict[country][scen_num]['BAU'] = sum([IFN.P_load.sum() * result['tariff'][y] /
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
                result_dict[f"{country},{item},{size1}"][scen_num]['BAU'] = sum([IFN.P_load.sum() * result['tariff'][y] /
                                                                               (1 + input_parameters_pulp.discount_rate) **
                                                                               (y - options.year0) for y in
                                                                               range(options.year0,
                                                                                     options.year1 + 1)])
            p = lib.PLOT.PlotOpt()
            fig_name = f'Frequency of constant NPV based on Monte Carlo, {country}, {item}, {size1}_new'
            label1 = 'Optimal_Scenario_NPV'
            label2 = 'BAU_scenario_average_NPV'

            p.opt5_bar(alpha, color1, label1, color2, label2, country, item, size1, result_dict, control1, control2,
                       options, fig_name)

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
                result_dict[f"{country},{item},{size2}"][scen_num]['BAU'] = sum([IFN.P_load.sum() * result['tariff'][y] /
                                                                                (1 + input_parameters_pulp.discount_rate) **
                                                                                (y - options.year0) for y in
                                                                                range(options.year0,
                                                                                      options.year1 + 1)])

            # Uncertainty analysis based on size of uncertainty range
            p = lib.PLOT.PlotOpt()
            fig_name = f'Frequency of constant NPV based on Monte Carlo, {country}, {item}, {size2}_new'

            label1 = 'Optimal_Scenario_NPV'
            label2 = 'BAU_scenario_average_NPV'

            p.opt5_bar(alpha, color1, label1, color2, label2, country, item, size2, result_dict, control1, control2,
                       options, fig_name)
                        
                        
    p = lib.PLOT.PlotOpt()
    # # Optimal portfolios for renewable electricity sourcing by year
    # if general_flag:
    #     for country in country_list:
    #         p.opt2_bar(result_dict, options, IFN, country, 'Optimal portfolios for renewable electricity sourcing by year')
    #
    # NPV of optimal sourcing  strategy for immediate RE100
    if general_flag:
        fig_name = 'Frequency of constant NPV based on Monte Carlo'
    if progressive_flag:
        fig_name = 'Frequency of progressive NPV based on Monte Carlo'

    if general_flag or progressive_flag:
        country = 'KR'
        alpha = 0.7
        color1 = 'b'
        color2 = 'r'
        label1 = 'Optimal_Scenario_NPV'
        label2 = 'BAU_scenario_average_NPV'
        control1 = 150
        control2 = 10
        p.opt4_bar(alpha, color1, label1, color2, label2, country, result_dict, control1, control2, options, fig_name)

    # # Decision analysis of FEMS installation
    # if fems_flag and general_flag:
    #     for country in country_list:
    #         fig_name = f"Decision analysis of FEMS installation in {country}"
    #         p.opt6_bar(country, options, 'dodgerblue', 'The time of FEMS decision', fig_name, result_dict)
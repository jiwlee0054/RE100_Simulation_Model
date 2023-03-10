import copy

import lib.parameters as para
import lib.re100_pln_model as re_pln
import lib.PLOT

import time
import pickle
import json
import os
import numpy as np
import pandas as pd
import math


def run_portfolio(options, IFN, flag_msg):
    y_resolution = 1
    y_overlap = 0

    count = 0
    input_parameters_pulp = para.ParameterPulpFrom(options, IFN)
    result_dict = dict()



    result_dict['u_y'] = dict()
    result_dict['p_sg_pv_y_d_h'] = dict()
    result_dict['p_sg_onshore_y_d_h'] = dict()
    result_dict['p_tariff_y_d_h'] = dict()
    result_dict['p_ppa_pv_y_d_h'] = dict()
    result_dict['p_ppa_onshore_y_d_h'] = dict()
    result_dict['capacity_cs_contract_y'] = dict()
    result_dict['p_eac_y'] = dict()
    result_dict['capacity_pv_y'] = dict()
    result_dict['capacity_onshore_y'] = dict()
    result_dict['c_sg_y'] = dict()
    result_dict['c_tariff_used_y'] = dict()
    result_dict['c_tariff_pre_dema_y'] = dict()
    result_dict['c_tariff_pro_dema_y'] = dict()
    result_dict['c_ppa_y'] = dict()
    result_dict['c_eac_y'] = dict()
    result_dict['c_residual_y'] = dict()
    result_dict['c_loss_payment_y'] = dict()
    result_dict['c_funding_tariff_y'] = dict()
    result_dict['c_funding_ppa_y'] = dict()
    result_dict['c_commission_kepco_y'] = dict()
    result_dict['c_commission_ppa_y'] = dict()

    while True:
        if count == 0:
            options.model_start_y = options.year0
            options.model_end_y = options.model_start_y + y_resolution - 1
        else:
            options.model_start_y = options.model_end_y - y_overlap + 1
            options.model_end_y = options.model_end_y + y_resolution

        if options.model_end_y > options.year1:
            options.model_end_y = options.year1

        if flag_msg:
            print(f"model time: {options.model_start_y} ~ {options.model_end_y}", end=' ')

        if count == 0:
            result = re_pln.solve_re100_milp(options, input_parameters_pulp, solver_name, result_dict=None)
        else:
            result = re_pln.solve_re100_milp(options, input_parameters_pulp, solver_name, result_dict=result_dict)

        for y in range(options.model_start_y, options.model_end_y + 1, 1):
            result_dict['u_y'].update(result['u_y'])
            result_dict['p_sg_pv_y_d_h'].update(result['p_sg_pv_y_d_h'])
            result_dict['p_sg_onshore_y_d_h'].update(result['p_sg_onshore_y_d_h'])
            result_dict['p_tariff_y_d_h'].update(result['p_tariff_y_d_h'])
            result_dict['p_ppa_pv_y_d_h'].update(result['p_ppa_pv_y_d_h'])
            result_dict['p_ppa_onshore_y_d_h'].update(result['p_ppa_onshore_y_d_h'])
            result_dict['capacity_cs_contract_y'].update(result['capacity_cs_contract_y'])
            result_dict['p_eac_y'].update(result['p_eac_y'])
            result_dict['capacity_pv_y'].update(result['capacity_pv_y'])
            result_dict['capacity_onshore_y'].update(result['capacity_onshore_y'])
            result_dict['c_sg_y'].update(result['c_sg_y'])
            result_dict['c_tariff_used_y'].update(result['c_tariff_used_y'])
            result_dict['c_tariff_pre_dema_y'].update(result['c_tariff_pre_dema_y'])
            result_dict['c_tariff_pro_dema_y'].update(result['c_tariff_pro_dema_y'])

            result_dict['c_ppa_y'].update(result['c_ppa_y'])
            result_dict['c_eac_y'].update(result['c_eac_y'])
            result_dict['c_residual_y'].update(result['c_residual_y'])
            result_dict['c_loss_payment_y'].update(result['c_loss_payment_y'])
            result_dict['c_funding_tariff_y'].update(result['c_funding_tariff_y'])
            result_dict['c_funding_ppa_y'].update(result['c_funding_ppa_y'])
            result_dict['c_commission_kepco_y'].update(result['c_commission_kepco_y'])
            result_dict['c_commission_ppa_y'].update(result['c_commission_ppa_y'])

        count += 1
        if options.model_end_y == options.year1:
            break

    result_dict['lambda_CAPEX_PV_y'] = input_parameters_pulp.lambda_CAPEX_PV_y
    result_dict['lambda_OPEX_PV_y'] = input_parameters_pulp.lambda_OPEX_PV_y
    result_dict['lambda_CAPEX_onshore_y'] = input_parameters_pulp.lambda_CAPEX_onshore_y
    result_dict['lambda_OPEX_onshore_y'] = input_parameters_pulp.lambda_OPEX_onshore_y
    result_dict['lambda_eac_y'] = input_parameters_pulp.lambda_eac_y
    result_dict['lambda_PPA_pv_y'] = input_parameters_pulp.lambda_PPA_pv_y
    result_dict['lambda_PPA_onshore_y'] = input_parameters_pulp.lambda_PPA_onshore_y
    result_dict['lambda_AS_payment_y'] = input_parameters_pulp.lambda_AS_payment_y
    result_dict['lambda_climate_y'] = input_parameters_pulp.lambda_climate_y
    result_dict['lambda_nt_c_y'] = input_parameters_pulp.lambda_nt_c_y
    result_dict['lambda_fuel_adjustment_y'] = input_parameters_pulp.lambda_fuel_adjustment_y
    result_dict['lambda_loss_payment_y'] = input_parameters_pulp.lambda_loss_payment_y
    result_dict['lambda_welfare_y'] = input_parameters_pulp.lambda_welfare_y
    result_dict['rate_pv'] = input_parameters_pulp.rate_pv
    result_dict['rate_onshore'] = input_parameters_pulp.rate_onshore
    result_dict['tariff_y'] = input_parameters_pulp.tariff_y

    result_dict['lambda_tariff_pre_y_d_h'] = dict()
    result_dict['lambda_tariff_pro_y_d_h'] = dict()
    for y in range(options.year0, options.year1 + 1, 1):
        result_dict['lambda_tariff_pre_y_d_h'][y] = dict()
        result_dict['lambda_tariff_pro_y_d_h'][y] = dict()
        for d in range(options.model_start_d, options.model_end_d + 1, 1):
            result_dict['lambda_tariff_pre_y_d_h'][y][d] = dict()
            result_dict['lambda_tariff_pro_y_d_h'][y][d] = dict()
            for h in range(options.model_start_h, options.model_end_h + 1, 1):
                result_dict['lambda_tariff_pre_y_d_h'][y][d][h] = input_parameters_pulp.lambda_tariff_pre_y_d_h[y, d, h]
                result_dict['lambda_tariff_pro_y_d_h'][y][d][h] = input_parameters_pulp.lambda_tariff_pro_y_d_h[y, d, h]

    return result_dict


def run_simulation(options, IFN):
    if options.run_simulation_flag:
        result = run_portfolio(options, IFN, flag_msg=False)
        if options.parallel_processing_flag:
            with open(f'{options.loc_pp_result}/{options.scenario_num}.json', 'w') as outfile:
                json.dump(result, outfile, indent=4)
        return result

    if options.run_simulation_by_case_flag:
        customize_exemption_list = [True, False]
        for flag in customize_exemption_list:
            options.customize_exemption_flag = flag
            options.set_result_loc(opt='????????????')
            result = run_portfolio(options, IFN, flag_msg=False)
            with open(f'{options.loc_pp_result}/{options.scenario_num}.json', 'w') as outfile:
                json.dump(result, outfile, indent=4)


def run_analysis(options, IFN):
    if options.result_analysis_flag:
        input_parameters_pulp = para.ParameterPulpFrom(options, IFN)
        if options.customize_exemption_flag:
            opt = '??????????????????'
        else:
            opt = '??????????????????'

        if os.path.isdir(f"{options.loc_plot}/{options.date_result}"):
            pass
        else:
            os.makedirs(f"{options.loc_plot}/{options.date_result}")

        if os.path.isdir(f"{options.loc_plot}/{options.date_result}/{opt}"):
            pass
        else:
            os.makedirs(f"{options.loc_plot}/{options.date_result}/{opt}")

        if os.path.isdir(f"{options.loc_excel}/{options.date_result}"):
            pass
        else:
            os.makedirs(f"{options.loc_excel}/{options.date_result}")

        if os.path.isdir(f"{options.loc_excel}/{options.date_result}/{opt}"):
            pass
        else:
            os.makedirs(f"{options.loc_excel}/{options.date_result}/{opt}")

        loc_plot = f"{options.loc_plot}/{options.date_result}/{opt}"
        loc_excel = f"{options.loc_excel}/{options.date_result}/{opt}"

        result_dict_s = dict()
        files = [f for f in os.listdir(f"{options.loc_pp_result}") if '.json' in f]
        for f in files:
            s = f.split('.')[0]
            result_dict_s[s] = dict()

            print(f'file reading scenario_number : {s}')

            with open(f'{options.loc_pp_result}/{s}.json', 'r') as f:
                result = json.load(f)

            result_dict_s[s]['u_y'] = result['u_y']
            result_dict_s[s]['p_sg_pv_y_d_h'] = result['p_sg_pv_y_d_h']
            result_dict_s[s]['p_sg_onshore_y_d_h'] = result['p_sg_onshore_y_d_h']
            result_dict_s[s]['p_tariff_y_d_h'] = result['p_tariff_y_d_h']
            result_dict_s[s]['p_ppa_pv_y_d_h'] = result['p_ppa_pv_y_d_h']
            result_dict_s[s]['p_ppa_onshore_y_d_h'] = result['p_ppa_onshore_y_d_h']
            result_dict_s[s]['p_ppa_onshore_y_d_h'] = result['p_ppa_onshore_y_d_h']
            result_dict_s[s]['capacity_cs_contract_y'] = result['capacity_cs_contract_y']

            result_dict_s[s]['p_eac_y'] = result['p_eac_y']
            result_dict_s[s]['capacity_pv_y'] = result['capacity_pv_y']
            result_dict_s[s]['capacity_onshore_y'] = result['capacity_onshore_y']

            result_dict_s[s]['c_sg_y'] = result['c_sg_y']
            result_dict_s[s]['c_tariff_used_y'] = result['c_tariff_used_y']
            result_dict_s[s]['c_tariff_pre_dema_y'] = result['c_tariff_pre_dema_y']
            result_dict_s[s]['c_tariff_pro_dema_y'] = result['c_tariff_pro_dema_y']

            result_dict_s[s]['c_ppa_y'] = result['c_ppa_y']
            result_dict_s[s]['c_eac_y'] = result['c_eac_y']

            result_dict_s[s]['c_residual_y'] = result['c_residual_y']
            result_dict_s[s]['c_loss_payment_y'] = result['c_loss_payment_y']
            result_dict_s[s]['c_funding_tariff_y'] = result['c_funding_tariff_y']
            result_dict_s[s]['c_funding_ppa_y'] = result['c_funding_ppa_y']

            result_dict_s[s]['c_commission_kepco_y'] = result['c_commission_kepco_y']
            result_dict_s[s]['c_commission_ppa_y'] = result['c_commission_ppa_y']

            result_dict_s[s]['lambda_CAPEX_PV_y'] = result['lambda_CAPEX_PV_y']
            result_dict_s[s]['lambda_OPEX_PV_y'] = result['lambda_OPEX_PV_y']
            result_dict_s[s]['lambda_CAPEX_onshore_y'] = result['lambda_CAPEX_onshore_y']
            result_dict_s[s]['lambda_OPEX_onshore_y'] = result['lambda_OPEX_onshore_y']
            result_dict_s[s]['lambda_eac_y'] = result['lambda_eac_y']
            result_dict_s[s]['lambda_PPA_pv_y'] = result['lambda_PPA_pv_y']
            result_dict_s[s]['lambda_PPA_onshore_y'] = result['lambda_PPA_onshore_y']
            result_dict_s[s]['lambda_AS_payment_y'] = result['lambda_AS_payment_y']
            result_dict_s[s]['lambda_climate_y'] = result['lambda_climate_y']
            result_dict_s[s]['lambda_nt_c_y'] = result['lambda_nt_c_y']
            result_dict_s[s]['lambda_fuel_adjustment_y'] = result['lambda_fuel_adjustment_y']
            result_dict_s[s]['lambda_loss_payment_y'] = result['lambda_loss_payment_y']
            result_dict_s[s]['lambda_welfare_y'] = result['lambda_welfare_y']

            result_dict_s[s]['lambda_tariff_pre_y_d_h'] = result['lambda_tariff_pre_y_d_h']
            result_dict_s[s]['lambda_tariff_pro_y_d_h'] = result['lambda_tariff_pro_y_d_h']


        if options.excel_unit_price_by_items_flag:
            year0 = options.year0
            year1 = options.year1

            x_data = np.arange(year0, year1 + 1, 1)

            ppa_pv_df = pd.DataFrame(columns=x_data, index=list(result_dict_s.keys()))
            ppa_onshore_df = pd.DataFrame(columns=x_data, index=list(result_dict_s.keys()))
            ppa_eac_df = pd.DataFrame(columns=x_data, index=list(result_dict_s.keys()))

            for s in result_dict_s.keys():
                for y in x_data:
                    ppa_pv_df.loc[s, y] = np.round(result_dict_s[s]['lambda_PPA_pv_y'][f"{y}"], 2)
                    ppa_onshore_df.loc[s, y] = np.round(result_dict_s[s]['lambda_PPA_onshore_y'][f"{y}"], 2)
                    ppa_eac_df.loc[s, y] = np.round(result_dict_s[s]['lambda_eac_y'][f"{y}"], 2)
            ppa_pv_df.to_excel(f"{loc_excel}/PPA_PV_??????.xlsx")
            print(f"PPA_PV_??????.xlsx ??????")
            ppa_onshore_df.to_excel(f"{loc_excel}/PPA_??????_??????.xlsx")
            print(f"PPA_??????_??????.xlsx ??????")
            ppa_eac_df.to_excel(f"{loc_excel}/?????????_??????.xlsx")
            print(f"?????????_??????.xlsx ??????")

        if options.plot_optimal_portfolio_flag:
            p = lib.PLOT.PlotOpt()
            fig_name = 'Optimal portfolios for renewable electricity sourcing by year'

            item_num = 6
            year0 = options.year0
            year1 = options.year1

            x_data = np.arange(year0, year1 + 1, 1)
            y_data = np.zeros((year1 - year0 + 1, item_num))
            for s in result_dict_s.keys():
                for y in options.set_y:
                    demand_y = sum([input_parameters_pulp.demand_y_d_h[y, d, h]
                                    for d in options.set_d
                                    for h in options.set_h]
                                   )
                    if demand_y == 0:
                        pass
                    else:
                        y_tilda = y - year0

                        y_data[y_tilda, 0] += sum(result_dict_s[s]['p_sg_pv_y_d_h'][f"{y}"][f"{d}"][f"{h}"]
                                                  for d in options.set_d
                                                  for h in options.set_h) / demand_y
                        y_data[y_tilda, 1] += sum(result_dict_s[s]['p_sg_onshore_y_d_h'][f"{y}"][f"{d}"][f"{h}"]
                                                  for d in options.set_d
                                                  for h in options.set_h) / demand_y

                        y_data[y_tilda, 2] += sum(result_dict_s[s]['p_ppa_pv_y_d_h'][f"{y}"][f"{d}"][f"{h}"]
                                                  for d in options.set_d
                                                  for h in options.set_h) / demand_y
                        y_data[y_tilda, 3] += sum(result_dict_s[s]['p_ppa_onshore_y_d_h'][f"{y}"][f"{d}"][f"{h}"]
                                                  for d in options.set_d
                                                  for h in options.set_h) / demand_y

                        y_data[y_tilda, 4] += result_dict_s[s]['p_eac_y'][f"{y}"] / demand_y
                        y_data[y_tilda, 5] += sum(input_parameters_pulp.ee_y_d_h[y, d, h]
                                                  for d in options.set_d
                                                  for h in options.set_h) / demand_y

            y_data /= len(result_dict_s.keys())
            y_data *= 100

            p.set_frame()
            p.axes.set_xticks(np.arange(options.year0, options.year1 + 1, 1).astype('int'))
            p.axes.set_xticklabels(np.arange(options.year0, options.year1 + 1, 1).astype('str'))
            p.axes.set_yticks(np.arange(0, 120, 20))
            p.axes.set_yticklabels(np.arange(0, 120, 20).astype('str'))
            p.axes.set_xlabel('Year', fontsize=30)
            p.axes.set_ylabel('%', fontsize=30)
            p.axes.tick_params(axis='x', labelsize=20)
            p.axes.tick_params(axis='y', labelsize=25)

            bottom = 0
            # ?????? ??????
            for num in range(y_data.shape[1]):
                # label, color = p.opt2_label_color(num=num)
                if num == 0:
                    label, color = '????????????(PV)', 'dimgray'
                elif num == 2:
                    label, color = '?????? PPA (PV)', 'goldenrod'
                elif num == 3:
                    label, color = '?????? PPA (Onshore Wind)', 'deepskyblue'
                elif num == 4:
                    label, color = '????????? ??????', 'b'

                if num == 0:
                    p.axes.bar(x_data, y_data[:, num], width=0.3, color=color, label=label)
                elif num == 1 or num == 5:
                    bottom += y_data[:, num - 1]
                    pass
                else:
                    bottom += y_data[:, num - 1]
                    p.axes.bar(x_data, y_data[:, num], width=0.3, bottom=bottom, color=color, label=label)
            p.axes.legend(loc='best', ncol=2, fontsize=20, frameon=True, shadow=True)
            p.save_fig(loc_plot, fig_name)
            print(f"{fig_name}.png ??????")

            if options.excel_optimal_portfolio_flag:
                y_data_df = pd.DataFrame(y_data,
                                         columns=['self generation (PV) [%]', 'self generation (onshore Wind) [%]',
                                                  'corporate PPAs with PV [%]', 'corporate PPAs with onshore Wind [%]',
                                                  'unbundled EAC (REC) [%]', 'Energy Efficiency [%]'],
                                         index=x_data)
                y_data_df.to_excel(f"{loc_excel}/????????? ?????? RE100 ?????? ???????????????(??????).xlsx")
                print(f"????????? ?????? RE100 ?????? ???????????????(??????).xlsx")

        if options.excel_optimal_portfolio_cost_flag:
            item_num = 19
            unit = options.unit
            year0 = options.year0
            year1 = options.year1

            x_data = np.arange(year0, year1 + 1, 1)
            y_data = np.zeros((year1 - year0 + 1, item_num))
            for s in result_dict_s.keys():
                z_data = np.zeros((year1 - year0 + 1, item_num))
                for y in options.set_y:
                    y_tilda = y - year0
                    if y == options.year0:
                        # ????????? ???????????? ?????? ??? ?????? ??????
                        y_data[y_tilda, 0] += (result_dict_s[s]['capacity_pv_y'][f"{y}"] *
                                               (result_dict_s[s]['lambda_CAPEX_PV_y'][f"{y}"] +
                                                result_dict_s[s]['lambda_OPEX_PV_y'][f"{y}"]))
                        # ???????????? ???????????? ?????? ??? ?????? ??????
                        y_data[y_tilda, 1] += (result_dict_s[s]['capacity_onshore_y'][f"{y}"] *
                                               (result_dict_s[s]['lambda_CAPEX_onshore_y'][f"{y}"] +
                                                result_dict_s[s]['lambda_OPEX_onshore_y'][f"{y}"]))
                    else:
                        y_data[y_tilda, 0] += ((result_dict_s[s]['capacity_pv_y'][f"{y}"] -
                                                result_dict_s[s]['capacity_pv_y'][f"{y - 1}"]) *
                                               result_dict_s[s]['lambda_CAPEX_PV_y'][f"{y}"] +
                                               result_dict_s[s]['capacity_pv_y'][f"{y}"] *
                                               result_dict_s[s]['lambda_OPEX_PV_y'][f"{y}"])

                        y_data[y_tilda, 1] += ((result_dict_s[s]['capacity_onshore_y'][f"{y}"] -
                                                result_dict_s[s]['capacity_onshore_y'][f"{y - 1}"]) *
                                               result_dict_s[s]['lambda_CAPEX_onshore_y'][f"{y}"] +
                                               result_dict_s[s]['capacity_onshore_y'][f"{y}"] *
                                               result_dict_s[s]['lambda_OPEX_onshore_y'][f"{y}"])

                    # ???????????? ????????????
                    y_data[y_tilda, 2] += result_dict_s[s]['c_tariff_pre_dema_y'][f"{y}"] + \
                                          result_dict_s[s]['c_tariff_pro_dema_y'][f"{y}"]

                    # ???????????? ???????????????
                    if result_dict_s[s]['u_y'][f"{y}"] == 1:
                        y_data[y_tilda, 3] += sum(result_dict_s[s]['p_tariff_y_d_h'][f"{y}"][f"{d}"][f"{h}"] *
                                                  result_dict_s[s]['lambda_tariff_pre_y_d_h'][f"{y}"][f"{d}"][f"{h}"]
                                                  for d in options.set_d
                                                  for h in options.set_h)
                    else:
                        y_data[y_tilda, 3] += sum(result_dict_s[s]['p_tariff_y_d_h'][f"{y}"][f"{d}"][f"{h}"] *
                                                  result_dict_s[s]['lambda_tariff_pro_y_d_h'][f"{y}"][f"{d}"][f"{h}"]
                                                  for d in options.set_d
                                                  for h in options.set_h)

                    # ???????????? ??????????????????
                    y_data[y_tilda, 4] += sum(result_dict_s[s]['p_tariff_y_d_h'][f"{y}"][f"{d}"][f"{h}"]
                                              for d in options.set_d
                                              for h in options.set_h) * \
                                          result_dict_s[s]['lambda_climate_y'][f"{y}"]
                    # ???????????? ??????????????????
                    y_data[y_tilda, 5] += sum(result_dict_s[s]['p_tariff_y_d_h'][f"{y}"][f"{d}"][f"{h}"]
                                              for d in options.set_d
                                              for h in options.set_h) * \
                                          result_dict_s[s]['lambda_fuel_adjustment_y'][f"{y}"]
                    # ???????????? ???????????????
                    y_data[y_tilda, 6] += result_dict_s[s]['c_commission_kepco_y'][f"{y}"]

                    # ???????????????????????? (??????)
                    y_data[y_tilda, 7] += result_dict_s[s]['c_funding_tariff_y'][f"{y}"]

                    if y == options.year0:
                        # ????????? ??????PPA ??????????????? ????????????
                        P_pv = sum(result_dict_s[s]['p_ppa_pv_y_d_h'][f"{y}"][f"{d}"][f"{h}"]
                                                  for d in options.set_d
                                                  for h in options.set_h) * result_dict_s[s]['lambda_PPA_pv_y'][f"{y}"]
                        P_wt = sum(result_dict_s[s]['p_ppa_onshore_y_d_h'][f"{y}"][f"{d}"][f"{h}"]
                                                  for d in options.set_d
                                                  for h in options.set_h) *  result_dict_s[s]['lambda_PPA_onshore_y'][f"{y}"]

                        y_data[y_tilda, 8] += P_pv
                        # ?????? ??????PPA ??????????????? ????????????
                        y_data[y_tilda, 9] += P_wt
                        z_data[y_tilda, 8] = P_pv
                        z_data[y_tilda, 9] = P_wt

                    else:
                        # ????????? ??????PPA ??????????????? ????????????
                        P_pv = (sum(result_dict_s[s]['p_ppa_pv_y_d_h'][f"{y}"][f"{d}"][f"{h}"] -
                                                   result_dict_s[s]['p_ppa_pv_y_d_h'][f"{y - 1}"][f"{d}"][f"{h}"]
                                                   for d in options.set_d
                                                   for h in options.set_h) * result_dict_s[s]['lambda_PPA_pv_y'][f"{y}"]
                                               + z_data[y_tilda - 1, 8])
                        P_wt = (sum(result_dict_s[s]['p_ppa_onshore_y_d_h'][f"{y}"][f"{d}"][f"{h}"] -
                                                   result_dict_s[s]['p_ppa_onshore_y_d_h'][f"{y - 1}"][f"{d}"][f"{h}"]
                                                   for d in options.set_d
                                                   for h in options.set_h) * result_dict_s[s]['lambda_PPA_onshore_y'][f"{y}"]
                                               + z_data[y_tilda - 1, 9])

                        y_data[y_tilda, 8] += P_pv
                        # ?????? ??????PPA ??????????????? ????????????
                        y_data[y_tilda, 9] += P_wt
                        z_data[y_tilda, 8] = P_pv
                        z_data[y_tilda, 9] = P_wt

                    # ??????PPA ???????????????
                    y_data[y_tilda, 10] += sum(result_dict_s[s]['p_ppa_pv_y_d_h'][f"{y}"][f"{d}"][f"{h}"] +
                                               result_dict_s[s]['p_ppa_onshore_y_d_h'][f"{y}"][f"{d}"][f"{h}"]
                                               for d in options.set_d
                                               for h in options.set_h) * \
                                           result_dict_s[s]['lambda_loss_payment_y'][f"{y}"]

                    # ??????PPA ??? ????????????
                    y_data[y_tilda, 11] += sum(result_dict_s[s]['p_ppa_pv_y_d_h'][f"{y}"][f"{d}"][f"{h}"] +
                                               result_dict_s[s]['p_ppa_onshore_y_d_h'][f"{y}"][f"{d}"][f"{h}"]
                                               for d in options.set_d
                                               for h in options.set_h) * \
                                           result_dict_s[s]['lambda_nt_c_y'][f"{y}"]

                    # ??????PPA ???????????????
                    y_data[y_tilda, 12] += sum(result_dict_s[s]['p_ppa_pv_y_d_h'][f"{y}"][f"{d}"][f"{h}"] +
                                               result_dict_s[s]['p_ppa_onshore_y_d_h'][f"{y}"][f"{d}"][f"{h}"]
                                               for d in options.set_d
                                               for h in options.set_h) * \
                                           result_dict_s[s]['lambda_AS_payment_y'][f"{y}"]

                    # ??????PPA ?????????????????????
                    y_data[y_tilda, 13] += sum(result_dict_s[s]['p_ppa_pv_y_d_h'][f"{y}"][f"{d}"][f"{h}"] +
                                               result_dict_s[s]['p_ppa_onshore_y_d_h'][f"{y}"][f"{d}"][f"{h}"]
                                               for d in options.set_d
                                               for h in options.set_h) * \
                                           result_dict_s[s]['lambda_welfare_y'][f"{y}"]
                    # ??????PPA ???????????????
                    y_data[y_tilda, 14] += result_dict_s[s]['c_ppa_y'][f"{y}"] * \
                                           input_parameters_pulp.ratio_commission_ppa_ratio_per_won_y[y]

                    # ???????????????????????? (PPA)
                    y_data[y_tilda, 15] += (sum(result_dict_s[s]['p_ppa_pv_y_d_h'][f"{y}"][f"{d}"][f"{h}"] +
                                               result_dict_s[s]['p_ppa_onshore_y_d_h'][f"{y}"][f"{d}"][f"{h}"]
                                               for d in options.set_d
                                               for h in options.set_h) * result_dict_s[s]['lambda_loss_payment_y'][f"{y}"] +
                                            result_dict_s[s]['c_ppa_y'][f"{y}"]) * \
                                           input_parameters_pulp.ratio_ppa_funding_y[y]

                    # REC ??????
                    y_data[y_tilda, 16] += result_dict_s[s]['c_eac_y'][f"{y}"]

                    # ?????????????????? ??????
                    y_data[y_tilda, 17] += input_parameters_pulp.lambda_ee_y[y]

                    # ????????????
                    y_data[y_tilda, 18] += result_dict_s[s]['c_residual_y'][f"{y}"]

            y_data /= len(result_dict_s.keys())
            y_data /= unit
            col_list = ['????????? ???????????? ?????? ??? ????????????',
                        '???????????? ???????????? ?????? ??? ????????????',
                        '?????? ????????????',
                        '?????? ???????????????',
                        '?????? ??????????????????',
                        '?????? ??????????????????',
                        '?????? ???????????????',
                        '????????????????????????(??????)',
                        '????????? ??????PPA ????????????????????? ??????',
                        '?????? ??????PPA ????????????????????? ??????',
                        '??????PPA ???????????????',
                        '??????PPA ??? ????????????',
                        '??????PPA ???????????????',
                        '??????PPA ?????? ??? ????????????',
                        '??????PPA ???????????????',
                        '????????????????????????(PPA)',
                        'REC ??????',
                        '?????????????????? ??????',
                        '????????????']

            y_data_df = pd.DataFrame(y_data, columns=col_list, index=x_data)
            y_data_df.to_excel(f"{loc_excel}/????????? ?????? RE100 ?????? ????????? ??????.xlsx")
            print(f"????????? ?????? RE100 ?????? ????????? ??????.xlsx ??????")

        if options.excel_optimal_portfolio_spec_flag:
            item_num = 6
            year0 = options.year0
            year1 = options.year1

            x_data = np.arange(year0, year1 + 1, 1)
            y_data = np.zeros((year1 - year0 + 1, item_num))
            for s in result_dict_s.keys():
                for y in options.set_y:
                    y_tilda = y - year0
                    y_data[y_tilda, 0] += result_dict_s[s]['capacity_pv_y'][f"{y}"]  # kW
                    y_data[y_tilda, 1] += result_dict_s[s]['capacity_onshore_y'][f"{y}"]  # kW
                    y_data[y_tilda, 2] += sum(result_dict_s[s]['p_ppa_pv_y_d_h'][f"{y}"][f"{d}"][f"{h}"]
                                              for d in options.set_d
                                              for h in options.set_h)  # kWh
                    y_data[y_tilda, 3] += sum(result_dict_s[s]['p_ppa_onshore_y_d_h'][f"{y}"][f"{d}"][f"{h}"]
                                              for d in options.set_d
                                              for h in options.set_h)  # kWh
                    y_data[y_tilda, 4] += sum(result_dict_s[s]['p_tariff_y_d_h'][f"{y}"][f"{d}"][f"{h}"]
                                              for d in options.set_d
                                              for h in options.set_h)  # kWh
                    y_data[y_tilda, 5] += result_dict_s[s]['p_eac_y'][f"{y}"]  # kWh

            y_data /= len(result_dict_s.keys())
            col_list = ['????????? ???????????? ??????(kW)', '???????????? ???????????? ??????(kW)', '??????PPA(PV) ????????? (kWh)',
                        '??????PPA(??????) ????????? (kWh)', '???????????? ????????? (kWh)', '????????? ????????? (kWh)']

            y_data_df = pd.DataFrame(y_data, columns=col_list, index=x_data)
            y_data_df.to_excel(f"{loc_excel}/????????? ?????? RE100 ????????? ?????? ?????? ??? ?????????.xlsx")
            print(f"????????? ?????? RE100 ????????? ?????? ?????? ??? ?????????.xlsx ??????")

        if options.excel_unit_price_by_year_flag:
            year0 = options.year0
            year1 = options.year1
            x_data = np.arange(year0, year1 + 1, 1)
            y_data = np.zeros((year1 - year0 + 1))
            for s in result_dict_s.keys():
                for y in options.set_y:
                    y_tilda = y - year0
                    demand_amount = sum(input_parameters_pulp.demand_y_d_h[y, d, h]
                                        for d in options.set_d
                                        for h in options.set_h)
                    if demand_amount == 0:
                        y_data[y_tilda] += 0
                    else:

                        y_data[y_tilda] += (result_dict_s[s]['c_sg_y'][f"{y}"] +
                                            result_dict_s[s]['c_tariff_used_y'][f"{y}"] +
                                            result_dict_s[s]['c_tariff_pre_dema_y'][f"{y}"] +
                                            result_dict_s[s]['c_tariff_pro_dema_y'][f"{y}"] +
                                            result_dict_s[s]['c_ppa_y'][f"{y}"] +
                                            result_dict_s[s]['c_eac_y'][f"{y}"] +
                                            result_dict_s[s]['c_loss_payment_y'][f"{y}"] +
                                            result_dict_s[s]['c_funding_tariff_y'][f"{y}"] +
                                            result_dict_s[s]['c_funding_ppa_y'][f"{y}"] +
                                            result_dict_s[s]['c_commission_kepco_y'][f"{y}"] +
                                            result_dict_s[s]['c_commission_ppa_y'][f"{y}"] -
                                            result_dict_s[s]['c_residual_y'][f"{y}"] +
                                            input_parameters_pulp.lambda_ee_y[y]) / demand_amount

            # unit price ?????? ??????/?????????/year = ???/kWh/year
            y_data /= len(result_dict_s.keys())

            y_data_df = pd.DataFrame(y_data.reshape(1, -1), columns=x_data, index=['??????(???/kWh)'])
            y_data_df.to_excel(f"{loc_excel}/?????? RE100 ????????? ?????? ?????? ??????.xlsx")
            print(f"?????? RE100 ????????? ?????? ?????? ??????.xlsx ??????")

        if options.plot_yearly_cost_flag:
            p = lib.PLOT.PlotOpt()
            fig_name = 'RE100 average achievement cost by year'
            year0 = options.year0
            year1 = options.year1
            x_data = np.arange(year0, year1 + 1, 1)
            y_data = np.zeros((year1 - year0 + 1))
            for s in result_dict_s.keys():
                for y in options.set_y:
                    y_tilda = y - year0
                    y_data[y_tilda] += result_dict_s[s]['c_sg_y'][f"{y}"] + \
                                       result_dict_s[s]['c_tariff_used_y'][f"{y}"] + \
                                       result_dict_s[s]['c_tariff_pre_dema_y'][f"{y}"] + \
                                       result_dict_s[s]['c_tariff_pro_dema_y'][f"{y}"] + \
                                       result_dict_s[s]['c_ppa_y'][f"{y}"] + \
                                       result_dict_s[s]['c_eac_y'][f"{y}"] + \
                                       result_dict_s[s]['c_loss_payment_y'][f"{y}"] + \
                                       result_dict_s[s]['c_funding_tariff_y'][f"{y}"] + \
                                       result_dict_s[s]['c_funding_ppa_y'][f"{y}"] + \
                                       result_dict_s[s]['c_commission_kepco_y'][f"{y}"] + \
                                       result_dict_s[s]['c_commission_ppa_y'][f"{y}"] - \
                                       result_dict_s[s]['c_residual_y'][f"{y}"]

            y_data /= len(result_dict_s.keys())
            y_data /= unit
            p.set_frame()
            y_min = math.floor(np.min(y_data))
            y_max = math.ceil(np.max(y_data))

            p.axes.set_xticks(np.arange(options.year0, options.year1 + 1, 1).astype('int'))
            p.axes.set_xticklabels(np.arange(options.year0, options.year1 + 1, 1).astype('str'))
            p.axes.set_yticks(np.arange(y_min, y_max, int((y_max - y_min) / 5)))
            p.axes.set_yticklabels(np.arange(y_min, y_max, int((y_max - y_min) / 5)).astype('str'))
            p.axes.set_xlabel('Year', fontsize=30)
            p.axes.set_ylabel('Average cost(??????)', fontsize=30)
            p.axes.tick_params(axis='x', labelsize=20)
            p.axes.tick_params(axis='y', labelsize=25)
            p.axes.plot(x_data, y_data, label='RE100 average achievement cost by year')
            p.axes.legend(loc='best', ncol=2, fontsize=20, frameon=True, shadow=True)
            p.save_fig(loc_plot, fig_name)
            print(f"{fig_name}.png ??????")

            if options.excel_average_cost_flag:
                y_data_df = pd.DataFrame(y_data.reshape(1, -1), columns=x_data, index=['Average Cost (??????)'])
                y_data_df.to_excel(f"{loc_excel}/?????? ?????? RE100 ????????????.xlsx")
                print(f"?????? ?????? RE100 ????????????.xlsx ??????")

        # ??????PPA ???????????? ??????
        if options.excel_ppa_capacity:
            item_num = 2
            unit = 1000
            year0 = options.year0
            year1 = options.year1
            y_data = np.zeros((year1 - year0 + 1, item_num))
            x_data = np.arange(year0, year1 + 1, 1)
            for s in result_dict_s.keys():
                for y in options.set_y:
                    y_tilda = y - year0
                    y_data[y_tilda, 0] += sum(result_dict_s[s]['p_ppa_pv_y_d_h'][f"{y}"][f"{d}"][f"{h}"]
                                              for d in options.set_d
                                              for h in options.set_h) / IFN.Apv_raw.sum()
                    y_data[y_tilda, 1] += sum(result_dict_s[s]['p_ppa_onshore_y_d_h'][f"{y}"][f"{d}"][f"{h}"]
                                              for d in options.set_d
                                              for h in options.set_h) / IFN.Aonshore_raw.sum()

            y_data /= len(result_dict_s.keys())
            y_data /= unit  # ?????? MW
            y_data_df = pd.DataFrame(y_data, columns=['PV ??????????????? ??????', '?????? ??????????????? ??????'], index=x_data)

            y_data_df.to_excel(f"{loc_excel}/??????PPA ???????????? ??????.xlsx")
            print(f"??????PPA ???????????? ??????.xlsx ??????")

        if options.excel_options_unit_price_by_year_flag:
            x_data = options.set_y
            p1 = lib.PLOT.PlotOpt()
            p1.set_frame()

            p2 = lib.PLOT.PlotOpt()
            p2.set_frame()

            p3 = lib.PLOT.PlotOpt()
            p3.set_frame()

            p4 = lib.PLOT.PlotOpt()
            p4.set_frame()

            p5 = lib.PLOT.PlotOpt()
            p5.set_frame()

            count = 1
            for s in result_dict_s.keys():
                if count == 1:
                    p1.axes.plot(x_data, result_dict_s[s]['lambda_PPA_pv_y'].values(), color='goldenrod',
                                 label='?????? PPA (PV) ??????')
                    p2.axes.plot(x_data, result_dict_s[s]['lambda_PPA_onshore_y'].values(), color='deepskyblue',
                                 label='?????? PPA (Onshore Wind) ??????')
                    p3.axes.plot(x_data, result_dict_s[s]['lambda_eac_y'].values(), color='b', label='????????? ??????')

                else:
                    p1.axes.plot(x_data, result_dict_s[s]['lambda_PPA_pv_y'].values(), color='goldenrod')
                    p2.axes.plot(x_data, result_dict_s[s]['lambda_PPA_onshore_y'].values(), color='deepskyblue')
                    p3.axes.plot(x_data, result_dict_s[s]['lambda_eac_y'].values(), color='b')

                count += 1
            p1.axes.legend(loc='best', ncol=2, fontsize=20, frameon=True, shadow=True)
            p1.save_fig(loc_plot, 'PPA_PV_UNIT_PRICE')

            p2.axes.legend(loc='best', ncol=2, fontsize=20, frameon=True, shadow=True)
            p2.save_fig(loc_plot, 'PPA_ONSHORE_UNIT_PRICE')

            p3.axes.legend(loc='best', ncol=2, fontsize=20, frameon=True, shadow=True)
            p3.save_fig(loc_plot, 'PPA_CETIFICATE_UNIT_PRICE')

            print(f"????????? ????????? ????????????.png ??????")

    if options.etc_plot_demand_pattern_flag:
        row = 59  # 3???1??? ???
        p = lib.PLOT.PlotOpt()
        for i in list(IFN.demand_factor.keys()):
            title = i
            y_data = (IFN.demand_factor[i] / np.max(IFN.demand_factor[i])).reshape(-1)
            x_data = list(range(0, 8760))
            p.set_frame()
            p.axes.set_xlabel('Hour', fontsize=30)
            p.axes.set_ylabel('factor', fontsize=30)
            p.axes.tick_params(axis='x', labelsize=15)
            p.axes.tick_params(axis='y', labelsize=25)
            p.axes.plot(x_data, y_data)
            p.save_fig(options.loc_plot, title)

    if options.result_BAU_analysis_flag:
        input_parameters_pulp = para.ParameterPulpFrom(options, IFN)
        tariff_average = input_parameters_pulp.tariff_average

        options.set_result_dir('BAU', options.loc_plot, options.date_result)
        options.set_result_dir('BAU', options.loc_excel, options.date_result)
        loc_plot = f"{options.loc_plot}/{options.date_result}/BAU"
        loc_excel = f"{options.loc_excel}/{options.date_result}/BAU"

        result_dict_s = dict()
        files = [f for f in os.listdir(f"{options.loc_pp_result}") if '.json' in f]
        for f in files:
            s = f.split('.')[0]
            result_dict_s[s] = dict()
            print(f'file reading scenario_number : {s}')

            with open(f'{options.loc_pp_result}/{s}.json', 'r') as f:
                result = json.load(f)

            # dict ??????
            result_dict_s[s]['lambda_tariff_pre_y_d_h'] = result['lambda_tariff_pre_y_d_h']
            result_dict_s[s]['lambda_climate_y'] = result['lambda_climate_y']
            result_dict_s[s]['lambda_fuel_adjustment_y'] = result['lambda_fuel_adjustment_y']
            result_dict_s[s]['capacity_cs_contract_y'] = result['capacity_cs_contract_y']

        if options.excel_yearly_cost_in_BAUscen_flag:
            year0 = options.year0
            year1 = options.year1
            x_data = np.arange(year0, year1 + 1, 1)
            y_data = np.zeros((year1 - year0 + 1))
            for s in result_dict_s.keys():
                for y in options.set_y:
                    y_tilda = y - year0
                    c_tariff_used = sum([input_parameters_pulp.demand_y_d_h[y, d, h] *
                                         (result_dict_s[s]['lambda_tariff_pre_y_d_h'][f"{y}"][f"{d}"][f"{h}"] +
                                          result_dict_s[s]['lambda_climate_y'][f"{y}"] +
                                          result_dict_s[s]['lambda_fuel_adjustment_y'][f"{y}"])
                                         for d in options.set_d
                                         for h in options.set_h])
                    c_tariff_demo = result_dict_s[s]['capacity_cs_contract_y'][f"{y}"] * input_parameters_pulp.lambda_tariff_dema_pre_y[y] * 12
                    c_funding_tariff = (c_tariff_used + c_tariff_demo) * input_parameters_pulp.ratio_tariff_funding_y[y]
                    c_commission_kepco = (c_tariff_used + c_tariff_demo) * \
                                         input_parameters_pulp.ratio_commission_tariff_ratio_per_won_y[y]

                    y_data[y_tilda] += c_tariff_used + c_tariff_demo + c_funding_tariff + c_commission_kepco


            y_data /= len(result_dict_s.keys())
            y_data /= options.unit

            y_data_df = pd.DataFrame(y_data.reshape(1, -1), columns=x_data, index=['Average Cost (??????)'])
            exl_name = f"BAU????????????_????????????.xlsx"
            y_data_df.to_excel(f"{loc_excel}/{exl_name}")
            print(f"{loc_excel}/{exl_name} ??????")

            if options.plot_yearly_cost_in_BAUscen_flag:
                y_min = math.floor(np.min(y_data))
                y_max = math.ceil(np.max(y_data))
                p = lib.PLOT.PlotOpt()
                fig_name = f'????????????_BAU????????????'
                p.set_frame()
                p.axes.set_xticks(np.arange(options.year0, options.year1 + 1, 1).astype('int'))
                p.axes.set_xticklabels(np.arange(options.year0, options.year1 + 1, 1).astype('str'))
                p.axes.set_yticks(np.arange(y_min, y_max, int((y_max - y_min) / 5)))
                p.axes.set_yticklabels(np.arange(y_min, y_max, int((y_max - y_min) / 5)).astype('str'))
                p.axes.set_xlabel('Year', fontsize=30)
                p.axes.set_ylabel('Average cost(??????)', fontsize=30)
                p.axes.tick_params(axis='x', labelsize=20)
                p.axes.tick_params(axis='y', labelsize=25)
                p.axes.plot(x_data, y_data, label='BAU scenario yearly cost by year')
                p.axes.legend(loc='best', ncol=2, fontsize=20, frameon=True, shadow=True)
                p.save_fig(loc_plot, fig_name)
                print(f"{fig_name}.png ??????")

            if options.excel_unit_price_by_year_in_BAUscen_flag:
                y_data_2 = np.zeros((options.year1 - options.year0 + 1))
                for y in options.set_y:
                    y_tilda = y - options.year0
                    demand_amount = sum(input_parameters_pulp.demand_y_d_h[y, d, h]
                                        for d in options.set_d
                                        for h in options.set_h)
                    if demand_amount == 0:
                        y_data_2[y_tilda] = 0
                    else:
                        y_data_2[y_tilda] = y_data[y_tilda] * options.unit / demand_amount

                y_data_df = pd.DataFrame(y_data_2.reshape(1, -1), columns=x_data, index=['??????(???/kWh)'])
                exl_name = f"??????????????????_BAU????????????.xlsx"
                y_data_df.to_excel(f"{loc_excel}/{exl_name}")
                print(f"{exl_name} ??????")

    if options.result_integration_analysis_flag:
        loc_plot = f"{options.loc_plot}/{options.date_result}/BAU"
        loc_excel = f"{options.loc_excel}/{options.date_result}/BAU"

        ppa_opt_list = ['??????????????????', '??????????????????']
        for po in ppa_opt_list:
            exl_ppa_name = f"?????? RE100 ????????? ?????? ?????? ??????.xlsx"
            p_unit_avg_y = pd.read_excel(f"{options.loc_excel}/{options.date_result}/{po}/{exl_ppa_name}", sheet_name='Sheet1', index_col=0)
            for tc in tariff_choice_list:
                exl_bau_name = f"??????????????????_BAU????????????.xlsx"
                p_unit_avg_BAU_y = pd.read_excel(f"{loc_excel}/{exl_bau_name}", sheet_name='Sheet1', index_col=0)

                y_min = math.floor(max(p_unit_avg_y.min().min(), p_unit_avg_BAU_y.min().min()))
                y_max = math.ceil(max(p_unit_avg_y.max().max(), p_unit_avg_BAU_y.max().max()))

                year0 = options.year0
                year1 = options.year1
                x_data = np.arange(year0, year1 + 1, 1)
                p = lib.PLOT.PlotOpt()
                p.set_frame()
                p.axes.set_xticks(np.arange(options.year0, options.year1 + 1, 1).astype('int'))
                p.axes.set_xticklabels(np.arange(options.year0, options.year1 + 1, 1).astype('str'))
                p.axes.set_yticks(np.arange(y_min, y_max, int((y_max - y_min) / 5)))
                p.axes.set_yticklabels(np.arange(y_min, y_max, int((y_max - y_min) / 5)).astype('str'))
                p.axes.set_xlabel('??????', fontsize=30)
                p.axes.set_ylabel('????????????(???/kWh)', fontsize=30)
                p.axes.tick_params(axis='x', labelsize=20)
                p.axes.tick_params(axis='y', labelsize=25)

                p.axes.plot(x_data, p_unit_avg_y.values.reshape(-1), label='RE100 scenario', c='k')
                p.axes.plot(x_data, p_unit_avg_BAU_y.values.reshape(-1), label='BAU scenario', c='r', linestyle='--')
                p.axes.legend(loc='best', ncol=2, fontsize=20, frameon=True, shadow=True)
                fig_name = f'??????????????????????????????_BAU({options.tariff_bau},{tc}),RE????????????({po})'
                p.save_fig(loc_plot, fig_name)
                print(f"{fig_name}.png ??????")


if __name__ == '__main__':
    start = time.time()
    options = para.ProbOptions()

    if options.parallel_processing_flag:
        loc_pp = f"{options.loc_result}/{options.date_result}"
        file_pp = 'result_profile.pkl'
        with open(f"{loc_pp}/{file_pp}", 'rb') as f:
            result_profile = pickle.load(f)
        if len([k for k, v in result_profile.items() if v == 0]) == 0:
            quit()
        else:
            scenario_num = [k for k, v in result_profile.items() if v == 0][-1]
            result_profile[scenario_num] = 1
            with open(f"{loc_pp}/{file_pp}", 'wb') as f:
                pickle.dump(result_profile, f, pickle.HIGHEST_PROTOCOL)
            print(f"num : {scenario_num}")

            options.scenario_num = f'{scenario_num}'

    # solver_name = 'gurobi'
    # solver_name = 'CBC'
    solver_name = 'cplex'
    IFN = para.ReadInputData(options)

    result_S = run_simulation(options, IFN)
    result_A = run_analysis(options, IFN)

    print(f"run time : {time.time() - start}")
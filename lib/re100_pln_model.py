import pulp as lp
import numpy as np
import copy

from lib.parameters import ParameterPulpFrom, ReadInputData, ProbOptions


# Next group case
def solve_re100_milp(options: ProbOptions, input_parameters_pulp: ParameterPulpFrom, solver_name, result_dict):
    if options.customize_exemption_flag:
        zero_value = input_parameters_pulp.fill_value_to_dict(options, 0)
        lambda_nt_c_y = zero_value
        lambda_AS_payment_y = zero_value
        lambda_loss_payment_y = zero_value
        ratio_ppa_funding_y = zero_value
        ratio_commission_ppa_ratio_per_won_y = zero_value
        
    else:
        lambda_nt_c_y = input_parameters_pulp.lambda_nt_c_y
        lambda_AS_payment_y = input_parameters_pulp.lambda_AS_payment_y
        lambda_loss_payment_y = input_parameters_pulp.lambda_loss_payment_y
        ratio_ppa_funding_y = input_parameters_pulp.ratio_ppa_funding_y
        ratio_commission_ppa_ratio_per_won_y = input_parameters_pulp.ratio_commission_ppa_ratio_per_won_y

    """
    인풋데이터 입력
    """
    set_y = np.arange(options.model_start_y, options.model_end_y + 1, 1).astype('int').tolist()
    set_d = np.arange(options.model_start_d, options.model_end_d + 1, 1).astype('int').tolist()
    set_h = np.arange(options.model_start_h, options.model_end_h + 1, 1).astype('int').tolist()

    set_y_d_h = [(y, d, h) for y in set_y for d in set_d for h in set_h]
    set_d_h = [(d, h) for d in set_d for h in set_h]

    # 재생e 이용률
    Apv_d_h = input_parameters_pulp.Apv_d_h
    Aonshore_d_h = input_parameters_pulp.Aonshore_d_h

    # 부하 pattern
    demand_y_d_h = input_parameters_pulp.demand_y_d_h
    # 에너지효율화 pattern
    ee_y_d_h = input_parameters_pulp.ee_y_d_h
    lambda_ee_y = input_parameters_pulp.lambda_ee_y

    # 자가발전 관련 가격
    lambda_CAPEX_PV_y = input_parameters_pulp.lambda_CAPEX_PV_y
    lambda_OPEX_PV_y = input_parameters_pulp.lambda_OPEX_PV_y
    lambda_CAPEX_onshore_y = input_parameters_pulp.lambda_CAPEX_onshore_y
    lambda_OPEX_onshore_y = input_parameters_pulp.lambda_OPEX_onshore_y

    # 보완공급
    lambda_tariff_pre_y_d_h = input_parameters_pulp.lambda_tariff_pre_y_d_h
    lambda_tariff_pro_y_d_h = input_parameters_pulp.lambda_tariff_pro_y_d_h
    lambda_tariff_dema_pre_y = input_parameters_pulp.lambda_tariff_dema_pre_y
    lambda_tariff_dema_pro_y = input_parameters_pulp.lambda_tariff_dema_pro_y
    lambda_climate_y = input_parameters_pulp.lambda_climate_y
    lambda_fuel_adjustment_y = input_parameters_pulp.lambda_fuel_adjustment_y
    ratio_tariff_funding_y = input_parameters_pulp.ratio_tariff_funding_y
    ratio_commission_tariff_ratio_per_won_y = input_parameters_pulp.ratio_commission_tariff_ratio_per_won_y

    # 설치 가능한 최대, 최소 용량
    cap_max_sg_pv_y = input_parameters_pulp.cap_max_sg_pv_y
    cap_max_sg_onshore_y = input_parameters_pulp.cap_max_sg_onshore_y
    cap_min_sg_pv_y = input_parameters_pulp.cap_min_sg_pv_y
    cap_min_sg_onshore_y = input_parameters_pulp.cap_min_sg_onshore_y

    # 계약 가능한 최대용량
    cap_max_ppa_pv_y = input_parameters_pulp.cap_max_ppa_pv_y
    cap_max_ppa_onshore_y = input_parameters_pulp.cap_max_ppa_onshore_y

    # 기업 PPA
    lambda_PPA_pv_y = input_parameters_pulp.lambda_PPA_pv_y
    lambda_PPA_onshore_y = input_parameters_pulp.lambda_PPA_onshore_y
    lambda_welfare_y = input_parameters_pulp.lambda_welfare_y

    # 인증서 관련
    lambda_eac_y = input_parameters_pulp.lambda_eac_y

    # simulation 관련 요소
    achievement_ratio_y = options.achievement_ratio_y
    discount_rate = input_parameters_pulp.discount_rate
    pv_life_span = input_parameters_pulp.pv_life_span
    onshore_life_span = input_parameters_pulp.onshore_life_span
    gaprel = options.gaprel
    year0 = options.year0
    year1 = options.year1
    Big_M = options.Big_M
    """
    결정변수 선언
    """
    p_tariff_y_d_h = lp.LpVariable.dicts("p_tariff_y_d_h", set_y_d_h, lowBound=0, cat='Continuous')
    p_ppa_pv_y_d_h = lp.LpVariable.dicts("p_ppa_pv_y_d_h", set_y_d_h, lowBound=0, cat='Continuous')
    p_ppa_onshore_y_d_h = lp.LpVariable.dicts("p_ppa_onshore_y_d_h", set_y_d_h, lowBound=0, cat='Continuous')
    capacity_cs_contract_y = lp.LpVariable.dicts("capacity_cs_contract_y", set_y, lowBound=0, cat='Continuous')
    p_eac_y = lp.LpVariable.dicts("p_eac_y", set_y, lowBound=0, cat='Continuous')
    capacity_pv_y = lp.LpVariable.dicts("capacity_pv_y", set_y, lowBound=0, cat='Continuous')
    capacity_onshore_y = lp.LpVariable.dicts("capacity_onshore_y", set_y, lowBound=0, cat='Continuous')

    c_sg_y = lp.LpVariable.dicts("c_sg_y", set_y, cat='Continuous')
    c_tariff_used_y = lp.LpVariable.dicts("c_tariff_used_y", set_y, cat='Continuous')
    c_tariff_pre_used_y_d_h = lp.LpVariable.dicts("c_tariff_pre_used_y_d_h", set_y_d_h, lowBound=0, cat='Continuous')
    c_tariff_pro_used_y_d_h = lp.LpVariable.dicts("c_tariff_pro_used_y_d_h", set_y_d_h, lowBound=0, cat='Continuous')

    c_tariff_dema_y = lp.LpVariable.dicts("c_tariff_dema_y", set_y, cat='Continuous')
    c_tariff_pre_dema_y = lp.LpVariable.dicts("c_tariff_pre_dema_y", set_y, lowBound=0, cat='Continuous')
    c_tariff_pro_dema_y = lp.LpVariable.dicts("c_tariff_pro_dema_y", set_y, lowBound=0, cat='Continuous')

    c_ppa_y = lp.LpVariable.dicts("c_ppa_y", set_y, cat='Continuous')
    c_eac_y = lp.LpVariable.dicts("c_eac_y", set_y, cat='Continuous')
    c_residual_y = lp.LpVariable.dicts("c_residual_y", set_y, cat='Continuous')
    c_loss_payment_y = lp.LpVariable.dicts("c_loss_payment_y", set_y, cat='Continuous')
    c_funding_tariff_y = lp.LpVariable.dicts("c_funding_tariff_y", set_y, cat='Continuous')
    c_funding_ppa_y = lp.LpVariable.dicts("c_funding_ppa_y", set_y, cat='Continuous')
    c_commission_kepco_y = lp.LpVariable.dicts("c_commission_kepco_y", set_y, cat='Continuous')
    c_commission_ppa_y = lp.LpVariable.dicts("c_commission_ppa_y", set_y, cat='Continuous')

    u_y = lp.LpVariable.dicts("u_y", set_y, cat='Binary')   # 산업용전기 선택유무

    model = lp.LpProblem("RE100_Problem", lp.LpMinimize)

    """
    미리 결정된 결정변수의 파라미터 입력
    """
    capacity_pv_init_y = dict()
    capacity_onshore_init_y = dict()
    p_ppa_pv_init_y_d_h = dict()
    p_ppa_onshore_init_y_d_h = dict()
    c_ppa_init_y = dict()
    u_init_y = dict()
    if result_dict == None:
        pass
    else:
        y_tilda = set_y[0] - 1
        capacity_pv_init_y[y_tilda] = result_dict['capacity_pv_y'][y_tilda]
        capacity_onshore_init_y[y_tilda] = result_dict['capacity_onshore_y'][y_tilda]
        c_ppa_init_y[y_tilda] = result_dict['c_ppa_y'][y_tilda]
        u_init_y[y_tilda] = result_dict['u_y'][y_tilda]
        for d in range(options.model_start_d, options.model_end_d + 1, 1):
            for h in range(options.model_start_h, options.model_end_h + 1, 1):
                p_ppa_pv_init_y_d_h[y_tilda, d, h] = result_dict['p_ppa_pv_y_d_h'][y_tilda][d][h]
                p_ppa_onshore_init_y_d_h[y_tilda, d, h] = result_dict['p_ppa_onshore_y_d_h'][y_tilda][d][h]

    # objective function
    model += lp.lpSum((c_sg_y[y] + c_tariff_used_y[y] + c_tariff_dema_y[y] + c_ppa_y[y] +
                       c_eac_y[y] + c_loss_payment_y[y] + c_funding_tariff_y[y] + c_funding_ppa_y[y] +
                       c_commission_kepco_y[y] + c_commission_ppa_y[y] - c_residual_y[y] + lambda_ee_y[y]) /
                      (1 + discount_rate) ** (y - year0) for y in set_y)

    # constraints
    for y in set_y:
        y_tilda = year1 - y + 1
        if y == year0:
            # 자가발전 비용
            model += c_sg_y[y] == capacity_pv_y[y] * (lambda_CAPEX_PV_y[y] + lambda_OPEX_PV_y[y]) + \
                     capacity_onshore_y[y] * (lambda_CAPEX_onshore_y[y] + lambda_OPEX_onshore_y[y])

            # 기업PPA 비용
            model += c_ppa_y[y] == lp.lpSum(p_ppa_pv_y_d_h[y, d, h] for d, h in set_d_h) * \
                     (lambda_PPA_pv_y[y] + lambda_nt_c_y[y] + lambda_AS_payment_y[y] + lambda_welfare_y[y]) + \
                     lp.lpSum(p_ppa_onshore_y_d_h[y, d, h] for d, h in set_d_h) * \
                     (lambda_PPA_onshore_y[y] + lambda_nt_c_y[y] + lambda_AS_payment_y[y] + lambda_welfare_y[y])

            # 잔존가치 비용
            model += c_residual_y[y] == lambda_CAPEX_PV_y[y] * capacity_pv_y[y] * (pv_life_span - y_tilda) / pv_life_span + \
                     lambda_CAPEX_onshore_y[y] * capacity_onshore_y[y] * (onshore_life_span - y_tilda) / onshore_life_span

        else:
            # 자가발전 비용
            if y == set_y[0]:
                model += c_sg_y[y] == lambda_CAPEX_PV_y[y] * (capacity_pv_y[y] - capacity_pv_init_y[y - 1]) + \
                         lambda_OPEX_PV_y[y] * capacity_pv_y[y] + \
                         lambda_CAPEX_onshore_y[y] * (capacity_onshore_y[y] - capacity_onshore_init_y[y - 1]) + \
                         lambda_OPEX_onshore_y[y] * capacity_onshore_y[y]

                model += c_ppa_y[y] == lp.lpSum(p_ppa_pv_y_d_h[y, d, h] - p_ppa_pv_init_y_d_h[y - 1, d, h] for d, h in set_d_h) * \
                         (lambda_PPA_pv_y[y] + lambda_nt_c_y[y] + lambda_AS_payment_y[y] + lambda_welfare_y[y]) + \
                         lp.lpSum(p_ppa_onshore_y_d_h[y, d, h] - p_ppa_onshore_init_y_d_h[y - 1, d, h] for d, h in set_d_h) * \
                         (lambda_PPA_onshore_y[y] + lambda_nt_c_y[y] + lambda_AS_payment_y[y] + lambda_welfare_y[y]) + \
                         c_ppa_init_y[y - 1]

                model += c_residual_y[y] == lambda_CAPEX_PV_y[y] * (capacity_pv_y[y] - capacity_pv_init_y[y - 1]) * (pv_life_span - y_tilda) / pv_life_span + \
                         lambda_CAPEX_onshore_y[y] * (capacity_onshore_y[y] - capacity_onshore_init_y[y - 1]) * (onshore_life_span - y_tilda) / onshore_life_span

            else:
                model += c_sg_y[y] == lambda_CAPEX_PV_y[y] * (capacity_pv_y[y] - capacity_pv_y[y - 1]) + \
                         lambda_OPEX_PV_y[y] * capacity_pv_y[y] + \
                         lambda_CAPEX_onshore_y[y] * (capacity_onshore_y[y] - capacity_onshore_y[y - 1]) + \
                         lambda_OPEX_onshore_y[y] * capacity_onshore_y[y]

                # 기업PPA 비용
                model += c_ppa_y[y] == lp.lpSum(p_ppa_pv_y_d_h[y, d, h] - p_ppa_pv_y_d_h[y - 1, d, h] for d, h in set_d_h) * \
                         (lambda_PPA_pv_y[y] + lambda_nt_c_y[y] + lambda_AS_payment_y[y] + lambda_welfare_y[y]) + \
                         lp.lpSum(p_ppa_onshore_y_d_h[y, d, h] - p_ppa_onshore_y_d_h[y - 1, d, h] for d, h in set_d_h) * \
                         (lambda_PPA_onshore_y[y] + lambda_nt_c_y[y] + lambda_AS_payment_y[y] + lambda_welfare_y[y]) + \
                         c_ppa_y[y - 1]

                # 잔존가치 비용
                model += c_residual_y[y] == lambda_CAPEX_PV_y[y] * (capacity_pv_y[y] - capacity_pv_y[y - 1]) * (pv_life_span - y_tilda) / pv_life_span + \
                         lambda_CAPEX_onshore_y[y] * (capacity_onshore_y[y] - capacity_onshore_y[y - 1]) * (onshore_life_span - y_tilda) / onshore_life_span

        # 이진변수 논리제약
        if y > year0:
            if y == set_y[0]:
                model += u_y[y] <= u_init_y[y - 1]
            else:
                model += u_y[y] <= u_y[y - 1]

        if y in [2022, 2023]:
            model += u_y[y] == 1

        # 보완공급
        model += c_tariff_used_y[y] == \
                 lp.lpSum(c_tariff_pre_used_y_d_h[y, d, h] for d, h in set_d_h) + \
                 lp.lpSum(c_tariff_pro_used_y_d_h[y, d, h] for d, h in set_d_h)
        model += c_tariff_dema_y[y] == c_tariff_pre_dema_y[y] + c_tariff_pro_dema_y[y]

        # 보완공급 (dema 비용)
        model += c_tariff_pre_dema_y[y] * (1 / lambda_tariff_dema_pre_y[y]) - capacity_cs_contract_y[y] * 12 <= (1 - u_y[y]) * Big_M
        model += c_tariff_pre_dema_y[y] * (1 / lambda_tariff_dema_pre_y[y]) - capacity_cs_contract_y[y] * 12 >= - (1 - u_y[y]) * Big_M

        model += c_tariff_pro_dema_y[y] * (1 / lambda_tariff_dema_pro_y[y]) - capacity_cs_contract_y[y] * 12 <= u_y[y] * Big_M
        model += c_tariff_pro_dema_y[y] * (1 / lambda_tariff_dema_pro_y[y]) - capacity_cs_contract_y[y] * 12 >= - u_y[y] * Big_M


        # 인증서 비용
        model += c_eac_y[y] == p_eac_y[y] * lambda_eac_y[y]

        # 손실정산금
        model += c_loss_payment_y[y] == lambda_loss_payment_y[y] * lp.lpSum(p_ppa_pv_y_d_h[y, d, h] + p_ppa_onshore_y_d_h[y, d, h] for d, h in set_d_h)

        # 전력산업기반기금 (tariff)
        model += c_funding_tariff_y[y] == c_tariff_used_y[y] * ratio_tariff_funding_y[y]
        
        # 전력산업기반기금 (PPA)
        model += c_funding_ppa_y[y] == (c_ppa_y[y] + c_loss_payment_y[y]) * ratio_ppa_funding_y[y]

        # 부가가치세
        model += c_commission_kepco_y[y] == c_tariff_used_y[y] * ratio_commission_tariff_ratio_per_won_y[y]
        model += c_commission_ppa_y[y] == c_ppa_y[y] * ratio_commission_ppa_ratio_per_won_y[y]

    for y in set_y:
        if y > year0:
            if y == set_y[0]:
                # 자가발전 용량 제약
                model += capacity_pv_y[y] - capacity_pv_init_y[y - 1] >= 0
                # 자가발전 용량 제약
                model += capacity_onshore_y[y] - capacity_onshore_init_y[y - 1] >= 0

                # PPA 장기계약 제약
                model += lp.lpSum(p_ppa_pv_init_y_d_h[y - 1, d, h] for d, h in set_d_h) <= \
                         lp.lpSum(p_ppa_pv_y_d_h[y, d, h] for d, h in set_d_h)
                model += lp.lpSum(p_ppa_onshore_init_y_d_h[y - 1, d, h] for d, h in set_d_h) <= \
                         lp.lpSum(p_ppa_onshore_y_d_h[y, d, h] for d, h in set_d_h)
            else:
                # 자가발전 용량 제약
                model += capacity_pv_y[y] - capacity_pv_y[y - 1] >= 0
                # 자가발전 용량 제약
                model += capacity_onshore_y[y] - capacity_onshore_y[y - 1] >= 0

                # PPA 장기계약 제약
                model += lp.lpSum(p_ppa_pv_y_d_h[y - 1, d, h] for d, h in set_d_h) <= \
                         lp.lpSum(p_ppa_pv_y_d_h[y, d, h] for d, h in set_d_h)
                model += lp.lpSum(p_ppa_onshore_y_d_h[y - 1, d, h] for d, h in set_d_h) <= \
                         lp.lpSum(p_ppa_onshore_y_d_h[y, d, h] for d, h in set_d_h)

    for y in set_y:
        # 자가발전 설치용량 제약
        model += capacity_pv_y[y] <= cap_max_sg_pv_y[y]
        model += capacity_pv_y[y] >= cap_min_sg_pv_y[y]
        model += capacity_onshore_y[y] <= cap_max_sg_onshore_y[y]
        model += capacity_onshore_y[y] >= cap_min_sg_onshore_y[y]

        # 보완공급 계약용량 제약
        model += capacity_cs_contract_y[y] == max([demand_y_d_h[y, d, h] for d, h in set_d_h])      # 임시 조치 - 다시 생각 필요
        # model += capacity_cs_contract_y[y] >= max([v for k, v in demand_y_d_h.items() if k[0] == y]) - (p_ppa_pv_y_d_h[y, d, h] + p_ppa_onshore_y_d_h[y, d, h])

    for y, d, h in set_y_d_h:
        # 한전으로부터 수급
        model += p_tariff_y_d_h[y, d, h] <= \
                 demand_y_d_h[y, d, h] - \
                 ee_y_d_h[y, d, h]

        # 보완공급 (used 비용) Big-M method 활용
        model += c_tariff_pre_used_y_d_h[y, d, h] * (1 / (lambda_tariff_pre_y_d_h[y, d, h] +
                                                          lambda_climate_y[y] +
                                                          lambda_fuel_adjustment_y[y])) - p_tariff_y_d_h[y, d, h] <= (1 - u_y[y]) * Big_M
        model += c_tariff_pre_used_y_d_h[y, d, h] * (1 / (lambda_tariff_pre_y_d_h[y, d, h] +
                                                          lambda_climate_y[y] +
                                                          lambda_fuel_adjustment_y[y])) - p_tariff_y_d_h[y, d, h] >= - (1 - u_y[y]) * Big_M

        model += c_tariff_pro_used_y_d_h[y, d, h] * (1 / (lambda_tariff_pro_y_d_h[y, d, h] +
                                                          lambda_climate_y[y] +
                                                          lambda_fuel_adjustment_y[y])) - p_tariff_y_d_h[y, d, h] <= u_y[y] * Big_M
        model += c_tariff_pre_used_y_d_h[y, d, h] * (1 / (lambda_tariff_pro_y_d_h[y, d, h] +
                                                          lambda_climate_y[y] +
                                                          lambda_fuel_adjustment_y[y])) - p_tariff_y_d_h[y, d, h] >= - u_y[y] * Big_M

        if demand_y_d_h[y, d, h] - ee_y_d_h[y, d, h] > Apv_d_h[d, h] * cap_max_ppa_pv_y[y]:
            model += p_ppa_pv_y_d_h[y, d, h] <= Apv_d_h[d, h] * cap_max_ppa_pv_y[y] * (1 - u_y[y])
        else:
            model += p_ppa_pv_y_d_h[y, d, h] <= (demand_y_d_h[y, d, h] - ee_y_d_h[y, d, h]) * (1 - u_y[y])

        if demand_y_d_h[y, d, h] - ee_y_d_h[y, d, h] > Aonshore_d_h[d, h] * cap_max_ppa_onshore_y[y]:
            model += p_ppa_onshore_y_d_h[y, d, h] <= Aonshore_d_h[d, h] * cap_max_ppa_onshore_y[y] * (1 - u_y[y])
        else:
            model += p_ppa_onshore_y_d_h[y, d, h] <= (demand_y_d_h[y, d, h] - ee_y_d_h[y, d, h]) * (1 - u_y[y])

    # RE100 제약
    if options.customize_achievement_flag:
        for y in set_y:
            model += lp.lpSum(Apv_d_h[d, h] * capacity_pv_y[y] + Aonshore_d_h[d, h] * capacity_onshore_y[y] +
                              p_ppa_pv_y_d_h[y, d, h] + p_ppa_onshore_y_d_h[y, d, h] for d, h in set_d_h) + p_eac_y[y] == \
                     lp.lpSum(demand_y_d_h[y, d, h] - ee_y_d_h[y, d, h] for d, h in set_d_h) * (achievement_ratio_y[y] / 100)

    else:
        for y in set_y:
            model += lp.lpSum(Apv_d_h[d, h] * capacity_pv_y[y] + Aonshore_d_h[d, h] * capacity_onshore_y[y] +
                              p_ppa_pv_y_d_h[y, d, h] + p_ppa_onshore_y_d_h[y, d, h] for d, h in set_d_h) + p_eac_y[y] >= \
                     lp.lpSum(demand_y_d_h[y, d, h] - ee_y_d_h[y, d, h] for d, h in set_d_h) * (achievement_ratio_y[y] / 100)

    # 수요공급
    for y, d, h in set_y_d_h:
        model += Apv_d_h[d, h] * capacity_pv_y[y] + Aonshore_d_h[d, h] * capacity_onshore_y[y] + \
                 p_tariff_y_d_h[y, d, h] + p_ppa_pv_y_d_h[y, d, h] + p_ppa_onshore_y_d_h[y, d, h] >= \
                 demand_y_d_h[y, d, h] - ee_y_d_h[y, d, h]

    print('RE100 simulator, start solve -> solve status:', end=' ')
    if solver_name == 'CBC':
        solver = lp.PULP_CBC_CMD(gapRel=gaprel, msg=options.solver_message)
    elif solver_name == 'gurobi':
        solver = lp.GUROBI_CMD(gapRel=gaprel, msg=options.solver_message)
    elif solver_name == 'cplex':
        solver = lp.CPLEX_CMD(gapRel=gaprel, msg=options.solver_message)
    else:
        print('solver selection error')

    model.solve(solver)
    print(lp.LpStatus[model.status])

    # 결과저장
    result = dict()

    u_y_dict = dict()

    p_sg_pv_y_d_h_dict = dict()
    p_sg_onshore_y_d_h_dict = dict()

    p_tariff_y_d_h_dict = dict()
    p_ppa_pv_y_d_h_dict = dict()
    p_ppa_onshore_y_d_h_dict = dict()
    capacity_cs_contract_y_dict = dict()

    p_eac_y_dict = dict()
    capacity_pv_y_dict = dict()
    capacity_onshore_y_dict = dict()
    c_sg_y_dict = dict()
    c_tariff_used_y_dict = dict()
    c_tariff_pre_dema_y_dict = dict()
    c_tariff_pro_dema_y_dict = dict()

    c_ppa_y_dict = dict()
    c_eac_y_dict = dict()
    c_residual_y_dict = dict()
    c_loss_payment_y_dict = dict()
    c_funding_tariff_y_dict = dict()
    c_funding_ppa_y_dict = dict()
    c_commission_kepco_y_dict = dict()
    c_commission_ppa_y_dict = dict()

    for y in set_y:
        u_y_dict[y] = u_y[y].value()

        p_eac_y_dict[y] = p_eac_y[y].value()
        capacity_pv_y_dict[y] = capacity_pv_y[y].value()
        capacity_onshore_y_dict[y] = capacity_onshore_y[y].value()
        capacity_cs_contract_y_dict[y] = capacity_cs_contract_y[y].value()
        c_sg_y_dict[y] = c_sg_y[y].value()
        c_tariff_used_y_dict[y] = c_tariff_used_y[y].value()
        c_tariff_pre_dema_y_dict[y] = c_tariff_pre_dema_y[y].value()
        c_tariff_pro_dema_y_dict[y] = c_tariff_pro_dema_y[y].value()

        c_ppa_y_dict[y] = c_ppa_y[y].value()
        c_eac_y_dict[y] = c_eac_y[y].value()
        c_residual_y_dict[y] = c_residual_y[y].value()
        c_loss_payment_y_dict[y] = c_loss_payment_y[y].value()
        c_funding_tariff_y_dict[y] = c_funding_tariff_y[y].value()
        c_funding_ppa_y_dict[y] = c_funding_ppa_y[y].value()
        c_commission_kepco_y_dict[y] = c_commission_kepco_y[y].value()
        c_commission_ppa_y_dict[y] = c_commission_ppa_y[y].value()

        p_sg_pv_y_d_h_dict[y] = dict()
        p_sg_onshore_y_d_h_dict[y] = dict()
        p_tariff_y_d_h_dict[y] = dict()
        p_ppa_pv_y_d_h_dict[y] = dict()
        p_ppa_onshore_y_d_h_dict[y] = dict()

        for d in set_d:
            p_sg_pv_y_d_h_dict[y][d] = dict()
            p_sg_onshore_y_d_h_dict[y][d] = dict()
            p_tariff_y_d_h_dict[y][d] = dict()
            p_ppa_pv_y_d_h_dict[y][d] = dict()
            p_ppa_onshore_y_d_h_dict[y][d] = dict()

            for h in set_h:
                p_sg_pv_y_d_h_dict[y][d][h] = Apv_d_h[d, h] * capacity_pv_y[y].value()
                p_sg_onshore_y_d_h_dict[y][d][h] = Aonshore_d_h[d, h] * capacity_onshore_y[y].value()
                p_tariff_y_d_h_dict[y][d][h] = p_tariff_y_d_h[y, d, h].value()
                p_ppa_pv_y_d_h_dict[y][d][h] = p_ppa_pv_y_d_h[y, d, h].value()
                p_ppa_onshore_y_d_h_dict[y][d][h] = p_ppa_onshore_y_d_h[y, d, h].value()

    result['u_y'] = u_y_dict

    result['p_sg_pv_y_d_h'] = p_sg_pv_y_d_h_dict
    result['p_sg_onshore_y_d_h'] = p_sg_onshore_y_d_h_dict

    result['p_tariff_y_d_h'] = p_tariff_y_d_h_dict
    result['p_ppa_pv_y_d_h'] = p_ppa_pv_y_d_h_dict
    result['p_ppa_onshore_y_d_h'] = p_ppa_onshore_y_d_h_dict
    result['capacity_cs_contract_y'] = capacity_cs_contract_y_dict

    result['p_eac_y'] = p_eac_y_dict
    result['capacity_pv_y'] = capacity_pv_y_dict
    result['capacity_onshore_y'] = capacity_onshore_y_dict
    result['c_sg_y'] = c_sg_y_dict
    result['c_tariff_used_y'] = c_tariff_used_y_dict
    result['c_tariff_pre_dema_y'] = c_tariff_pre_dema_y_dict
    result['c_tariff_pro_dema_y'] = c_tariff_pro_dema_y_dict

    result['c_ppa_y'] = c_ppa_y_dict
    result['c_eac_y'] = c_eac_y_dict
    result['c_residual_y'] = c_residual_y_dict
    result['c_loss_payment_y'] = c_loss_payment_y_dict
    result['c_funding_tariff_y'] = c_funding_tariff_y_dict
    result['c_funding_ppa_y'] = c_funding_ppa_y_dict

    result['c_commission_kepco_y'] = c_commission_kepco_y_dict
    result['c_commission_ppa_y'] = c_commission_ppa_y_dict

    return result